"""User-run CPU full Karpathy ROI preparation. No neural imports/execution."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from audit_assets import DATA_SHA,DETECTION_SHA,coco_id,detection_index,sha,text
from data import encode,roi_pool
from prepare_real_assets import visual_index,pack_regions,VOCAB_SHA
from real_bank_data import read,save

def select_full(dataset):
    counts=Counter(x['split'] for x in dataset['images'])
    if counts!=dict(train=82783,restval=30504,val=5000,test=5000):raise ValueError('Karpathy split counts differ')
    seen=set();groups=dict(train=[],val=[])
    for row in dataset['images']:
        identity=coco_id(row['filename'])
        if identity!=int(row['cocoid']) or identity in seen:raise ValueError('Duplicate/mismatched COCO ID')
        seen.add(identity)
        if row['split'] in ('train','restval'):groups['train'].append(row)
        elif row['split']=='val':groups['val'].append(row)
    return groups

def chunks(rows,size=1000):
    return [rows[i:i+size] for i in range(0,len(rows),size)]

def prepare(config):
    import h5py
    from PIL import Image
    started=time.monotonic();deadline=started+35400
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists() or (out/'progress.json').exists():raise ValueError('Preserve existing shards; this worker does not auto-resume')
    for name,h in [('dataset_json',DATA_SHA),('detections',DETECTION_SHA)]:
        print('Hashing pinned input:',name,flush=True)
        if sha(config[name])!=h:raise ValueError('Changed input: '+name)
    words_root=Path(config['prepared_root'])
    candidates=[p for p in words_root.rglob('vocabulary.json') if sha(p)==VOCAB_SHA]
    if len(candidates)!=1:raise ValueError('Expected unique original vocabulary')
    vocab=read(candidates[0]);shutil.copyfile(candidates[0],out/'vocabulary.json')
    if len(vocab)!=10000:raise ValueError('Vocabulary size differs')
    dataset=read(config['dataset_json']);groups=select_full(dataset)
    detections=detection_index(read(config['detections']))
    # Index metadata once. Per-image filenames are checked below using cached handles,
    # avoiding118287 separate HDF5 opens in the small-pilot selected-row checker.
    index,profile=visual_index(config['visual_cache'],[])
    for group,rows in groups.items():
        for row in rows:
            if coco_id(row['filename']) not in index or row['filename'] not in detections:raise ValueError('Missing feature/detector record')
        rows.sort(key=lambda x:(str(index[coco_id(x['filename'])][0]),index[coco_id(x['filename'])][1]))
    save(out/'input_receipt.json',dict(dataset_json_sha256=DATA_SHA,detections_sha256=DETECTION_SHA,vocabulary_sha256=VOCAB_SHA,
        feature_profile=profile,full_h5_payload_hashed=False,bank_reencoded=False,test_selected=False,
        source=read(out/'source_receipt.json'),order='WithinTRAIN/val rawH5path,row for sequential I/O'))
    progress=dict(status='RUNNING',images=0,train_images=0,val_images=0,test_images=0,regions=0,caption_events=0,clipped_boxes=0,
        fallback_images=0,shards=[],max_regions=0)
    save(out/'progress.json',progress);handles={};processing=time.monotonic();shard_number=0
    try:
        for group,rows in groups.items():
            for local_shard,selected in enumerate(chunks(rows)):
                records=[];metadata=[]
                for image in selected:
                    if time.monotonic()>deadline:raise TimeoutError('Internal9h50cap; completed shards retained')
                    name=image['filename'];identity=coco_id(name);path,row=index[identity]
                    if path not in handles:handles[path]=h5py.File(path,'r')
                    f=handles[path]
                    if text(f['filename'][row])!=name:raise ValueError('Raw row filename differs')
                    directory=image.get('filepath','train2014' if 'train2014' in name else 'val2014')
                    image_paths=[Path(config['image_root'])/directory/name,Path(config['image_root'])/name]
                    actual=next((p for p in image_paths if p.is_file()),None)
                    if actual is None:raise FileNotFoundError('RGB dimensions unavailable: '+name)
                    with Image.open(actual) as im:width,height=im.size
                    raw=f['features'][row]
                    if raw.shape!=(197,768) or not np.isfinite(raw).all():raise ValueError('Invalid raw feature row')
                    objects=detections[name];features,boxes,kinds=roi_pool(raw.astype(np.float32)[1:].reshape(14,14,768),objects,width,height)
                    chosen=sorted(enumerate(objects),key=lambda x:(-float(x[1]['conf']),x[0]))[:150]
                    scores=np.asarray([float(o['conf']) for _,o in chosen] if chosen else [0.],np.float32)
                    clipped=sum(any(v<0 for v in o['bbox'][:2]) or o['bbox'][2]>width or o['bbox'][3]>height for o in objects)
                    records.append((features,boxes,scores,identity))
                    sequences=[encode(s['tokens'],vocab,maximum=54) for s in image['sentences']]
                    metadata.append(dict(coco_id=identity,filename=name,source_split=image['split'],group=group,width=width,height=height,
                        image_dimensions_source='JPEG header, no pixelhash',raw_selected_row_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                        raw_selected_row_dtype=str(raw.dtype),region_kinds=kinds,source_detections=len(objects),boxes_needing_boundary_clip=clipped,
                        captions=[s['raw'] for s in image['sentences']],sequences=sequences))
                    progress['clipped_boxes']+=clipped;progress['fallback_images']+=int(not objects)
                    if (progress['images']+len(records))%100==0:print('ROI',group,progress['images']+len(records),'/118287',flush=True)
                prefix=f'{group}_{local_shard:04d}';payload=pack_regions(records)
                np.savez_compressed(out/(prefix+'.npz'),**payload)
                save(out/(prefix+'.json'),dict(schema='csa_full_roi_shard_v1',group=group,images=metadata))
                entry=dict(group=group,images=len(selected),regions=int(payload['offsets'][-1]),
                    first_id=metadata[0]['coco_id'],last_id=metadata[-1]['coco_id'],
                    id_order_sha256=hashlib.sha256(json.dumps([m['coco_id'] for m in metadata]).encode()).hexdigest(),
                    files={prefix+s:sha(out/(prefix+s)) for s in ('.npz','.json')})
                progress['shards'].append(entry);progress['images']+=len(selected);progress[group+'_images']+=len(selected)
                progress['regions']+=entry['regions'];progress['caption_events']+=sum(len(m['sequences']) for m in metadata)
                progress['max_regions']=max(progress['max_regions'],max(np.diff(payload['offsets']).tolist()))
                progress['worker_seconds']=time.monotonic()-started;save(out/'progress.json',progress);shard_number+=1
                print('SHARD COMPLETE',prefix,progress['images'],'/118287',flush=True)
                if shard_number==1:
                    elapsed=time.monotonic()-processing;remaining=elapsed/len(selected)*(118287-len(selected))
                    estimate=dict(first_shard_images=len(selected),processing_seconds=elapsed,projected_remaining_seconds=remaining,
                        excludes_setup=True,guaranteed=False)
                    save(out/'throughput.json',estimate);print('First-shard projection:',estimate,flush=True)
                    if remaining>deadline-time.monotonic():raise RuntimeError('Measured projection exceeds remaining CPUcap; preserve first shard/review before continuation')
        if (progress['images'],progress['train_images'],progress['val_images'],len(progress['shards']))!=(118287,113287,5000,119):raise ValueError('Incomplete fullcoverage')
        progress['status']='FULL_ROI_PREPARATION_COMPLETE';progress['worker_seconds']=time.monotonic()-started
        progress['payload_bytes']=sum((out/n).stat().st_size for e in progress['shards'] for n in e['files'])
        progress['vocabulary_sha256']=VOCAB_SHA;progress['feature_profile']=profile
        save(out/'progress.json',progress);save(out/'manifest.json',progress)
        save(out/'receipt.json',dict(status='FULL_ROI_PREPARATION_COMPLETE',manifest_sha256=sha(out/'manifest.json'),
            vocabulary_sha256=VOCAB_SHA,input_receipt_sha256=sha(out/'input_receipt.json'),images=118287,test_images=0))
        print('FULL ROI COMPLETE. Preserve entire output folder.',flush=True)
    finally:
        for handle in handles.values():handle.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args();prepare(read(args.config))
