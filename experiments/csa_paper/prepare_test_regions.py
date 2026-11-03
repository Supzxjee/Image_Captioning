"""USER-RUN CPU pooling of fixed KarpathyTEST raw cachedCLIP; no neural model."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import time
import numpy as np
from real_bank_data import HERE,read,save,sha,validate_arrays
from audit_assets import DATA_SHA,DETECTION_SHA,detection_index,coco_id,text
from prepare_real_assets import visual_index,pack_regions,VOCAB_SHA
from data import roi_pool,encode
from run_best9_decode import saved_text


def selected_test(dataset):
    if Counter(r['split'] for r in dataset['images'])!=dict(train=82783,restval=30504,val=5000,test=5000):raise ValueError('Not full Karpathy split')
    ids=[int(r['cocoid']) for r in dataset['images']]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate split IDs')
    test=[r for r in dataset['images'] if r['split']=='test'];test.sort(key=lambda r:int(r['cocoid']))
    if len(test)!=5000 or any(coco_id(r['filename'])!=int(r['cocoid']) for r in test):raise ValueError('Wrong TEST IDs')
    return test


def prepare(c):
    import h5py
    from PIL import Image
    start=time.monotonic();out=Path(c['output'])
    if (out/'test_receipt.json').exists():raise ValueError('Preserve existing TEST cache')
    if sha(c['dataset_json'])!=DATA_SHA or sha(c['detections'])!=DETECTION_SHA:raise ValueError('Changed split/detector source')
    original,_=saved_text();raw_vocab=original['vocabulary.json']
    if hashlib.sha256(raw_vocab).hexdigest()!=VOCAB_SHA:raise ValueError('Shared vocabulary differs')
    import json
    vocab=json.loads(raw_vocab);(out/'vocabulary.json').write_bytes(raw_vocab)
    rows=selected_test(read(c['dataset_json']));objects=detection_index(read(c['detections']))
    index,profile=visual_index(c['visual_cache'],[])
    expected=read(HERE/'full_roi_review_text/input_receipt.json')['feature_profile']
    if profile!=expected:raise ValueError('TEST rawCLIP profile differs from TRAIN/VALID')
    records=[];metadata=[];handles={}
    try:
        for number,row in enumerate(rows,1):
            if time.monotonic()-start>3600:raise TimeoutError('TEST preparation1hcap')
            name=row['filename'];identity=int(row['cocoid'])
            if identity not in index or name not in objects:raise ValueError('Missing TEST feature/detector '+name)
            path,i=index[identity]
            if path not in handles:handles[path]=h5py.File(path,'r')
            f=handles[path]
            if text(f['filename'][i])!=name:raise ValueError('Wrong TEST raw row')
            raw=f['features'][i]
            if raw.shape!=(197,768) or not np.isfinite(raw).all():raise ValueError('Invalid TEST features')
            root=Path(c['image_root']);directory=row.get('filepath','train2014' if 'train2014' in name else 'val2014')
            image=next((p for p in (root/directory/name,root/name) if p.is_file()),None)
            if image is None:raise FileNotFoundError('JPEG dimensions missing: '+name)
            with Image.open(image) as im:width,height=im.size
            detected=objects[name];features,boxes,kinds=roi_pool(raw.astype(np.float32)[1:].reshape(14,14,768),detected,width,height)
            chosen=sorted(enumerate(detected),key=lambda x:(-float(x[1]['conf']),x[0]))[:150]
            scores=np.asarray([float(o['conf']) for _,o in chosen] if chosen else [0.],np.float32)
            records.append((features,boxes,scores,identity))
            metadata.append(dict(coco_id=identity,filename=name,source_split='test',group='test',width=width,height=height,
                region_kinds=kinds,raw_selected_row_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                captions=[s['raw'] for s in row['sentences']],sequences=[encode(s['tokens'],vocab,maximum=54) for s in row['sentences']]))
            if number%100==0:print('TEST ROI',number,'/5000 seconds',round(time.monotonic()-start,1),flush=True)
    finally:
        for f in handles.values():f.close()
    arrays=pack_regions(records);validate_arrays(arrays,metadata,vocab);np.savez_compressed(out/'test_regions.npz',**arrays)
    save(out/'test_metadata.json',metadata)
    receipt=dict(status='KARPATHY_TEST_ROI_COMPLETE',images=5000,group='test',dataset_json_sha256=DATA_SHA,detector_sha256=DETECTION_SHA,
        feature_profile=profile,full_h5_payload_hashed=False,neural_extraction=False,training_images=0,validation_images=0,
        files={n:sha(out/n) for n in ('test_regions.npz','test_metadata.json','vocabulary.json')},
        test_id_order_sha256=hashlib.sha256(str([r['coco_id'] for r in metadata]).encode()).hexdigest(),seconds=time.monotonic()-start)
    save(out/'test_receipt.json',receipt);print('TEST ROI CACHE COMPLETE; preserve NPZ/metadata/receipt/vocabulary.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);prepare(read(p.parse_args().config))
