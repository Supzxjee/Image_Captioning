"""User-run CPU staging of existing caches; no neural forward or new features."""
import gzip
import json
from pathlib import Path
import shutil
import time
import numpy as np
from readout_data import *


def prepare(cfg):
    import h5py
    import torch
    from captioning.visual_cache import VisualCache
    from captioning.semantic_prompts import align_prompt_cache
    from captioning.object_reward import make_targets
    start=time.monotonic();out=Path(cfg['output']);cache_root=Path(cfg['cache'])
    if out.exists():raise FileExistsError('Existing output preserved; use a fresh session/output')
    if cfg['policy']!=POLICY or cfg['arm'] not in ('I','J'):raise ValueError('Locked protocol changed')
    for name,h in FILES.items():
        print('Hashing v105:',name,flush=True)
        if sha(cache_root/name)!=h:raise ValueError('Wrong v105 payload: '+name)
    for key,h in [('checkpoint',CHECKPOINT_SHA),('dataset_json',DATA_SHA),('detections',DETECTION_SHA),
                  ('legacy_prompt',LEGACY_SHA),('dimensions',DIMENSIONS_SHA)]:
        print('Hashing pinned input:',key,flush=True)
        if sha(cfg[key])!=h:raise ValueError('Wrong pinned input: '+key)
    if sha(Path(cfg['chair'])/'chair.pkl')!=CHAIR_SHA:raise ValueError('Wrong CHAIR cache')
    prior=vocabulary(cfg['metadata']);vocab=prior['word2idx']
    data=json.loads(Path(cfg['dataset_json']).read_bytes())
    by_coco={r['cocoid']:(i,r) for i,r in enumerate(data['images'])}
    if len(by_coco)!=len(data['images']):raise ValueError('Duplicate COCO identities')
    manifest=json.loads(gzip.decompress((cache_root/'relation_frames_manifest.bin').read_bytes()))
    rows=json.loads((cache_root/'images.json').read_bytes())
    objects=json.loads((cache_root/'detector_boxes.json').read_bytes())
    dev=json.loads(Path(cfg['dimensions']).read_bytes())['rows']
    if [{k:r[k] for k in ('image_id','coco_id','filename','split')} for r in dev]!=prior['groups']['val']:raise ValueError('Fixed dev differs')
    train_ids={p[role]['image_id'] for p in manifest['train'] for role in ('positive','negative')}
    held_ids={p[role]['image_id'] for p in manifest['heldout'] for role in ('positive','negative')}
    dev_ids={r['coco_id'] for r in dev}
    if len(train_ids)!=484 or len(held_ids)!=128 or train_ids&held_ids or (train_ids|held_ids)&dev_ids:raise ValueError('Cohort/leakage guard failed')
    det={}
    for filename,entry in json.loads(Path(cfg['detections']).read_bytes()).items():
        iid=int(Path(filename).stem.rsplit('_',1)[-1])
        if iid in dev_ids:
            if iid in det:raise ValueError('Duplicate detector image')
            det[iid]=entry['objects']
    if set(det)!=dev_ids:raise ValueError('Missing fixed-dev detector records')
    # Dimensions are verified source metadata; all model boxes remain detector predictions.
    for r in rows:r['objects']=objects[str(r['coco_id'])]
    for r in dev:rows.append({**r,'objects':bank_row(det[r['coco_id']],r['width'],r['height'])})
    for r in rows:
        image_id,source=by_coco[r['coco_id']]
        if source['filename']!=r['filename'] or (r['coco_id'] in train_ids|held_ids and source['split'] not in ('train','restval')):raise ValueError('Caption/source identity differs')
        if 'image_id' in r and r['image_id']!=image_id:raise ValueError('Karpathy index differs')
        r.update(image_id=image_id,split=source['split'],references=[s['raw'] for s in source['sentences'][:5]])
        if len(r['references'])!=5:raise ValueError('Need five references')
        region_arrays(r)
    index={r['coco_id']:i for i,r in enumerate(rows)}
    pairs={}
    for split in ('train','heldout'):
        pairs[split]=[]
        for pair in manifest[split]:
            a=assertion(pair,vocab)
            a.update(positive_index=index[a['positive']],negative_index=index[a['negative']])
            pairs[split].append(a)
            for role in ('positive','negative'):
                r=pair[role];pred={b['original_index']:b for b in rows[index[r['image_id']]]['objects']}
                for entity in ('subject','object'):
                    b=r['predicted'][entity];found=pred[b['index']]
                    if found['bbox_xyxy']!=b['bbox_xyxy'] or found['label']!=b['label']:raise ValueError('Frozen target referent differs')
    out.mkdir(parents=True)
    if shutil.disk_usage(out).free<3*1024**3:raise RuntimeError('Need3GiB output space')
    save(out/'protocol.json',cfg)
    groups=dict(train=[{k:r[k] for k in ('image_id','coco_id','filename')} for r in rows if r['coco_id'] in train_ids],
                val=[{k:r[k] for k in ('image_id','coco_id','filename')} for r in rows if r['coco_id'] not in train_ids])
    targets=make_targets(groups,data,Path(cfg['chair'])/'chair.pkl',json.loads(Path(cfg['detections']).read_bytes()),'aliases_v2','unique_count_v2')
    save(out/'object_targets.json',targets)
    source=VisualCache([cfg['visual_cache']]);source.require_ids(dev_ids,'fixed dev1000')
    expected=dict(dataset_json_sha256=DATA_SHA,image_size=[224,224],preprocessing='bilinear',
        model_id='openai/clip-vit-base-patch16',output='vision_model.last_hidden_state',
        mean=[.48145466,.4578275,.40821073],std=[.26862954,.26130258,.27577711],antialias=True,compute_dtype='float32',autocast=False)
    for path in source.paths:
        with h5py.File(path,'r') as f:
            info=json.loads(f.attrs['metadata_json'])
            if any(info.get(k)!=v for k,v in expected.items()) or str(f['features'].dtype)!='float16':raise ValueError('Visual preprocessing/schema differs')
    bundle=torch.load(cfg['legacy_prompt'],map_location='cpu',weights_only=False)
    prompts=align_prompt_cache(bundle['data'],[r['filename'] for r in dev]);del bundle
    if set(prompts)!={r['filename'] for r in dev}:raise ValueError('Dev original prompt coverage incomplete')
    try:
        with h5py.File(out/'inputs.h5','x') as f,h5py.File(cache_root/'inputs.h5','r') as original:
            f.attrs['complete']=False
            for key,shape,dtype in [('visual',(197,768),'f2'),('prompt',(20,512),'f4'),('mask',(20,),'u1')]:
                f.create_dataset(key,(1612,*shape),dtype=dtype)
                f[key][:612]=original[key][:]
            if original['coco_ids'][:].tolist()!=[r['coco_id'] for r in rows[:612]]:raise ValueError('v105 HDF5 order differs')
            f.create_dataset('coco_ids',data=[r['coco_id'] for r in rows])
            for count,r in enumerate(sorted(dev,key=lambda r:source.index[r['coco_id']]),1):
                i=index[r['coco_id']];entry=prompts[r['filename']]
                visual=source.read(r['coco_id']);p=entry['tokens'].float().numpy();mask=entry['mask'].numpy()
                if p.shape!=(20,512) or mask.shape!=(20,) or not np.isfinite(p).all() or not np.isin(mask,[0,1]).all() or not mask.any():raise ValueError('Invalid original prompt')
                f['visual'][i]=visual;f['prompt'][i]=p;f['mask'][i]=mask
                if not np.array_equal(f['visual'][i].astype('float32'),visual) or not np.array_equal(f['prompt'][i],p):raise ValueError('Copy differs from original cache')
                if count%100==0:print(f'Staged exact dev caches {count}/1000; {time.monotonic()-start:.1f}s',flush=True)
                if time.monotonic()-start>POLICY['prepare_seconds']:raise TimeoutError('CPU preparation cap exceeded')
            f.attrs['complete']=True
    finally:source.close()
    meta=dict(schema='shared-pair-readout-data-v1',rows=rows,pairs=pairs,word2idx=vocab,
        train_indices=[index[i] for i in sorted(train_ids)],dev_indices=[index[r['coco_id']] for r in dev],
        held_indices=[index[i] for i in sorted(held_ids)],baseline=prior['baseline'])
    meta['semantic_panel_pair_ids']=panel_ids(pairs['heldout'])
    if len(rows)!=1612:raise ValueError('Wrong image count')
    save(out/'metadata.json',meta)
    import hashlib
    content=hashlib.sha256()
    with h5py.File(out/'inputs.h5','r') as f:
        for key in ('coco_ids','visual','prompt','mask'):
            content.update(str((key,str(f[key].dtype),tuple(f[key].shape))).encode())
            for first in range(0,len(rows),64):content.update(f[key][first:first+64].tobytes())
    receipt=dict(status='PREPARATION_COMPLETE',images=1612,caption_train_images=484,relation_train_pairs=256,
        heldout_pairs=64,dev_images=1000,feature_extraction=False,neural_forward=False,gt_model_boxes=False,
        input_hashes=dict(cache=FILES,checkpoint=CHECKPOINT_SHA,data=DATA_SHA,detections=DETECTION_SHA,legacy=LEGACY_SHA,dimensions=DIMENSIONS_SHA),
        tensor_content_sha256=content.hexdigest(),
        payload_sha256={n:sha(out/n) for n in ('inputs.h5','metadata.json','object_targets.json')},seconds=time.monotonic()-start)
    save(out/'preparation.json',receipt)
    print('Full preparation COMPLETE; no model forward. Ready for user-run model preflight.',flush=True)
