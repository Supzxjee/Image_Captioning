"""TRAIN-only raw-patch smoke stage; importing never imports a neural library."""
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from data import vocabulary,encode
from real_bank_data import sha,save,read

DATA_SHA='2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
COUNTS=dict(train=82783,restval=30504,val=5000,test=5000)

def selected_rows(dataset,count=256):
    from collections import Counter
    rows=dataset['images']
    if dict(Counter(r['split'] for r in rows))!=COUNTS:raise ValueError('Wrong Karpathy splits')
    ids=[int(r['cocoid']) for r in rows]
    if len(set(ids))!=len(ids):raise ValueError('Duplicate COCO ID')
    pool=[r for r in rows if r['split'] in ('train','restval')]
    if not 1<=count<=len(pool):raise ValueError('Invalid TRAIN smoke size')
    return sorted(pool,key=lambda r:(hashlib.sha256(f"patch-baseline/42/{r['cocoid']}".encode()).hexdigest(),int(r['cocoid'])))[:count]

def check_features(raw):
    if raw.ndim!=3 or raw.shape[1:]!=(197,768) or not np.isfinite(raw).all():raise ValueError('Expected finite raw [B,197,768]')
    if raw.dtype.kind!='f':raise ValueError('Expected floating raw features')
    return raw[:,1:,:].astype(np.float16)

def batch(data,events):
    events=np.asarray(events,dtype=np.int64)
    if events.ndim!=1 or not len(events) or np.any(events<0) or np.any(events>=len(data['sequences'])):raise ValueError('Invalid caption event')
    seq=np.asarray(data['sequences'][events],np.int64);seq=seq[:,:int((seq!=0).sum(1).max())]
    return np.asarray(data['patches'][data['caption_image'][events]],np.float32),seq

def prepare(cfg):
    import h5py
    started=time.monotonic();out=Path(cfg['output']);stage=Path(cfg['stage']);stage.mkdir(parents=True,exist_ok=False)
    if sha(cfg['dataset_json'])!=DATA_SHA:raise ValueError('Changed COCO JSON')
    dataset=read(cfg['dataset_json']);rows=selected_rows(dataset);words=vocabulary(dataset)
    if len(words)!=10000:raise ValueError('Wrong TRAIN vocabulary size')
    wanted={int(r['cocoid']):j for j,r in enumerate(rows)};found=set();profile=None;shards=[]
    patches=np.empty((len(rows),196,768),np.float16);row_hashes={}
    paths=sorted(Path(cfg['visual_cache']).rglob('*.h5'))
    if not paths:raise FileNotFoundError('Add raw visual-cache Dataset; ROI134 cannot replace raw patches')
    for path in paths:
        if time.monotonic()-started>cfg['prepare_seconds']:raise TimeoutError('Raw-cache scan/staging cap exceeded')
        with h5py.File(path,'r') as f:
            text=lambda x:x.decode() if isinstance(x,bytes) else str(x)
            if not bool(f.attrs.get('complete',False)) or text(f.attrs.get('id_key',''))!='coco_id':raise ValueError('Incomplete or ambiguous cache IDs')
            meta=json.loads(text(f.attrs['metadata_json']))
            expected=dict(model_id='openai/clip-vit-base-patch16',output='vision_model.last_hidden_state',preprocessing='bilinear',image_size=[224,224],antialias=True,dataset_json_sha256=DATA_SHA)
            if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('Raw feature profile differs')
            np.testing.assert_allclose(meta['mean'],[.48145466,.4578275,.40821073],rtol=0,atol=1e-8)
            np.testing.assert_allclose(meta['std'],[.26862954,.26130258,.27577711],rtol=0,atol=1e-8)
            current={k:meta.get(k) for k in (*expected,'model_revision','mean','std','compute_dtype','storage_dtype','autocast')}
            if profile is not None and current!=profile:raise ValueError('Mixed raw extraction metadata')
            profile=current;ids=f['imgids'][:];features=f['features'];names=f['filename']
            if features.shape!=(len(ids),197,768) or features.dtype.kind!='f' or features.dtype.itemsize not in (2,4) or len(names)!=len(ids):raise ValueError('Raw cache shape/dtype differs')
            positions=[i for i,v in enumerate(ids) if int(v) in wanted]
            if positions:
                raw=features[positions];values=check_features(raw)
                for pos,full,value in zip(positions,raw,values):
                    coco=int(ids[pos]);j=wanted[coco]
                    if coco in found or text(names[pos])!=rows[j]['filename']:raise ValueError('Duplicate/mismatched selected raw ID')
                    found.add(coco);patches[j]=value;row_hashes[str(coco)]=hashlib.sha256(np.ascontiguousarray(full).tobytes()).hexdigest()
            shards.append(dict(name=path.name,bytes=path.stat().st_size,shape=list(features.shape),selected_rows=len(positions)))
        print('Raw patch scan:',path.name,'TRAIN extracted',len(found),'/',len(rows),flush=True)
    if found!=set(wanted):raise ValueError('Missing selected TRAIN features')
    sequences=[];caption_image=[]
    for j,r in enumerate(rows):
        for sentence in r['sentences']:
            sequences.append(encode(sentence['tokens'],words));caption_image.append(j)
    arrays=dict(patches=patches,sequences=np.asarray(sequences,np.int32),caption_image=np.asarray(caption_image,np.int32),coco_ids=np.asarray([int(r['cocoid']) for r in rows],np.int64))
    for k,v in arrays.items():np.save(stage/(k+'.npy'),v)
    save(stage/'vocabulary.json',words)
    manifest=dict(status='PATCH_TRAIN_SMOKE_STAGE_COMPLETE',images=len(rows),caption_events=len(sequences),split='TRAIN_only',test_used=False,validation_used=False,
        dataset_sha256=DATA_SHA,feature_profile=profile,shards=shards,raw_selected_row_sha256=row_hashes,
        files_sha256={p.name:sha(p) for p in stage.iterdir() if p.is_file()},seconds=time.monotonic()-started,
        limits='Inherited producer/revision/preprocessing; no full H5 payload hash or new CLIP extraction. Only selected TRAIN feature rows read.')
    save(stage/'manifest.json',manifest);save(out/'stage_summary.json',manifest)

def load_stage(root):
    root=Path(root);m=read(root/'manifest.json')
    if m['status']!='PATCH_TRAIN_SMOKE_STAGE_COMPLETE' or m['images']!=256 or m['test_used'] or m['validation_used']:raise ValueError('Wrong stage scope')
    for n,h in m['files_sha256'].items():
        if Path(n).name!=n or sha(root/n)!=h:raise ValueError('Changed stage '+n)
    data={k:np.load(root/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in ('patches','sequences','caption_image','coco_ids')}
    return data,read(root/'vocabulary.json'),m
