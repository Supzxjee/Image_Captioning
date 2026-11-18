"""Fixed TEST199 metadata/staging contracts; no neural imports."""
from collections import Counter
import hashlib
from pathlib import Path
import time
import numpy as np
from real_bank_data import read,save,sha
from data import vocabulary
from full_data import encode_tokens
from patch_data import DATA_SHA,COUNTS

HERE=Path(__file__).resolve().parent;TEST_SIZE=5000
CHECKPOINT_SHA='5061e305d98edf54930a200090a41c983c354d1ba2dfcf700b715cf20a5b80b7'
VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/199'

def runtime_policy(training,current):
    if training!='2.11.0+cu128' or current not in ('2.11.0+cu128','2.11.0+cu130'):
        raise RuntimeError('Unreviewed Torch runtime; expected training2.11.0+cu128 and CUDA inference2.11.0+cu128/cu130. Preserve log; do not blindly reinstall.')
    return dict(training_torch=training,inference_torch=current,cuda_wheel_changed=training!=current,
        policy='Observed CUDA builds of the same publicTorch2.11.0; strictweights and GPU beam-path checks required.',
        bitwise_cross_build_equivalence_established=False)

def validate_lock(lock):
    if lock['source_version']!=VERSION or lock['checkpoint_sha256']!=CHECKPOINT_SHA or lock['epoch']!=10 or lock['updates']!=177110:raise ValueError('Wrong fixed199 checkpoint')
    if lock['beam_width']!=5 or lock['max_new_tokens']!=20 or lock['length_normalization'] or lock['reranking']:raise ValueError('Changed TEST decoding rule')
    ids=lock['expected_test_ids']
    if len(ids)!=TEST_SIZE or ids!=sorted(set(ids)) or any(type(x)!=int for x in ids):raise ValueError('Wrong fixed TEST cohort')
    import json
    if hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()!=lock['test_id_sha256']:raise ValueError('Changed cohort SHA')
    return lock

def lock():return validate_lock(read(HERE/'TEST199_LOCK.json'))

def checkpoint(root):
    root=Path(root);files=[root] if root.is_file() else sorted([*root.rglob('best.bin'),*root.rglob('checkpoint.bin')],key=lambda p:(p.name!='best.bin',str(p)))
    matches=[p for p in files if p.is_file() and sha(p)==CHECKPOINT_SHA]
    if not matches:raise ValueError('Need exact199 checkpoint bytes; no latest/other version fallback')
    return matches[0]

def selected_test_rows(ds,locked):
    if dict(Counter(r['split'] for r in ds['images']))!=COUNTS:raise ValueError('Wrong Karpathy split counts')
    if len({int(r['cocoid']) for r in ds['images']})!=len(ds['images']):raise ValueError('Duplicate source COCO IDs')
    rows=sorted([r for r in ds['images'] if r['split']=='test'],key=lambda r:int(r['cocoid']))
    if [int(r['cocoid']) for r in rows]!=locked['expected_test_ids']:raise ValueError('Wrong TEST IDs')
    return rows

def prepare(cfg,locked=None,weight_path=None):
    import h5py
    started=time.monotonic();out=Path(cfg['output']);stage=Path(cfg['stage']);stage.mkdir(parents=True,exist_ok=False);locked=lock() if locked is None else locked;weight_path=checkpoint(cfg['checkpoint_root']) if weight_path is None else weight_path
    if sha(cfg['dataset_json'])!=DATA_SHA:raise ValueError('Changed COCO dataset JSON')
    ds=read(cfg['dataset_json']);rows=selected_test_rows(ds,locked);wanted={int(r['cocoid']):i for i,r in enumerate(rows)};words=vocabulary(ds);lookup={w:i for i,w in enumerate(words)}
    gold=read(HERE/'SMOKE_SOURCE.json');save(out/'vocabulary.json',words)
    if sha(out/'vocabulary.json')!=gold['vocabulary_sha256']:raise ValueError('TRAIN vocabulary differs from199')
    save(out/'test_references.json',[dict(coco_id=int(r['cocoid']),references=[s['raw'] for s in r['sentences']]) for r in rows])
    sequences=[];caption_image=[]
    for i,r in enumerate(rows):
        for s in r['sentences']:sequences.append(encode_tokens(s['tokens'],lookup));caption_image.append(i)
    np.save(stage/'sequences.npy',np.asarray(sequences,np.int32));np.save(stage/'caption_image.npy',np.asarray(caption_image,np.int32));np.save(stage/'ids.npy',np.asarray([r['cocoid'] for r in rows],np.int64))
    patches=np.lib.format.open_memmap(stage/'patches.npy',mode='w+',dtype=np.float16,shape=(len(rows),196,768));seen=set();sources=[]
    try:
        for path in sorted(Path(cfg['visual_cache']).rglob('*.h5')):
            with h5py.File(path,'r',rdcc_nbytes=2*1024**2) as f:
                text=lambda x:x.decode() if isinstance(x,bytes) else str(x)
                import json
                if not f.attrs.get('complete',False) or text(f.attrs.get('id_key',''))!='coco_id':raise ValueError('Incomplete/ambiguous H5')
                meta=json.loads(text(f.attrs['metadata_json']))
                if {k:meta.get(k) for k in gold['feature_profile']}!=gold['feature_profile']:raise ValueError('Feature profile differs from199')
                ids=f['imgids'][:];features=f['features']
                if features.shape!=(len(ids),197,768) or features.dtype!=np.float16:raise ValueError('Invalid original H5 shape/dtype')
                positions=[i for i,x in enumerate(ids) if int(x) in wanted]
                for start in range(0,len(positions),128):
                    pos=positions[start:start+128];raw=features[pos]
                    if not np.isfinite(raw).all():raise ValueError('Nonfinite TEST patches')
                    for j,p in enumerate(pos):
                        coco=int(ids[p]);image=wanted[coco]
                        if coco in seen or text(f['filename'][p])!=rows[image]['filename']:raise ValueError('Repeated TEST ID/filename mismatch')
                        values=raw[j,1:];patches[image]=values
                        if not np.array_equal(patches[image].view(np.uint16),values.view(np.uint16)):raise ValueError('TEST patch bytes changed')
                        seen.add(coco)
                    print('TEST patches staged',len(seen),'/',len(rows),round(time.monotonic()-started),'s',flush=True)
                sources.append(dict(path=str(path),bytes=path.stat().st_size,selected_TEST_rows=len(positions)))
        if seen!=set(wanted):raise ValueError('Missing TEST features')
        patches.flush()
    finally:patches._mmap.close()
    receipt=dict(status='PATCH_TEST199_STAGE_COMPLETE',images=len(rows),caption_events=len(sequences),feature_profile=gold['feature_profile'],sources=sources,
        dataset_sha256=DATA_SHA,training=False,TRAIN_VALID_features_read=False,seconds=time.monotonic()-started,
        stage_files_sha256={p.name:sha(p) for p in stage.iterdir() if p.is_file()},
        output_files_sha256={n:sha(out/n) for n in ('vocabulary.json','test_references.json')})
    save(out/'stage_receipt.json',receipt);save(out/'test_lock.json',dict(**locked,checkpoint=str(weight_path),stage_receipt_sha256=sha(out/'stage_receipt.json')))

def load(cfg,verify=True,validator=None):
    out=Path(cfg['output']);stage=Path(cfg['stage']);r=read(out/'stage_receipt.json');locked=(validator or validate_lock)(read(out/'test_lock.json'))
    if r['status']!='PATCH_TEST199_STAGE_COMPLETE' or r['images']!=TEST_SIZE or r['TRAIN_VALID_features_read']:raise ValueError('Invalid TEST stage scope')
    if locked['stage_receipt_sha256']!=sha(out/'stage_receipt.json'):raise ValueError('Changed TEST receipt')
    if verify:
        for n,h in r['stage_files_sha256'].items():
            if Path(n).name!=n or sha(stage/n)!=h:raise ValueError('Changed staged TEST '+n)
        for n,h in r['output_files_sha256'].items():
            if Path(n).name!=n or sha(out/n)!=h:raise ValueError('Changed TEST output '+n)
    return {n:np.load(stage/(n+'.npy'),mmap_mode='r',allow_pickle=False) for n in ('patches','ids','sequences','caption_image')},read(out/'vocabulary.json'),r,locked
