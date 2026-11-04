"""Data-only CSA SCST contracts. No neural imports."""
import hashlib
from pathlib import Path
import numpy as np
from real_bank_data import read, save, sha
from full_cache_data import stage_full
from full_xe_data import validation_references

CHECKPOINT_SHA='e02593ea079bdb563a1afe0c5898abdbf81297aa5e688b3b0df839cb820e5b25'
DATA_SHA='2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
DETECTION_SHA='498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c'
MODEL_SHA='139edbb1b73caf3dadaf5c960b25542e282f88baf2ce9bdf141de922ceacbc09'


def exact_checkpoint(root):
    root=Path(root)
    candidates=[root] if root.is_file() else [p for p in root.rglob('*.bin') if p.name in ('best.bin','checkpoint.bin')]
    matches=[p for p in candidates if sha(p)==CHECKPOINT_SHA]
    if not matches:raise ValueError('Attach exact CSA15 SaveVersion147 weights, SHA '+CHECKPOINT_SHA)
    return sorted(matches)[0] # best/latest15 are byte-identical; either is valid.


def study_updates(cfg):
    value=cfg.get('updates',500)
    if type(value)!=int or value not in (500,2500):raise ValueError('Only approved500/2500 horizons supported')
    return value


def schedule(train_ids, pool=10000, updates=500, batch=8):
    ids=[int(i) for i in train_ids]
    study_updates(dict(updates=updates))
    if pool!=10000 or batch!=8 or len(set(ids))!=len(ids) or len(ids)<pool:raise ValueError('Invalid pilot TRAIN IDs/budget')
    selected=sorted(ids,key=lambda i:hashlib.sha256(f'42:{i}'.encode()).hexdigest())[:pool]
    ordered=np.concatenate([np.random.default_rng(42+round_index).permutation(selected)
        for round_index in range((updates*batch+pool-1)//pool)])
    return selected,ordered[:updates*batch].reshape(updates,batch)


def event_schedule(data, ids):
    index={int(i):j for j,i in enumerate(data['ids'])}
    result=np.empty(ids.shape,np.int64)
    first_reference={};visits={}
    for step,row in enumerate(ids):
        for j,coco in enumerate(row):
            image=index[int(coco)]
            if data['group'][image]!=0:raise ValueError('Held-out image in SCST schedule')
            lo,hi=np.searchsorted(data['caption_image'],[image,image+1])
            if hi<=lo:raise ValueError('Missing CE reference')
            key=int(coco);first_reference.setdefault(key,(step+j)%(hi-lo))
            visit=visits.get(key,0);result[step,j]=int(lo+(first_reference[key]+visit)%(hi-lo))
            visits[key]=visit+1
    return result


def prepare(cfg):
    from captioning.object_reward import make_targets
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    ck=exact_checkpoint(cfg['checkpoint_root'])
    for key,h in [('dataset_json',DATA_SHA),('detections',DETECTION_SHA)]:
        if sha(cfg[key])!=h:raise ValueError('Changed pinned input: '+key)
    dataset=read(cfg['dataset_json']);train=[];heldout=[]
    for i,r in enumerate(dataset['images']):
        coco=int(Path(r['filename']).stem.split('_')[-1])
        row=dict(image_id=i,coco_id=coco,filename=r['filename'])
        (train if r['split'] in ('train','restval') else heldout).append(row)
    if len(train)!=113287:raise ValueError('Wrong Karpathy TRAIN count')
    updates=study_updates(cfg)
    pool,ids=schedule([r['coco_id'] for r in train],updates=updates);wanted=set(pool)
    targets=make_targets(dict(train=[r for r in train if r['coco_id'] in wanted],val=heldout),
        dataset,Path(cfg['chair'])/'chair.pkl',read(cfg['detections']),
        parser_version='aliases_v2',penalty_version='unique_count_v2')
    # Reward/scorer key is COCO ID; preserve original dataset index separately.
    for r in targets['data']:r['dataset_index']=r['image_id'];r['image_id']=r['coco_id']
    refs={str(r['coco_id']):[s['raw'] for s in dataset['images'][r['image_id']]['sentences'][:5]] for r in train}
    if any(len(v)!=5 for v in refs.values()):raise ValueError('Five TRAIN reward references required')
    del dataset
    data,vocab,staged=stage_full(cfg['full_root'],cfg['stage'])
    events=event_schedule(data,ids)
    if not np.array_equal(data['ids'][data['caption_image'][events]],ids):raise ValueError('Schedule join differs')
    save(out/'object_targets.json',targets);save(out/'training_references.json',refs)
    save(out/'validation_references.json',validation_references(cfg['full_root']));save(out/'vocabulary.json',vocab)
    np.save(out/'events.npy',events);np.save(out/'coco_ids.npy',ids)
    files=['object_targets.json','training_references.json','validation_references.json','vocabulary.json','events.npy','coco_ids.npy']
    receipt=dict(status='CSA_SCST_STAGE_COMPLETE',checkpoint_sha256=CHECKPOINT_SHA,
        dataset_json_sha256=DATA_SHA,detections_sha256=DETECTION_SHA,pool_ids=pool,
        train_images=113287,pool_images=10000,unique_exposed=len(np.unique(ids)),updates=updates,global_batch=8,
        files_sha256={n:sha(out/n) for n in files},source_stage=staged['files_sha256'])
    if updates==2500:
        unique,counts=np.unique(ids,return_counts=True)
        if len(unique)!=10000 or not (counts==2).all():raise ValueError('Expected two exposures perTRAINpool image')
        receipt.update(image_events=20000,image_exposure_histogram={'2':10000},
            sampler='two_complete_permutations_seed42_43_reference_rotation')
    save(out/'pilot_stage.json',receipt)
    print(f'SCST staged:{updates}updates/global8/{len(np.unique(ids))}unique TRAIN images; fixed full-TRAIN reward IDF.',flush=True)
