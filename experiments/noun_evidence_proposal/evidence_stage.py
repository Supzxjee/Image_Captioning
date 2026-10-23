"""Pinned data preparation and persistent HDF5 loaders. No model runs on import."""
from collections import defaultdict
import json
from pathlib import Path
import shutil
import time
import numpy as np
from evidence_data import A_SHA,DATA_SHA,digest,save,validate_package,encoded

INPUTS=dict(checkpoint=A_SHA,dataset_json=DATA_SHA,
    legacy_prompt='d7910e9bf87e3aad86d806c77ec336b34793e3bef68a3f662fdcb2e9456ca38d',
    detections='498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c')
CHAIR_SHA='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3'


def verify_sources(cfg):
    for name,expected in cfg['implementation_sha256'].items():
        if digest(Path(__file__).parent/name)!=expected:raise ValueError('Changed source: '+name)


def package(cfg):
    root=Path(cfg['labels']);receipt=validate_package(root,cfg['label_receipt_sha256'])
    values={k:json.loads((root/(k+'.json')).read_text()) for k in ('groups','lexicon','targets','prior')}
    groups,targets=values['groups'],values['targets']
    if len(groups['train'])!=10000 or len(groups['val'])!=5000:raise ValueError('Cohort size changed.')
    if set(targets)!={str(r['image_id']) for r in groups['train']}:raise ValueError('Labels must be TRAIN only.')
    if {r['coco_id'] for r in groups['train']}&{r['coco_id'] for r in groups['val']}:raise ValueError('TRAIN/val overlap.')
    y=np.asarray([v['y'] for v in targets.values()]);o=np.asarray([v['observed'] for v in targets.values()])
    if y.shape!=(10000,80) or not np.isin(y,[0,1]).all() or not np.isin(o,[0,1]).all() or (y>o).any():raise ValueError('Invalid presence.')
    if not np.array_equal(y.sum(0),values['prior']['positive']) or not np.array_equal(o.sum(0),values['prior']['observed']):raise ValueError('Prior counts changed.')
    if not np.allclose((y.sum(0)+.5)/(o.sum(0)+1),values['prior']['probability'],rtol=0,atol=0):raise ValueError('Prior formula differs.')
    region_count=0
    for target in targets.values():
        for r in target['regions']:
            mask=np.asarray(r['mask'])
            if mask.shape!=(196,) or not np.isfinite(mask).all() or not ((mask>=0)&(mask<=1)).all() or mask.sum()<=0:raise ValueError('Invalid fractional mask.')
            if not 0<=r['class_id']<80 or not target['y'][r['class_id']]:raise ValueError('Invalid region class.')
            region_count+=1
    if region_count!=5602:raise ValueError('Expected 5602 unique regions.')
    return receipt,values


def prepare(config):
    import hashlib
    import h5py
    import torch
    from captioning.visual_cache import VisualCache
    from captioning.semantic_prompts import align_prompt_cache
    from captioning.object_reward import make_targets,_AnnotationReader
    from run_pilot import corpus
    cfg=json.loads(Path(config).read_text());verify_sources(cfg)
    if cfg['arm'] not in ('G','S') or cfg['updates']!=2500:raise ValueError('Locked recipe differs.')
    out,stage=Path(cfg['output']),Path(cfg['stage'])
    if out.exists() or stage.exists():raise FileExistsError('Preserve old output/stage; use a fresh session.')
    out.mkdir(parents=True);begin=time.monotonic()
    for key,expected in INPUTS.items():
        print('Hashing pinned input:',key,flush=True)
        if digest(cfg[key])!=expected:raise ValueError('Pinned input differs: '+key)
    if digest(Path(cfg['chair'])/'chair.pkl')!=CHAIR_SHA:raise ValueError('Different CHAIR cache.')
    receipt,values=package(cfg);groups=values['groups']
    dataset,tokenizer=corpus(cfg)
    vocab_sha=hashlib.sha256(json.dumps(tokenizer.word2idx,sort_keys=True).encode()).hexdigest()
    if vocab_sha!=receipt['vocabulary_sha256']:raise ValueError('A vocabulary differs from lexical tables.')
    for split,rows in groups.items():
        for r in rows:
            item=dataset['images'][r['image_id']]
            if item['cocoid']!=r['coco_id'] or item['filename']!=r['filename'] or (item['split']=='val')!=(split=='val'):raise ValueError('ID/split mismatch.')
    with (Path(cfg['chair'])/'chair.pkl').open('rb') as stream:annotation=_AnnotationReader(stream).load()
    if any(r['coco_id'] not in annotation.imid_to_objects for r in groups['val']):raise ValueError('CHAIR lacks val images.')
    del annotation
    cfg.update(groups=groups,cohort_sha256=hashlib.sha256(encoded(groups)).hexdigest(),input_sha256=INPUTS,
        test_used=False,validation_scope='Historically reused Karpathy val5000; old dev1000 is secondary; no virgin holdout.',
        secondary_dev_ids=receipt['secondary_dev_ids'],source_checkpoint_sha256=A_SHA)
    save(out/'groups.json',groups)
    targets=make_targets(groups,dataset,Path(cfg['chair'])/'chair.pkl',json.loads(Path(cfg['detections']).read_text()),'aliases_v2','unique_count_v2')
    save(out/'train_object_targets.json',targets)
    print('Loading legacy once; staging unchanged image/prompt inputs for 15000 images.',flush=True)
    bundle=torch.load(cfg['legacy_prompt'],map_location='cpu',weights_only=False)
    rows=groups['train']+groups['val']
    names=[str(Path('/images')/dataset['images'][r['image_id']]['filepath']/r['filename']) for r in rows]
    legacy=align_prompt_cache(bundle['data'],names)
    if len(legacy)!=len(rows):raise ValueError('Missing legacy prompt tokens.')
    length=legacy[names[0]]['tokens'].shape[0]
    required=len(rows)*(197*768*4+length*512*4+length*8)
    stage.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(stage.parent).free<required+2*1024**3:raise RuntimeError('Insufficient staging disk plus2GiB reserve.')
    cache=VisualCache([cfg['visual_cache']],id_key='coco_id');metadata=[]
    try:
        cache.require_ids([r['coco_id'] for r in rows],'noun evidence train10k/val5000')
        for path in cache.paths:
            with h5py.File(path) as f:
                m=json.loads(f.attrs['metadata_json'])
                if not (m['model_id']=='openai/clip-vit-base-patch16' and m['output']=='vision_model.last_hidden_state' and m['preprocessing']=='bilinear' and m['image_size']==[224,224] and m['antialias'] is True and m['dataset_json_sha256']==DATA_SHA):raise ValueError('Visual geometry mismatch.')
                metadata.append(dict(path=str(path),metadata=m))
        by_file=defaultdict(list)
        for index,row in enumerate(rows):
            file_id,position=cache.index[row['coco_id']];by_file[file_id].append((position,index))
        with h5py.File(stage,'w') as f:
            for key,shape,dtype in [('visual',(len(rows),197,768),'f4'),('legacy',(len(rows),length,512),'f4'),('legacy_mask',(len(rows),length),'i8')]:f.create_dataset(key,shape=shape,dtype=dtype,chunks=(1,*shape[1:]))
            f.create_dataset('image_ids',data=np.asarray([r['coco_id'] for r in rows]));done=0
            for file_id,positions in sorted(by_file.items()):
                with h5py.File(cache.paths[file_id],'r') as source:
                    features=source['features'];dest={k:f[k] for k in ('visual','legacy','legacy_mask')}
                    for position,index in sorted(positions):
                        old=legacy[names[index]];tokens=old['tokens'].float().numpy();mask=old['mask'].long().numpy();visual=features[position].astype(np.float32)
                        if tokens.shape!=(length,512) or mask.shape!=(length,) or not np.isfinite(tokens).all() or not np.isfinite(visual).all() or not np.isin(mask,[0,1]).all():raise ValueError('Invalid cached values.')
                        for key,value in [('visual',visual),('legacy',tokens),('legacy_mask',mask)]:dest[key][index]=value
                        done+=1
                        if done==1 or done%250==0:print(f'Staged CLIP/legacy: {done}/{len(rows)}',flush=True)
            f.attrs.update(complete=True,cohort_sha256=cfg['cohort_sha256'])
    finally:cache.close()
    cfg.update(stage_sha256=digest(stage),object_targets_sha256=digest(out/'train_object_targets.json'),preparation_seconds=time.monotonic()-begin)
    save(out/'visual_metadata.json',metadata);save(out/'label_receipt.json',receipt);save(out/'protocol.json',cfg)
    print('PREPARE PASS. Next: two-GPU preflight, then training.',flush=True)


def loader(cfg,dataset,tokenizer,split,rank,shard=True):
    import h5py
    import torch
    from torch.utils.data import Dataset,DataLoader,DistributedSampler,Subset
    rows=cfg['groups'][split];offset=0 if split=='train' else 10000
    labels=json.loads((Path(cfg['labels'])/'targets.json').read_text()) if split=='train' else None
    slots=max(len(v['regions']) for v in labels.values()) if labels else 0
    class Staged(Dataset):
        def __init__(self):
            self.f=h5py.File(cfg['stage'],'r')
            if not self.f.attrs.get('complete') or self.f.attrs['cohort_sha256']!=cfg['cohort_sha256']:raise ValueError('Stage identity mismatch.')
            self.arrays={k:self.f[k] for k in ('visual','legacy','legacy_mask','image_ids')}
        def __len__(self):return len(rows)
        def __getitem__(self,index):
            row=rows[index];i=offset+index
            if int(self.arrays['image_ids'][i])!=row['coco_id']:raise ValueError('Stage row identity mismatch.')
            result={k:torch.from_numpy(self.arrays[k][i]) for k in ('visual','legacy','legacy_mask')}
            text=[tokenizer.encode(s['raw']) for s in dataset['images'][row['image_id']]['sentences'][:5]]
            if len(text)!=5:raise ValueError('Expected five references.')
            result.update(captions=torch.stack([e[0] for e in text]),caption_mask=torch.stack([e[1] for e in text]),image_id=torch.tensor(row['image_id']))
            if labels is not None:
                target=labels[str(row['image_id'])];regions=target['regions'];n=len(regions)
                masks=torch.zeros(slots,196);classes=torch.zeros(slots,dtype=torch.long);valid=torch.zeros(slots,dtype=torch.bool)
                if n:
                    masks[:n]=torch.tensor([r['mask'] for r in regions]);classes[:n]=torch.tensor([r['class_id'] for r in regions]);valid[:n]=True
                result.update(presence=torch.tensor(target['y'],dtype=torch.float32),observed=torch.tensor(target['observed'],dtype=torch.float32),region_masks=masks,region_class=classes,region_valid=valid)
            return result
    staged=Staged()
    if split=='train' and shard:
        sampler=DistributedSampler(staged,num_replicas=2,rank=rank,shuffle=True,seed=42,drop_last=False)
        return DataLoader(staged,batch_size=4,sampler=sampler,num_workers=0,pin_memory=True)
    return DataLoader(Subset(staged,list(range(rank,len(rows),2))) if shard else staged,batch_size=16,num_workers=0,pin_memory=True)
