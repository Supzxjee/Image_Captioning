"""Reuse the existing A-compatible CLIP/legacy staging and five-caption loader; no hidden-pair input."""
from collections import defaultdict
import json
from pathlib import Path
import shutil
import time
import numpy as np
from pilot_data import digest, save
from hidden_data import INPUT_HASHES, COHORT_SHA, locked_groups


def verify_sources(cfg):
    for name, expected in cfg['implementation_sha256'].items():
        if digest(Path(__file__).parent/name) != expected:
            raise ValueError('Source differs: '+name)


def prepare(config):
    import h5py
    import torch
    from captioning.visual_cache import VisualCache
    from captioning.semantic_prompts import align_prompt_cache
    from captioning.object_reward import make_targets
    cfg = json.loads(Path(config).read_text())
    verify_sources(cfg)
    out, stage = Path(cfg['output']), Path(cfg['stage'])
    if (out/'protocol.json').exists() or stage.exists():
        raise FileExistsError('Preserve existing stage/protocol; use fresh output/session.')
    out.mkdir(parents=True,exist_ok=True)
    begin = time.monotonic()
    for key, expected in INPUT_HASHES.items():
        print('Hashing pinned input:',key,flush=True)
        if digest(cfg[key]) != expected:
            raise ValueError('Pinned input differs: '+key)
    cfg['input_sha256'] = INPUT_HASHES
    dataset = json.loads(Path(cfg['dataset_json']).read_text())
    groups = locked_groups(dataset)
    rows = groups['train']+groups['val']
    cfg.update(groups=groups,cohort_sha256=COHORT_SHA)
    save(out/'groups.json',groups)
    targets = make_targets(groups,dataset,Path(cfg['chair'])/'chair.pkl',
                           json.loads(Path(cfg['detections']).read_text()),'aliases_v2','unique_count_v2')
    save(out/'train_object_targets.json',targets)
    print('Loading legacy tokens once; staging CLIP/legacy only, no UPT features.',flush=True)
    bundle = torch.load(cfg['legacy_prompt'],map_location='cpu',weights_only=False)
    names = [str(Path('/images')/dataset['images'][r['image_id']]['filepath']/r['filename']) for r in rows]
    legacy = align_prompt_cache(bundle['data'],names)
    if len(legacy) != len(rows):
        raise ValueError('Missing legacy prompt rows.')
    length = legacy[names[0]]['tokens'].shape[0]
    required = len(rows)*(197*768*4+length*512*4+length*8)
    stage.parent.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(stage.parent).free < required+2*1024**3:
        raise RuntimeError('Insufficient stage disk space plus2GiB reserve.')
    cache = VisualCache([cfg['visual_cache']],id_key='coco_id')
    try:
        cache.require_ids([r['coco_id'] for r in rows],'COCO routing train10k/dev1000')
        by_file = defaultdict(list)
        for index,row in enumerate(rows):
            file_id,position = cache.index[row['coco_id']]
            by_file[file_id].append((position,index))
        with h5py.File(stage,'w') as f:
            for key,shape,dtype in [('visual',(len(rows),197,768),'f4'),('legacy',(len(rows),length,512),'f4'),
                                    ('legacy_mask',(len(rows),length),'i8')]:
                f.create_dataset(key,shape=shape,dtype=dtype,chunks=(1,*shape[1:]))
            f.create_dataset('image_ids',data=np.asarray([r['coco_id'] for r in rows]))
            done = 0
            for file_id,positions in sorted(by_file.items()):
                # Keep the source dataset handle alive, avoiding metadata lookup per image.
                with h5py.File(cache.paths[file_id],'r') as source:
                    features = source['features']
                    for position,index in sorted(positions):
                        old = legacy[names[index]]
                        tokens,mask = old['tokens'].float().numpy(),old['mask'].long().numpy()
                        visual = features[position].astype(np.float32)
                        if tokens.shape != (length,512) or mask.shape != (length,):
                            raise ValueError('Legacy shape differs.')
                        if not np.isfinite(visual).all() or not np.isfinite(tokens).all() or not np.isin(mask,[0,1]).all():
                            raise ValueError('Invalid staged values.')
                        for key,value in [('visual',visual),('legacy',tokens),('legacy_mask',mask)]:
                            f[key][index] = value
                        done += 1
                        if done == 1 or done % 250 == 0:
                            print(f'Staged exact CLIP/legacy: {done}/{len(rows)}',flush=True)
            f.attrs.update(complete=True,cohort_sha256=COHORT_SHA)
        cfg.update(stage_sha256=digest(stage),object_targets_sha256=digest(out/'train_object_targets.json'),
                   preparation_seconds=time.monotonic()-begin)
    finally:
        cache.close()
    save(out/'protocol.json',cfg)
    print('Preparation complete; ready for user-run routing pilot.',flush=True)


def loader(cfg,dataset,tokenizer,split,rank):
    import h5py
    import torch
    from torch.utils.data import Dataset,DataLoader,DistributedSampler,Subset
    rows = cfg['groups'][split]
    offset = 0 if split == 'train' else 10000
    class Staged(Dataset):
        def __init__(self):
            self.f = h5py.File(cfg['stage'],'r')
            if not self.f.attrs.get('complete') or self.f.attrs['cohort_sha256'] != COHORT_SHA:
                raise ValueError('Incomplete/different stage.')
            # Cache dataset handles for the whole loader lifetime.
            self.arrays = {k:self.f[k] for k in ('visual','legacy','legacy_mask','image_ids')}
        def __len__(self):
            return len(rows)
        def __getitem__(self,index):
            row = rows[index]
            i = offset+index
            if int(self.arrays['image_ids'][i]) != row['coco_id']:
                raise ValueError('Stage image identity differs.')
            result = {k:torch.from_numpy(self.arrays[k][i]) for k in ('visual','legacy','legacy_mask')}
            encoded = [tokenizer.encode(s['raw']) for s in dataset['images'][row['image_id']]['sentences'][:5]]
            if len(encoded) != 5:
                raise ValueError('Expected five references.')
            result.update(captions=torch.stack([e[0] for e in encoded]),caption_mask=torch.stack([e[1] for e in encoded]),
                          image_id=torch.tensor(row['image_id']))
            return result
    staged = Staged()
    if split == 'train':
        sampler = DistributedSampler(staged,num_replicas=2,rank=rank,shuffle=True,seed=42,drop_last=False)
        return DataLoader(staged,batch_size=4,sampler=sampler,num_workers=0,pin_memory=True,drop_last=False)
    return DataLoader(Subset(staged,list(range(rank,len(rows),2))),batch_size=16,num_workers=0,pin_memory=True)


def predict(model,batches,tokenizer,condition):
    import torch
    from captioning.scst import rollout,decode
    from run_pilot import cuda_batch
    records = []
    with torch.no_grad():
        for i,batch in enumerate(batches):
            batch = cuda_batch(batch)
            memory,padding = model.memory(batch,condition)
            seq,_ = rollout(model,memory,padding,tokenizer,sample=False)
            records.extend(dict(image_id=iid,caption=text) for iid,text in
                           zip(batch['image_id'].cpu().tolist(),decode(seq,tokenizer)))
            if i % 10 == 0:
                print(f'Evaluation {condition}: {len(records)} local images',flush=True)
    return records
