"""Exact A input staging, H/L cache loader. No model inference on import."""
from collections import defaultdict
import json
from pathlib import Path
import shutil
import time
from detail_data import *


def prepare(cfg):
    import numpy as np
    import h5py
    import torch
    from captioning.visual_cache import VisualCache
    from captioning.semantic_prompts import align_prompt_cache
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    for key,pin in [('checkpoint',A_SHA),('dataset_json',DATA_SHA),('legacy_prompt',LEGACY_SHA)]:
        print('Hashing pinned input:',key,flush=True)
        if digest(cfg[key])!=pin:raise ValueError('Wrong input: '+key)
    if digest(Path(cfg['chair'])/'chair.pkl')!='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':raise ValueError('Different CHAIR annotation cache.')
    inventory=json.loads(Path(cfg['inventory']).read_text(encoding='utf-8'));manifest,groups=verify_cache(cfg['tile_cache'],inventory)
    data=json.loads(Path(cfg['dataset_json']).read_text(encoding='utf-8'));rows=groups['train']+groups['val']
    for r in rows:
        item=data['images'][r['image_id']]
        if item['filename']!=r['filename'] or item['cocoid']!=r['coco_id'] or (item['split']=='val')!=(r['group']=='val'):raise ValueError('Dataset/cache ID or split mismatch.')
    cfg.update(groups=groups,hl_manifest=manifest,schedule=schedule(groups['train']),test_used=False,
        input_sha256=dict(checkpoint=A_SHA,dataset_json=DATA_SHA,legacy_prompt=LEGACY_SHA,HL_receipt=CACHE_RECEIPT_SHA))
    save(out/'groups.json',groups);save(out/'cache_receipt.json',json.loads((Path(cfg['tile_cache'])/'cache_receipt.json').read_text(encoding='utf-8')))
    print('Loading legacy tokens once for unchanged A staging.',flush=True)
    bundle=torch.load(cfg['legacy_prompt'],map_location='cpu',weights_only=False)
    names=[str(Path('/images')/r['filepath']/r['filename']) for r in rows];legacy=align_prompt_cache(bundle['data'],names)
    length=legacy[names[0]]['tokens'].shape[0];needed=len(rows)*(197*768*2+length*512*4+length)
    path=Path(cfg['stage']);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Preserve previous stage; use fresh OUTPUT/STAGE.')
    if shutil.disk_usage(path.parent).free<needed+1024**3:raise RuntimeError('Not enough disk for exact A stage plus1GiB.')
    cache=VisualCache([cfg['visual_cache']]);cache.require_ids([r['coco_id'] for r in rows],'full H/L pilot')
    by_file=defaultdict(list)
    for i,r in enumerate(rows):
        f,p=cache.index[r['coco_id']];by_file[f].append((p,i))
    try:
        with h5py.File(path,'w') as stage:
            arrays={k:stage.create_dataset(k,shape=shape,dtype=dtype) for k,shape,dtype in [('visual',(15000,197,768),'f2'),('legacy',(15000,length,512),'f4'),('legacy_mask',(15000,length),'u1')]}
            stage.create_dataset('coco_ids',data=np.asarray([r['coco_id'] for r in rows],dtype='i8'));done=0
            for file_id,positions in sorted(by_file.items()):
                with h5py.File(cache.paths[file_id],'r') as source:
                    meta=json.loads(source.attrs['metadata_json']);features=source['features']
                    if meta['model_id']!='openai/clip-vit-base-patch16' or meta['output']!='vision_model.last_hidden_state' or meta['preprocessing']!='bilinear' or meta['image_size']!=[224,224] or meta['antialias'] is not True or meta['mean']!=[.48145466,.4578275,.40821073] or meta['std']!=[.26862954,.26130258,.27577711] or meta['compute_dtype']!='float32' or meta['autocast'] is not False or meta['dataset_json_sha256']!=DATA_SHA or features.dtype!=np.dtype('float16'):raise ValueError('A visual input geometry/storage differs.')
                    for position,i in sorted(positions):
                        old=legacy[names[i]];v=features[position];p=old['tokens'].float().numpy();m=old['mask'].long().numpy()
                        if p.shape!=(length,512) or m.shape!=(length,) or not np.isfinite(v).all() or not np.isfinite(p).all() or not np.isin(m,[0,1]).all():raise ValueError('Invalid staged input.')
                        for key,value in [('visual',v),('legacy',p),('legacy_mask',m)]:arrays[key][i]=value
                        done+=1
                        if done==1 or done%250==0:print(f'Staged exact A inputs: {done}/15000',flush=True)
            stage.attrs['complete']=True
        with h5py.File(path,'r') as staged:
            for i in (0,4999,9999,10000,14999):
                np.testing.assert_array_equal(staged['visual'][i].astype(np.float32),cache.read(rows[i]['coco_id']))
                np.testing.assert_array_equal(staged['legacy'][i],legacy[names[i]]['tokens'].float().numpy())
                np.testing.assert_array_equal(staged['legacy_mask'][i],legacy[names[i]]['mask'].long().numpy())
    finally:cache.close()
    cfg.update(stage_sha256=digest(path),preparation_seconds=time.monotonic()-started,stage_bytes=path.stat().st_size,stage_readback_indices=[0,4999,9999,10000,14999])
    save(out/'protocol.json',cfg);print('PREPARE COMPLETE; no training yet.',flush=True)
    return cfg


def loader(cfg,data,tokenizer,split,rank,epoch=0,condition=None,limit=None):
    import h5py
    import numpy as np
    import torch
    from torch.utils.data import Dataset,DataLoader,DistributedSampler,Subset
    rows=cfg['groups'][split];offset=0 if split=='train' else 10000
    mapping={r['coco_id']:(s['name'],i) for s in cfg['hl_manifest'] for i,r in enumerate(s['rows'])}
    class Cached(Dataset):
        def __init__(self):
            self.f=h5py.File(cfg['stage'],'r');self.arrays={k:self.f[k] for k in ('visual','legacy','legacy_mask','coco_ids')};self.handles={}
            if not bool(self.f.attrs['complete']):raise ValueError('Incomplete A stage.')
        def __len__(self):return len(rows)
        def __getitem__(self,index):
            row=rows[index];i=offset+index
            if int(self.arrays['coco_ids'][i])!=row['coco_id']:raise ValueError('Stage ID mismatch.')
            result={k:torch.from_numpy(self.arrays[k][i]) for k in ('visual','legacy','legacy_mask')}
            choice=condition if condition is not None else ('H' if cfg['schedule'][epoch][index] else 'L')
            if choice=='A':tiles=np.zeros((196,512),dtype=np.float32)
            else:
                name,j=mapping[row['coco_id']]
                if name not in self.handles:
                    f=h5py.File(Path(cfg['tile_cache'])/(name+'.h5'),'r');self.handles[name]=(f,{c:f[c] for c in ('H','L','coco_ids','xy')})
                f,arrays=self.handles[name]
                if int(arrays['coco_ids'][j])!=row['coco_id']:raise ValueError('H/L row ID mismatch.')
                tiles=arrays[choice][j].astype(np.float32)
            from tile_data import tile_centers
            if split=='train':
                encoded=[tokenizer.encode(s['raw']) for s in data['images'][row['image_id']]['sentences'][:5]]
                if len(encoded)!=5:raise ValueError('Exactly five natural captions required.')
                for ids,mask in encoded:
                    if not bool(ids.eq(tokenizer.eos_idx).any()):ids[-1]=tokenizer.eos_idx;mask[-1]=False
                result.update(captions=torch.stack([e[0] for e in encoded]),caption_mask=torch.stack([e[1] for e in encoded]))
            result.update(tiles=torch.from_numpy(tiles),xy=torch.from_numpy(tile_centers()),image_id=torch.tensor(row['image_id']),condition=torch.tensor(int(choice=='H')))
            return result
        def __del__(self):
            if hasattr(self,'f'):self.f.close()
            for f,_ in getattr(self,'handles',{}).values():f.close()
    ds=Cached()
    if split=='train':
        sampler=DistributedSampler(ds,num_replicas=2,rank=rank,shuffle=True,seed=42,drop_last=False);sampler.set_epoch(epoch)
        return DataLoader(ds,batch_size=4 if condition is None else cfg['eval_batch_size'],sampler=sampler,num_workers=0,pin_memory=True)
    ids=list(range(rank,len(rows),2));ids=ids[:limit] if limit is not None else ids
    return DataLoader(Subset(ds,ids),batch_size=cfg['eval_batch_size'],num_workers=0,pin_memory=True)
