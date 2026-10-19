"""Reuse the audited caption runner; add fit-only GT pairs and matcher diagnostics."""
import argparse
import json
import os
from pathlib import Path
import shutil
import numpy as np
import base_runner as runner
from pilot_data import digest,identity,save
from verified_data import training_schedule,geometry_summary
from probe_data import summarize

ORIGINAL_LOADER=runner.make_loader
ORIGINAL_PREDICT=runner.predict


def manifest(cfg):
    path=Path(cfg['verified_manifest'])
    if digest(path)!=cfg['verified_manifest_sha256']: raise ValueError('Verified labels changed')
    data=json.loads(path.read_text())
    assert list(data['splits'])==['fit','calibration','probe']
    ids={k:{r['coco_id'] for r in rows} for k,rows in data['splits'].items()}
    assert not ids['fit']&ids['calibration'] and not ids['fit']&ids['probe'] and not ids['calibration']&ids['probe']
    for key,rows in data['splits'].items(): assert identity(rows)==data['split_sha256'][key]
    assert data['schedule']==training_schedule(data['splits']['fit'])
    assert len(data['schedule'])==10000 and all(0<=i<len(data['splits']['fit']) for i in data['schedule'])
    assert not data['test_used']
    return data


def prepare(config):
    import h5py
    import torch
    from transformers import CLIPTextModel,AutoTokenizer
    from captioning.visual_cache import VisualCache
    cfg=json.loads(Path(config).read_text()); labels=manifest(cfg)
    runner.prepare(config)
    protocol=Path(cfg['output'])/'protocol.json';cfg=json.loads(protocol.read_text())
    stage=Path(cfg['aux_stage'])
    if stage.exists(): raise FileExistsError(stage)
    rows=[r for split in ('fit','calibration','probe') for r in labels['splits'][split]]
    ids=sorted({r['coco_id'] for r in rows});index={iid:i for i,iid in enumerate(ids)}
    required=len(ids)*197*768*4+len(rows)*2*512*4
    if shutil.disk_usage(stage.parent).free<required+1024**3: raise RuntimeError('Not enough space for auxiliary stage')
    cache=VisualCache([cfg['visual_cache']],id_key='coco_id');cache.require_ids(ids,'verified relation partitions')
    try:
        with h5py.File(stage,'w') as f:
            visual=f.create_dataset('visual',(len(ids),197,768),dtype='f4',chunks=(1,197,768))
            for count,iid in enumerate(sorted(ids,key=lambda x:cache.index[x]),1):
                visual[index[iid]]=cache.read(iid)
                if count%100==0 or count==len(ids): print(f'Verified image features: {count}/{len(ids)}',flush=True)
            f.create_dataset('image_index',data=np.asarray([index[r['coco_id']] for r in rows],dtype=np.int64))
            f.create_dataset('bounds',data=np.asarray([[c['bounds'] for c in r['candidates']] for r in rows],dtype=np.float32))
            f.create_dataset('positive',data=np.asarray([r['positive_index'] for r in rows],dtype=np.int64))
    finally: cache.close()
    encoder=CLIPTextModel.from_pretrained(runner.CLIP,revision=runner.CLIP_REVISION).cuda().eval()
    tokenizer=AutoTokenizer.from_pretrained(runner.CLIP,revision=runner.CLIP_REVISION)
    texts=sorted({c['text'] for r in rows for c in r['candidates']});embedding={}
    with torch.inference_mode():
        for start in range(0,len(texts),128):
            batch=texts[start:start+128]
            token=tokenizer(batch,padding=True,truncation=True,max_length=77,return_tensors='pt').to('cuda')
            for text,value in zip(batch,encoder(**token).pooler_output.float().cpu().numpy()): embedding[text]=value
    with h5py.File(stage,'r+') as f:
        f.create_dataset('text',data=np.asarray([[embedding[c['text']] for c in r['candidates']] for r in rows],dtype=np.float32))
        f.attrs['verified_manifest_sha256']=cfg['verified_manifest_sha256'];f.attrs['complete']=True
    # The underlying runner's generic E description is replaced before workers lock protocol SHA.
    cfg.update(method='verified_same_image_F',alignment_weight=.1,alignment_margin=.2,aux_stage_sha256=digest(stage),
        negative_policy=labels['policy'],verified_counts=labels['counts'],
        loss='CE + scheduled(.05)*SCST(CIDEr-D+.5*object_count_v2) + .1*verified_same_image_R_hinge_margin.2',
        comparison_limits='Same caption cohort/updates/architecture as E; additional verified fit-only relation labels and auxiliary image exposures. Not equal annotation/compute budget to A/E.',
        module_claim='Unchanged E module; modified relation supervision, not an established novel architecture')
    fit_ids={r['coco_id'] for r in labels['splits']['fit']}
    probe_ids={r['coco_id'] for r in labels['splits']['probe']}
    main_train={r['coco_id'] for r in cfg['groups']['train']}
    assert not main_train&probe_ids
    cfg['auxiliary_main_caption_overlap']=len(main_train&fit_ids)
    save(protocol,cfg);save(Path(cfg['output'])/'verified_manifest.json',labels)
    print('VERIFIED RELATION PREPARATION PASS',labels['counts'],flush=True)


def make_loader(cfg,dataset,tokenizer,split,rank,world):
    import h5py
    import torch
    from torch.utils.data import Dataset,DataLoader,DistributedSampler
    original=ORIGINAL_LOADER(cfg,dataset,tokenizer,split,rank,world)
    original.verified_cfg=cfg
    if split!='train': return original
    labels=manifest(cfg);schedule=labels['schedule']
    class Augmented(Dataset):
        def __init__(self):
            self.aux=h5py.File(cfg['aux_stage'],'r')
            assert self.aux.attrs.get('complete') and self.aux.attrs['verified_manifest_sha256']==cfg['verified_manifest_sha256']
        def __len__(self):return len(original.dataset)
        def __getitem__(self,index):
            row=original.dataset[index];i=schedule[index]
            for key,source in [('aux_visual',self.aux['visual'][int(self.aux['image_index'][i])]),
                               ('aux_text',self.aux['text'][i]),('aux_bounds',self.aux['bounds'][i])]:
                row[key]=torch.from_numpy(source)
            row['aux_positive']=torch.tensor(int(self.aux['positive'][i]),dtype=torch.long)
            row['aux_valid']=torch.tensor(True)
            return row
    data=Augmented();sampler=DistributedSampler(data,num_replicas=world,rank=rank,shuffle=True,seed=42,drop_last=False)
    return DataLoader(data,batch_size=cfg['global_batch_size']//world,sampler=sampler,num_workers=0,pin_memory=True)


def score_matcher(model,cfg,phase):
    import h5py
    import torch
    rank=int(os.environ['RANK']);labels=manifest(cfg);offset=0
    with h5py.File(cfg['aux_stage'],'r') as f,torch.no_grad():
        for split,rows in labels['splits'].items():
            if split=='fit': offset+=len(rows);continue
            selected=set(sorted({r['coco_id'] for r in rows})[rank::2]);records=[]
            for j,row in enumerate(rows):
                if row['coco_id'] not in selected:continue
                i=offset+j
                visual=torch.from_numpy(f['visual'][int(f['image_index'][i])][1:]).cuda()[None].expand(2,-1,-1)
                text=torch.from_numpy(f['text'][i]).cuda();boxes=torch.from_numpy(f['bounds'][i]).cuda()
                roles=torch.full((2,),2,dtype=torch.long,device=text.device);scores={}
                for method in ('boxed','global_view','zero_visual'):
                    p,b=visual,boxes
                    if method=='global_view': b=torch.zeros_like(boxes);b[:,2,2:]=1
                    if method=='zero_visual':p=torch.zeros_like(visual)
                    _,values,_=model.extra.attend(p,text,roles,b)
                    if not torch.isfinite(values).all():raise FloatingPointError('Nonfinite verified diagnostic')
                    scores[method]=values.cpu().tolist()
                records.append(dict(row,scores=scores))
            save(Path(cfg['output'])/f'matcher_{phase}_{split}_rank{rank}.json',records)
            offset+=len(rows)


def predict(model,loader,tokenizer,intervention):
    result=ORIGINAL_PREDICT(model,loader,tokenizer,intervention)
    if not getattr(model,'_verified_before',False):
        score_matcher(model,loader.verified_cfg,'before');model._verified_before=True
    elif intervention=='on': score_matcher(model,loader.verified_cfg,'after')
    return result


def evaluate(protocol):
    runner.evaluate(protocol)
    cfg=json.loads(Path(protocol).read_text());labels=manifest(cfg);out=Path(cfg['output']);result={}
    for phase in ('before','after'):
        result[phase]={}
        for split in ('calibration','probe'):
            rows=sum((json.loads((out/f'matcher_{phase}_{split}_rank{r}.json').read_text()) for r in (0,1)),[])
            rows.sort(key=lambda r:(r['task'],r['case_id']))
            assert [{k:v for k,v in r.items() if k!='scores'} for r in rows]==labels['splits'][split]
            result[phase][split]=summarize(rows)
            save(out/f'matcher_{phase}_{split}.json',rows)
    result['geometry_controls']={split:geometry_summary(labels['splits'][split]) for split in ('calibration','probe')}
    save(out/'verified_matcher_summary.json',result)
    summary=json.loads((out/'run_summary.json').read_text());summary.update(method=cfg['method'],verified_counts=cfg['verified_counts'],verified_matcher=result)
    save(out/'run_summary.json',summary)


runner.make_loader=make_loader
runner.predict=predict


def worker(protocol):
    cfg=json.loads(Path(protocol).read_text());manifest(cfg)
    assert cfg['arm']=='F' and cfg['alignment_weight']==.1 and cfg['alignment_margin']==.2
    if digest(cfg['aux_stage'])!=cfg['aux_stage_sha256']:raise ValueError('Auxiliary features changed')
    runner.worker(protocol)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','evaluate','archive']);p.add_argument('--config',required=True)
    a=p.parse_args()
    dict(prepare=prepare,worker=worker,evaluate=evaluate,archive=runner.archive)[a.mode](a.config)
