"""One user-run matched C/T arm; shared audited objective, no sweep/selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import numpy as np
from pilot_contract import POLICY,CHAIR_SHA,sha,locate_cache,checked_payloads,schedule,validate_auxiliary
from diagnostic_data import A_SHA,BASE_SHA,DATA_SHA,LEGACY_SHA,save


def prepare(cfg):
    import h5py
    import torch
    from captioning.semantic_prompts import align_prompt_cache
    from captioning.visual_cache import VisualCache
    from hidden_data import locked_groups
    root=locate_cache(cfg['cache_root']);checked_payloads(root)
    old=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    if old['rows']!=manifest['rows'] or len(old['rows'])!=2000:raise ValueError('Auxiliary manifest binding differs.')
    for key,h in [('checkpoint',A_SHA),('dataset_json',DATA_SHA),('legacy_prompt',LEGACY_SHA)]:
        print('Hashing pinned input:',key,flush=True)
        if sha(cfg[key])!=h:raise ValueError('Pinned input differs: '+key)
    if sha(Path(cfg['chair'])/'chair.pkl')!=CHAIR_SHA:raise ValueError('CHAIR annotations differ.')
    dataset=json.loads(Path(cfg['dataset_json']).read_text(encoding='utf-8'));groups=locked_groups(dataset)
    if groups!=old['dataset_groups']:raise ValueError('Cohort differs from audited preparation.')
    out=Path(cfg['output']);stage=Path(cfg['stage'])
    if out.exists() or stage.exists():raise FileExistsError('Preserve existing output/stage; use fresh paths/session.')
    for rank in (0,1):
        with h5py.File(root/f'aux_rank{rank}.h5','r') as f:validate_auxiliary(f,old['rows'][rank::2])
    out.mkdir(parents=True);stage.parent.mkdir(parents=True,exist_ok=True)
    for name in cfg['implementation_sha256']:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(Path(__file__).parent/name,target)
    cfg.update(cache_root=str(root),rows=old['rows'],word2idx=old['word2idx'],groups=groups,policy=POLICY,
        preparation_receipt_sha256=sha(root/'receipt.json'),test_used=False,independent_review='Argon response pending; bounded C/T authorized by user')
    rows=groups['train']+groups['val'];names=[r['filename'] for r in rows]
    bundle=torch.load(cfg['legacy_prompt'],map_location='cpu',weights_only=False)
    legacy=align_prompt_cache(bundle['data'],names)
    if len(legacy)!=11000:raise ValueError('Original prompt coverage differs.')
    del bundle
    from captioning.tokenizer import CaptionTokenizer
    tok=CaptionTokenizer([]);tok.word2idx=cfg['word2idx']
    from captioning.object_reward import _AnnotationReader
    with (Path(cfg['chair'])/'chair.pkl').open('rb') as stream:annotation=_AnnotationReader(stream).load()
    if any(r['coco_id'] not in annotation.imid_to_objects for r in groups['val']):raise ValueError('CHAIR dev coverage incomplete.')
    del annotation
    required=11000*(197*768+20*512)*4
    if shutil.disk_usage(stage.parent).free<required+1024**3:raise RuntimeError('Need about8GiB free for local staged inputs.')
    visual=VisualCache([cfg['visual_cache']]);visual.require_ids([r['coco_id'] for r in rows],'stale pilot')
    checked=set();start=time.monotonic()
    try:
        with h5py.File(stage,'w') as f:
            for split in ('train','val'):
                items=groups[split];g=f.create_group(split)
                for name,shape,dtype in [('visual',(197,768),'f4'),('prompt',(20,512),'f4'),('mask',(20,),'u1')]:
                    g.create_dataset(name,(len(items),*shape),dtype=dtype)
                g.create_dataset('coco_ids',data=[r['coco_id'] for r in items])
                caps=[];pads=[]
                for r in items:
                    refs=dataset['images'][r['image_id']]['sentences'][:5]
                    if len(refs)!=5:raise ValueError('Expected five captions per image.')
                    encoded=[tok.encode(s['raw']) for s in refs];caps.append([e[0].tolist() for e in encoded]);pads.append([e[1].tolist() for e in encoded])
                g.create_dataset('captions',data=np.asarray(caps,dtype='i8'));g.create_dataset('padding',data=np.asarray(pads,dtype=bool))
                # Physical source order reduces repeated remote HDF5 chunk reads.
                order=sorted(range(len(items)),key=lambda i:visual.index[items[i]['coco_id']])
                for count,i in enumerate(order,1):
                    r=items[i];fi,_=visual.index[r['coco_id']]
                    if fi not in checked:
                        with h5py.File(visual.paths[fi],'r') as source:
                            meta=json.loads(source.attrs['metadata_json'])
                            expected=dict(dataset_json_sha256=DATA_SHA,image_size=[224,224],preprocessing='bilinear',model_id='openai/clip-vit-base-patch16',
                                output='vision_model.last_hidden_state',mean=[.48145466,.4578275,.40821073],std=[.26862954,.26130258,.27577711],antialias=True,compute_dtype='float32',autocast=False)
                            if any(meta.get(k)!=v for k,v in expected.items()):raise ValueError('Original visual cache metadata differs.')
                        checked.add(fi)
                    g['visual'][i]=visual.read(r['coco_id']);entry=legacy[r['filename']]
                    p=entry['tokens'].float().numpy();m=entry['mask'].numpy()
                    if p.shape!=(20,512) or m.shape!=(20,) or not np.isfinite(p).all() or not np.isin(m,[0,1]).all():raise ValueError('Original prompt shape/value differs.')
                    g['prompt'][i]=p;g['mask'][i]=m
                    if count%250==0:print(f'Stage {split}: {count}/{len(items)}; elapsed {time.monotonic()-start:.1f}s; no training yet.',flush=True)
                    if time.monotonic()-start>POLICY['staging_max_seconds']:raise RuntimeError('Main staging exceeds2h; preserve output, do not auto-reduce cohort.')
            index={r['coco_id']:i for i,r in enumerate(groups['train'])}
            for rank in (0,1):
                with h5py.File(root/f'aux_rank{rank}.h5','r') as aux:
                    for j,r in enumerate(cfg['rows'][rank::2]):
                        i=index[r['coco_id']]
                        for key,original in [('visual','visual_original'),('prompt','prompt_original'),('mask','mask_original')]:
                            if not np.array_equal(f['train'][key][i],aux[original][j]):raise ValueError('Main/aux factual inputs differ: '+key)
            f.attrs['complete']=True
    finally:visual.close()
    cfg.update(stage_sha256=sha(stage),staging_seconds=time.monotonic()-start)
    save(out/'protocol.json',cfg);print('PREPARED: verified immutable cache + staged original TRAIN10k/dev1000.',flush=True)


def frozen_digest(model):
    from captioning.scst_runtime import tensor_digest
    trainable={n for n,p in model.named_parameters() if p.requires_grad};h=hashlib.sha256()
    for n,v in sorted(model.state_dict().items()):
        if n not in trainable:h.update(n.encode());h.update(tensor_digest(v).encode())
    return h.hexdigest()


def predict(model,source,cfg,rank,condition):
    import torch
    from captioning.tokenizer import CaptionTokenizer
    from captioning.scst import rollout,decode
    from profile_stale import tensor_batch
    tok=CaptionTokenizer([]);tok.word2idx=cfg['word2idx'];tok.idx2word={i:w for w,i in tok.word2idx.items()}
    result=[];positions=list(range(rank,1000,2))
    with torch.no_grad():
        for start in range(0,len(positions),16):
            idx=positions[start:start+16];b=tensor_batch({k:source[k] for k in ('visual','prompt','mask')},idx)
            memory,mask=model.build_memory(b['visual'],b['prompt'],b['mask']);seq,_=rollout(model,memory,mask,tok,sample=False)
            for i,text in zip(idx,decode(seq,tok)):result.append(dict(image_id=cfg['groups']['val'][i]['image_id'],coco_id=cfg['groups']['val'][i]['coco_id'],caption=text))
    save(Path(cfg['output'])/f'{condition}_rank{rank}.json',result)


def worker(cfg):
    import h5py
    import torch
    from profile_stale import setup,trainable,Objective,tensor_batch,auxiliary
    from run_diagnostic import build
    from captioning.scst_runtime import model_digest,runtime_manifest
    rank=setup(True);out=Path(cfg['output']);root=Path(cfg['cache_root'])
    try:
        if sha(cfg['stage'])!=cfg['stage_sha256'] or sha(root/f'aux_rank{rank}.h5')!=json.loads((root/'receipt.json').read_text())['payload_sha256'][f'aux_rank{rank}.h5']:raise ValueError('Inputs changed since prepare.')
        if sha(cfg['checkpoint'])!=A_SHA:raise ValueError('Starting A checkpoint changed.')
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        student=build(cfg);anchor=build(cfg);parameters=trainable(student)
        frozen=frozen_digest(student);anchor_digest=model_digest(anchor)
        if anchor_digest!=BASE_SHA:raise ValueError('Wrong anchor.')
        with h5py.File(cfg['stage'],'r') as mainfile,h5py.File(root/f'aux_rank{rank}.h5','r') as auxfile:
            if not mainfile.attrs.get('complete'):raise ValueError('Incomplete staged main inputs.')
            validate_auxiliary(auxfile,cfg['rows'][rank::2])
            for split in ('train','val'):
                if mainfile[split]['coco_ids'][:].tolist()!=[r['coco_id'] for r in cfg['groups'][split]]:raise ValueError('Staged cohort differs.')
            predict(anchor,mainfile['val'],cfg,rank,'A')
            objective=Objective(student,anchor,cfg['arm']).eval()
            ddp=torch.nn.parallel.DistributedDataParallel(objective,device_ids=[rank],broadcast_buffers=False)
            optimizer=torch.optim.AdamW(parameters,lr=5e-6,betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=False)
            main_order,aux_order=schedule();main_group={k:mainfile['train'][k] for k in ('visual','prompt','mask','captions','padding')}
            torch.manual_seed(42);torch.cuda.manual_seed_all(42);torch.distributed.barrier();torch.cuda.synchronize();start=time.monotonic()
            for step,indices in enumerate(main_order):
                batch=tensor_batch(main_group,indices[rank*4:rank*4+4]);aux=None
                if (step+1)%5==0:aux=auxiliary(cfg,auxfile,aux_order[step//5])
                optimizer.zero_grad(set_to_none=True);loss=ddp(batch,aux)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss.')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
                if not torch.isfinite(norm):raise FloatingPointError('Nonfinite gradients.')
                optimizer.step()
                if (step+1)%25==0:
                    torch.cuda.synchronize();elapsed=time.monotonic()-start
                    measured=torch.tensor([elapsed,elapsed/(step+1)*1250*1.5+120,float(loss.detach()),float(norm)],dtype=torch.float64,device='cuda')
                    torch.distributed.all_reduce(measured,op=torch.distributed.ReduceOp.MAX)
                    if rank==0:
                        record=dict(step=step+1,images_seen=(step+1)*8,loss=float(measured[2]),grad_norm=float(measured[3]),seconds=float(measured[0]),projected_seconds=float(measured[1]))
                        with (out/'training.jsonl').open('a') as log:log.write(json.dumps(record)+'\n')
                        print(json.dumps(record),flush=True)
                    if measured[0]>7200 or measured[1]>7200:raise RuntimeError('Measured throughput exceeds2h/arm budget; no auto-resize/restart.')
                    if frozen_digest(student)!=frozen or model_digest(anchor)!=anchor_digest:raise ValueError('Frozen/anchor weights changed.')
            if frozen_digest(student)!=frozen or model_digest(anchor)!=BASE_SHA:raise ValueError('Frozen state changed.')
            final=model_digest(student);all_final=[None,None];torch.distributed.all_gather_object(all_final,final)
            if len(set(all_final))!=1 or final==BASE_SHA:raise ValueError('Rank state mismatch or student unchanged.')
            if rank==0:
                temporary=out/'checkpoint.bin.partial'
                torch.save(dict(format='stale_prompt_CT_v1',model_state_dict={'base.'+k:v.detach().cpu() for k,v in student.state_dict().items()},caption_word2idx=cfg['word2idx'],
                    arm=cfg['arm'],updates=1250,base_sha256=BASE_SHA,preparation_receipt_sha256=cfg['preparation_receipt_sha256']),temporary,_use_new_zipfile_serialization=False)
                temporary.replace(out/'checkpoint.bin')
            predict(student,mainfile['val'],cfg,rank,'end');torch.cuda.synchronize()
            seconds=time.monotonic()-start
            if seconds>7200:raise RuntimeError('Training + end greedy exceeds2h/arm.')
            save(out/f'worker_rank{rank}.json',dict(status='COMPLETE',arm=cfg['arm'],updates=1250,aux_images_consumed=1000,initial_sha256=BASE_SHA,final_sha256=final,
                frozen_sha256=frozen,anchor_unchanged=True,seconds=seconds,runtime=runtime_manifest(),peak_allocated_bytes=torch.cuda.max_memory_allocated()))
    finally:torch.distributed.destroy_process_group()


def evaluate(cfg):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    out=Path(cfg['output']);done=[json.loads((out/f'worker_rank{r}.json').read_text()) for r in (0,1)]
    if any(d['status']!='COMPLETE' or d['updates']!=1250 for d in done) or done[0]['final_sha256']!=done[1]['final_sha256']:raise ValueError('Incomplete/mismatched workers.')
    data=json.loads(Path(cfg['dataset_json']).read_text(encoding='utf-8'));expected={r['image_id']:r for r in cfg['groups']['val']};tokenizer=PTBTokenizer()
    refs=tokenizer.tokenize({i:[dict(caption=s['raw']) for s in data['images'][i]['sentences'][:5]] for i in sorted(expected)})
    metrics={}
    for condition in ('A','end'):
        rows=sum([json.loads((out/f'{condition}_rank{r}.json').read_text()) for r in (0,1)],[]);rows.sort(key=lambda r:r['image_id'])
        if len(rows)!=1000 or len({r['image_id'] for r in rows})!=1000 or {r['image_id'] for r in rows}!=set(expected):raise ValueError('Dev predictions incomplete/duplicated.')
        if any(r['coco_id']!=expected[r['image_id']]['coco_id'] for r in rows):raise ValueError('Prediction IDs differ.')
        hyp=tokenizer.tokenize({r['image_id']:[dict(caption=r['caption'])] for r in rows});score,per_image=Cider().compute_score(refs,hyp)
        save(out/f'{condition}_predictions.json',rows);save(out/f'{condition}_per_image_cider.json',[dict(image_id=r['image_id'],coco_id=r['coco_id'],CIDEr=float(v)) for r,v in zip(rows,per_image)])
        path=out/f'{condition}_chair_input.json';result=out/f'{condition}_chair_output.json'
        save(path,[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
        subprocess.run([sys.executable,'-u',str(Path(cfg['chair'])/'chair.py'),'--cap_file',str(path),'--cache',str(Path(cfg['chair'])/'chair.pkl'),'--save_path',str(result)],cwd=cfg['chair'],check=True)
        chair=json.loads(result.read_text());sentences=chair['sentences']
        if len(sentences)!=1000 or {s['image_id'] for s in sentences}!={r['coco_id'] for r in expected.values()}:raise ValueError('CHAIR IDs differ.')
        counts=dict(images=1000,hallucinated_captions=sum(bool(s['mscoco_hallucinated_words']) for s in sentences),hallucinated_mentions=sum(len(s['mscoco_hallucinated_words']) for s in sentences),
            object_mentions=sum(len(s['mscoco_generated_words']) for s in sentences),recalled_unique_objects=sum(len(set(s['mscoco_generated_words'])&set(s['mscoco_gt_words'])) for s in sentences),reference_unique_objects=sum(len(set(s['mscoco_gt_words'])) for s in sentences))
        aggregate=dict(CHAIRs=counts['hallucinated_captions']/1000,CHAIRi=counts['hallucinated_mentions']/max(1,counts['object_mentions']),Recall=counts['recalled_unique_objects']/max(1,counts['reference_unique_objects']))
        if any(abs(v-chair['overall_metrics'][k])>1e-12 for k,v in aggregate.items()):raise ValueError('CHAIR aggregation differs.')
        metrics[condition]=dict(CIDEr=float(score),chair=aggregate,counts=counts,mean_caption_words=float(np.mean([len(r['caption'].split()) for r in rows])))
    save(out/'summary.json',dict(status='PILOT_COMPLETE_WAIT_MATCHED_CT_COMPARISON',arm=cfg['arm'],metrics=metrics,updates=1250,checkpoint_sha256=sha(out/'checkpoint.bin'),
        preparation_receipt_sha256=cfg['preparation_receipt_sha256'],test_used=False,full_pilot_run=True,policy=POLICY,
        CIDEr_IDF='Standard pycocoevalcap Cider on the identical locked dev1000 reference set; matches prior A/H evaluation, not minibatch IDF.',
        limits=['One seed and reused development set; not unbiased final test.','No gain or novelty established before matched comparison.','Masks preserve silhouettes/context; objective success alone does not establish semantic alignment.','Argon response not received at packaging.'],staging_seconds=cfg['staging_seconds']))
    print(json.dumps(metrics,indent=2),flush=True)


def archive(cfg):
    out=Path(cfg['output'])
    with zipfile.ZipFile(str(out)+'_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.jsonl','.log','.py','.md'):z.write(p,p.relative_to(out))
    print('SEND:',str(out)+'_review.zip','; KEEP:',out/'checkpoint.bin',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','evaluate','archive']);p.add_argument('--config',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text(encoding='utf-8'))
    if cfg['arm'] not in ('C','T'):raise ValueError('Expected C or T.')
    for name,h in cfg['implementation_sha256'].items():
        if sha(Path(__file__).parent/name)!=h:raise ValueError('Runtime source changed: '+name)
    globals()[a.mode](cfg)
