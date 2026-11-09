"""USER-RUN full10XE; direct cached patch IO, exact cursor and saved-caption scoring."""
import argparse
from datetime import timedelta
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
from real_bank_data import read,save,sha
from full_data import prepare,load_stage,CaptionBatches,ImageBatches,loader,order_for_epoch,learning_rate,validate_state,STEPS,TRAIN_EVENTS,TRAIN_IMAGES,VALID_IMAGES

def atomic(path,value):
    temp=path.with_suffix('.tmp');save(temp,value);os.replace(temp,path)

def distributed(cfg,guard=False):
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from patch_model import PatchCaptioner
    from patch_beam import beam_search,text_from_ids
    from full_runtime import preflight,tensor_digest
    from cuda_runtime_guard import check_runtime
    rank=int(os.environ['LOCAL_RANK']);out=Path(cfg['output']);check_runtime(out,rank)
    if int(os.environ['WORLD_SIZE'])!=2:raise ValueError('Select T4x2')
    torch.set_num_threads(1);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(seconds=1200));started=time.monotonic();deadline=started+cfg['worker_seconds']
    try:
        data,words,stage=load_stage(cfg['stage'],verify=rank==0);dist.barrier(device_ids=[rank])
        if cfg.get('feature_storage') and stage.get('feature_storage')!=cfg['feature_storage']:raise ValueError('Required local stage incomplete; do not fall back to mounted H5')
        context=None
        torch.manual_seed(42)
        if cfg.get('semantic'):
            from semantic_training import make_model,context_for
            if stage.get('semantic',{}).get('method')!='bank_prompt_patch_v1':raise ValueError('Semantic stage missing/incomplete')
            model=make_model(len(words),cfg);context=context_for(cfg)
        else:model=PatchCaptioner(len(words))
        model=model.to(device);wrapped=DDP(model,device_ids=[rank],broadcast_buffers=False)
        def predict(net,x,prefix,events):return net(x,prefix,context.events(data,events,device)) if context else net(x,prefix)
        def memory_for(x,indices):return model.encode(x,context.images(indices,device)) if context else model.encode(x)
        if guard:preflight(cfg,model,wrapped,data,words,rank,device,out,deadline);return
        guards=[read(out/f'preflight_rank{r}.json') for r in (0,1)]
        if not all(g['passed'] and g['initial_tensor_sha256']==tensor_digest(model) for g in guards):raise ValueError('Fresh initialization/preflight mismatch')
        step_est=max(g['full_IO_step_p90_seconds'] for g in guards);val_est=max(g['decode_seconds']/g['decode_images']*2500 for g in guards)
        refs=read(Path(cfg['stage'])/'validation_references.json')
        identity=dict(schema='clip_patch_full_xe10_v1',source=read(out/'source_receipt.json'),stage_files=stage['files_sha256'],raw_sources=stage['sources'],feature_profile=stage['feature_profile'],
            initial_sha256=guards[0]['initial_tensor_sha256'],global_batch=32,epochs=10,lr='warmup1epoch8e-6to1e-4_then_cosine_to1e-5',dropout=.1,clip1=True,seed=42,
            method='bank_prompt_patch_v1' if context else 'plain_patch_v1',software={k:importlib.metadata.version(k) for k in ('torch','numpy','pycocoevalcap')},test_used=False)
        optimizer=torch.optim.Adam(model.parameters(),lr=8e-6,betas=(.9,.99),eps=1e-8,weight_decay=0)
        state=dict(epoch=1,batch_cursor=0,phase='train',updates=0,caption_events=0,history=[],best_epoch=None,best_CIDEr=None,epoch_nll_sum=0.,epoch_tokens=0)
        random.seed(42+rank);np.random.seed(42+rank);torch.manual_seed(4242+rank);torch.cuda.manual_seed(4242+rank)
        if cfg.get('resume_root'):
            root=Path(cfg['resume_root']);receipt=read(root/'checkpoint_receipt.json')
            if sha(root/'checkpoint.bin')!=receipt['sha256']:raise ValueError('Changed resume checkpoint')
            ck=torch.load(root/'checkpoint.bin',map_location='cpu',weights_only=False)
            if ck['identity']!=identity:raise ValueError('Resume recipe/data/source/software differs')
            state=ck['state'];validate_state(state);model.load_state_dict(ck['model'],strict=True);optimizer.load_state_dict(ck['optimizer'])
            rng=ck['rng_by_rank'][rank];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'],device)
            if rank==0:
                for row in state['history']:
                    folder=root/'evaluations'/f"epoch_{row['epoch']:02d}"
                    for name,h in row['files_sha256'].items():
                        if sha(folder/name)!=h:raise ValueError('Changed resume validation '+name)
                    shutil.copytree(folder,out/'evaluations'/folder.name)
                if state['best_epoch'] is not None:
                    b=read(root/'best_receipt.json') if (root/'best_receipt.json').is_file() else {}
                    if b.get('epoch')!=state['best_epoch'] and state['batch_cursor']==0 and state['phase']=='train' and state['best_epoch']==state['epoch']-1:
                        shutil.copy2(root/'checkpoint.bin',out/'best.bin');save(out/'best_receipt.json',dict(sha256=receipt['sha256'],epoch=state['best_epoch'],CIDEr=state['best_CIDEr'],test_used=False))
                    else:
                        if b.get('epoch')!=state['best_epoch'] or sha(root/'best.bin')!=b['sha256']:raise ValueError('Changed resume best')
                        for name in ('best.bin','best_receipt.json'):shutil.copy2(root/name,out/name)
                if receipt.get('training_log_sha256'):
                    if sha(root/'training.jsonl')!=receipt['training_log_sha256']:raise ValueError('Changed resume training log')
                    shutil.copy2(root/'training.jsonl',out/'training.jsonl')
            del ck;dist.barrier(device_ids=[rank])
        if rank==0:save(out/'identity.json',identity)
        def checkpoint(promote=False):
            validate_state(state)
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device));all_rng=[None,None];dist.all_gather_object(all_rng,rng)
            if rank==0:
                temp=out/'checkpoint.tmp';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),state=state,identity=identity,rng_by_rank=all_rng),temp);os.replace(temp,out/'checkpoint.bin')
                receipt=dict(sha256=sha(out/'checkpoint.bin'),updates=state['updates'],epoch=state['epoch'],phase=state['phase'],training_log_sha256=sha(out/'training.jsonl') if (out/'training.jsonl').is_file() else None)
                atomic(out/'checkpoint_receipt.json',receipt);atomic(out/'progress.json',state)
                if promote:
                    shutil.copy2(out/'checkpoint.bin',out/'best.tmp');os.replace(out/'best.tmp',out/'best.bin')
                    atomic(out/'best_receipt.json',dict(sha256=receipt['sha256'],epoch=state['best_epoch'],CIDEr=state['best_CIDEr'],test_used=False))
            dist.barrier(device_ids=[rank])
        remaining_steps=10*STEPS-state['updates'];remaining_epochs=11-state['epoch']
        projected=1.35*(remaining_steps*step_est+remaining_epochs*(val_est+90))+900
        if rank==0:save(out/'measured_projection.json',dict(remaining_steps=remaining_steps,remaining_epochs=remaining_epochs,seconds=projected,available=deadline-time.monotonic(),not_guaranteed=True))
        if projected>deadline-time.monotonic():checkpoint();raise TimeoutError('Measured full IO/train/decode projection exceeds cap; no automatic train extension')
        def validate(epoch):
            wrapped.eval();folder=out/'evaluations'/f'epoch_{epoch:02d}';folder.mkdir(parents=True,exist_ok=True);rows=[];nll_sum=0.;tokens=0
            images=np.arange(TRAIN_IMAGES,TRAIN_IMAGES+VALID_IMAGES)[rank::2]
            with torch.inference_mode():
                for x,index in loader(ImageBatches(cfg['stage'],images)):
                    if time.monotonic()>deadline-60:raise TimeoutError('VALID decode deadline; preserve training checkpoint')
                    memory=memory_for(x.to(device,non_blocking=True),index);seq,scores=beam_search(model,memory)
                    for i,s,v in zip(index.tolist(),seq[:,0].cpu().tolist(),scores[:,0].cpu().tolist()):rows.append(dict(coco_id=int(data['ids'][i]),caption=text_from_ids(s,words),ids=s,score=float(v)))
                ev=np.flatnonzero(np.isin(data['caption_image'],images))
                for x,seq,event_ids in loader(CaptionBatches(cfg['stage'],ev,rank=None)):
                    if time.monotonic()>deadline-60:raise TimeoutError('VALID NLL deadline')
                    x=x.to(device,non_blocking=True);seq=seq.to(device,non_blocking=True);logits=predict(model,x,seq[:,:-1],event_ids)
                    nll_sum+=float(F.cross_entropy(logits.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum'));tokens+=int((seq[:,1:]!=0).sum())
            if len(rows)!=2500:raise ValueError('Incomplete VALID worker')
            save(folder/f'val_rank{rank}.json',rows);save(folder/f'worker{rank}.json',dict(complete=True,epoch=epoch,images=2500,NLL_sum=nll_sum,NLL_tokens=tokens))
            dist.barrier(device_ids=[rank]);point=[None]
            if rank==0:
                from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
                from pycocoevalcap.cider.cider import Cider
                all_rows=read(folder/'val_rank0.json')+read(folder/'val_rank1.json')
                if len({r['coco_id'] for r in all_rows})!=5000 or {r['coco_id'] for r in all_rows}!={r['coco_id'] for r in refs}:raise ValueError('VALID IDs differ')
                ptb=PTBTokenizer();gts=ptb.tokenize({r['coco_id']:[dict(caption=t) for t in r['references']] for r in refs});hyp=ptb.tokenize({r['coco_id']:[dict(caption=r['caption'])] for r in all_rows})
                cider,per=Cider().compute_score(gts,hyp)
                if not np.isfinite(cider):raise ValueError('Nonfinite CIDEr')
                save(folder/'cider_per_image.json',[dict(coco_id=i,CIDEr=float(s)) for i,s in zip(gts,per)])
                workers=[read(folder/f'worker{r}.json') for r in (0,1)]
                value=dict(epoch=epoch,CIDEr=float(cider),NLL=sum(w['NLL_sum'] for w in workers)/sum(w['NLL_tokens'] for w in workers),TRAIN_online_NLL=state['epoch_nll_sum']/state['epoch_tokens'])
                save(folder/'result.json',value);value['files_sha256']={p.name:sha(p) for p in folder.iterdir() if p.is_file()};point[0]=value
                print('FULL VALID',value,flush=True)
            dist.broadcast_object_list(point,src=0);return point[0]
        while state['epoch']<=10:
            epoch=state['epoch'];wrapped.train();order=order_for_epoch(data,epoch);cursor=state['batch_cursor'];speed_start=None
            if state['phase']=='train':
                it=iter(loader(CaptionBatches(cfg['stage'],order,rank,cursor)))
                for offset in range(cursor,STEPS):
                    stop=torch.tensor(int(time.monotonic()>deadline-900),device=device);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
                    if int(stop):checkpoint();raise TimeoutError('Saved before parent/worker cap')
                    x,seq,ev=next(it);x=x.to(device,non_blocking=True);seq=seq.to(device,non_blocking=True)
                    lr=learning_rate(state['updates']+1)
                    for g in optimizer.param_groups:g['lr']=lr
                    optimizer.zero_grad(set_to_none=True);logits=predict(wrapped,x,seq[:,:-1],ev);count=torch.tensor([int((seq[:,1:]!=0).sum()),len(ev)],device=device);dist.all_reduce(count)
                    nll=F.cross_entropy(logits.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum');loss=nll*(2/int(count[0]))
                    if not torch.isfinite(loss):raise ValueError('Nonfinite XE')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
                    total=nll.detach().double();dist.all_reduce(total)
                    state.update(batch_cursor=offset+1,updates=state['updates']+1,caption_events=(epoch-1)*TRAIN_EVENTS+min((offset+1)*32,TRAIN_EVENTS),epoch_nll_sum=state['epoch_nll_sum']+float(total),epoch_tokens=state['epoch_tokens']+int(count[0]))
                    if (offset+1)%100==0 or offset+1==STEPS:
                        if rank==0:
                            report=dict(epoch=epoch,updates=state['updates'],batch_cursor=offset+1,caption_events=state['caption_events'],NLL=float(total)/int(count[0]),lr=lr,grad_norm=float(norm),seconds=time.monotonic()-started)
                            with (out/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(report)+'\n')
                            print('FULL XE',report,flush=True)
                        completed=offset+1-cursor
                        if completed==100:speed_start=time.monotonic()
                        if completed>=200 and speed_start is not None:
                            measured=(time.monotonic()-speed_start)/(completed-100);remaining=(10*STEPS-state['updates'])*measured+(11-epoch)*(val_est+90)+900
                            stop=torch.tensor(int(remaining>deadline-time.monotonic()),device=device);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
                            if int(stop):checkpoint();raise TimeoutError('Measured training projection exceeds cap; resume explicitly if authorized')
                    if state['updates']%500==0:checkpoint()
                del it;state['phase']='validate';checkpoint()
            point=validate(epoch);better=state['best_CIDEr'] is None or point['CIDEr']>state['best_CIDEr']
            if better:state.update(best_CIDEr=point['CIDEr'],best_epoch=epoch)
            state['history'].append(point);state.update(epoch=epoch+1,batch_cursor=0,phase='train',epoch_nll_sum=0.,epoch_tokens=0);checkpoint(promote=better)
        if rank==0:save(out/'training_complete.json',dict(status='PATCH_FULL_XE10_TRAINING_COMPLETE',**state,worker_seconds=time.monotonic()-started,timing_scope='this worker invocation only; excludes setup, scoring and earlier resume segments'))
    finally:
        if dist.is_initialized():dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','preflight','worker'));p.add_argument('--config',required=True)
    a=p.parse_args();cfg=read(a.config)
    if a.mode=='prepare':
        deadline=time.monotonic()+cfg.get('prepare_seconds',1800)
        if cfg.get('semantic'):
            from semantic_data import checked_inputs
            checked_inputs(cfg)
            print('Frozen bank130/prompt175 exact source precheck PASS before patch copy.',flush=True)
        prepare(cfg)
        if cfg.get('feature_storage')=='local_fp16_memmap_v2':
            from local_stage import stage_local
            stage_local(cfg,deadline)
        if cfg.get('semantic'):
            from semantic_data import stage_semantics
            stage_semantics(cfg,deadline)
    else:distributed(cfg,guard=a.mode=='preflight')
