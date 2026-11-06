"""USER-RUN ONLY: selected C178/P58, one full SCST+CE epoch, saved5beams."""
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
from prompt_scst_data import SOURCES,PHASES,EVAL_NAMES,UPDATES,TRAIN_IMAGES,prepare,source_checkpoint,check_source,validate_state,copy_evaluations,validate_candidates


def atomic(path,value):
    temp=path.with_suffix('.tmp');save(temp,value);os.replace(temp,path)


def distributed(cfg,preflight=False):
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from model import Config,CSACaptioner,beam_search
    from prompt_scratch_model import ScratchCaptioner,to_device
    from prompt_scratch_data import batch,checked_cache
    from prompt_pilot_model import PromptPolicyReader
    from scst_pilot_model import rollout,selected_logps,policy_loss,captions
    from scst_full_data import local_batch,image_events,copy_training_log
    from cuda_runtime_guard import check_runtime
    rank=int(os.environ['LOCAL_RANK']);out=Path(cfg['output']);check_runtime(out,rank)
    if int(os.environ['WORLD_SIZE'])!=2:raise ValueError('Select T4x2')
    torch.set_num_threads(1);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(seconds=600))
    started=time.monotonic();deadline=started+cfg['worker_seconds'];enabled=cfg['arm']=='PROMPT'
    try:
        prepared=read(out/'prepared.json');stage=read(Path(cfg['stage'])/'stage_receipt.json')
        if prepared['arm']!=cfg['arm'] or prepared['source']!=SOURCES[cfg['arm']]:raise ValueError('Wrong prepared arm')
        for n,h in prepared['files_sha256'].items():
            if sha(out/n)!=h:raise ValueError('Changed prepared payload:'+n)
        if stage['files_sha256']!=prepared['stage_sha256']:raise ValueError('Wrong staged manifest')
        if rank==0:
            for n,h in stage['files_sha256'].items():
                if sha(Path(cfg['stage'])/n)!=h:raise ValueError('Changed staged array:'+n)
        dist.barrier(device_ids=[rank])
        data={k:np.load(Path(cfg['stage'])/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in stage['shapes']}
        _,links,emb,texts,inventory=checked_cache(cfg['prompt_root'],verify=False)
        if inventory['cache_files']!=prepared['prompt_files']:raise ValueError('Changed prompt175 manifest')
        ck=torch.load(source_checkpoint(cfg['checkpoint_root'],cfg['arm']),map_location='cpu',weights_only=False)
        check_source(ck,cfg['arm'],stage['files_sha256'],prepared['prompt_files'])
        for name in ('model.py','prompt_adapter_model.py','prompt_scratch_model.py'):
            if sha(Path(__file__).with_name(name))!=ck['identity']['source']['experiments/csa_paper/'+name]:raise ValueError('Changed source model behavior:'+name)
        torch.manual_seed(42)
        wrapper=ScratchCaptioner(CSACaptioner(Config(vocab=10000),ck['model']['model.base.nouns'],ck['model']['model.base.verbs']),enabled)
        wrapper.load_state_dict(ck['model'],strict=True);del ck
        model=wrapper.model.to(device).eval();del wrapper
        banks={n:getattr(model.base,n).detach().clone() for n in ('nouns','verbs')}
        params=[p for p in model.parameters() if p.requires_grad]
        reader=DDP(PromptPolicyReader(model,enabled),device_ids=[rank],broadcast_buffers=False,find_unused_parameters=False).eval()
        optimizer=torch.optim.Adam(params,lr=5e-6,betas=(.9,.99),eps=1e-8,weight_decay=0)
        events=np.load(out/'events.npy',allow_pickle=False);ids=np.load(out/'coco_ids.npy',allow_pickle=False);vocab=read(out/'vocabulary.json')
        if events.shape!=(UPDATES,8) or ids.shape!=events.shape:raise ValueError('Wrong full1 schedule')
        identity=dict(schema='prompt_scratch_scst_full1_v1',arm=cfg['arm'],source_checkpoint=SOURCES[cfg['arm']],prepared={k:v for k,v in prepared.items() if k!='checkpoint'},
            sources=read(out/'source_receipt.json'),lr=5e-6,scst_weight=.05,object_weight=0,dropout=False,global_batch=8,
            software={n:importlib.metadata.version(n) for n in ('torch','numpy','pycocoevalcap')},cuda=torch.version.cuda,
            trainable_parameters=sum(p.numel() for p in params),test_used=False)
        random.seed(42+rank);np.random.seed(42+rank);torch.manual_seed(4242+rank);torch.cuda.manual_seed(4242+rank)
        state=dict(updates=0,image_events=0,validated=[],validation_files={})

        def get(ev):
            x,v,seq,p,_=to_device(batch(data,ev,links,emb,texts),device)
            return x,v,seq,p

        if preflight:
            ev,_=local_batch(events,ids,0,rank);x,v,ce,p=get(ev)
            with torch.random.fork_rng(devices=[rank]),torch.no_grad():
                memory=model.encode(x,v,p,enabled=enabled)
                sample,mask,logps=rollout(model,*memory,v,True)
                replay=selected_logps(model.decode(sample[:,:-1],*memory,v),sample[:,1:],mask)
                torch.testing.assert_close(logps,replay,rtol=1e-4,atol=2e-5)
                empty={k:a.clone() for k,a in p.items()};empty['object_valid'].zero_();empty['relation_valid'].zero_()
                for a,b in zip(model.base.encode(x,v),model.encode(x,v,empty,enabled=enabled)):torch.testing.assert_close(a,b,rtol=0,atol=0)
                max_abs=float((logps-replay).abs().max())
            optimizer.zero_grad(set_to_none=True)
            logits,ce_logits=reader(x,v,p,sample[:,:-1],ce[:,:-1])
            tokens=torch.tensor(int((ce[:,1:]!=0).sum()),device=device);dist.all_reduce(tokens)
            loss=F.cross_entropy(ce_logits.flatten(0,1),ce[:,1:].flatten(),ignore_index=0,reduction='sum')*(2/int(tokens))
            loss+=.05*policy_loss(logits,sample[:,1:],mask,torch.ones(len(x),device=device))
            loss.backward()
            for name,param in model.named_parameters():
                if param.requires_grad and (param.grad is None or not torch.isfinite(param.grad).all()):raise ValueError('Missing/nonfinite loaded-model gradient:'+name)
            norm=torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);optimizer.step()
            checksum=torch.stack([p.detach().double().sum() for p in params]).sum();copies=[torch.zeros_like(checksum) for _ in range(2)];dist.all_gather(copies,checksum)
            if not torch.equal(copies[0],copies[1]):raise ValueError('DDP parameter checksums differ')
            save(out/f'preflight_rank{rank}.json',dict(passed=True,arm=cfg['arm'],source_sha256=SOURCES[cfg['arm']]['sha256'],
                policy_replay_max_abs=max_abs,empty_fallback_exact=True,gradient_norm=float(norm),updates_discarded=True))
            print('Loaded-model preflight PASS; all updates discarded.',rank,cfg['arm'],flush=True);return

        for r in (0,1):
            guard=read(out/f'preflight_rank{r}.json')
            if not guard['passed'] or guard['arm']!=cfg['arm'] or guard['source_sha256']!=SOURCES[cfg['arm']]['sha256']:raise ValueError('Missing/wrong actual guard')
        if cfg.get('resume_root'):
            root=Path(cfg['resume_root']);receipt=read(root/'checkpoint_receipt.json')
            if sha(root/'checkpoint.bin')!=receipt['sha256']:raise ValueError('Changed resume checkpoint')
            restored=torch.load(root/'checkpoint.bin',map_location='cpu',weights_only=False)
            if restored['identity']!=identity:raise ValueError('Resume arm/source/data/software differs')
            state=restored['state'];validate_state(state);model.load_state_dict(restored['model'],strict=True);optimizer.load_state_dict(restored['optimizer'])
            rng=restored['rng_by_rank'][rank];random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'],device)
            if rank==0:copy_evaluations(root,out,state['validation_files']);copy_training_log(root,out,receipt)
            del restored;dist.barrier(device_ids=[rank])
        if rank==0:save(out/'identity.json',identity)

        def checkpoint():
            validate_state(state)
            for n,expected in banks.items():
                value=getattr(model.base,n)
                if value.requires_grad or not torch.equal(value,expected):raise ValueError('Frozen semantic bank changed')
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device));rngs=[None,None];dist.all_gather_object(rngs,rng)
            if rank==0:
                temp=out/'checkpoint.tmp';torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),state=state,identity=identity,rng_by_rank=rngs),temp);os.replace(temp,out/'checkpoint.bin')
                atomic(out/'checkpoint_receipt.json',dict(sha256=sha(out/'checkpoint.bin'),arm=cfg['arm'],**state,
                    training_log_sha256=sha(out/'training.jsonl') if (out/'training.jsonl').is_file() else None))
                atomic(out/'progress.json',state)
            dist.barrier(device_ids=[rank])

        def validate(phase):
            d=out/'evaluations'/phase;d.mkdir(parents=True,exist_ok=True);rows=[];beams=[];total=0.;tokens=0
            images=np.flatnonzero(data['group']==1)[rank::2]
            with torch.inference_mode():
                for offset in range(0,len(images),8):
                    if time.monotonic()>deadline:raise TimeoutError('VALID deadline; preserve checkpoint')
                    indices=images[offset:offset+8];x,v,_,p=get(np.searchsorted(data['caption_image'],indices));memory=model.encode(x,v,p,enabled=enabled)
                    for i,(seq,scores) in zip(indices,beam_search(model.base,*memory,v,beams=5,maximum=20)):
                        text=captions(seq,vocab);candidates=[dict(rank=j,caption=t,ids=seq[j].cpu().tolist(),logprob=float(scores[j])) for j,t in enumerate(text)]
                        beams.append(dict(coco_id=int(data['ids'][i]),candidates=candidates));rows.append(dict(coco_id=int(data['ids'][i]),caption=text[0],ids=candidates[0]['ids']))
                    if offset%400==0:print('VALID',phase,rank,len(rows),'/2500',flush=True)
                ev=np.flatnonzero(np.isin(data['caption_image'],images))
                for offset in range(0,len(ev),16):
                    if time.monotonic()>deadline:raise TimeoutError('VALID NLL deadline; preserve checkpoint')
                    x,v,seq,p=get(ev[offset:offset+16]);logits=model.decode(seq[:,:-1],*model.encode(x,v,p,enabled=enabled),v)
                    value=F.cross_entropy(logits.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum')
                    if not torch.isfinite(value):raise ValueError('Nonfinite VALID NLL')
                    total+=float(value);tokens+=int((seq[:,1:]!=0).sum())
            if len(rows)!=2500:raise ValueError('Incomplete VALID rank')
            save(d/f'val_rank{rank}.json',rows);save(d/f'candidates_rank{rank}.json',beams)
            save(d/f'worker{rank}.json',dict(complete=True,arm=cfg['arm'],phase=phase,updates=state['updates'],image_events=state['image_events'],images=len(rows),nll_sum=total,nll_tokens=tokens))
            dist.barrier(device_ids=[rank]);ok=[None]
            if rank==0:
                from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
                from pycocoevalcap.cider.cider import Cider
                from saved_quality_data import check_rows
                combined=rows+read(d/'val_rank1.json');refs=read(out/'validation_references.json');expected={r['coco_id'] for r in refs};check_rows(combined,expected)
                validate_candidates(beams+read(d/'candidates_rank1.json'),expected)
                ptb=PTBTokenizer();reference=ptb.tokenize({r['coco_id']:[dict(caption=t) for t in r['references']] for r in refs});hyp=ptb.tokenize({r['coco_id']:[dict(caption=r['caption'])] for r in combined})
                cider,_=Cider().compute_score(reference,hyp)
                if not np.isfinite(cider):raise ValueError('Nonfinite CIDEr')
                if phase=='original' and abs(cider-SOURCES[cfg['arm']]['cider'])>1e-7:raise ValueError('ORIGINAL CIDEr replay differs; no optimizer update authorized')
                save(d/'cider_replay.json',dict(CIDEr=float(cider),historical=SOURCES[cfg['arm']]['cider'] if phase=='original' else None))
                ok[0]={f'evaluations/{phase}/{n}':sha(d/n) for n in EVAL_NAMES}
            dist.broadcast_object_list(ok,src=0);state['validation_files'].update(ok[0]);state['validated'].append(phase);checkpoint()

        validate_state(state)
        if 'original' not in state['validated']:validate('original')
        # Build TRAIN-only reward after ORIGINAL replay; no validation captions enter reward.
        from captioning.scst import TrainingCider
        cider=TrainingCider({int(k):v for k,v in read(out/'training_references.json').items()},reference_cache_size=2048)
        began=time.monotonic();initial=state['updates']
        for step in range(initial,UPDATES):
            stop=torch.tensor(int(time.monotonic()>deadline-900),device=device);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            if int(stop):checkpoint();raise TimeoutError('Saved before worker deadline; resume same arm only')
            ev,batch_ids=local_batch(events,ids,step,rank);x,v,ce,p=get(ev)
            with torch.no_grad():
                memory=model.encode(x,v,p,enabled=enabled);sample,mask,_=rollout(model,*memory,v,True);greedy,_,_=rollout(model,*memory,v,False)
            sc=cider.score(batch_ids.tolist(),captions(sample,vocab));gc=cider.score(batch_ids.tolist(),captions(greedy,vocab));advantage=torch.from_numpy(sc-gc).to(device)
            if not torch.isfinite(advantage).all():raise ValueError('Nonfinite advantage')
            optimizer.zero_grad(set_to_none=True);logits,ce_logits=reader(x,v,p,sample[:,:-1],ce[:,:-1])
            rl=policy_loss(logits,sample[:,1:],mask,advantage);nll=F.cross_entropy(ce_logits.flatten(0,1),ce[:,1:].flatten(),ignore_index=0,reduction='sum')
            counts=torch.tensor([int((ce[:,1:]!=0).sum()),len(ev)],device=device);dist.all_reduce(counts);tokens,images=map(int,counts)
            loss=nll*(2/tokens)+.05*rl*(2*len(ev)/images)
            if not torch.isfinite(loss):raise ValueError('Nonfinite SCST+CE loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);optimizer.step()
            state.update(updates=step+1,image_events=image_events(step+1))
            if (step+1)%25==0 or step+1==UPDATES:
                stats=torch.tensor([float(nll.detach()),float(rl.detach())*len(ev),float(sc.sum()),float(gc.sum())],device=device,dtype=torch.float64);dist.all_reduce(stats)
                if rank==0:
                    report=dict(updates=state['updates'],image_events=state['image_events'],ce_loss=float(stats[0])/tokens,scst_loss=float(stats[1])/images,
                        sample_cider=float(stats[2])/images,greedy_cider=float(stats[3])/images,grad_norm=float(norm),seconds=time.monotonic()-started)
                    print('SCST',cfg['arm'],report,flush=True)
                    with (out/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(report,allow_nan=False)+'\n')
                remaining=(time.monotonic()-began)/(step+1-initial)*(UPDATES-step-1)+900
                stop=torch.tensor(int(remaining>deadline-time.monotonic()),device=device);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
                if int(stop):checkpoint();raise TimeoutError('Measured projection exceeds cap; weights saved, no auto-extension')
            if (step+1)%500==0:checkpoint()
        checkpoint()
        if 'end' not in state['validated']:validate('end')
        if rank==0:save(out/'training_complete.json',dict(complete=True,**state,seconds=time.monotonic()-started))
    finally:
        if dist.is_initialized():dist.destroy_process_group()


def score(cfg):
    from run_scst_pilot import score as score_phase
    out=Path(cfg['output']);state=read(out/'progress.json');validate_state(state)
    if state['updates']!=UPDATES or state['validated']!=list(PHASES):raise ValueError('Full epoch incomplete')
    copy_evaluations(out,out,state['validation_files']);metrics={}
    for phase,step in PHASES.items():
        d=out/'evaluations'/phase
        for n in ('vocabulary.json','validation_references.json'):shutil.copy2(out/n,d/n)
        score_phase({**cfg,'output':str(d),'study':'full1','evaluation_step':step})
        metrics[phase]=read(d/'result.json')['metrics']
        if abs(metrics[phase]['CIDEr']-read(d/'cider_replay.json')['CIDEr'])>1e-7:raise ValueError('CIDEr score replay differs')
    save(out/'training_result.json',dict(status='PROMPT_SCST_FULL1_COMPLETE',arm=cfg['arm'],source=SOURCES[cfg['arm']],
        updates=UPDATES,image_events=TRAIN_IMAGES,validation_images=5000,test_used=False,selection='fixed END; no checkpoint selection',metrics=metrics))
    print('SCST FULL1 COMPLETE',cfg['arm'],metrics,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','metric_preflight','preflight','worker','score'));p.add_argument('--config',required=True)
    a=p.parse_args();cfg=read(a.config)
    if a.mode in ('preflight','worker'):distributed(cfg,preflight=a.mode=='preflight')
    elif a.mode=='metric_preflight':
        from run_scst_pilot import metric_preflight
        metric_preflight(cfg)
    else:globals()[a.mode](cfg)
