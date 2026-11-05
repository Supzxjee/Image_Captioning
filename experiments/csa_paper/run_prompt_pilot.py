"""USER-RUN ONLY: exact161/169 paired500 training, actual CUDA guards and fullVALID."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import time
import numpy as np
from real_bank_data import read,save,sha
from prompt_pilot_data import prepare,checkpoint,cache_root,records,batch,CHECKPOINT_SHA,MODEL_SHA


def atomic(path,value):
    temp=path.with_suffix('.tmp');save(temp,value);os.replace(temp,path)


def load(cfg,rank):
    import torch
    from model import Config,CSACaptioner
    from prompt_pilot_model import configure,reseed
    out=Path(cfg['output']);stage=Path(cfg['stage']);receipt=read(out/'pilot_stage.json')
    for n,h in receipt['files_sha256'].items():
        if sha(out/n)!=h:raise ValueError('Changed pilot payload:'+n)
    staged=read(stage/'stage_receipt.json')
    if staged['files_sha256']!=receipt['source_stage']:raise ValueError('Wrong stage manifest')
    # Rank0 verifies all staged arrays; both ranks verify exact input weights/cache.
    if rank==0:
        for n,h in staged['files_sha256'].items():
            if sha(stage/n)!=h:raise ValueError('Changed staged array:'+n)
    if sha(Path(__file__).with_name('model.py'))!=MODEL_SHA:raise ValueError('Changed CSA core')
    root,emb,texts,_=cache_root(cfg['input_root']);links=records(root)
    data={k:np.load(stage/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in staged['shapes']}
    ck=torch.load(checkpoint(cfg['input_root']),map_location='cpu',weights_only=False)
    if ck['state']['updates']!=14161 or ck['identity']['arm']!='SCST_CE' or ck['state']['validated']!=['original','mid','end']:
        raise ValueError('Wrong161 checkpoint cursor/arm')
    torch.manual_seed(42)
    base=CSACaptioner(Config(vocab=10000),ck['model']['nouns'],ck['model']['verbs'])
    base.load_state_dict(ck['model'],strict=True);del ck
    enabled=cfg['arm']=='PROMPT';model=configure(base,enabled).to(torch.device('cuda',rank)).eval();reseed(rank)
    return model,data,links,emb,texts,receipt


def distributed(cfg,mode):
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from prompt_pilot_model import PromptPolicyReader,to_device
    from scst_pilot_model import rollout,selected_logps,policy_loss,captions
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=2 or torch.cuda.device_count()!=2:raise ValueError('Select T4x2')
    torch.set_num_threads(1);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(seconds=600))
    start=time.monotonic();deadline=start+cfg['worker_seconds'];out=Path(cfg['output']);enabled=cfg['arm']=='PROMPT'
    try:
        model,data,links,emb,texts,receipt=load(cfg,rank);dist.barrier(device_ids=[rank])
        vocab=read(out/'vocabulary.json');events=np.load(out/'events.npy',allow_pickle=False);ids=np.load(out/'coco_ids.npy',allow_pickle=False)
        if events.shape!=(500,8) or ids.shape!=(500,8):raise ValueError('Wrong500/global8 schedule')
        params=[p for p in model.parameters() if p.requires_grad]
        reader=DDP(PromptPolicyReader(model,enabled),device_ids=[rank],broadcast_buffers=False,find_unused_parameters=False).eval()
        optimizer=torch.optim.Adam(params,lr=5e-6,betas=(.9,.99),eps=1e-8,weight_decay=0)

        def get(ev):return to_device(batch(data,ev,links,emb,texts),device)

        if mode=='preflight':
            # Actual paired records, uneven slot coverage across ranks, no held-out supervision.
            rich=sorted(i for i,r in links.items() if r['group']=='train' and r['objects'] and r['relations'])
            if len(rich)<8:raise ValueError('Need8TRAIN records with mapped relations for actual smoke')
            index={int(i):j for j,i in enumerate(data['ids'])}
            ev=[int(np.searchsorted(data['caption_image'],index[i])) for i in rich[rank*4:rank*4+4]]
            (x,v,ce),p,_=get(ev)
            with torch.no_grad(),torch.random.fork_rng(devices=[rank]):
                old=model.base.encode(x,v);new=model.encode(x,v,p)
                for a,b in zip(old,new):torch.testing.assert_close(a,b,rtol=0,atol=0)
                zero={k:t.clone() for k,t in p.items()};zero['object_valid'].zero_();zero['relation_valid'].zero_()
                for a,b in zip(old,model.encode(x,v,zero)):torch.testing.assert_close(a,b,rtol=0,atol=0)
                sample,mask,logp=rollout(model,*new,v,True)
                replay=selected_logps(model.decode(sample[:,:-1],*new,v),sample[:,1:],mask)
                torch.testing.assert_close(logp,replay,rtol=1e-4,atol=2e-5)
                maxdiff=float((logp-replay).abs().max())
            del old,new,sample,mask,logp,replay,zero
            if rank==0:p['object_valid'].zero_();p['relation_valid'].zero_()
            gradient_reports=[]
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                logits,ce_logits=reader(x,v,p,ce[:,:-1],ce[:,:-1])
                nll=F.cross_entropy(ce_logits.flatten(0,1),ce[:,1:].flatten(),ignore_index=0,reduction='sum')
                tokens=torch.tensor(int((ce[:,1:]!=0).sum()),device=device);dist.all_reduce(tokens)
                # Exercise both reader outputs while preserving the CE gradient objective.
                loss=nll*world/int(tokens)+logits.sum()*0
                loss.backward()
                for name,param in model.named_parameters():
                    if param.requires_grad and (param.grad is None or not torch.isfinite(param.grad).all()):raise ValueError('Missing/nonfinite DDP gradient:'+name)
                report={}
                if enabled:
                    for name in ('object_prompt','relation_prompt'):
                        module=getattr(model,name);g=float(module.output.weight.grad.abs().sum());u=float(module.project[0].weight.grad.abs().sum())
                        if g<=0 or (step==1 and u<=0):raise ValueError('Actual adapter gradient path did not open:'+name)
                        report[name]=dict(output_grad_l1=g,upstream_grad_l1=u)
                norm=torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);optimizer.step()
                checksum=torch.stack([t.detach().double().sum() for t in params]).sum()
                copies=[torch.zeros_like(checksum) for _ in range(world)];dist.all_gather(copies,checksum)
                if any(not torch.equal(copies[0],z) for z in copies[1:]):raise ValueError('DDP post-step parameter checksum differs')
                gradient_reports.append(dict(step=step+1,grad_norm=float(norm),adapters=report))
            save(out/f'actual_preflight_rank{rank}.json',dict(passed=True,arm=cfg['arm'],checkpoint_sha256=CHECKPOINT_SHA,
                torch_version=torch.__version__,closed_memory_exact=True,empty_memory_exact=True,policy_replay_max_abs=maxdiff,
                gradient_reports=gradient_reports,smoke_updates_discarded=True,seconds=time.monotonic()-start))
            print('Actual161 CUDA/DDP preflight PASS',cfg['arm'],'rank',rank,'smoke discarded; worker reloads161.',flush=True)
            return

        for r in (0,1):
            check=read(out/f'actual_preflight_rank{r}.json')
            if not check['passed'] or check['arm']!=cfg['arm'] or check['checkpoint_sha256']!=CHECKPOINT_SHA:raise ValueError('Actual preflight missing/wrong arm')
        from captioning.scst import TrainingCider
        references={int(k):v for k,v in read(out/'training_references.json').items()}
        cider=TrainingCider(references,reference_cache_size=2048)
        state=dict(updates=0,image_events=0,validated=[])
        identity=dict(schema='csa_prompt_pilot500_v1',arm=cfg['arm'],checkpoint_sha256=CHECKPOINT_SHA,stage=receipt,
            scst_weight=.05,object_weight=0,seed=42,policy_seed=1234,dropout=False,lr=5e-6,
            trainable_parameters=sum(p.numel() for p in params),test_used=False)
        if rank==0:save(out/'identity.json',identity)

        def save_checkpoint():
            import random
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device))
            rngs=[None,None];dist.all_gather_object(rngs,rng)
            if rank==0:
                temp=out/'checkpoint.tmp'
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),state=state,identity=identity,rng_by_rank=rngs),temp)
                os.replace(temp,out/'checkpoint.bin');atomic(out/'checkpoint_receipt.json',dict(sha256=sha(out/'checkpoint.bin'),arm=cfg['arm'],**state))
                atomic(out/'progress.json',state)
            dist.barrier(device_ids=[rank])

        def validate(phase):
            from model import beam_search
            destination=out/'evaluations'/phase;destination.mkdir(parents=True,exist_ok=True)
            model.eval();rows=[];candidates=[];nll_sum=0.;nll_tokens=0
            images=np.flatnonzero(data['group']==1)[rank::2]
            if len(images)!=2500:raise ValueError('Need2500VALID/rank')
            with torch.inference_mode():
                for offset in range(0,len(images),8):
                    if time.monotonic()>deadline:raise TimeoutError('VALID cap; preserve original output')
                    ix=images[offset:offset+8];ev=np.searchsorted(data['caption_image'],ix)
                    (x,v,_),p,batch_ids=get(ev);memory=model.encode(x,v,p,enabled=enabled)
                    # Original per-image beam algorithm; decoder consumes adapted memory.
                    for i,(seq,scores) in zip(batch_ids,beam_search(model.base,*memory,v,beams=5,maximum=20)):
                        texts_out=captions(seq,vocab)
                        beams=[dict(rank=j,caption=t,ids=seq[j].cpu().tolist(),score=float(scores[j])) for j,t in enumerate(texts_out)]
                        if len(beams)!=5:raise ValueError('Need5beams/image')
                        rows.append(dict(coco_id=int(i),caption=beams[0]['caption'],ids=beams[0]['ids'],score=beams[0]['score']))
                        candidates.append(dict(coco_id=int(i),candidates=beams))
                    if offset%400==0:print('VALID',phase,rank,len(rows),'/2500',flush=True)
                val_events=np.flatnonzero(np.isin(data['caption_image'],images))
                for offset in range(0,len(val_events),16):
                    if time.monotonic()>deadline:raise TimeoutError('VALID NLL cap')
                    (x,v,seq),p,_=get(val_events[offset:offset+16])
                    memory=model.encode(x,v,p,enabled=enabled);logits=model.decode(seq[:,:-1],*memory,v)
                    nll_sum+=float(F.cross_entropy(logits.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum'))
                    nll_tokens+=int((seq[:,1:]!=0).sum())
            save(destination/f'val_rank{rank}.json',rows);save(destination/f'candidates_rank{rank}.json',candidates)
            save(destination/f'worker{rank}.json',dict(complete=True,arm=cfg['arm'],phase=phase,**state,images=2500,
                nll_sum=nll_sum,nll_tokens=nll_tokens,torch_version=torch.__version__,seconds=time.monotonic()-start))
            dist.barrier(device_ids=[rank]);state['validated'].append(phase)

        validate('original');save_checkpoint();began=time.monotonic()
        for step in range(500):
            stop=torch.tensor(int(time.monotonic()>deadline-900),device=device);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            if int(stop):save_checkpoint();raise TimeoutError('Worker cap; checkpoint preserved, no automatic restart')
            local=slice(rank*4,rank*4+4);(x,v,ce),p,batch_ids=get(events[step,local])
            with torch.no_grad():
                memory=model.encode(x,v,p,enabled=enabled)
                sample,mask,_=rollout(model,*memory,v,True);greedy,_,_=rollout(model,*memory,v,False)
            sc=cider.score(batch_ids.tolist(),captions(sample,vocab));gc=cider.score(batch_ids.tolist(),captions(greedy,vocab))
            advantage=torch.from_numpy(sc-gc).to(device)
            if not torch.isfinite(advantage).all():raise ValueError('Nonfinite reward')
            optimizer.zero_grad(set_to_none=True)
            logits,ce_logits=reader(x,v,p,sample[:,:-1],ce[:,:-1])
            rl=policy_loss(logits,sample[:,1:],mask,advantage)
            nll=F.cross_entropy(ce_logits.flatten(0,1),ce[:,1:].flatten(),ignore_index=0,reduction='sum')
            denominators=torch.tensor([int((ce[:,1:]!=0).sum()),len(batch_ids)],device=device);dist.all_reduce(denominators)
            tokens,images_in_batch=map(int,denominators)
            loss=nll*(world/tokens)+.05*rl*(world*len(batch_ids)/images_in_batch)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1,error_if_nonfinite=True);optimizer.step()
            state.update(updates=step+1,image_events=(step+1)*8)
            if (step+1)%25==0:
                stats=torch.tensor([float(nll.detach()),float(rl.detach())*len(batch_ids),float(sc.sum()),float(gc.sum())],device=device,dtype=torch.float64);dist.all_reduce(stats)
                if rank==0:
                    report=dict(**state,ce_loss=float(stats[0])/tokens,scst_loss=float(stats[1])/images_in_batch,
                        sample_cider=float(stats[2])/images_in_batch,greedy_cider=float(stats[3])/images_in_batch,grad_norm=float(norm),seconds=time.monotonic()-start)
                    print('PROMPT PILOT',cfg['arm'],report,flush=True)
                    with (out/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(report,allow_nan=False)+'\n')
                projection=(time.monotonic()-began)/(step+1)*(500-step-1)+900
                stop=torch.tensor(int(projection>deadline-time.monotonic()),device=device);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
                if int(stop):save_checkpoint();raise TimeoutError('Measured projection exceeds cap; checkpoint preserved')
            if (step+1)%100==0:save_checkpoint()
        validate('end');save_checkpoint()
        if enabled:probe(cfg,model,data,links,emb,texts,vocab,rank,device,deadline)
        save(out/f'training_complete_rank{rank}.json',dict(complete=True,arm=cfg['arm'],**state,seconds=time.monotonic()-start))
    finally:
        if dist.is_initialized():dist.destroy_process_group()


def probe(cfg,model,data,links,emb,texts,vocab,rank,device,deadline):
    import torch
    from torch.nn import functional as F
    from prompt_pilot_model import to_device
    from scst_pilot_model import rollout,captions
    chosen=sorted(int(i) for i in data['ids'][data['group']==1])[:64];index={int(i):j for j,i in enumerate(data['ids'])}
    rows=[];model.eval()
    with torch.inference_mode():
        for offset in range(rank*4,len(chosen),8):
            if time.monotonic()>deadline:raise TimeoutError('Probe cap; endpoint weights preserved')
            ids=chosen[offset:offset+4];ev=[int(np.searchsorted(data['caption_image'],index[i])) for i in ids]
            (x,v,ce),p,_=to_device(batch(data,ev,links,emb,texts),device)
            results={}
            for mode in ('normal','zero','shuffle','off'):
                memory=model.encode(x,v,p,text_mode='normal' if mode=='off' else mode,enabled=mode!='off')
                seq,_,_=rollout(model,*memory,v,False);logits=model.decode(ce[:,:-1],*memory,v)
                nll=F.cross_entropy(logits.flatten(0,1),ce[:,1:].flatten(),ignore_index=0,reduction='none').reshape(ce.shape[0],-1)
                results[mode]=dict(captions=captions(seq,vocab),nll_sum=nll.sum(1).cpu().tolist(),logits=logits)
            for j,i in enumerate(ids):
                rows.append(dict(coco_id=i,target_tokens=int((ce[j,1:]!=0).sum()),
                    modes={m:dict(caption=r['captions'][j],nll_sum=r['nll_sum'][j],
                        logit_max_abs_from_normal=float((r['logits'][j]-results['normal']['logits'][j]).abs().max())) for m,r in results.items()}))
    save(Path(cfg['output'])/f'probe_rank{rank}.json',dict(rows=rows,split='VALID',label='sensitivity_only_not_relation_accuracy',test_used=False))


def score(cfg):
    import subprocess,sys
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from saved_quality_data import check_rows,chair_counts
    out=Path(cfg['output']);refs=read(out/'validation_references.json');expected={r['coco_id'] for r in refs};points={}
    for r in (0,1):
        done=read(out/f'training_complete_rank{r}.json')
        if not done['complete'] or done['updates']!=500 or done['image_events']!=4000 or done['validated']!=['original','end']:
            raise ValueError('Incomplete endpoint, do not score a partial run')
    if len(refs)!=5000:raise ValueError('Full5000VALID required')
    chair=Path(cfg['chair'])
    if sha(chair/'chair.pkl')!='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':raise ValueError('Changed CHAIR cache')
    for phase,step in [('original',0),('end',500)]:
        d=out/'evaluations'/phase;rows=read(d/'val_rank0.json')+read(d/'val_rank1.json');check_rows(rows,expected)
        workers=[read(d/f'worker{r}.json') for r in (0,1)]
        if not all(w['complete'] and w['updates']==step and w['images']==2500 and w['arm']==cfg['arm'] for w in workers):raise ValueError('Wrong evaluation cursor/coverage')
        candidates=read(d/'candidates_rank0.json')+read(d/'candidates_rank1.json')
        if len(candidates)!=5000 or {r['coco_id'] for r in candidates}!=expected or any(len(r['candidates'])!=5 for r in candidates):raise ValueError('Incomplete5beam cache')
        lookup={r['coco_id']:r['caption'] for r in rows}
        if any(r['candidates'][0]['caption']!=lookup[r['coco_id']] for r in candidates):raise ValueError('Rank0 candidate differs')
        ptb=PTBTokenizer();reference=ptb.tokenize({r['coco_id']:[dict(caption=c) for c in r['references']] for r in refs})
        hyp=ptb.tokenize({r['coco_id']:[dict(caption=r['caption'])] for r in rows})
        cider,per=Cider().compute_score(reference,hyp);bleu,_=Bleu(4).compute_score(reference,hyp);rouge,_=Rouge().compute_score(reference,hyp)
        ci=d/'chair_input.json';co=d/'chair_output.json';save(ci,[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
        subprocess.run([sys.executable,'-u',chair/'chair.py','--cap_file',ci,'--cache',chair/'chair.pkl','--save_path',co],cwd=chair,check=True,timeout=300)
        metric,counts=chair_counts(read(co),rows)
        point=dict(CIDEr=float(cider),Bleu_4=float(bleu[3]),ROUGE_L=float(rouge),**metric,counts=counts,
            NLL=sum(w['nll_sum'] for w in workers)/sum(w['nll_tokens'] for w in workers),
            mean_words=sum(len(r['caption'].split()) for r in rows)/5000,short_le2=sum(len(r['caption'].split())<=2 for r in rows))
        if any(not np.isfinite(v) for k,v in point.items() if k!='counts'):raise ValueError('Nonfinite metric')
        save(d/'cider_per_image.json',[dict(coco_id=int(i),CIDEr=float(v)) for i,v in zip(reference,per)]);save(d/'result.json',point);points[phase]=point
    if cfg['arm']=='PROMPT':
        probes=read(out/'probe_rank0.json')['rows']+read(out/'probe_rank1.json')['rows']
        if len(probes)!=64 or len({r['coco_id'] for r in probes})!=64 or not {r['coco_id'] for r in probes}<=expected:raise ValueError('Incomplete sensitivity probe')
        save(out/'probe_summary.json',dict(images=64,caption_changes={m:sum(r['modes'][m]['caption']!=r['modes']['normal']['caption'] for r in probes) for m in ('zero','shuffle','off')},
            interpretation='Sensitivity only; not verified relation accuracy or isolated text contribution.'))
    result=dict(status='PROMPT_PILOT500_COMPLETE',arm=cfg['arm'],updates=500,image_events=4000,validation_images=5000,
        test_used=False,selection='Fixed endpoint500, no checkpoint selection',rerank_measured=False,metrics=points)
    save(out/'result.json',result);print('PROMPT PILOT500 COMPLETE',cfg['arm'],points,flush=True)


def metric_preflight(cfg):
    from run_scst_pilot import metric_preflight as existing
    existing(cfg)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','metric_preflight','preflight','worker','score'));p.add_argument('--config',required=True)
    a=p.parse_args();cfg=read(a.config)
    if a.mode in ('preflight','worker'):distributed(cfg,a.mode)
    else:globals()[a.mode](cfg)
