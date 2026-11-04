"""USER-RUN ONLY: paired two-GPU CSA15 SCST, approved fixed500/2500 endpoint."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
from real_bank_data import read,save,sha
from scst_pilot_data import prepare,exact_checkpoint,CHECKPOINT_SHA,MODEL_SHA,study_updates,seen_events
from full_cache_data import collate


def atomic(path,value):
    tmp=path.with_suffix('.tmp');save(tmp,value);os.replace(tmp,path)


def worker(cfg):
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from model import Config,CSACaptioner,beam_search
    from scst_pilot_model import PolicyReader,rollout,captions,policy_loss,selected_logps
    from captioning.scst import TrainingCider
    from captioning.object_reward import TrainingObjectReward
    updates=study_updates(cfg);full=cfg.get('study')=='full1'
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=2 or torch.cuda.device_count()!=2:raise ValueError('Select T4x2')
    torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(seconds=600))
    out=Path(cfg['output']);stage=Path(cfg['stage']);start=time.monotonic();deadline=start+cfg['worker_seconds']
    try:
        receipt=read(out/'pilot_stage.json');staged=read(stage/'stage_receipt.json')
        if rank==0:
            for n,h in receipt['files_sha256'].items():
                if sha(out/n)!=h:raise ValueError('Changed pilot payload:'+n)
            for n,h in staged['files_sha256'].items():
                if sha(stage/n)!=h:raise ValueError('Changed stage:'+n)
        dist.barrier(device_ids=[rank])
        if sha(Path(__file__).with_name('model.py'))!=MODEL_SHA:raise ValueError('CSA core changed')
        data={k:np.load(stage/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in staged['shapes']}
        events=np.load(out/'events.npy',allow_pickle=False);ids=np.load(out/'coco_ids.npy',allow_pickle=False)
        if events.shape!=(updates,8) or ids.shape!=events.shape or receipt['updates']!=updates:raise ValueError('Study schedule/horizon differs')
        vocab=read(out/'vocabulary.json');references={int(k):v for k,v in read(out/'training_references.json').items()}
        cider=TrainingCider(references,reference_cache_size=2048)
        target=read(out/'object_targets.json')
        objects=TrainingObjectReward(target['data'],parser_version='aliases_v2',penalty_version='unique_count_v2')
        ck=torch.load(exact_checkpoint(cfg['checkpoint_root']),map_location='cpu',weights_only=False)
        if ck['state']['updates']!=265665 or ck['state']['epoch']!=16:raise ValueError('Wrong CSA15 cursor')
        torch.manual_seed(42)
        base=CSACaptioner(Config(vocab=10000),ck['model']['nouns'],ck['model']['verbs'])
        base.load_state_dict(ck['model'],strict=True);del ck;base.to(device).eval()
        reader=DDP(PolicyReader(base),device_ids=[rank],broadcast_buffers=False,find_unused_parameters=False)
        reader.eval() # eval disables dropout; gradients remain enabled.
        optimizer=torch.optim.Adam(base.parameters(),lr=5e-6,betas=(.9,.99),eps=1e-8,weight_decay=0)
        if cfg['arm'] not in ('SCST_CE','SCST_CE_OBJECT'):raise ValueError('Unknown arm')
        weight=0. if cfg['arm']=='SCST_CE' else .5
        identity=dict(schema='csa15_scst_full1_v1' if full else 'csa15_scst_pilot_v1',arm=cfg['arm'],object_weight=weight,scst_weight=.05,
            checkpoint_sha256=CHECKPOINT_SHA,stage=receipt,source=read(out/'source_receipt.json'),
            updates=updates,global_batch=8,seed=42,lr=5e-6,maximum_actions=20,dropout=False)
        state=dict(updates=0,image_events=0)
        if full:state.update(validated=[],validation_files={})
        random.seed(42+rank);np.random.seed(42+rank)
        torch.manual_seed(4242+rank);torch.cuda.manual_seed(4242+rank)
        if cfg.get('resume_root'):
            root=Path(cfg['resume_root']);r=read(root/'checkpoint_receipt.json')
            if sha(root/'checkpoint.bin')!=r['sha256']:raise ValueError('Changed resume checkpoint')
            restored=torch.load(root/'checkpoint.bin',map_location='cpu',weights_only=False)
            if restored['identity']!=identity:raise ValueError('Resume source/arm/data/protocol differs')
            base.load_state_dict(restored['model']);optimizer.load_state_dict(restored['optimizer']);state=restored['state']
            rng=restored['rng_by_rank'][rank];random.setstate(rng['python']);np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch']);torch.cuda.set_rng_state(rng['cuda'],device);del restored
            if full:
                from scst_full_data import copy_evaluations,copy_training_log,MID
                if rank==0:
                    copy_evaluations(root,out,state['validation_files'])
                    copy_training_log(root,out,r)
                    if state['updates']>=MID:
                        mr=read(root/'mid_receipt.json')
                        if sha(root/'mid.bin')!=mr['sha256']:raise ValueError('Missing/changed MID resume weights')
                        shutil.copy2(root/'mid.bin',out/'mid.bin');save(out/'mid_receipt.json',mr)
                dist.barrier(device_ids=[rank])
        if not 0<=state['updates']<=updates or state['image_events']!=seen_events(cfg,state['updates']):raise ValueError('Bad resume cursor')
        if full:
            from scst_full_data import validate_state,local_batch
            validate_state(state)
        if rank==0:save(out/'identity.json',identity)

        def checkpoint(tag=None):
            if full:validate_state(state)
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device))
            rngs=[None,None];dist.all_gather_object(rngs,rng)
            if rank==0:
                temp=out/'checkpoint.tmp'
                torch.save(dict(model=base.state_dict(),optimizer=optimizer.state_dict(),state=state,identity=identity,rng_by_rank=rngs),temp)
                os.replace(temp,out/'checkpoint.bin')
                receipt=dict(sha256=sha(out/'checkpoint.bin'),**state,arm=cfg['arm'])
                if full:receipt['training_log_sha256']=sha(out/'training.jsonl') if (out/'training.jsonl').is_file() else None
                atomic(out/'checkpoint_receipt.json',receipt)
                atomic(out/'progress.json',state)
                if tag=='mid':
                    shutil.copy2(out/'checkpoint.bin',out/'mid.bin')
                    save(out/'mid_receipt.json',read(out/'checkpoint_receipt.json'))
            dist.barrier(device_ids=[rank])

        # Actual loaded-model rollout/rescore replay, before any optimizer update.
        x,v,_=[torch.from_numpy(a).to(device) for a in collate(data,events[0,rank*4:rank*4+4])]
        with torch.random.fork_rng(devices=[rank]),torch.no_grad():
            vo,vr=base.encode(x,v);seq,mask,logp=rollout(base,vo,vr,v,True)
            replay=selected_logps(base.decode(seq[:,:-1],vo,vr,v),seq[:,1:],mask)
            torch.testing.assert_close(logp,replay,rtol=1e-4,atol=2e-5)
        save(out/f'policy_replay_rank{rank}.json',dict(passed=True,max_abs=float((logp-replay).abs().max()),dropout=False))
        del x,v,vo,vr,seq,mask,logp,replay
        def validate(destination):
            destination.mkdir(parents=True,exist_ok=True)
            base.eval();rows=[];nll_total=0.;nll_tokens=0
            images=np.flatnonzero(data['group']==1)[rank::2]
            if len(images)!=2500:raise ValueError('VALID must have2500images/rank')
            with torch.inference_mode():
                for offset in range(0,len(images),8):
                    if time.monotonic()>deadline:raise TimeoutError('VALID cap; endpoint weights preserved')
                    ix=images[offset:offset+8];ev=np.searchsorted(data['caption_image'],ix)
                    x,v,_=[torch.from_numpy(a).to(device) for a in collate(data,ev)]
                    vo,vr=base.encode(x,v)
                    for i,(seq,scores) in zip(ix,beam_search(base,vo,vr,v,beams=5,maximum=20)):
                        rows.append(dict(coco_id=int(data['ids'][i]),caption=captions(seq[:1],vocab)[0],ids=seq[0].cpu().tolist(),score=float(scores[0])))
                    if offset%400==0:print('VALID',rank,len(rows),'/2500',flush=True)
                ev=np.flatnonzero(np.isin(data['caption_image'],images))
                for offset in range(0,len(ev),16):
                    if time.monotonic()>deadline:raise TimeoutError('VALID NLL cap; endpoint preserved')
                    x,v,seq=[torch.from_numpy(a).to(device) for a in collate(data,ev[offset:offset+16])]
                    value=F.cross_entropy(base(x,v,seq[:,:-1]).flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum')
                    nll_total+=float(value);nll_tokens+=int((seq[:,1:]!=0).sum())
            save(destination/f'val_rank{rank}.json',rows)
            save(destination/f'worker{rank}.json',dict(complete=True,**state,images=len(rows),nll_sum=nll_total,nll_tokens=nll_tokens,
                torch_version=torch.__version__,seconds=time.monotonic()-start))
            dist.barrier(device_ids=[rank])

        def full_validation(name):
            from scst_full_data import PHASE_STEPS
            if state['updates']!=PHASE_STEPS[name]:raise ValueError('Evaluation at wrong full1 cursor')
            destination=out/'evaluations'/name
            validate(destination)
            files=None
            if rank==0:
                files={str((destination/n).relative_to(out)).replace('\\','/'):sha(destination/n)
                    for n in ('val_rank0.json','val_rank1.json','worker0.json','worker1.json')}
            payload=[files];dist.broadcast_object_list(payload,src=0)
            state['validation_files'].update(payload[0]);state['validated'].append(name)
            checkpoint()

        if full:
            from scst_full_data import MID
            if 'original' not in state['validated']:full_validation('original')
            if state['updates']==MID and 'mid' not in state['validated']:full_validation('mid')
        began=time.monotonic();initial=state['updates']
        for step in range(initial,updates):
            margin=900*(2-int('mid' in state['validated'])) if full else 900
            stop=torch.tensor(int(time.monotonic()>deadline-margin),device=device);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            if int(stop):checkpoint();raise TimeoutError('Saved before worker cap; resume same arm only')
            if full:
                local_events,local_ids=local_batch(events,ids,step,rank)
            else:
                local=slice(rank*4,rank*4+4);local_events=events[step,local];local_ids=ids[step,local]
            batch_ids=local_ids.tolist()
            if not batch_ids:raise ValueError('Unexpected empty SCST rank')
            x,v,ce=[torch.from_numpy(a).to(device) for a in collate(data,local_events)]
            with torch.no_grad():
                vo,vr=base.encode(x,v)
                sample,mask,_=rollout(base,vo,vr,v,True);greedy,_,_=rollout(base,vo,vr,v,False)
            sample_text=captions(sample,vocab);greedy_text=captions(greedy,vocab)
            sc=cider.score(batch_ids,sample_text);gc=cider.score(batch_ids,greedy_text)
            so,sd=objects.score(batch_ids,sample_text);go,gd=objects.score(batch_ids,greedy_text)
            advantage=torch.from_numpy((sc+weight*so)-(gc+weight*go)).to(device)
            if not torch.isfinite(advantage).all():raise ValueError('Nonfinite reward')
            optimizer.zero_grad(set_to_none=True)
            logits,ce_logits=reader(x,v,sample[:,:-1],ce[:,:-1])
            rl=policy_loss(logits,sample[:,1:],mask,advantage)
            nll=F.cross_entropy(ce_logits.flatten(0,1),ce[:,1:].flatten(),ignore_index=0,reduction='sum')
            denominators=torch.tensor([int((ce[:,1:]!=0).sum()),len(batch_ids)],device=device);dist.all_reduce(denominators)
            tokens=int(denominators[0]);images_in_batch=int(denominators[1])
            loss=nll*(world/tokens)+.05*rl*(world*len(batch_ids)/images_in_batch)
            if not torch.isfinite(loss):raise ValueError('Nonfinite SCST loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(base.parameters(),1,error_if_nonfinite=True);optimizer.step()
            state.update(updates=step+1,image_events=seen_events(cfg,step+1))
            stats=torch.tensor([float(nll.detach()),float(rl.detach())*len(batch_ids),float(sc.sum()),float(gc.sum()),float(so.sum()),float(go.sum()),
                sum(d['unsupported_count'] for d in sd),sum(d['unsupported_count'] for d in gd)],device=device,dtype=torch.float64)
            dist.all_reduce(stats)
            if (step+1)%25==0 or step+1==updates:
                report=dict(updates=state['updates'],image_events=state['image_events'],ce_loss=float(stats[0])/int(tokens),scst_loss=float(stats[1])/images_in_batch,
                    sample_cider=float(stats[2])/images_in_batch,greedy_cider=float(stats[3])/images_in_batch,
                    sample_object_reward=float(stats[4])/images_in_batch,greedy_object_reward=float(stats[5])/images_in_batch,
                    sample_unsupported_count=float(stats[6])/images_in_batch,greedy_unsupported_count=float(stats[7])/images_in_batch,
                    grad_norm=float(norm),seconds=time.monotonic()-start)
                if rank==0:
                    print('SCST',cfg['arm'],report,flush=True)
                    with (out/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(report,allow_nan=False)+'\n')
                remaining=(time.monotonic()-began)/(step+1-initial)*(updates-step-1)+margin
                projected_stop=torch.tensor(int(remaining>deadline-time.monotonic()),device=device)
                dist.all_reduce(projected_stop,op=dist.ReduceOp.MAX)
                if int(projected_stop):checkpoint();raise TimeoutError('Measured pilot projection exceeds worker cap; checkpoint preserved')
            if (step+1)%(500 if full else 100)==0:checkpoint()
            if full and step+1==MID:
                checkpoint('mid');full_validation('mid')
        checkpoint()
        if full:
            if 'end' not in state['validated']:full_validation('end')
        else:validate(out)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


def score(cfg):
    import subprocess,sys
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    from saved_quality_data import check_rows,chair_counts
    updates=study_updates(cfg)
    evaluation_step=cfg.get('evaluation_step',updates)
    if cfg.get('study')=='full1':
        from scst_full_data import PHASE_STEPS
        if evaluation_step not in PHASE_STEPS.values():raise ValueError('Unapproved evaluation cursor')
    elif evaluation_step!=updates:raise ValueError('Pilot must score only its endpoint')
    out=Path(cfg['output']);chair=Path(cfg['chair']);refs=read(out/'validation_references.json')
    rows=read(out/'val_rank0.json')+read(out/'val_rank1.json');workers=[read(out/f'worker{r}.json') for r in (0,1)]
    if len(refs)!=5000 or not all(w['complete'] and w['updates']==evaluation_step and w['image_events']==seen_events(cfg,evaluation_step) and w['images']==2500 for w in workers):raise ValueError('Incomplete endpoint/VALID')
    check_rows(rows,{r['coco_id'] for r in refs})
    ptb=PTBTokenizer();ref=ptb.tokenize({r['coco_id']:[dict(caption=c) for c in r['references']] for r in refs})
    hyp=ptb.tokenize({r['coco_id']:[dict(caption=r['caption'])] for r in rows})
    cider,per=Cider().compute_score(ref,hyp);bleu,_=Bleu(4).compute_score(ref,hyp);rouge,_=Rouge().compute_score(ref,hyp)
    if sha(chair/'chair.pkl')!='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':raise ValueError('Changed CHAIR')
    ci=out/'chair_input.json';co=out/'chair_output.json';save(ci,[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
    subprocess.run([sys.executable,'-u',chair/'chair.py','--cap_file',ci,'--cache',chair/'chair.pkl','--save_path',co],cwd=chair,check=True,timeout=600)
    metric,counts=chair_counts(read(co),rows)
    point=dict(CIDEr=float(cider),Bleu_4=float(bleu[3]),ROUGE_L=float(rouge),**metric,counts=counts,
        NLL=sum(w['nll_sum'] for w in workers)/sum(w['nll_tokens'] for w in workers),
        mean_words=sum(len(r['caption'].split()) for r in rows)/5000,short_le2=sum(len(r['caption'].split())<=2 for r in rows))
    if any(not np.isfinite(v) for k,v in point.items() if k!='counts'):raise ValueError('Nonfinite metric')
    save(out/'cider_per_image.json',[dict(coco_id=int(i),CIDEr=float(v)) for i,v in zip(ref,per)])
    save(out/'result.json',dict(status='CSA_SCST_PILOT_COMPLETE',arm=cfg['arm'],updates=evaluation_step,image_events=seen_events(cfg,evaluation_step),
        validation_images=5000,test_used=False,selection=f'fixed endpoint{evaluation_step}; no checkpoint selection',metrics=point))
    print('CSA SCST PILOT COMPLETE',cfg['arm'],point,flush=True)


def metric_preflight(cfg):
    import subprocess,sys
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from saved_quality_data import chair_counts
    out=Path(cfg['output']);chair=Path(cfg['chair'])
    if sha(chair/'chair.pkl')!='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':raise ValueError('Changed CHAIR cache')
    refs=read(out/'validation_references.json')[:2]
    rows=[dict(coco_id=r['coco_id'],caption=r['references'][0]) for r in refs]
    tokens=PTBTokenizer().tokenize({r['coco_id']:[dict(caption=r['caption'])] for r in rows})
    value,_=Cider().compute_score(tokens,tokens)
    if not np.isfinite(value) or value<=0:raise ValueError('Java/PTB/CIDEr failed')
    ci=out/'metric_smoke_input.json';co=out/'metric_smoke_chair.json'
    save(ci,[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
    subprocess.run([sys.executable,'-u',chair/'chair.py','--cap_file',ci,'--cache',chair/'chair.pkl','--save_path',co],cwd=chair,check=True,timeout=180)
    chair_counts(read(co),rows);save(out/'metric_preflight.json',dict(passed=True,images=2,CIDEr=float(value)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','metric_preflight','worker','score'));p.add_argument('--config',required=True)
    a=p.parse_args();cfg=read(a.config)
    if a.mode=='score' and cfg.get('study')=='full1':
        from scst_full_data import score as full_score
        full_score(cfg)
    else:globals()[a.mode](cfg)
