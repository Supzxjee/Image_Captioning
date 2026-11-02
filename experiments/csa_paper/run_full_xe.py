"""USER-RUN ONLY: ordinary two-GPU full CSA XE and validation, first10 epochs."""
import argparse
from datetime import timedelta
import importlib.metadata
import os
from pathlib import Path
import random
import shutil
import time
import numpy as np
from real_bank_data import read,save,sha
from full_cache_data import stage_full,collate,target_tokens
from full_xe_data import TRAIN_EVENTS,STEPS,EPOCHS,learning_rate,epoch_order,validation_references,validate_predictions,validate_state,is_best
from sharing_data import rank_events,ddp_loss_multiplier
from run_sharing_benchmark import bank_arrays


def atomic_json(path,value):
    temp=path.with_suffix('.json.tmp');save(temp,value);os.replace(temp,path)


def prepare(args):
    _,_,summary=stage_full(args.full_root,args.stage)
    banks,package=bank_arrays(args.bank_root)
    save(args.output/'stage_summary.json',summary)
    save(args.output/'bank_identity.json',package)
    refs=validation_references(args.full_root)
    data_ids=np.load(args.stage/'ids.npy',mmap_mode='r',allow_pickle=False)[113287:]
    if data_ids.tolist()!=[x['coco_id'] for x in refs]:raise ValueError('Validation image/reference order differs')
    save(args.output/'validation_references.json',refs)
    expected=summary['source_package']['files']['vocabulary.json']
    matches=[p for p in Path(args.full_root).rglob('vocabulary.json') if sha(p)==expected]
    if len(matches)!=1:raise ValueError('Expected unique verified vocabulary')
    shutil.copy2(matches[0],args.output/'vocabulary.json')
    print('Full TRAIN staged. Validation references separate; no test data.',flush=True)


def metric_preflight(args):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    if importlib.metadata.version('pycocoevalcap')!='1.2':raise ValueError('Expected pycocoevalcap1.2')
    ptb=PTBTokenizer()
    sample={0:[dict(caption='a dog is on a chair')],1:[dict(caption='a bicycle is beside a bus')]}
    tokens=ptb.tokenize(sample);score,_=Cider().compute_score(tokens,tokens)
    if not np.isfinite(score) or score<=0:raise ValueError('CIDEr/Java preflight failed')
    refs=read(args.output/'validation_references.json')
    raw={x['coco_id']:[dict(caption=y) for y in x['references']] for x in refs}
    tokenized=ptb.tokenize(raw)
    save(args.output/'validation_ptb.json',{str(k):v for k,v in tokenized.items()})
    save(args.output/'metric_preflight.json',dict(passed=True,pycocoevalcap='1.2',metric='CIDEr',reference_images=len(raw),sample_score=float(score)))
    print('PTB/Java/CIDEr preflight PASS;5000 validation reference sets tokenized once.',flush=True)


def worker(args):
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from model import Config,CSACaptioner
    from full_xe_decode import batched_beam,beam_parity
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=2 or torch.cuda.device_count()!=2:raise ValueError('Select GPU T4x2')
    torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(seconds=600))
    start=time.monotonic();deadline=start+args.wall_seconds
    try:
        stage=read(args.stage/'stage_receipt.json')
        if rank==0:
            for n,h in stage['files_sha256'].items():
                if sha(args.stage/n)!=h:raise ValueError('Changed staged cache:'+n)
        dist.barrier(device_ids=[rank])
        data={k:np.load(args.stage/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in stage['shapes']}
        banks,bank_package=bank_arrays(args.bank_root)
        refs=read(args.output/'validation_references.json')
        # Vocabulary is pinned as part of ROI source package; preserve it beside source.
        vocab=read(args.output/'vocabulary.json')
        if len(vocab)!=10000:raise ValueError('Expected vocabulary10000')
        identity=dict(schema='csa_full_xe10_v1',source=read(args.output/'source_receipt.json')['sha256'],
            stage_sha256=stage['files_sha256'],bank_package=bank_package,
            validation_references_sha256=sha(args.output/'validation_references.json'),
            ptb_sha256=sha(args.output/'validation_ptb.json'),
            software={k:importlib.metadata.version(k) for k in ('torch','numpy','pycocoevalcap')},
            cuda=torch.version.cuda,world_size=2,global_batch=32,seed=42,precision='FP32',epochs=10)
        random.seed(42+rank);np.random.seed(42+rank);torch.manual_seed(42)
        base=CSACaptioner(Config(vocab=10000),*[torch.from_numpy(x) for x in banks]).to(device)
        model=DDP(base,device_ids=[rank],output_device=rank,broadcast_buffers=False,find_unused_parameters=False)
        torch.cuda.manual_seed(4242+rank)
        optimizer=torch.optim.Adam(base.parameters(),lr=8e-6,betas=(.9,.99),eps=1e-8,weight_decay=0)
        state=dict(epoch=1,batch_cursor=0,phase='train',updates=0,caption_events=0,target_tokens=0,
                   epoch_nll_sum=0.,epoch_tokens=0,best_cider=None,best_epoch=None,history=[])
        last_receipt=None

        def tensors(indices):return [torch.from_numpy(x).to(device) for x in collate(data,indices)]
        # Check batching implementation in eval mode before training or restoring RNG.
        first_val=np.searchsorted(data['caption_image'],[113287,113288])
        x,valid,_=tensors(first_val)
        parity=beam_parity(base,x,valid)
        if rank==0:save(args.output/'beam_preflight.json',parity)
        del x,valid

        if args.resume_root:
            root=Path(args.resume_root);receipt=read(root/'checkpoint_receipt.json')
            checkpoint=root/'checkpoint.bin'
            if sha(checkpoint)!=receipt['sha256']:raise ValueError('Resume checkpoint SHA differs')
            ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
            if ck['identity']!=identity:raise ValueError('Resume source/input/policy/software identities differ')
            state=ck['state'];validate_state(state)
            base.load_state_dict(ck['model'],strict=True);optimizer.load_state_dict(ck['optimizer'])
            rng=ck['rng_by_rank'][rank]
            random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.set_rng_state(rng['torch'])
            torch.cuda.set_rng_state(rng['cuda'],device)
            if rank==0:
                shutil.copy2(checkpoint,args.output/'checkpoint.bin')
                atomic_json(args.output/'checkpoint_receipt.json',receipt)
                if state['best_epoch'] is not None:
                    best=read(root/'best_receipt.json')
                    if sha(root/'best.bin')!=best['sha256']:raise ValueError('Missing/changed best checkpoint; preserve whole output folder')
                    shutil.copy2(root/'best.bin',args.output/'best.bin');atomic_json(args.output/'best_receipt.json',best)
            last_receipt=receipt
            del ck
            print('RESUMED',rank,state['epoch'],state['batch_cursor'],state['phase'],flush=True)
        validate_state(state)
        if rank==0:
            save(args.output/'input_identity.json',identity)
            save(args.output/'runtime.json',dict(world_size=world,devices=[torch.cuda.get_device_name(i) for i in range(2)],
                software=identity['software'],precision='FP32',resume_root=args.resume_root))

        def bank_guard():
            for name,array in zip(('nouns','verbs'),banks):
                value=getattr(base,name)
                if value.requires_grad or not torch.equal(value.detach().cpu(),torch.from_numpy(array)):raise ValueError('Frozen bank changed')

        def checkpoint():
            nonlocal last_receipt
            validate_state(state);bank_guard()
            rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device))
            rngs=[None,None];dist.all_gather_object(rngs,rng)
            if rank==0:
                temp=args.output/'checkpoint.tmp'
                torch.save(dict(model=base.state_dict(),optimizer=optimizer.state_dict(),state=state,identity=identity,rng_by_rank=rngs),temp)
                os.replace(temp,args.output/'checkpoint.bin')
                last_receipt=dict(sha256=sha(args.output/'checkpoint.bin'),epoch=state['epoch'],batch_cursor=state['batch_cursor'],
                    phase=state['phase'],updates=state['updates'],caption_events=state['caption_events'])
                atomic_json(args.output/'checkpoint_receipt.json',last_receipt)
                atomic_json(args.output/'progress.json',state)
            dist.barrier(device_ids=[rank])

        def stop_due():
            flag=torch.tensor(int(time.monotonic()>deadline-180),device=device,dtype=torch.long)
            dist.all_reduce(flag,op=dist.ReduceOp.MAX)
            return bool(int(flag))

        def evaluate(epoch):
            base.eval();indices=np.flatnonzero(data['group']==1);local_images=indices[rank::2]
            local_events=np.arange(TRAIN_EVENTS,len(data['sequences']),dtype=np.int64)
            local_events=local_events[np.isin(data['caption_image'][local_events],local_images)]
            total=0.;count=0;rows=[];began=time.monotonic()
            with torch.inference_mode():
                for offset in range(0,len(local_events),16):
                    if time.monotonic()>deadline:raise TimeoutError('Validation wall cap; resume from pre-validation checkpoint')
                    x,v,seq=tensors(local_events[offset:offset+16]);logits=base(x,v,seq[:,:-1])
                    nll=F.cross_entropy(logits.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum')
                    if not torch.isfinite(nll):raise ValueError('Nonfinite validation NLL')
                    total+=float(nll);count+=int((seq[:,1:]!=0).sum())
                # Eight unique images/rank, each expanded tofive decoder beams.
                first_events=np.searchsorted(data['caption_image'],local_images)
                for offset in range(0,len(first_events),8):
                    if time.monotonic()>deadline:raise TimeoutError('Validation wall cap; resume from pre-validation checkpoint')
                    x,v,_=tensors(first_events[offset:offset+8]);vo,vr=base.encode(x,v)
                    seq,scores=batched_beam(base,vo,vr,v)
                    for j,image in enumerate(local_images[offset:offset+8]):
                        ids=seq[j,0].cpu().tolist()[1:];ended=2 in ids;content=ids[:ids.index(2)] if ended else ids
                        rows.append(dict(coco_id=int(data['ids'][image]),caption=' '.join(vocab[t] for t in content if t!=0),
                            ids=ids,eos=ended,score=float(scores[j,0])))
                    if offset%400==0:print('VAL beam',epoch,'rank',rank,offset,'/2500',flush=True)
            save(args.output/f'val_epoch{epoch:02d}_rank{rank}.json',rows)
            sums=torch.tensor([total,count,len(local_events),len(rows)],device=device,dtype=torch.float64)
            dist.all_reduce(sums)
            if (int(sums[2]),int(sums[3]))!=(25010,5000):raise ValueError('Incomplete validation coverage')
            metrics=[None]
            if rank==0:
                from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
                from pycocoevalcap.cider.cider import Cider
                rows+=read(args.output/f'val_epoch{epoch:02d}_rank1.json');validate_predictions(rows,refs)
                refs_ptb={int(k):v for k,v in read(args.output/'validation_ptb.json').items()}
                hyps=PTBTokenizer().tokenize({x['coco_id']:[dict(caption=x['caption'])] for x in rows})
                score,_=Cider().compute_score(refs_ptb,hyps)
                if not np.isfinite(score):raise ValueError('Nonfinite CIDEr')
                metrics[0]=dict(epoch=epoch,updates=state['updates'],val_nll=float(sums[0]/sums[1]),val_target_tokens=int(sums[1]),
                    cider=float(score),validation_images=5000,references=25010,beam=5,max_new_tokens=20,
                    eos_images=sum(x['eos'] for x in rows),mean_words=float(np.mean([len(x['caption'].split()) for x in rows])),
                    validation_seconds=time.monotonic()-began)
                save(args.output/f'metrics_epoch{epoch:02d}.json',metrics[0])
                print('VALIDATION',metrics[0],flush=True)
            dist.broadcast_object_list(metrics,src=0)
            return metrics[0]

        while state['epoch']<=EPOCHS:
            epoch=state['epoch'];began=time.monotonic()
            if state['phase']=='train':
                order=epoch_order(data,epoch);base.train()
                for batch in range(state['batch_cursor'],STEPS):
                    if batch%25==0 and stop_due():
                        checkpoint()
                        if rank==0:save(args.output/'interrupted.json',dict(reason='wall budget',state=state))
                        return
                    indices=order[batch*32:(batch+1)*32];local=rank_events(indices,rank)
                    x,v,seq=tensors(local);denominator=target_tokens(data,indices)
                    token_count=torch.tensor(int((seq[:,1:]!=0).sum()),device=device,dtype=torch.long)
                    dist.all_reduce(token_count)
                    if int(token_count)!=denominator:raise ValueError('DDP token denominator mismatch')
                    step=state['updates']+1;lr=learning_rate(step)
                    for group in optimizer.param_groups:group['lr']=lr
                    optimizer.zero_grad(set_to_none=True);logits=model(x,v,seq[:,:-1])
                    nll=F.cross_entropy(logits.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum')
                    if not torch.isfinite(nll) or not torch.isfinite(logits).all():raise ValueError('Nonfinite TRAIN logits/loss')
                    (nll*ddp_loss_multiplier(world,denominator)).backward()
                    if step==1:
                        for name,param in base.named_parameters():
                            if param.grad is None or not torch.isfinite(param.grad).all():raise ValueError('Missing/nonfinite gradient:'+name)
                    norm=torch.nn.utils.clip_grad_norm_(base.parameters(),1,error_if_nonfinite=True);optimizer.step()
                    observed=nll.detach().clone();dist.all_reduce(observed)
                    state['updates']=step;state['batch_cursor']=batch+1;state['caption_events']+=len(indices)
                    state['target_tokens']+=denominator;state['epoch_tokens']+=denominator;state['epoch_nll_sum']+=float(observed)
                    if step%100==0 and rank==0:
                        print('TRAIN',dict(epoch=epoch,step=step,total_updates=EPOCHS*STEPS,batch=batch+1,
                            lr=lr,nll=float(observed)/denominator,grad_norm=float(norm),seconds=time.monotonic()-start),flush=True)
                    if step%1000==0:checkpoint()
                state['phase']='validate';checkpoint()
            if stop_due():
                checkpoint()
                if rank==0:save(args.output/'interrupted.json',dict(reason='wall budget before validation',state=state))
                return
            metrics=evaluate(epoch)
            metrics['observed_train_nll']=state['epoch_nll_sum']/state['epoch_tokens']
            metrics['phase_seconds_this_session']=time.monotonic()-began
            better=is_best(metrics['cider'],state['best_cider'])
            if better:state['best_cider']=metrics['cider'];state['best_epoch']=epoch
            state['history'].append(metrics);state['epoch']+=1;state['batch_cursor']=0;state['phase']='train'
            state['epoch_nll_sum']=0.;state['epoch_tokens']=0;checkpoint()
            if rank==0 and better:
                temp=args.output/'best.tmp';shutil.copy2(args.output/'checkpoint.bin',temp);os.replace(temp,args.output/'best.bin')
                atomic_json(args.output/'best_receipt.json',dict(**last_receipt,validated_epoch=epoch,cider=metrics['cider'],selection='max validation CIDEr; earlier ties'))
            dist.barrier(device_ids=[rank])
        if rank==0:
            save(args.output/'result.json',dict(status='FULL_XE_10_COMPLETE',completed_epochs=10,updates=state['updates'],
                caption_events=state['caption_events'],train_images=113287,val_images=5000,test_images=0,
                history=state['history'],best_epoch=state['best_epoch'],best_cider=state['best_cider'],
                session_seconds=time.monotonic()-start,selection='max validation CIDEr; earlier ties',
                precision='FP32',global_batch=32,feature_sharing=False,rl=False,chair_computed=False))
            print('FULL XE10 COMPLETE. Preserve entire output folder including checkpoint.bin and best.bin.',flush=True)
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','metric_preflight','worker'])
    p.add_argument('--full-root');p.add_argument('--bank-root',required=True);p.add_argument('--stage',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume-root',default='');p.add_argument('--wall-seconds',type=float,default=36900)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);globals()[args.mode](args)
