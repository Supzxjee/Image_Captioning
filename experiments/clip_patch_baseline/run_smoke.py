"""USER-RUN ONLY: two-GPU TRAIN learning/throughput smoke, no quality scoring."""
import argparse
import copy
from datetime import timedelta
import json
import os
from pathlib import Path
import time
import numpy as np
from real_bank_data import read,save,sha
from patch_data import prepare,load_stage,batch

def worker(cfg):
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.nn import functional as F
    from patch_model import PatchCaptioner
    from cuda_runtime_guard import check_runtime
    rank=int(os.environ['LOCAL_RANK']);out=Path(cfg['output']);check_runtime(out,rank)
    if int(os.environ['WORLD_SIZE'])!=2:raise ValueError('Select T4 x2')
    torch.set_num_threads(1);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(seconds=300));started=time.monotonic()
    try:
        data,words,manifest=load_stage(cfg['stage'])
        def get(indices):
            x,y=batch(data,indices)
            return torch.from_numpy(x).to(device),torch.from_numpy(y).to(device)
        torch.manual_seed(42);model=PatchCaptioner(len(words)).to(device).eval()
        count=sum(p.numel() for p in model.parameters() if p.requires_grad)
        wrapped=nn.parallel.DistributedDataParallel(model,device_ids=[rank],broadcast_buffers=False).eval()
        x,seq=get(np.arange(32));local=slice(rank*16,(rank+1)*16)
        # Math checks do not update weights; compare the same global TRAIN batch.
        with torch.no_grad():
            before=model(x,seq[:,:-1]);changed=seq[:,:-1].clone();changed[:,3]=(changed[:,3]+7)%len(words)
            future=model(x,changed)
            torch.testing.assert_close(before[:,:3],future[:,:3],rtol=1e-4,atol=3e-5)
            memory_change=float((before-model(x.flip(0),seq[:,:-1])).abs().mean())
            if memory_change<=1e-7:raise ValueError('Logits insensitive to patch memory')
            original_nll=float(F.cross_entropy(before.flatten(0,1),seq[:,1:].flatten(),ignore_index=0))
        reference=copy.deepcopy(model).eval() if rank==0 else None
        logits=wrapped(x[local],seq[local,:-1]);tokens=torch.tensor(int((seq[local,1:]!=0).sum()),device=device);dist.all_reduce(tokens)
        loss=F.cross_entropy(logits.flatten(0,1),seq[local,1:].flatten(),ignore_index=0,reduction='sum')*(2/int(tokens));loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('Missing/nonfinite gradient')
        visual_grad=float(model.project[0].weight.grad.norm())
        if visual_grad==0:raise ValueError('No visual projection gradient')
        error=[None]
        if rank==0:
            try:
                expected=reference(x,seq[:,:-1]);ref_loss=F.cross_entropy(expected.flatten(0,1),seq[:,1:].flatten(),ignore_index=0);ref_loss.backward()
                for a,b in zip(model.parameters(),reference.parameters()):torch.testing.assert_close(a.grad,b.grad,rtol=1e-3,atol=1e-5)
                del expected,ref_loss
            except BaseException as exc:error[0]=str(exc)
        dist.broadcast_object_list(error,src=0)
        if error[0]:raise ValueError('Global-token DDP gradient parity failed: '+error[0])
        del reference,before,future,logits,loss,x,seq;model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
        save(out/f'preflight_rank{rank}.json',dict(passed=True,causal_future_invariance=True,memory_logit_mean_change=memory_change,
            token_normalized_DDP_gradient_parity=True,visual_projection_grad_norm=visual_grad,TRAIN_batch_NLL_before=original_nll,parameters=count,updates_discarded=0))
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.99),eps=1e-8,weight_decay=0)
        rng=np.random.default_rng(42);events=[]
        while len(events)<cfg['updates']*32:events.extend(rng.permutation(len(data['sequences'])).tolist())
        events=np.asarray(events[:cfg['updates']*32]).reshape(cfg['updates'],32)
        torch.manual_seed(4242+rank);wrapped.train();times=[];loading=[];losses=[]
        for step,row in enumerate(events):
            stop=torch.tensor(int(time.monotonic()-started>cfg['worker_seconds']-60),device=device);dist.all_reduce(stop,op=dist.ReduceOp.MAX)
            if int(stop):raise TimeoutError('Smoke cap exceeded; no full run implied')
            torch.cuda.synchronize(device);t=time.monotonic();x,seq=get(row[local]);torch.cuda.synchronize(device);loaded=time.monotonic()
            lr=8e-6+(1e-4-8e-6)*min(step/19,1)
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.zero_grad(set_to_none=True);logits=wrapped(x,seq[:,:-1])
            tokens=torch.tensor(int((seq[:,1:]!=0).sum()),device=device);dist.all_reduce(tokens)
            nll=F.cross_entropy(logits.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum')
            loss=nll*(2/int(tokens))
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
            value=nll.detach().double();dist.all_reduce(value)
            torch.cuda.synchronize(device);now=time.monotonic();times.append(now-t);loading.append(loaded-t);losses.append(float(value)/int(tokens))
            if (step+1)%25==0 and rank==0:print('PATCH XE smoke',dict(updates=step+1,TRAIN_NLL=losses[-1],lr=lr,grad_norm=float(norm),seconds=now-started),flush=True)
        wrapped.eval();x,seq=get(np.arange(32))
        with torch.no_grad():
            final_logits=model(x,seq[:,:-1]);final_nll=float(F.cross_entropy(final_logits.flatten(0,1),seq[:,1:].flatten(),ignore_index=0))
            final_memory_change=float((final_logits-model(x.flip(0),seq[:,:-1])).abs().mean())
        if not final_nll<original_nll:raise ValueError('Same TRAIN probe NLL did not decrease')
        samples=model.greedy(x[:8]).cpu().tolist()
        texts=[]
        for row in samples:
            content=[]
            for word in row[1:]:
                if word==2:break
                if word not in (0,1):content.append(words[word])
            texts.append(' '.join(content))
        report=dict(complete=True,rank=rank,updates=cfg['updates'],TRAIN_images=256,caption_events=cfg['updates']*32,
            TRAIN_probe_NLL_before=original_nll,TRAIN_probe_NLL_after=final_nll,training_loss_last25=float(np.mean(losses[-25:])),parameters=count,
            steady_step_median_seconds=float(np.median(times[25:])),steady_step_p90_seconds=float(np.quantile(times[25:],.9)),
            global_caption_events_per_second=32/float(np.median(times[25:])),GPU_name=torch.cuda.get_device_name(device),final_memory_logit_mean_change=final_memory_change,
            steady_load_H2D_median_seconds=float(np.median(loading[25:])),
            peak_allocated_GiB=torch.cuda.max_memory_allocated(device)/1024**3,peak_reserved_GiB=torch.cuda.max_memory_reserved(device)/1024**3,
            seconds=time.monotonic()-started,sample_TRAIN_captions=texts,test_used=False,validation_used=False,
            limits='Tiny TRAIN pool learning/benchmark only; page-cache throughput does not guarantee full-corpus IO. Peak memory includes final global32 TRAIN probe. Not CIDEr or convergence evidence.')
        save(out/f'worker{rank}.json',report);dist.barrier(device_ids=[rank])
        if rank==0:
            reports=[read(out/f'worker{r}.json') for r in (0,1)]
            if not all(r['complete'] and r['updates']==200 for r in reports):raise ValueError('Incomplete smoke ranks')
            save(out/'result.json',dict(status='CLIP_PATCH_BASELINE_TRAIN_SMOKE_COMPLETE',workers=reports,stage_seconds=manifest['seconds'],
                architecture='frozen CLIP-B16 raw196patches -> Linear/LayerNorm512 -> standard4layer causal Transformer decoder -> vocab',
                test_used=False,validation_used=False,full_training_complete=False,training_quality_evaluated=False,
                recommended_next='Review learning/runtime/IO before deciding full XE epochs and a full-corpus staging strategy.'))
    finally:
        if dist.is_initialized():dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','worker'));p.add_argument('--config',required=True)
    a=p.parse_args();cfg=read(a.config)
    if cfg['updates']!=200:raise ValueError('Only200 smoke updates approved in this notebook')
    globals()[a.mode](cfg)
