"""Full-data Accelerate guards/benchmark; neural calls only in user's worker."""
import copy
from pathlib import Path
import time
import numpy as np
from real_bank_data import save
from full_runtime import tensor_digest
from csa_full_data import SharedCaptionBatches,order_for_epoch
from full_data import CaptionBatches,ImageBatches,loader
from csa_sharing_data import image_plan,local_events
from gradient_agreement import gradient_agreement


def make_model(vocab,cfg):
    import torch
    from csa_shared_model import SharedCSAPatchCaptioner
    if not cfg.get('csa_shared_full') or cfg.get('semantic_strength_gate'):
        raise ValueError('Need CSAsharedfull configuration, no old strength gate')
    banks=[torch.from_numpy(np.load(Path(cfg['stage'])/('sem_'+k+'_bank.npy'),allow_pickle=False)) for k in ('noun','verb')]
    return SharedCSAPatchCaptioner(vocab,*banks)


def preflight(cfg,model,wrapped,data,words,rank,device,out,deadline,accelerator):
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    from semantic_data import SemanticContext
    from patch_beam import beam_search,check_paths
    context=SemanticContext(cfg['stage']);order=order_for_epoch(data,1);model.eval();initial=tensor_digest(model)
    def inputs(events,shared):
        ds=(SharedCaptionBatches if shared else CaptionBatches)(cfg['stage'],events)
        x,seq,ev=ds[0];ds.reader.close()
        first,inverse=image_plan(data['caption_image'],ev,shared)
        return torch.from_numpy(x).to(device),torch.from_numpy(seq).to(device),context.events(data,ev[first],device),None if inverse is None else torch.from_numpy(inverse).to(device)
    parity=[]
    for label,events in [('first32',order[:32]),('tail27',order[-27:])]:
        denominator=int((data['sequences'][events,1:]!=0).sum());reference_grads=None;reference_nll=None
        if rank==0:
            reference=copy.deepcopy(model).eval();reference_nll=0.
            for rr in (0,1):
                x,seq,p,_=inputs(local_events(events,rr),False);logits=reference(x,seq[:,:-1],p)
                loss=F.cross_entropy(logits.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum')/denominator
                loss.backward();reference_nll+=float(loss.detach())
                del x,seq,p,logits,loss
            reference_grads={n:v.grad.detach().cpu().clone() for n,v in reference.named_parameters()};del reference
        ev=local_events(events,rank);x,seq,p,inverse=inputs(ev,True)
        with torch.no_grad():
            expected=wrapped(x,seq[:,:-1],p,inverse);changed=seq[:,:-1].clone();changed[:,3]=(changed[:,3]+7)%len(words)
            torch.testing.assert_close(expected[:,:3],model(x,changed,p,inverse)[:,:3],rtol=1e-4,atol=5e-5)
        wrapped.zero_grad(set_to_none=True);logits=wrapped(x,seq[:,:-1],p,inverse)
        nll=F.cross_entropy(logits.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum')
        total=torch.tensor(int((seq[:,1:]!=0).sum()),device=device);dist.all_reduce(total)
        if int(total)!=denominator:raise ValueError('Wrong full-data globaltokens')
        accelerator.backward(nll*(2/denominator))
        stats=torch.tensor([float(nll.detach()),len(x),len(seq)],dtype=torch.float64,device=device);dist.all_reduce(stats)
        error=[None];diagnostics={}
        if rank==0:
            try:
                if abs(float(stats[0])/denominator-reference_nll)>5e-5:raise ValueError('Full-data reference NLL differs')
                for name,v in model.named_parameters():
                    if v.grad is None or not torch.isfinite(v.grad).all():raise ValueError('Missing/nonfinitegradient '+name)
                    diagnostics[name]=gradient_agreement(v.grad.detach().cpu().numpy(),reference_grads[name].numpy())
                failed=[n for n,d in diagnostics.items() if not d['passed']]
                if failed:raise ValueError('Full-data gradient parity '+','.join(failed))
            except BaseException as exc:error[0]=str(exc)
            parity.append(dict(case=label,passed=error[0] is None,error=error[0],NLL=float(stats[0])/denominator,reference_NLL=reference_nll,
                               global_tokens=denominator,caption_events=int(stats[2]),encoded_images=int(stats[1]),gradient_diagnostics=diagnostics))
            save(out/'full_parity.json',parity)
        dist.broadcast_object_list(error,src=0)
        if error[0]:raise ValueError(error[0])
        del x,seq,p,inverse,logits,nll,reference_grads,expected,changed
    guard_peak=torch.cuda.max_memory_allocated(device)/1024**3;wrapped.zero_grad(set_to_none=True)
    optimizer=accelerator.prepare_optimizer(torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.99),eps=1e-8,weight_decay=0))
    wrapped.train();times=[];encoded=[];torch.cuda.reset_peak_memory_stats(device)
    ds=SharedCaptionBatches(cfg['stage'],order[:64*32],rank)
    iterator=iter(loader(ds))
    for _ in range(64):
        if time.monotonic()>deadline-120:raise TimeoutError('Full data preflight cap')
        torch.cuda.synchronize(device);started=time.monotonic()
        x,seq,events=next(iterator)
        x=x.to(device,non_blocking=True);seq=seq.to(device,non_blocking=True)
        first,inverse=image_plan(data['caption_image'],events,True);p=context.events(data,events[first],device)
        inverse=torch.from_numpy(inverse).to(device);optimizer.zero_grad(set_to_none=True)
        logits=wrapped(x,seq[:,:-1],p,inverse);count=torch.tensor(int((seq[:,1:]!=0).sum()),device=device);dist.all_reduce(count)
        loss=F.cross_entropy(logits.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum')*(2/int(count))
        accelerator.backward(loss);norm=accelerator.clip_grad_norm_(model.parameters(),1)
        if not torch.isfinite(loss) or not torch.isfinite(norm):raise ValueError('Nonfinite preflighttrain')
        optimizer.step();torch.cuda.synchronize(device);times.append(time.monotonic()-started);encoded.append(len(x))
    if len(times)!=64:raise ValueError('Incomplete64step preflight')
    del optimizer,ds,iterator,x,seq,events,logits,loss,p,inverse
    probe=copy.deepcopy(model).eval();indices=np.arange(64)[rank::2]
    with torch.inference_mode():
        # Check real beam scoring before altering ONLY the separate timing copy.
        images=ImageBatches(cfg['stage'],indices[:2]);xx,ii=images[0];images.reader.close()
        memory=probe.encode(torch.from_numpy(xx).to(device),context.images(ii,device));seq,scores=beam_search(probe,memory);check_paths(probe,memory,seq,scores)
        probe.head.bias[2]=-1e9
        torch.cuda.synchronize(device);seen=0;decode_times=[]
        batches=ImageBatches(cfg['stage'],indices);iterator=iter(loader(batches))
        for batch_index in range(len(batches)):
            if time.monotonic()>deadline-30:raise TimeoutError('Forced beam preflight cap')
            began=time.monotonic();xx,ii=next(iterator)
            memory=probe.encode(xx.to(device),context.images(ii,device));seq,scores=beam_search(probe,memory)
            if seq.shape[-1]!=21 or (seq[:,:,1:]==2).any():raise ValueError('Forced20token timing not realized')
            torch.cuda.synchronize(device)
            if batch_index:seen+=len(ii);decode_times.append(time.monotonic()-began)
        decode_seconds=sum(decode_times);del batches,iterator
    save(out/f'preflight_rank{rank}.json',dict(passed=True,rank=rank,initial_tensor_sha256=initial,discarded_benchmark_updates=64,
        causal=True,shared_DDP_gradient_parity=True,beam_score_rescore=True,forced20token_decode=True,
        full_IO_step_median_seconds=float(np.median(times[16:])),full_IO_step_p90_seconds=float(np.quantile(times[16:],.9)),
        mean_unique_images_local=float(np.mean(encoded)),decode_images=seen,decode_seconds=decode_seconds,
        peak_allocated_GiB=torch.cuda.max_memory_allocated(device)/1024**3,gradient_guard_peak_GiB=guard_peak,
        limits='TRAIN-only sampled full-loader preflight; forcedEOS-suppressed beam timing copy, not captions/VALID quality/billing.64updates discarded, mainworker recreated fresh.'))
    del probe;dist.barrier(device_ids=[rank])
