"""User-run neural guards/real IO benchmark. Importing does not import Torch."""
import copy
import hashlib
import time
import numpy as np
from real_bank_data import save
from full_data import CaptionBatches,ImageBatches,loader,order_for_epoch

def tensor_digest(model):
    h=hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(str(tuple(value.shape)).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def preflight(cfg,model,wrapped,data,words,rank,device,out,deadline):
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    from patch_model import PatchCaptioner
    from patch_beam import beam_search,check_paths
    context=None;semantic_report=None
    if cfg.get('semantic'):
        from semantic_training import context_for,neural_guards
        context=context_for(cfg)
    def predict(net,x,prefix,ev):return net(x,prefix,context.events(data,ev,device)) if context else net(x,prefix)
    model.eval();initial=tensor_digest(model);events=order_for_epoch(data,1)
    probe=CaptionBatches(cfg['stage'],events[:32]);x,seq,ev=probe[0];probe.reader.close()
    x=torch.from_numpy(x).to(device);seq=torch.from_numpy(seq).to(device);part=slice(rank*16,(rank+1)*16)
    if context:semantic_report=neural_guards(model,x,context.events(data,ev,device))
    with torch.no_grad():
        logits=predict(model,x,seq[:,:-1],ev);changed=seq[:,:-1].clone();changed[:,3]=(changed[:,3]+7)%len(words)
        torch.testing.assert_close(logits[:,:3],predict(model,x,changed,ev)[:,:3],rtol=1e-4,atol=3e-5)
        if float((logits-predict(model,x.flip(0),seq[:,:-1],ev)).abs().mean())<=1e-7:raise ValueError('Patch conditioning absent')
    ref=copy.deepcopy(model).eval() if rank==0 else None
    pred=predict(wrapped,x[part],seq[part,:-1],ev[part]);count=torch.tensor(int((seq[part,1:]!=0).sum()),device=device);dist.all_reduce(count)
    loss=F.cross_entropy(pred.flatten(0,1),seq[part,1:].flatten(),ignore_index=0,reduction='sum')*(2/int(count));loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()) or float(model.project[0].weight.grad.norm())<=0:raise ValueError('Missing/nonfinite visual/model gradient')
    error=[None]
    if rank==0:
        try:
            F.cross_entropy(predict(ref,x,seq[:,:-1],ev).flatten(0,1),seq[:,1:].flatten(),ignore_index=0).backward()
            for a,b in zip(model.parameters(),ref.parameters()):torch.testing.assert_close(a.grad,b.grad,rtol=1e-3,atol=1e-5)
        except BaseException as exc:error[0]=str(exc)
    dist.broadcast_object_list(error,src=0)
    if error[0]:raise ValueError('DDP gradient parity '+error[0])
    if context:
        for module in (model.object_bank,model.relation_bank,model.object_prompt,model.relation_prompt):
            if float(module.attention.out_proj.weight.grad.norm())<=0:raise ValueError('No semantic output gradient on real probe; send coverage')
    del ref,logits,pred,loss,x,seq;model.zero_grad(set_to_none=True);torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.99),eps=1e-8);wrapped.train();times=[];started=time.monotonic()
    dl=loader(CaptionBatches(cfg['stage'],events[:64*32],rank));iterator=iter(dl)
    for step in range(64):
        if time.monotonic()>deadline-120:raise TimeoutError('Preflight deadline')
        t=time.monotonic();x,seq,ev=next(iterator);x=x.to(device,non_blocking=True);seq=seq.to(device,non_blocking=True)
        opt.zero_grad(set_to_none=True);pred=predict(wrapped,x,seq[:,:-1],ev);n=torch.tensor(int((seq[:,1:]!=0).sum()),device=device);dist.all_reduce(n)
        loss=F.cross_entropy(pred.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum')*(2/int(n))
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);opt.step();torch.cuda.synchronize(device);times.append(time.monotonic()-t)
    del iterator,dl,opt
    torch.manual_seed(42)
    if context:
        from semantic_training import make_model
        decode_model=make_model(len(words),cfg).to(device).eval()
    else:decode_model=PatchCaptioner(len(words)).to(device).eval()
    image_indices=np.arange(128)[rank::2];decode_started=time.monotonic()
    with torch.inference_mode():
        for x,indices in loader(ImageBatches(cfg['stage'],image_indices)):
            xx=x.to(device,non_blocking=True);memory=decode_model.encode(xx,context.images(indices,device)) if context else decode_model.encode(xx);seq,scores=beam_search(decode_model,memory)
            check_paths(decode_model,memory,seq,scores)
    torch.cuda.synchronize(device)
    report=dict(passed=True,rank=rank,semantic=semantic_report,initial_tensor_sha256=initial,discarded_benchmark_updates=64,
        causal=True,memory_conditioning=True,global_token_DDP_gradient_parity=True,beam_score_rescore=True,
        full_IO_step_median_seconds=float(np.median(times[16:])),full_IO_step_p90_seconds=float(np.quantile(times[16:],.9)),
        decode_images=len(image_indices),decode_seconds=time.monotonic()-decode_started,
        peak_allocated_GiB=torch.cuda.max_memory_allocated(device)/1024**3,seconds=time.monotonic()-started,
        caution='Sampled direct-H5/prefetch IO; full-corpus/cache conditions may differ. TRAIN-only beam timing, no quality scoring.')
    save(out/f'preflight_rank{rank}.json',report);dist.barrier(device_ids=[rank])
