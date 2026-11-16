import sys,time
from datetime import timedelta
from pathlib import Path
from statistics import median
import numpy as np
from real_bank_data import read,save
from full_data import prepare,load_stage,FeatureReader,TRAIN_IMAGES
from csa_full_data import order_for_epoch
from csa_sharing_data import image_plan,local_events
from semantic_data import checked_inputs,record_arrays,SemanticContext
from gradient_agreement import gradient_agreement

def stage(cfg):
    start=time.monotonic()
    prepare(cfg)
    meta=time.monotonic()-start
    root=Path(cfg['stage'])
    data,_,_=load_stage(root)
    order=order_for_epoch(data,1)
    events=order[:32*24]
    tail=order[-27:]
    validation=np.arange(TRAIN_IMAGES,TRAIN_IMAGES+64)
    selected=np.unique(np.r_[data['caption_image'][np.r_[events,tail]],validation])
    ids=[int(x) for x in data['ids'][selected]]
    remap={int(v):i for i,v in enumerate(selected)}
    local_images=np.asarray([remap[int(i)] for i in data['caption_image'][np.r_[events,tail]]],dtype=np.int64)
    _,source,_,texts,vectors,_,banks,_=checked_inputs(cfg)
    wanted=set(ids)
    records={}
    with (source/'prompt_links.jsonl').open(encoding='utf8') as stream:
        for row in map(__import__('json').loads,stream):
            cid=row['coco_id']
            if cid in wanted:
                if cid in records:
                    raise ValueError('Duplicate selected prompt')
                records[cid]=record_arrays(row,texts)
    if set(records)!=wanted:
        raise ValueError('Missing selected prompts')
    for name in records[ids[0]]:
        np.save(root/f'sem_{name}.npy',np.stack([records[i][name] for i in ids]))
    np.save(root/'sem_text_vectors.npy',vectors)
    for name,value in banks.items():
        np.save(root/f'sem_{name}_bank.npy',value)
    reader=FeatureReader(root)
    fetch_start=time.monotonic()
    try:
        patches=np.stack([reader.get(i) for i in selected])
    finally:
        reader.close()
    reads=time.monotonic()-fetch_start
    np.savez(root/'probe_inputs.npz',patches=patches,sequences=data['sequences'][np.r_[events,tail]],
             image=local_images,valid=np.asarray([remap[int(i)] for i in validation]))
    save(Path(cfg['output'])/'stage_probe.json',dict(images=len(ids),metadata_seconds=meta,
         selected_patch_read_seconds=reads,total_stage_seconds=time.monotonic()-start,
         train_caption_events=len(events),tail_events=len(tail),valid_images=len(validation)))

def bench(cfg):
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs,InitProcessGroupKwargs
    from csa_shared_model import SharedCSAPatchCaptioner
    from patch_beam import beam_search,check_paths
    root=Path(cfg['stage'])
    out=Path(cfg['output'])
    accelerator=Accelerator(mixed_precision='no',dynamo_backend='no',kwargs_handlers=[
        DistributedDataParallelKwargs(broadcast_buffers=False),
        InitProcessGroupKwargs(backend='nccl',timeout=timedelta(seconds=300))])
    if accelerator.num_processes!=2 or accelerator.device.type!='cuda':
        raise ValueError('Select T4 x2')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    rank=accelerator.process_index
    device=accelerator.device
    with np.load(root/'probe_inputs.npz',allow_pickle=False) as z:
        data={k:z[k] for k in z.files}
    context=SemanticContext(root)
    words=read(root/'vocabulary.json')
    banks=[torch.from_numpy(np.load(root/f'sem_{k}_bank.npy',allow_pickle=False)) for k in ('noun','verb')]
    def build(train):
        torch.manual_seed(42)
        raw=SharedCSAPatchCaptioner(len(words),*banks).to(device)
        raw.train(train)
        return raw,accelerator.prepare_model(raw)
    def step(model,events,shared):
        local=local_events(events,rank)
        first,inverse=image_plan(data['image'],local,shared)
        frames=data['image'][local[first]]
        x=torch.from_numpy(np.asarray(data['patches'][frames],np.float32)).to(device)
        seq=data['sequences'][local].astype(np.int64)
        seq=torch.from_numpy(np.ascontiguousarray(seq[:,:int(np.count_nonzero(seq,axis=1).max())])).to(device)
        prompt=context.images(frames,device)
        gather=None if inverse is None else torch.from_numpy(inverse).to(device)
        scores=model(x,seq[:,:-1],prompt,gather)
        nll=F.cross_entropy(scores.flatten(0,1),seq[:,1:].reshape(-1),ignore_index=0,reduction='sum')
        if not torch.isfinite(nll):
            raise ValueError('Nonfinite benchmark loss')
        tokens=int(np.count_nonzero(data['sequences'][events,1:]))
        accelerator.backward(nll*(2/tokens))
        total=torch.tensor([nll.detach().item(),len(frames)],device=device,dtype=torch.float64)
        dist.all_reduce(total)
        return float(total[0]/tokens),int(total[1].item())

    raw,model=build(False)
    reports=[]
    for label,events in [('first32',np.arange(32)),('tail27',np.arange(768,795))]:
        losses=[]
        grads=None
        failed=[]
        for shared in (False,True):
            model.zero_grad(set_to_none=True)
            loss,_=step(model,events,shared)
            losses.append(loss)
            if rank==0:
                if grads is None:
                    grads={k:v.grad.detach().cpu().numpy().copy() for k,v in raw.named_parameters()}
                else:
                    failed=[k for k,v in raw.named_parameters()
                            if not gradient_agreement(v.grad.detach().cpu().numpy(),grads[k])['passed']]
        if rank==0:
            reports.append(dict(case=label,losses=losses,failed=failed,
                                passed=not failed and abs(losses[0]-losses[1])<=5e-5))
    model.zero_grad(set_to_none=True)
    dist.barrier()
    torch.cuda.synchronize(device)
    started=time.monotonic()
    with torch.inference_mode():
        examples=data['valid'][rank::2]
        for start in range(0,len(examples),8):
            frames=examples[start:start+8]
            x=torch.from_numpy(np.asarray(data['patches'][frames],np.float32)).to(device)
            memory=raw.encode(x,context.images(frames,device))
            seq,scores=beam_search(raw,memory,width=5,maximum=20)
            check_paths(raw,memory,seq,scores)
    torch.cuda.synchronize(device)
    decode=torch.tensor(time.monotonic()-started,device=device,dtype=torch.float64)
    dist.all_reduce(decode,op=dist.ReduceOp.MAX)
    del raw,model
    accelerator.free_memory()
    passes=[]
    for shared in (False,True,True,False):
        raw,model=build(True)
        torch.cuda.manual_seed(4242+rank)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        timings=[]
        images=[]
        torch.cuda.reset_peak_memory_stats(device)
        for i in range(24):
            dist.barrier()
            torch.cuda.synchronize(device)
            started=time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            _,n=step(model,np.arange(i*32,(i+1)*32),shared)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize(device)
            elapsed=torch.tensor(time.monotonic()-started,device=device,dtype=torch.float64)
            dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
            if i>=8:
                timings.append(float(elapsed))
                images.append(n)
        peak=torch.tensor(torch.cuda.max_memory_allocated(device),device=device,dtype=torch.int64)
        dist.all_reduce(peak,op=dist.ReduceOp.MAX)
        passes.append(dict(shared=shared,p50_seconds=median(timings),
                           seconds=timings,encoder_rows=images,peak_bytes=int(peak)))
        del optimizer,raw,model
        accelerator.free_memory()
        torch.cuda.empty_cache()
    if rank==0:
        save(out/'result.json',dict(complete=True,parity=reports,passes=passes,
              valid_decode64_seconds=float(decode),device=torch.cuda.get_device_name(device),
              full_train=False,quality_scored=False))
    accelerator.end_training()

if __name__=='__main__':
    {'stage':stage,'bench':bench}[sys.argv[1]](read(Path(sys.argv[2])))
