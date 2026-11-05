"""USER-RUN CUDA/DDP parity for the exact prompt reader; never author-run."""
import argparse
from datetime import timedelta
import os
from pathlib import Path
import numpy as np
from real_bank_data import read,save
from full_cache_data import target_tokens
from sharing_data import rank_events,ddp_loss_multiplier,grouped_order
from prompt_scratch_data import checked_cache,batch


def worker(args):
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from model import Config,CSACaptioner
    from prompt_scratch_model import ScratchCaptioner,to_device
    from run_sharing_benchmark import bank_arrays
    rank=int(os.environ['LOCAL_RANK'])
    from cuda_runtime_guard import check_runtime
    check_runtime(args.output,rank)
    torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    if int(os.environ['WORLD_SIZE'])!=2 or torch.cuda.device_count()!=2:raise ValueError('Select T4x2')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(seconds=600))
    try:
        stage=read(args.stage/'stage_receipt.json');data={k:np.load(args.stage/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in stage['shapes']}
        _,links,emb,texts,_=checked_cache(args.prompt_root,verify=False);banks,_=bank_arrays(args.bank_root)
        order=grouped_order(data,566747);train_ids=set(map(int,data['ids'][data['group']==0]))
        chosen=[]
        for predicate in (lambda r:bool(r['relations']),lambda r:bool(r['objects']) and not r['relations'],lambda r:not r['objects'] and not r['relations']):
            chosen.append(next(i for i in sorted(train_ids) if predicate(links[i])))
        image_index={int(i):j for j,i in enumerate(data['ids'])}
        events=[int(np.searchsorted(data['caption_image'],image_index[i])) for i in chosen]
        cases=[('mixed32',np.resize(np.asarray(events,np.int64),32)),('tail27',order[-27:])]
        reports=[]
        for opened in (False,True):
            for label,indices in cases:
                def build():
                    torch.manual_seed(42)
                    m=ScratchCaptioner(CSACaptioner(Config(vocab=10000,dropout=0),*[torch.from_numpy(a) for a in banks]),args.arm=='PROMPT').to(device).eval()
                    if opened and m.enabled:
                        with torch.no_grad():
                            m.model.object_prompt.output.weight.normal_(std=.005);m.model.relation_prompt.output.weight.normal_(std=.005)
                    return m
                model=DDP(build(),device_ids=[rank],broadcast_buffers=False,find_unused_parameters=False)
                local=rank_events(indices,rank);x,v,seq,p,inverse=to_device(batch(data,local,links,emb,texts,shared=True),device)
                logits=model(x,v,seq[:,:-1],p,inverse)
                loss=F.cross_entropy(logits.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum')
                (loss*ddp_loss_multiplier(2,target_tokens(data,indices))).backward()
                total=loss.detach().double();dist.all_reduce(total);total/=target_tokens(data,indices)
                passed=torch.ones((),dtype=torch.long,device=device);errors={}
                if rank==0:
                    reference=build();reference.load_state_dict(model.module.state_dict(),strict=True)
                    x,v,seq,p,_=to_device(batch(data,indices,links,emb,texts,shared=False),device)
                    expected=reference(x,v,seq[:,:-1],p)
                    ref_loss=F.cross_entropy(expected.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum')/target_tokens(data,indices)
                    ref_loss.backward();difference=abs(float(total)-float(ref_loss.detach()))
                    if not np.isfinite(difference) or difference>5e-5:passed.zero_()
                    params=dict(model.module.named_parameters());refs=dict(reference.named_parameters())
                    probes=['model.base.project.weight','model.base.noun_agg.wa.weight','model.base.head.bias']
                    if args.arm=='PROMPT':probes+=['model.object_prompt.output.weight','model.relation_prompt.output.weight']
                    for name in probes:
                        a=params[name].grad;e=refs[name].grad
                        if a is None or e is None or not torch.isfinite(a).all() or not torch.isfinite(e).all():raise ValueError('Missing/nonfinite probe gradient:'+name)
                        delta=a-e;absolute=float(delta.abs().max());norm=float(e.norm());relative=float(delta.norm())/max(norm,1e-12)
                        errors[name]=dict(max_abs=absolute,relative_l2=relative,reference_norm=norm)
                        if absolute>3e-5 or (norm>1e-10 and relative>.005):passed.zero_()
                    reports.append(dict(arm=args.arm,opened=opened,case=label,global_captions=len(indices),passed=bool(int(passed)),loss_abs_difference=difference,gradients=errors))
                    save(args.output/'prompt_preflight.json',reports);del reference,params,refs,expected,ref_loss
                dist.broadcast(passed,src=0)
                if not int(passed):raise ValueError('Prompt sharing/DDP loss-gradient parity failed')
                del model,logits,loss,x,v,seq,p,inverse;torch.cuda.empty_cache();dist.barrier(device_ids=[rank])
        if rank==0:save(args.output/'prompt_preflight_result.json',dict(passed=True,cases=4,arm=args.arm,training_started=False,guard_weights_discarded=True))
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--bank-root',required=True);p.add_argument('--prompt-root',required=True);p.add_argument('--arm',choices=('CONTROL','PROMPT'),required=True)
    a=p.parse_args();worker(a)
