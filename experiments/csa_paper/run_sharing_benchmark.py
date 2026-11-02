"""USER-RUN: matched two-GPU ordinary/shared CSA benchmark; never run locally."""
import argparse
from datetime import timedelta
import math
import os
from pathlib import Path
import time
import numpy as np
from real_bank_data import HERE,read,save,sha
from pilot_data import load_banks
from full_cache_data import stage_full,collate,target_tokens,stress_events
from sharing_data import grouped_order,rank_events,shared_collate,ddp_loss_multiplier

PROBES=('base.project.weight','base.noun_agg.wa.weight','base.verb_agg.wa.weight','base.decoder.0.gate_o.weight','base.head.bias')

def bank_arrays(root):
    inputs=read(HERE/'REAL_BANK_INPUT_PACKAGE.json');path=HERE/'lemma_correction_review_text/bank_words_corrected.json'
    if sha(path)!=inputs['overlay_sha256']['bank_words_corrected.json']:raise ValueError('Corrected bank SHA differs')
    return load_banks(root,read(path),inputs)

def prepare(args):
    import torch
    if torch.cuda.device_count()!=2:raise RuntimeError('Select GPU T4x2 for this benchmark')
    _,bank_package=bank_arrays(args.bank_root)
    _,_,summary=stage_full(args.full_root,args.stage)
    save(args.output/'stage_summary.json',summary)
    save(args.output/'bank_identity.json',bank_package)
    print('Sharedreadonlystage ready. Next launch two workers.',flush=True)

def worker(args):
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.nn import functional as F
    from torch.nn.parallel import DistributedDataParallel as DDP
    from model import Config,CSACaptioner
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    if world!=2:raise ValueError('Exactlytwo workers required')
    torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(seconds=120))
    try:
        staging=read(args.stage/'stage_receipt.json')
        if rank==0:
            for name,h in staging['files_sha256'].items():
                if sha(args.stage/name)!=h:raise ValueError('Stage bytes differ: '+name)
        dist.barrier(device_ids=[rank])
        data={k:np.load(args.stage/(k+'.npy'),mmap_mode='r',allow_pickle=False) for k in staging['shapes']}
        banks,bank_package=bank_arrays(args.bank_root)
        order=grouped_order(data,566747);batches=order[:1280].reshape(40,32)
        save(args.output/f'order_rank{rank}.json',dict(first1280=order[:1280].tolist(),last27=order[-27:].tolist()))

        class Reader(nn.Module):
            def __init__(self,dropout):
                super().__init__();self.base=CSACaptioner(Config(vocab=10000,dropout=dropout),*[torch.from_numpy(x) for x in banks])
            def forward(self,features,valid,prefix,inverse=None):
                vo,vr=self.base.encode(features,valid)
                if inverse is not None:
                    vo=vo.index_select(0,inverse);vr=vr.index_select(0,inverse);valid=valid.index_select(0,inverse)
                return self.base.decode(prefix,vo,vr,valid)

        def build(dropout):
            torch.manual_seed(42);model=Reader(dropout).to(device);torch.cuda.manual_seed(4242+rank)
            return DDP(model,device_ids=[rank],output_device=rank,broadcast_buffers=False,find_unused_parameters=False)

        def tensors(indices,shared):
            values=shared_collate(data,indices) if shared else (*collate(data,indices),None)
            return [None if x is None else torch.from_numpy(x).to(device) for x in values]

        def backward(model,global_indices,shared,inspect=False):
            local=rank_events(global_indices,rank);features,valid,seq,inverse=tensors(local,shared)
            local_tokens=int((seq[:,1:]!=0).sum());total=torch.tensor(local_tokens,dtype=torch.long,device=device)
            dist.all_reduce(total)
            denominator=int(total)
            if denominator!=target_tokens(data,global_indices):raise ValueError('Global token denominator differs')
            logits=model(features,valid,seq[:,:-1],inverse)
            nll=F.cross_entropy(logits.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum')
            if not torch.isfinite(logits).all() or not torch.isfinite(nll):raise ValueError('Nonfinite DDP forward')
            (nll*ddp_loss_multiplier(world,denominator)).backward()
            if inspect:
                for name,param in model.named_parameters():
                    if param.grad is None or not torch.isfinite(param.grad).all():raise ValueError('Missing/nonfinite gradient '+name)
            stats=torch.tensor([float(nll.detach()),local_tokens,len(features),len(local)],dtype=torch.float64,device=device)
            dist.all_reduce(stats)
            return dict(global_nll=float(stats[0])/denominator,global_tokens=denominator,encoded_images_global=int(stats[2]),
                caption_events_global=int(stats[3]),local_tokens=local_tokens,local_captions=len(local),encoded_images_local=len(features),
                regions=features.shape[1],prefix_length=seq.shape[1]-1)

        def reference_gradients(model,global_indices):
            reference=Reader(0).to(device);reference.load_state_dict(model.module.state_dict(),strict=True);reference.eval()
            features,valid,seq,_=tensors(global_indices,False)
            logits=reference(features,valid,seq[:,:-1]);loss=F.cross_entropy(logits.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum')/target_tokens(data,global_indices)
            loss.backward();params=dict(reference.named_parameters())
            grads={name:params[name].grad.detach().cpu().clone() for name in PROBES};nll=float(loss.detach())
            del reference
            return nll,grads

        preflight=[]
        for shared in (False,True):
            model=build(0);model.eval()
            for label,indices in [('batch32',batches[0]),('tail27',order[-27:]),('stress32',stress_events(data,566747))]:
                if rank==0:ref_nll,ref_grads=reference_gradients(model,indices)
                model.zero_grad(set_to_none=True);stats=backward(model,indices,shared,inspect=True)
                passed=torch.ones((),device=device,dtype=torch.long);metrics={}
                if rank==0:
                    passed.fill_(int(abs(stats['global_nll']-ref_nll)<=5e-5));params=dict(model.module.named_parameters())
                    for name in PROBES:
                        actual=params[name].grad.detach().cpu();expected=ref_grads[name]
                        norm=float(expected.norm());error=actual-expected
                        relative=float(error.norm())/max(norm,1e-12);absolute=float(error.abs().max())
                        metrics[name]=dict(relative_l2=relative,max_abs=absolute,reference_norm=norm)
                        if not all(math.isfinite(x) for x in (norm,relative,absolute)) or norm<=1e-12 or relative>.005 or absolute>3e-5:passed.zero_()
                    entry=dict(shared=shared,case=label,global_nll=stats['global_nll'],reference_nll=ref_nll,gradients=metrics,**{k:v for k,v in stats.items() if k!='global_nll'})
                    preflight.append(entry);save(args.output/'preflight.json',preflight)
                    print('Preflight',shared,label,'loss error',abs(stats['global_nll']-ref_nll),'passed',bool(int(passed)),flush=True)
                    del params
                dist.broadcast(passed,src=0)
                if not int(passed):raise ValueError('DDP/sharing loss-gradient parity guard failed; see preflight.json')
            del model;torch.cuda.empty_cache();dist.barrier(device_ids=[rank])

        combined=[]
        for shared in (False,True):
            name='shared' if shared else 'ordinary';torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            model=build(.2);model.train();optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.99),eps=1e-8,weight_decay=0)
            history=[]
            for i,indices in enumerate(batches):
                dist.barrier(device_ids=[rank]);torch.cuda.synchronize(device);started=time.monotonic()
                optimizer.zero_grad(set_to_none=True);stats=backward(model,indices,shared,inspect=i==0)
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
                torch.cuda.synchronize(device);seconds=torch.tensor(time.monotonic()-started,dtype=torch.float64,device=device)
                dist.all_reduce(seconds,op=dist.ReduceOp.MAX)
                stats.update(step=i+1,timed=i>=10,seconds=float(seconds),grad_norm=float(norm))
                history.append(stats)
                if rank==0:
                    save(args.output/(name+'_steps.json'),history)
                    print(name,'DDP update',i+1,'/40;',float(seconds),'s; encoderrows',stats['encoded_images_global'],'/32',flush=True)
            for kind,original in zip(('nouns','verbs'),banks):
                if getattr(model.module.base,kind).requires_grad or not torch.equal(getattr(model.module.base,kind).detach().cpu(),torch.from_numpy(original)):raise ValueError('Bank changed')
            per_rank=dict(rank=rank,device=torch.cuda.get_device_name(rank),local_captions=16,
                torch_version=str(torch.__version__),numpy_version=np.__version__,cuda_version=torch.version.cuda,
                peak_allocated_mib=torch.cuda.max_memory_allocated(device)/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved(device)/2**20)
            reports=[None,None];dist.all_gather_object(reports,per_rank)
            if rank==0:
                times=np.array([x['seconds'] for x in history[10:]]);median=float(np.median(times))
                result=dict(mode=name,global_batch=32,microbatch_per_rank=16,accumulation=1,world_size=2,precision='FP32',warmup_updates=10,timed_updates=30,
                    median_seconds=median,p90_seconds=float(np.quantile(times,.9)),mean_seconds=float(times.mean()),
                    mean_encoder_images_global=float(np.mean([x['encoded_images_global'] for x in history[10:]])),
                    per_rank=reports,bank_buffers_unchanged=True,train_only_hours_per_epoch=median*17711/3600,train_only_hours_15epochs=median*17711*15/3600)
                combined.append(result);save(args.output/(name+'_result.json'),result)
            del model,optimizer;torch.cuda.empty_cache();dist.barrier(device_ids=[rank])
        if rank==0:
            save(args.output/'result.json',dict(status='DDP_SHARING_BENCHMARK_COMPLETE',modes=combined,
                speed_ratio_ordinary_over_shared=combined[0]['median_seconds']/combined[1]['median_seconds'],preflight_cases=len(preflight),
                optimizer_updates_per_mode=40,checkpoint_saved=False,quality_metrics=False,full_training_started=False,
                bank_package=bank_package,full_cache_manifest_sha256=staging['source_package']['files']['manifest.json'],
                limitations='Matchedgroupedorder32captions; sharingchangesencoderdropoutcorrelations. Old1GPUmicro4/randomcaptiontimingnotmatched. Noauto-fulltraining/billingestimate.'))
            print('DDP SHARING BENCHMARK COMPLETE',flush=True)
    finally:
        if dist.is_initialized():dist.destroy_process_group()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker']);p.add_argument('--full-root');p.add_argument('--bank-root',required=True)
    p.add_argument('--stage',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    globals()[args.mode](args)
