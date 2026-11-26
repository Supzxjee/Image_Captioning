"""USER-RUN ABBA timing from the same v199 state; every update is discarded."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import os
from pathlib import Path
import statistics
import time
import numpy as np
from real_bank_data import read,save,sha
from beam_policy import train_beams,image_losses,rank_microbatch,mechanism_guard
from full_epoch_data import FullTrain
from speed_candidate import batch_rows,train_beams_fast

FULL_SHA='5b8d33fc0b26f2c77ad1cee4a58d3e4a718f13107a2d2bd9a0884c427faed88c'
POSITIONS=(64,96,40000,40032,80000,80032,113280)


def run(cfg):
    import torch
    import torch.distributed as dist
    from torch import nn
    from cuda_runtime_guard import check_runtime
    from csa_patch_model import CSAPatchCaptioner
    from patch_beam import text_from_ids
    from scst_cache import load_reward
    rank=int(os.environ['LOCAL_RANK']);out=Path(cfg['output']);check_runtime(out,rank)
    if int(os.environ['WORLD_SIZE'])!=2:raise ValueError('Select T4 x2')
    torch.set_num_threads(1);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    dist.init_process_group('nccl',timeout=timedelta(seconds=1200))
    deadline=time.monotonic()+cfg['probe_seconds']
    try:
        path=Path(cfg['checkpoint_root'])
        if sha(path)!=FULL_SHA:raise ValueError('Need exact v199 full optimizer/RNG checkpoint')
        ck=torch.load(path,map_location='cpu',weights_only=False)
        expected=read(Path(__file__).with_name('PARENT_CHECKPOINT_RECEIPT.json'))
        if ck['identity']!=expected['identity'] or ck['updates']!=3541 or ck['cursor_images']!=113287:
            raise ValueError('Wrong v199 full state')
        if str(torch.__version__)!=ck['identity']['torch'] or np.__version__!=ck['identity']['numpy']:
            raise RuntimeError('Software differs; preserve log, no reinstall fallback')
        words=read(out/'vocabulary.json')
        if len(words)!=10000:raise ValueError('Wrong vocabulary')
        model=CSAPatchCaptioner(len(words),ck['model']['core.nouns'],ck['model']['core.verbs']).to(device).train()
        if sum(p.numel() for p in model.parameters())!=53183736:raise ValueError('Wrong architecture')
        data=FullTrain(cfg['train_roots']);rewarder=load_reward(cfg['train_roots'][0])
        if set(map(int,np.concatenate(data.ids)))!=set(rewarder.references):raise ValueError('Wrong TRAIN reward cohort')

        class Policy(nn.Module):
            def __init__(self,captioner):super().__init__();self.captioner=captioner;self.fast=False
            def forward(self,patches,prompts):
                function=train_beams_fast if self.fast else train_beams
                return function(self.captioner,self.captioner.encode(patches,prompts))

        policy=Policy(model)
        wrapped=nn.parallel.DistributedDataParallel(policy,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=True)
        optimizer=torch.optim.Adam(wrapped.parameters(),lr=4e-6,betas=(.9,.99),eps=1e-8,weight_decay=0)

        def restore():
            model.load_state_dict(ck['model'],strict=True);optimizer.load_state_dict(ck['optimizer'])
            optimizer.zero_grad(set_to_none=True);rewarder.vectors.clear()
            state=ck['rng'][rank]
            torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state(state['cuda'],device);np.random.set_state(state['numpy'])

        def update(position,fast,capture=False):
            if time.monotonic()>deadline-60:raise TimeoutError('Bounded speed probe; no training weights saved')
            dist.barrier(device_ids=[rank]);torch.cuda.synchronize(device)
            started=time.monotonic();events={};cpu={};paths=[]
            def gpu(name,fn):
                a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                a.record();value=fn();b.record();events.setdefault(name,[]).append((a,b));return value
            policy.fast=fast
            real=min(32,113287-position);optimizer.zero_grad(set_to_none=True)
            summaries=torch.zeros(5,dtype=torch.float64,device=device)
            for micro in range(4):
                rows,keep=rank_microbatch(position,rank,micro)
                weights=torch.tensor(keep,dtype=torch.float32,device=device)
                load=lambda:batch_rows(data,rows,device) if fast else data.batch(rows,device)
                clock=time.monotonic();x,prompt,ids=gpu('batch_H2D_span',load)
                cpu['batch_wall']=cpu.get('batch_wall',0)+time.monotonic()-clock
                sync=wrapped.no_sync() if micro<3 else nullcontext()
                with sync:
                    seq,logps=gpu('beam_forward_span',lambda:wrapped(x,prompt))
                    clock=time.monotonic()
                    tokens=seq.detach().reshape(20,21).cpu().tolist()
                    cpu['caption_copy_wait']=cpu.get('caption_copy_wait',0)+time.monotonic()-clock
                    clock=time.monotonic();captions=[text_from_ids(t,words) for t in tokens]
                    ids5=[image for image in ids for _ in range(5)]
                    rewards=torch.from_numpy(rewarder.score(ids5,captions).reshape(4,5)).to(device)
                    cpu['CPU_reward_and_transfer']=cpu.get('CPU_reward_and_transfer',0)+time.monotonic()-clock
                    if not torch.isfinite(rewards).all():raise FloatingPointError('Nonfinite reward')
                    per=image_losses(logps,rewards);loss=(per*weights).sum()*(2/real)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite SCST loss')
                    if capture:paths.append(dict(seq=seq.detach().cpu(),logps=logps.detach().cpu(),loss=loss.detach().cpu()))
                    gpu('backward_span',lambda:loss.backward())
                summaries[0]+=(per.detach()*weights).sum()
                summaries[1]+=(rewards.mean(-1)*weights).sum()
                summaries[2]+=((rewards-rewards.mean(-1,keepdim=True)).abs().mean(-1)*weights).sum()
                summaries[3]+=weights.sum()
                del x,prompt,seq,logps,loss,rewards,per
            norm=nn.utils.clip_grad_norm_(wrapped.parameters(),float('inf'),error_if_nonfinite=True)
            if capture:
                gradients={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.grad is not None}
            gpu('optimizer_span',lambda:optimizer.step())
            summaries[4]=float(norm)
            gpu('summary_allreduce_span',lambda:dist.all_reduce(summaries))
            values=summaries.cpu().tolist();torch.cuda.synchronize(device)
            result=dict(position=position,real_images=real,fast=fast,wall_seconds=time.monotonic()-started,
                GPU_event_spans_seconds={k:sum(a.elapsed_time(b) for a,b in pairs)/1000 for k,pairs in events.items()},
                CPU_wall_seconds=cpu,SCST=values[0]/values[3],mean_reward=values[1]/values[3])
            if capture:
                return result,dict(paths=paths,gradients=gradients,
                    parameters={n:p.detach().cpu().clone() for n,p in model.named_parameters()},
                    cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(device),numpy_rng=np.random.get_state())
            return result

        # Data equivalence exercises duplicates and unsorted cross-shard batches.
        for rows in ([0,1,2,3],[39999,40000,0,113286],[80000,79999,80000,0],[113286,0,0,0]):
            a,pa,ia=data.batch(rows,device);b,pb,ib=batch_rows(data,rows,device)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            if ia!=ib or pa.keys()!=pb.keys():raise ValueError('Grouped data changed IDs/keys')
            for key in pa:torch.testing.assert_close(pa[key],pb[key],rtol=0,atol=0)
        mechanism_guard(device)
        # Candidate must reject nonfinite logits, even for an unselected vocabulary item.
        class Bad(nn.Module):
            def __init__(self):super().__init__();self.head=nn.Linear(1,9)
            def decode(self,memory,prefix):
                x=torch.zeros((len(prefix),prefix.shape[1],9),device=device);x[:,:,-1]=float('nan');return x
        try:train_beams_fast(Bad().to(device),torch.zeros((1,1,1),device=device),5,2)
        except FloatingPointError:pass
        else:raise ValueError('Candidate silently accepted nonfinite logits')
        restore();base,reference=update(64,False,True)
        restore();candidate,actual=update(64,True,True)
        for a,b in zip(reference['paths'],actual['paths']):
            torch.testing.assert_close(a['seq'],b['seq'],rtol=0,atol=0)
            for key in ('logps','loss'):torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
        for key in ('gradients','parameters'):
            if reference[key].keys()!=actual[key].keys():raise ValueError('Gradient/parameter keys changed')
            for name,value in reference[key].items():torch.testing.assert_close(value,actual[key][name],rtol=1e-6,atol=1e-7)
        for key in ('cpu_rng','cuda_rng'):torch.testing.assert_close(reference[key],actual[key],rtol=0,atol=0)
        for a,b in zip(reference['numpy_rng'],actual['numpy_rng']):
            if not np.array_equal(a,b):raise ValueError('NumPy RNG changed')
        del reference,actual
        save(out/f'equivalence_rank{rank}.json',dict(passed=True,exact_data_paths_logps_loss_and_RNG=True,
            gradients_parameters_rtol=1e-6,gradients_parameters_atol=1e-7,nonfinite_rejected=True,
            optimizer_steps_discarded=2,no_weights_saved=True,full_source_sha256=FULL_SHA))

        blocks=[];torch.cuda.reset_peak_memory_stats(device)
        for number,fast in enumerate((False,True,True,False)):
            restore()
            for position in (0,32):update(position,fast)
            values=[update(position,fast) for position in POSITIONS]
            blocks.append(dict(block=number,variant='candidate' if fast else 'reference',measurements=values))
            save(out/f'timing_rank{rank}.json',dict(blocks=blocks,complete=len(blocks)==4))
            print('SCST speed probe rank',rank,'block',number,'median seconds',statistics.median(v['wall_seconds'] for v in values),flush=True)
        save(out/f'probe_rank{rank}.json',dict(complete=True,blocks=blocks,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            discarded_updates=38,checkpoint_sha256=FULL_SHA,models_trained_for_use=False,training_checkpoint_saved=False))
        dist.barrier(device_ids=[rank])
        if rank==0:
            ranks=[read(out/f'probe_rank{r}.json') for r in (0,1)]
            totals={'reference':[],'candidate':[]};paired=[]
            for block in range(4):
                for j,position in enumerate(POSITIONS):
                    if position==113280:continue # Report tail separately; do not extrapolate seven images as32.
                    seconds=max(r['blocks'][block]['measurements'][j]['wall_seconds'] for r in ranks)
                    variant=ranks[0]['blocks'][block]['variant'];totals[variant].append(seconds)
                    paired.append(dict(block=block,position=position,variant=variant,seconds=seconds))
            median={k:statistics.median(v) for k,v in totals.items()}
            save(out/'result.json',dict(status='SCST_SPEED_PILOT_COMPLETE',checkpoint_source='ducanh3/notebook7bc83c4c76/199',
                median_global32_update_seconds=median,ratio_reference_over_candidate=median['reference']/median['candidate'],
                measured_samples=paired,measured_parent_order='ABBA',reference_epoch1_worker_seconds=11249.117766055,
                one_epoch_training_only_projection={k:v*3541 for k,v in median.items()},
                projections_are_not_ETA_or_quota=True,all_updates_discarded=True,no_VALID_or_TEST=True,
                limits=['Small same-session sample and warm cache; not a completed epoch or accuracy evaluation.',
                        'CUDA event spans may include launch/host gaps; phases overlap and are not additive.',
                        'Accept only after checking per-shard/block timings, invariance and peak memory. No automatic production switch.']))
        dist.barrier(device_ids=[rank])
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('config');run(read(p.parse_args().config))
