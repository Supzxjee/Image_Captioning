"""Kaggle-only four-condition scores and no-update forward/backward cost profiles."""
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from diagnostic_data import digest,save
from stale_data import POLICY,contrast_values


def class_score(logits,batch):
    values=[]
    for i,row in enumerate(batch['rows']):
        p=logits[i,row['reference']['position']].float().log_softmax(-1)
        values.append(p[row['alias_ids']].logsumexp(0).clamp(np.log(1e-8),np.log(1-1e-8)))
    return torch.stack(values)


def teacher(model,batch,v,p,m):
    caps,pads=batch['captions'],batch['padding'];repeats=5 if caps.ndim==3 else 1
    caps=caps.flatten(0,1) if repeats==5 else caps;pads=pads.flatten(0,1) if repeats==5 else pads
    memory,mask=model.build_memory(batch[v],batch[p],batch[m])
    logits=model.decoder(caps[:,:-1],memory.repeat_interleave(repeats,0),tgt_key_padding_mask=pads[:,:-1],memory_key_padding_mask=mask.repeat_interleave(repeats,0))
    return logits,caps[:,1:],~pads[:,1:]


def token_mean(values,valid):
    count=valid.sum().float();world=1
    if torch.distributed.is_initialized():
        world=torch.distributed.get_world_size();count=count.detach().clone();torch.distributed.all_reduce(count)
    return (values*valid).sum()*world/count.clamp_min(1)


def stale_loss(stale,neutral):return F.relu(stale-neutral.detach()-.1).mean()


class Objective(nn.Module):
    def __init__(self,student,anchor,arm):
        super().__init__();self.student,self.anchor,self.arm=student,anchor,arm
    def forward(self,main,aux=None):
        logits,targets,valid=teacher(self.student,main,'visual','prompt','mask')
        with torch.no_grad():reference,_,_=teacher(self.anchor,main,'visual','prompt','mask')
        ce=token_mean(F.cross_entropy(logits.transpose(1,2),targets,reduction='none'),valid)
        kl=token_mean(F.kl_div(logits.float().log_softmax(-1),reference.float().softmax(-1),reduction='none').sum(-1),valid)
        loss=ce+.1*kl
        if aux is not None:
            scores={}
            for name,v,p,m in [('11','visual_original','prompt_original','mask_original'),('10','visual_original','prompt_neutral','mask_neutral'),
                              ('01','visual_occluded','prompt_original','mask_original'),('00','visual_occluded','prompt_neutral','mask_neutral')]:
                output,tokens,mask=teacher(self.student,aux,v,p,m);scores[name]=class_score(output,aux)
                if name=='10':neutral=token_mean(F.cross_entropy(output.transpose(1,2),tokens,reduction='none'),mask)
            visual=.5*(F.relu(.2-scores['11']+scores['01'])+F.relu(.2-scores['10']+scores['00'])).mean()
            loss=loss+.1*neutral+.1*visual
            if self.arm=='T':loss=loss+.1*stale_loss(scores['01'],scores['00'])
        return loss


def setup(distributed=False):
    from captioning.scst_runtime import configure_strict_runtime
    from datetime import timedelta
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);torch.set_num_threads(2);configure_strict_runtime(True)
    if int(os.environ['WORLD_SIZE'])!=2:raise ValueError('Select T4x2.')
    if distributed:torch.distributed.init_process_group('nccl',timeout=timedelta(minutes=30))
    return rank


def tensor_batch(g,indices):
    return {k:torch.from_numpy(np.stack([g[k][i] for i in indices])).cuda() for k in g}


def auxiliary(cfg,f,indices):
    from captioning.tokenizer import CaptionTokenizer
    rank=int(os.environ['LOCAL_RANK']);rows=cfg['rows'][rank::2]
    batch=tensor_batch({k:f[k] for k in ('visual_original','visual_occluded','prompt_original','prompt_neutral','mask_original','mask_neutral')},indices)
    tok=CaptionTokenizer([]);tok.word2idx=cfg['word2idx']
    selected=[rows[i] for i in indices];encoded=[tok.encode(r['reference']['caption']) for r in selected]
    batch.update(captions=torch.stack([e[0] for e in encoded]).cuda(),padding=torch.stack([e[1] for e in encoded]).cuda(),rows=selected)
    return batch


def scores(cfg):
    import h5py
    from run_diagnostic import build
    from captioning.scst_runtime import model_digest
    from diagnostic_data import BASE_SHA
    rank=setup();out=Path(cfg['output']);model=build(cfg);records=[]
    with h5py.File(out/f'aux_rank{rank}.h5','r') as f,torch.no_grad():
        if not f.attrs.get('complete'):raise ValueError('Incomplete cache.')
        count=min(len(cfg['rows']),POLICY['frozen_score_images']);mine=list(range(len(cfg['rows'][rank:count:2])))
        for start in range(0,len(mine),4):
            batch=auxiliary(cfg,f,mine[start:start+4]);values={}
            for name,v,p,m in [('11','visual_original','prompt_original','mask_original'),('10','visual_original','prompt_neutral','mask_neutral'),
                              ('01','visual_occluded','prompt_original','mask_original'),('00','visual_occluded','prompt_neutral','mask_neutral')]:
                logits,_,_=teacher(model,batch,v,p,m);values[name]=class_score(logits,batch).cpu().tolist()
            for j,row in enumerate(batch['rows']):
                s={k:v[j] for k,v in values.items()};records.append(dict(coco_id=row['coco_id'],target=row['target'],scores=s,**contrast_values(s['11'],s['10'],s['01'],s['00'])))
    if model_digest(model)!=BASE_SHA:raise ValueError('Frozen diagnostic changed A.')
    save(out/f'four_scores_rank{rank}.json',records)


def trainable(student):
    student.requires_grad_(False)
    for module in (student.encoder.qformer.layers[-1],student.encoder.prompt_projection,student.encoder.prompt_to_visual_attn,
                   student.encoder.prompt_visual_gate,student.encoder.prompt_attn_norm,student.decoder):module.requires_grad_(True)
    return [p for p in student.parameters() if p.requires_grad]


def benchmark(cfg):
    import h5py
    from run_diagnostic import build
    from captioning.scst import rollout
    from captioning.tokenizer import CaptionTokenizer
    from captioning.scst_runtime import runtime_manifest,model_digest
    from diagnostic_data import BASE_SHA
    rank=setup(True);out=Path(cfg['output']);profiles={}
    if digest(out/'benchmark_inputs.h5')!=cfg['benchmark_sha256']:raise ValueError('Benchmark inputs changed.')
    try:
        with h5py.File(out/'benchmark_inputs.h5','r') as source,h5py.File(out/f'aux_rank{rank}.h5','r') as auxfile:
            main_group={k:source['main'][k] for k in source['main']};eval_group={k:source['eval'][k] for k in source['eval']}
            for arm in ('C','T'):
                torch.manual_seed(42);torch.cuda.manual_seed_all(42)
                student=build(cfg);anchor=build(cfg);parameters=trainable(student)
                objective=Objective(student,anchor,arm).eval()
                ddp=nn.parallel.DistributedDataParallel(objective,device_ids=[rank],broadcast_buffers=False)
                torch.distributed.barrier();torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
                for step in range(50):
                    indices=list(range(step*8+rank*4,step*8+rank*4+4))
                    main=tensor_batch(main_group,indices)
                    aux=None
                    if (step+1)%5==0:
                        n=len(auxfile['coco_ids']);base=(step//5)*4;aux=auxiliary(cfg,auxfile,[(base+i)%n for i in range(4)])
                    objective.zero_grad(set_to_none=True);loss=ddp(main,aux)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite benchmark loss.')
                    loss.backward();norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
                    if not torch.isfinite(norm):raise FloatingPointError('Nonfinite benchmark gradient.')
                    if step==9:
                        torch.cuda.synchronize()
                        early=torch.tensor((time.monotonic()-start)/10*1250*1.5+120,device='cuda')
                        torch.distributed.all_reduce(early,op=torch.distributed.ReduceOp.MAX)
                        if float(early)>7200:raise RuntimeError('Ten-batch conservative projection exceeds2h/arm; stop profile, no pilot or automatic batch change.')
                    if step%10==0 and rank==0:print(f'{arm} forward/backward batch {step+1}/50; NO optimizer step',flush=True)
                torch.cuda.synchronize();seconds=time.monotonic()-start
                tok=CaptionTokenizer([]);tok.word2idx=cfg['word2idx'];tok.idx2word={i:w for w,i in tok.word2idx.items()}
                torch.cuda.synchronize();begin=time.monotonic()
                with torch.no_grad():
                    for first in (0,16):
                        b=tensor_batch(eval_group,list(range(first+rank*32,first+rank*32+16)))
                        memory,mask=student.build_memory(b['visual'],b['prompt'],b['mask']);rollout(student,memory,mask,tok,sample=False)
                torch.cuda.synchronize();evaluation=time.monotonic()-begin
                times=torch.tensor([seconds,evaluation,torch.cuda.max_memory_allocated()],device='cuda',dtype=torch.float64)
                torch.distributed.all_reduce(times,op=torch.distributed.ReduceOp.MAX)
                projected=POLICY['projection_safety_factor']*(float(times[0])*25+float(times[1])*1000/64)+120
                optimizer_states=8*sum(p.numel() for p in parameters)
                estimated_memory=int(times[2])+optimizer_states
                if model_digest(student)!=BASE_SHA or model_digest(anchor)!=BASE_SHA:raise ValueError('No-step profile changed weights.')
                profiles[arm]=dict(forward_backward_batches=50,aux_batches=10,optimizer_steps=0,forward_backward_seconds=float(times[0]),greedy64_seconds=float(times[1]),
                    peak_allocated_bytes=int(times[2]),trainable_parameters=sum(p.numel() for p in parameters),
                    estimated_AdamW_state_bytes=optimizer_states,estimated_peak_with_AdamW_bytes=estimated_memory,
                    projected_arm_seconds=projected,within_2h=projected<=7200,
                    estimated_memory_within_T4_headroom=estimated_memory<.9*torch.cuda.get_device_properties(rank).total_memory,
                    projection='1.5*(50-forward/backward time*25+64image greedy time*1000/64)+120sec reserve. Optimizer latency unmeasured; memory states estimated, not allocated; not pilot authorization or guaranteed ETA.')
                del ddp,objective,student,anchor,parameters,loss,main,aux;torch.cuda.empty_cache();torch.distributed.barrier()
    finally:
        torch.distributed.destroy_process_group()
    save(out/f'benchmark_rank{rank}.json',dict(status='DISPOSABLE_PROFILE_COMPLETE',arms=profiles,runtime=runtime_manifest(),
        saved_checkpoint=False,full_pilot_run=False,reset_from_A_for_each_arm=True))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['scores','benchmark']);p.add_argument('--config',required=True)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text(encoding='utf-8'))
    for name,sha in cfg['implementation_sha256'].items():
        if digest(Path(__file__).parent/name)!=sha:raise ValueError('Runtime source changed: '+name)
    globals()[a.mode](cfg)
