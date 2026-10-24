"""User-run shared H/L pilot; frozen A, CE only, endpoint raw-greedy evaluation."""
import argparse
import copy
import json
import os
from pathlib import Path
import random
import time
import zipfile
from detail_data import *
import detail_stage as stage


def seed():
    import numpy as np
    import torch
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)


def build(cfg):
    import torch
    from captioning.models import UniversalVisionEncoder,CaptionDecoder,ImageCaptioningModel
    from captioning.tokenizer import CaptionTokenizer
    from captioning.scst_runtime import model_digest
    from detail_model import DetailModel
    state=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)
    tokenizer=CaptionTokenizer([]);tokenizer.word2idx=state['caption_word2idx'];tokenizer.idx2word={v:k for k,v in tokenizer.word2idx.items()}
    base=ImageCaptioningModel(UniversalVisionEncoder(load_backbone=False,visual_adapter='qformer',num_visual_queries=32,qformer_layers=2),CaptionDecoder(len(tokenizer.word2idx)))
    if any(not k.startswith('base.') for k in state['model_state_dict']):raise ValueError('Expected exact A state without extra branch.')
    base.load_state_dict({k[5:]:v for k,v in state['model_state_dict'].items()},strict=True);del state
    if model_digest(base)!=BASE_SHA:raise ValueError('A tensor identity mismatch.')
    return DetailModel(base).cuda().eval(),tokenizer


def cuda(batch):return {k:v.cuda(non_blocking=True) for k,v in batch.items()}


def generate(model,loader,tokenizer,condition,deadline):
    import torch
    from captioning.scst import rollout,decode
    result=[]
    with torch.no_grad():
        for i,b in enumerate(loader):
            if time.monotonic()>deadline:raise RuntimeError('Evaluation exceeded the wall budget; keep endpoint, return incomplete review.')
            b=cuda(b);memory,padding=model.memory(b,'A' if condition=='A' else 'on')
            tokens,_=rollout(model,memory,padding,tokenizer,sample=False,max_len=30)
            result.extend(dict(image_id=int(iid),caption=text) for iid,text in zip(b['image_id'].cpu().tolist(),decode(tokens,tokenizer)))
            if i%25==0:print(f'Raw greedy {condition}: {len(result)} local images',flush=True)
    return result


def worker(cfg):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from datetime import timedelta
    from captioning.scst_runtime import configure_strict_runtime,model_digest,runtime_manifest
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK']);out=Path(cfg['output'])
    torch.cuda.set_device(local);torch.set_num_threads(2);configure_strict_runtime(True)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    try:
        if dist.get_world_size()!=2:raise ValueError('Requires one shared model on two GPUs.')
        worker_started=time.monotonic();seed();model,tokenizer=build(cfg);save(out/f'runtime_rank{rank}.json',runtime_manifest())
        data=json.loads(Path(cfg['dataset_json']).read_text(encoding='utf-8'))
        train=stage.loader(cfg,data,tokenizer,'train',rank,epoch=0)
        first=cuda(next(iter(train)))
        frozen=model_digest(model.base);initial=copy.deepcopy(model.head.state_dict());initial_sha=model_digest(model.head)
        initial_hashes=[None,None];dist.all_gather_object(initial_hashes,initial_sha)
        if len(set(initial_hashes))!=1:raise ValueError('Initial adapter differs between ranks.')
        with torch.no_grad():
            a,pa=model.memory(first,'A');on,po=model.memory(first)
            torch.testing.assert_close(a,on,rtol=0,atol=0);torch.testing.assert_close(pa,po,rtol=0,atol=0)
            tgt=first['captions'].flatten(0,1)[:,:-1];mask=first['caption_mask'].flatten(0,1)[:,:-1]
            torch.testing.assert_close(model.teacher(a,pa,tgt,mask),model.teacher(on,po,tgt,mask),rtol=1e-5,atol=1e-5)
            from captioning.scst import rollout
            ga,_=rollout(model,a,pa,tokenizer,sample=False);go,_=rollout(model,on,po,tokenizer,sample=False)
            if not torch.equal(ga,go):raise ValueError('Zero-init greedy does not recover A.')
        # DDP averaging must yield a globally token-weighted mean with uneven counts.
        probe=torch.tensor(1.,device='cuda',requires_grad=True);count=torch.tensor(float(rank+1),device='cuda');total_count=count.clone();dist.all_reduce(total_count)
        (probe*count*2/total_count).backward();g=probe.grad.detach();dist.all_reduce(g);g/=2
        torch.testing.assert_close(g,torch.ones_like(g),rtol=0,atol=1e-6)
        ddp=DDP(model,device_ids=[local],broadcast_buffers=False)
        optimizer=torch.optim.AdamW(model.head.parameters(),lr=1e-4,weight_decay=.01)
        def step(b):
            optimizer.zero_grad(set_to_none=True);loss,stats=ddp(b,tokenizer.pad_idx);loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.head.parameters(),1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):raise ValueError('Nonfinite CE/gradient.')
            if any(p.grad is not None for p in model.base.parameters()):raise ValueError('Frozen A received parameter gradients.')
            optimizer.step();return float(stats[0]/stats[1]),float(stats[1]),float(norm)
        model.train();iterator=iter(train)
        # Profiling updates are discarded, including optimizer state. The actual pilot starts from zero output again.
        for _ in range(3):step(cuda(next(iterator)))
        torch.cuda.synchronize();dist.barrier();start=time.monotonic()
        for _ in range(25):step(cuda(next(iterator)))
        torch.cuda.synchronize();cost=torch.tensor(time.monotonic()-start,device='cuda');dist.all_reduce(cost,op=dist.ReduceOp.MAX)
        if not model.head.o.weight.detach().abs().sum()>0:raise ValueError('No learned adapter output after CE profile.')
        if not all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in model.head.parameters()):raise ValueError('Gradient did not reach all adapter projections.')
        if model_digest(model.base)!=frozen:raise ValueError('A changed during profile.')
        model.head.load_state_dict(initial);model.head.zero_grad(set_to_none=True);optimizer=torch.optim.AdamW(model.head.parameters(),lr=1e-4,weight_decay=.01);model.eval();seed()
        if model_digest(model.head)!=initial_sha:raise ValueError('Profile did not restore exact initial adapter.')
        start=time.monotonic()
        # TRAIN-only decoding sample measures runtime; no validation quality is inspected before training.
        with torch.no_grad():
            for condition in ('A','L','H'):
                it=iter(stage.loader(cfg,data,tokenizer,'train',rank,epoch=0,condition=condition));b=cuda(next(it))
                memory,padding=model.memory(b,'A' if condition=='A' else 'on');rollout(model,memory,padding,tokenizer,sample=False,max_len=30)
        torch.cuda.synchronize();eval_cost=torch.tensor(time.monotonic()-start,device='cuda');dist.all_reduce(eval_cost,op=dist.ReduceOp.MAX)
        projected_train=float(cost)/25*5000;projected_eval=float(eval_cost)*2500/cfg['eval_batch_size']
        projected_total=cfg['preparation_seconds']+(time.monotonic()-worker_started)+1.5*(projected_train+projected_eval)
        profile=dict(status='PASS',training_steps_measured=25,discarded_profile_updates=28,initial_head_restored=True,adapter_parameters=262656,
            zero_init_A_memory_logits_greedy='PASS',global_token_CE_scaling='PASS',all_adapter_gradients='PASS',frozen_A='PASS',
            two_gpu_seconds_per_step=float(cost)/25,projected_training_seconds=projected_train,projected_A_H_L_eval_seconds=projected_eval,
            conservative_projected_total_seconds=projected_total,wall_budget_seconds=cfg['wall_budget_seconds'],
            qualification='Planning estimate only; warm/cold I/O, batch size and caption lengths may change. Not a measurement of Kaggle billed GPU quota.',reported_gpu_quota_hours=cfg['reported_gpu_quota_hours'])
        save(out/f'profile_rank{rank}.json',profile)
        if rank==0:print('PROFILE:',json.dumps(profile),flush=True)
        if projected_total>cfg['wall_budget_seconds']:raise RuntimeError('Projected pilot exceeds the locked8-hour wall budget; preserve profile and review before training.')
        seed();model.train();history=[];completed=0;began=time.monotonic();torch.cuda.reset_peak_memory_stats()
        def checkpoint():
            if rank==0:
                state=dict(format='native_detail_shared_HL_v1',head={k:v.detach().cpu() for k,v in model.head.state_dict().items()},
                    source_A_sha256=A_SHA,completed_updates=completed,caption_word2idx=tokenizer.word2idx,protocol_sha256=digest(out/'protocol.json'))
                tmp=out/'head.bin.partial';torch.save(state,tmp,_use_new_zipfile_serialization=False);tmp.replace(out/'head.bin')
        for epoch in range(4):
            train=stage.loader(cfg,data,tokenizer,'train',rank,epoch)
            for b in train:
                image_ids=b['image_id'].tolist();conditions=b['condition'].tolist();ce,count,norm=step(cuda(b));completed+=1
                history.append(dict(step=completed,epoch=epoch,image_ids=image_ids,conditions=conditions,CE_global=ce,global_target_tokens=count,grad_norm=norm,seconds=time.monotonic()-began))
                if completed==1 or completed%25==0:
                    save(out/f'history_rank{rank}.json',history)
                    if rank==0:print(f'Shared H/L CE {completed}/5000: {ce:.6f}',flush=True)
                    elapsed=torch.tensor(time.monotonic()-worker_started+cfg['preparation_seconds'],device='cuda');dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
                    if float(elapsed)>cfg['wall_budget_seconds']:
                        checkpoint();raise RuntimeError('Wall budget reached; partial head is not a completed pilot.')
            checkpoint()
        save(out/f'history_rank{rank}.json',history)
        if completed!=5000 or model_digest(model.base)!=BASE_SHA:raise ValueError('Wrong endpoint or changed A.')
        model.eval();head_hash=model_digest(model.head)
        head_hashes=[None,None];dist.all_gather_object(head_hashes,head_hash)
        if len(set(head_hashes))!=1:raise ValueError('Trained adapter differs between ranks; do not evaluate.')
        save(out/f'train_complete_rank{rank}.json',dict(status='COMPLETE',updates=completed,initial_head_sha256=initial_sha,final_head_sha256=head_hash,
            final_base_sha256=model_digest(model.base),parameters=262656,trainable_base_parameters=0,training_seconds=time.monotonic()-began,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated()))
        # Training collectives are complete. Unequal decoding runtimes cannot stall the other GPU's loss.
        for condition in ('A','L','H'):
            records=generate(model,stage.loader(cfg,data,tokenizer,'val',rank,condition=condition),tokenizer,condition,worker_started+cfg['wall_budget_seconds']-cfg['preparation_seconds'])
            save(out/f'{condition}_rank{rank}.json',records)
        save(out/f'worker_complete_rank{rank}.json',dict(status='COMPLETE',updates=5000,final_head_sha256=head_hash,final_base_sha256=BASE_SHA))
    finally:dist.destroy_process_group()


def archive(cfg):
    out=Path(cfg['output'])
    with zipfile.ZipFile(str(out)+'_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.log','.py'):z.write(p,p.relative_to(out))
    print('SEND:',str(out)+'_review.zip','KEEP:',out/'head.bin',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','evaluate','archive']);p.add_argument('--config',required=True);args=p.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    locked=dict(epochs=4,updates=5000,global_batch_size=8,eval_batch_size=16,seed=42,head_lr=1e-4,weight_decay=.01,clip_norm=1.,head_parameters=262656)
    if any(cfg.get(k)!=v for k,v in locked.items()):raise ValueError('Locked shared-adapter pilot recipe differs.')
    for name,pin in cfg['implementation_sha256'].items():
        if digest(Path(__file__).resolve().parent/name)!=pin:raise ValueError('Source changed: '+name)
    if args.mode=='prepare':stage.prepare(cfg)
    elif args.mode=='worker':worker(cfg)
    elif args.mode=='evaluate':
        from detail_metrics import evaluate
        evaluate(cfg)
    else:archive(cfg)
