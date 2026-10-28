"""User-run preparation, strict two-GPU preflight, fixed endpoint training/evaluation."""
import argparse
from datetime import timedelta
import gzip
import json
import os
from pathlib import Path
import time
import numpy as np
from frame_data import Cache, CHECKPOINT_SHA, FILES, META_SHA, metadata, save, schedule, sha


def prepare(config):
    out=Path(config['output'])
    if out.exists():raise FileExistsError('Existing output preserved; use a fresh run directory')
    if config['arm'] not in ('RelPos','RelContrast') or config['updates']!=200 or config['seed']!=42:raise ValueError('Locked protocol differs')
    if config['cache_sha256']!=FILES or config['checkpoint_sha256']!=CHECKPOINT_SHA or config['metadata_sha256']!=META_SHA:raise ValueError('Pinned identities differ')
    if not 0<config['device_gpu_hours_cap']<=.5:raise ValueError('At most .5 device-GPU-hour per arm')
    print('Verifying every v105 payload, including full HDF5.',flush=True)
    for n,h in FILES.items():
        print('Hashing:',n,flush=True)
        if sha(Path(config['cache'])/n)!=h:raise ValueError('Changed input '+n)
    print('Hashing pinned A checkpoint.',flush=True)
    if sha(config['checkpoint'])!=CHECKPOINT_SHA:raise ValueError('Wrong A checkpoint')
    vocab,images,pairs=metadata(config['cache'],config['metadata'],verify=False)
    cache=Cache(config['cache'],config['metadata'])
    for split in pairs:
        for i in range(len(pairs[split])):
            b=cache.batch(split,[i])
            if any(not np.isfinite(a).all() for a in b.values()):raise ValueError('Nonfinite prepared batch')
    out.mkdir(parents=True)
    save(out/'protocol.json',config)
    save(out/'preparation.json',dict(status='PREPARATION_PASS',images=len(images),train_pairs=len(pairs['train']),
        heldout_pairs=len(pairs['heldout']),payload_sha256=FILES,checkpoint_sha256=CHECKPOINT_SHA,
        manifest_unchanged=True,gt_used_as_model_input=False,original_prompt_used=False,
        note='Prompt cache preserved in dataset but not consumed by frame-only pretraining. Human audit accepted separately.'))
    print('Preparation PASS; no model forward. Ready for T4 x2.',flush=True)


def run(config):
    import hashlib
    import torch
    import torch.distributed as dist
    from frame_model import FrameModel
    from scst_runtime import configure_strict_runtime, model_digest, tensor_digest
    configure_strict_runtime(True)
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=2 or torch.cuda.device_count()!=2:raise RuntimeError('Require exactly two visible GPUs')
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    dist.init_process_group('nccl',timeout=timedelta(minutes=25))
    start=time.monotonic();out=Path(config['output']);arm=config['arm'];completed=0

    def sync_status(value):
        values=[None,None];dist.all_gather_object(values,value);return values

    def tensor_batch(indices,split='train'):
        return {k:torch.from_numpy(v).to(device) for k,v in cache.batch(split,indices).items()}

    def reduce_gradients(model):
        params=[p for p in model.parameters() if p.requires_grad]
        used=torch.tensor([int(p.grad is not None) for p in params],device=device)
        dist.all_reduce(used,op=dist.ReduceOp.MAX)
        used=used.tolist()
        flat=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in params])
        if not torch.isfinite(flat).all():raise FloatingPointError('Nonfinite gradient')
        dist.all_reduce(flat);flat.div_(2)
        offset=0
        for i,p in enumerate(params):
            p.grad=flat[offset:offset+p.numel()].view_as(p).clone() if used[i] else None
            offset+=p.numel()
        return flat

    def check_budget():
        expired=torch.tensor(int((time.monotonic()-start)*2/3600>config['device_gpu_hours_cap']),device=device)
        dist.all_reduce(expired,op=dist.ReduceOp.MAX)
        if expired.item():
            raise TimeoutError('Device-GPU budget reached; incomplete, no endpoint selection')

    try:
        if rank==0:
            for n,h in FILES.items():
                if sha(Path(config['cache'])/n)!=h:raise ValueError('Changed cache before GPU: '+n)
            if sha(config['checkpoint'])!=CHECKPOINT_SHA:raise ValueError('Changed checkpoint before GPU')
        dist.barrier()
        if json.loads((out/'protocol.json').read_text())!=config:raise ValueError('Protocol differs from preparation')
        cache=Cache(config['cache'],config['metadata'])
        torch.manual_seed(42);np.random.seed(42)
        state=torch.load(config['checkpoint'],map_location='cpu',weights_only=False)
        if state['caption_word2idx']!=cache.vocab:raise ValueError('Checkpoint vocabulary differs')
        emb=state['model_state_dict']['base.decoder.embedding.weight'].float().clone();del state
        model=FrameModel(emb).to(device);del emb
        initial=model_digest(model);frozen=tensor_digest(model.embedding.weight)
        if len(set(sync_status(initial)))!=1:raise ValueError('Rank initialization differs')
        original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        model.eval();b=tensor_batch([0,1])
        with torch.no_grad():
            a=b['areas'][:,:2]
            if not torch.isfinite(a).all() or not (a.sum(-1)>0).all():raise ValueError('Invalid ROI normalization')
            constant=torch.bmm(a/a.sum(-1,keepdim=True),torch.ones_like(b['visual']))
            torch.testing.assert_close(constant,torch.ones_like(constant),rtol=1e-5,atol=1e-5)
            whole=model.encode(b)
            individual=torch.cat([model.encode({k:v[i:i+1] for k,v in b.items() if k in ('visual','areas','geometry','classes')}) for i in range(4)])
            max_batch_error=float((whole-individual).abs().max())
            torch.testing.assert_close(whole,individual,rtol=1e-5,atol=1e-5)
            gaps=model.ranking_gap(b)
            if not torch.isfinite(gaps).all():raise ValueError('Nonfinite ranking')
        # cuDNN GRU must run its forward in training mode to support backward.
        model.train()
        # Actual model/autograd checks stay in the user-run notebook, never local.
        def require_gradient(name):
            grads=[p.grad for p in getattr(model,name).parameters()]
            if not any(g is not None and torch.isfinite(g).all() and g.abs().sum()>0 for g in grads):raise ValueError('Missing/nonfinite gradient: '+name)

        model.zero_grad(set_to_none=True)
        positive_loss,_=model.objective(b,'RelPos');positive_loss.backward()
        for name in ('entity','query','key','value','classes','frame','initial','gru','output'):require_gradient(name)
        if model.truth.weight.grad is not None:raise ValueError('Positive-only loss trained truth head')
        model.zero_grad(set_to_none=True)
        r=model.encode(b);d=model.truth(r).gather(1,b['relation'].repeat_interleave(2)[:,None]).squeeze(1)
        aux=.25*(torch.nn.functional.softplus(-d[::2]).mean()+torch.nn.functional.softplus(d[1::2]).mean())+.5*torch.nn.functional.softplus(-(d[::2]-d[1::2])).mean()
        aux.backward()
        for name in ('frame','entity','truth'):require_gradient(name)
        if any(p.grad is not None for p in model.gru.parameters()):raise ValueError('Truth-only loss reached auxiliary GRU')
        model.zero_grad(set_to_none=True);one=tensor_batch([0]);loss,_=model.objective(one,'RelContrast');loss.backward()
        observed=int(one['relation'][0]);other=[i for i in range(16) if i!=observed]
        if torch.count_nonzero(model.truth.weight.grad[other]) or torch.count_nonzero(model.truth.bias.grad[other]):raise ValueError('Unknown predicates received supervision')
        for name in ('entity','query','key','value','classes','frame','initial','gru','output','truth'):
            require_gradient(name)
        if model.embedding.weight.grad is not None:raise ValueError('Frozen embedding gradient')
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
        model.zero_grad(set_to_none=True)
        serial=tensor_batch(schedule(0));serial_loss,_=model.objective(serial,arm);serial_loss.backward()
        expected_grad=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in model.parameters() if p.requires_grad]).detach().clone()
        model.zero_grad(set_to_none=True)
        sample=tensor_batch(schedule(0)[rank*8:(rank+1)*8])
        loss,_=model.objective(sample,arm);loss.backward();actual_grad=reduce_gradients(model)
        max_gradient_error=float((actual_grad-expected_grad).abs().max())
        torch.testing.assert_close(actual_grad,expected_grad,rtol=1e-5,atol=1e-5)
        if arm=='RelPos' and model.truth.weight.grad is not None:raise ValueError('Unused RelPos truth parameters received optimizer gradients')
        del actual_grad,expected_grad,serial,serial_loss
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
        optimizer.step()
        if len(set(sync_status(model_digest(model))))!=1:raise ValueError('Synchronized optimizer states differ')
        model.load_state_dict(original,strict=True);model.zero_grad(set_to_none=True);del original,optimizer
        if model_digest(model)!=initial or tensor_digest(model.embedding.weight)!=frozen:raise ValueError('Preflight did not restore initial model')
        preflight=dict(status='MODEL_PREFLIGHT_PASS',initial_model_digest=initial,frozen_embedding_digest=frozen,
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
            frozen_parameters=sum(p.numel() for p in model.parameters() if not p.requires_grad),
            auxiliary_output_head_parameters=sum(p.numel() for p in model.output.parameters()),
            max_batch_error=max_batch_error,comparison_atol=1e-5,comparison_rtol=1e-5,
            max_distributed_vs_serial_gradient_error=max_gradient_error,
            ranks=2,unknown_predicate_gradients_zero=True,synchronized_step_checked=True,
            isolated_positive_gradient_to_frame=True,isolated_truth_gradient_to_frame=True,
            constant_feature_pooling_invariant=True,state_restored=True,torch=torch.__version__,gpu=torch.cuda.get_device_name(local))
        if preflight['auxiliary_output_head_parameters']!=2570000:raise ValueError('Output parameter count differs')
        if rank==0:save(out/'preflight.json',preflight);print(json.dumps(preflight),flush=True)
        check_budget()
        model.train();optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=.01)
        for step in range(200):
            check_budget();optimizer.zero_grad(set_to_none=True)
            b=tensor_batch(schedule(step)[rank*8:(rank+1)*8])
            loss,terms=model.objective(b,arm)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            loss.backward();reduce_gradients(model)
            norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
            optimizer.step();completed=step+1
            if completed==1 or completed%10==0:
                row=dict(step=completed,arm=arm,loss=float(loss.detach()),grad_norm=float(norm),
                    seconds=time.monotonic()-start,**{k:float(v) for k,v in terms.items()})
                if rank==0:
                    print(json.dumps(row),flush=True)
                    with (out/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\n')
        if tensor_digest(model.embedding.weight)!=frozen:raise ValueError('Frozen A embedding changed')
        if len(set(sync_status(model_digest(model))))!=1:raise ValueError('Final rank states differ')
        if rank==0:
            torch.save(dict(schema='relation_frame_pretrain_v1',arm=arm,updates=completed,protocol=config,
                initial_model_digest=initial,model_state_dict=model.state_dict(),optimizer_state_dict=optimizer.state_dict(),
                caption_word2idx=cache.vocab,relations=list(__import__('frame_data').RELATIONS),
                classes=__import__('frame_data').CLASSES),out/'checkpoint.bin')
        model.eval();rows=[]
        with torch.no_grad():
            for begin in range(rank*32,(rank+1)*32,8):
                check_budget();indices=list(range(begin,begin+8));gap=model.ranking_gap(tensor_batch(indices,'heldout')).cpu().tolist()
                for i,g in zip(indices,gap):
                    p=cache.pairs['heldout'][i]
                    rows.append(dict(pair_id=p['pair_id'],relation=p['relation'],stratum=p['stratum'],
                        likelihood_gap=g,correct=float(g>0),tie=bool(g==0)))
        all_rows=sync_status(rows)
        if rank==0:
            rows=all_rows[0]+all_rows[1]
            if len({r['pair_id'] for r in rows})!=64:raise ValueError('Heldout identity differs')
            save(out/'result.json',dict(status='ENDPOINT_200_COMPLETE',arm=arm,updates=completed,seed=42,
                initial_model_digest=initial,checkpoint_sha256=sha(out/'checkpoint.bin'),protocol=config,
                correct_image_accuracy=float(np.mean([r['correct'] for r in rows])),
                score='negative equally weighted subject/predicate/object teacher-forced CE; same GRU likelihood in both arms',
                ties='count incorrect and report',rows=rows,device_gpu_hours=(time.monotonic()-start)*2/3600,
                caption_training=False,novelty_or_caption_gain_established=False))
            print('ENDPOINT200 COMPLETE; send review ZIP, keep checkpoint.bin. Compare BOTH arms before caption training.',flush=True)
    except Exception as e:
        save(out/f'failure_rank{rank}.json',dict(status='INCOMPLETE',completed_updates=completed,error=repr(e)))
        raise
    finally:
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--config',required=True)
    args=p.parse_args();globals()[args.mode](json.loads(Path(args.config).read_text()))
