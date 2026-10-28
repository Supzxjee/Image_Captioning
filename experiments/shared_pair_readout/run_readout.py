"""Kaggle-only I/J runner. Import starts no work; root never executes neural code."""
import argparse
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from readout_data import *


def tensor_batch(data,indices,device,relation=False,split='train'):
    import torch
    b=data.relation_batch(split,indices) if relation else data.batch(indices)
    return {k:torch.from_numpy(v).to(device) for k,v in b.items()}


def references_batch(data,indices,step,device):
    import torch
    from captioning.tokenizer import CaptionTokenizer
    tok=tokenizer(data.meta);epoch=step//math.ceil(len(data.meta['train_indices'])/8)
    return torch.stack([tok.encode(data.meta['rows'][int(i)]['references'][epoch%5])[0] for i in indices]).to(device)


def tokenizer(meta):
    from captioning.tokenizer import CaptionTokenizer
    tok=CaptionTokenizer([]);tok.word2idx=meta['word2idx'];tok.idx2word={i:w for w,i in tok.word2idx.items()};return tok


def build(cfg,device):
    import torch
    from captioning.models import UniversalVisionEncoder,CaptionDecoder,ImageCaptioningModel
    from captioning.scst_runtime import model_digest
    from readout_model import Captioner
    meta=vocabulary(cfg['metadata'])
    state=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)
    if state['caption_word2idx']!=meta['word2idx']:raise ValueError('Wrong A vocabulary')
    weights=state['model_state_dict']
    if not weights or any(not k.startswith('base.') for k in weights):raise ValueError('Wrong A wrapper')
    base=ImageCaptioningModel(UniversalVisionEncoder(load_backbone=False,visual_adapter='qformer',num_visual_queries=32,qformer_layers=2),CaptionDecoder(10000))
    base.load_state_dict({k[5:]:v for k,v in weights.items()},strict=True);del state,weights
    if model_digest(base)!=BASE_SHA:raise ValueError('A tensor identity differs')
    return Captioner(base,cfg['arm']).to(device).eval()


def frozen_digest(model):
    from captioning.scst_runtime import tensor_digest
    h=hashlib.sha256()
    for n,p in list(model.named_parameters())+list(model.named_buffers()):
        if not getattr(p,'requires_grad',False):h.update(n.encode());h.update(tensor_digest(p).encode())
    return h.hexdigest()


def parameters(model):return [p for p in model.parameters() if p.requires_grad]


def optimizer(model):
    import torch
    return torch.optim.AdamW([{'params':model.reader.parameters(),'lr':1e-4},
        {'params':[p for p in model.base.decoder.parameters() if p.requires_grad],'lr':5e-6}],
        weight_decay=.01,betas=(.9,.999),eps=1e-8)


def sync(model):
    import torch
    import torch.distributed as dist
    ps=parameters(model);used=torch.tensor([int(p.grad is not None) for p in ps],device=ps[0].device)
    dist.all_reduce(used,op=dist.ReduceOp.MAX);used=used.tolist()
    flat=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in ps])
    bad=torch.tensor(int(not torch.isfinite(flat).all()),device=flat.device);dist.all_reduce(bad,op=dist.ReduceOp.MAX)
    if bad.item():raise FloatingPointError('Nonfinite gradient on a rank')
    dist.all_reduce(flat);flat.div_(2);offset=0
    for i,p in enumerate(ps):
        p.grad=flat[offset:offset+p.numel()].view_as(p).clone() if used[i] else None;offset+=p.numel()
    return flat


def same_ranks(value):
    import torch.distributed as dist
    values=[None,None];dist.all_gather_object(values,value)
    if values[0]!=values[1]:raise ValueError('Rank identities diverged')


def preflight(model,data,rank,device):
    import torch
    from captioning.scst_runtime import model_digest
    initial=model_digest(model);frozen=frozen_digest(model);same_ranks(initial)
    for name,p in model.named_parameters():
        expected=name.startswith('reader.') or name.startswith(('base.decoder.transformer_decoder.layers.1.','base.decoder.transformer_decoder.layers.2.'))
        if p.requires_grad!=expected:raise ValueError('Wrong trainable scope: '+name)
    original={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    tok=tokenizer(data.meta);ix=data.meta['train_indices'][:2];b=tensor_batch(data,ix,device);x=model.inputs(b)
    ref=references_batch(data,ix,0,device)[:,:-1]
    torch.manual_seed(813);torch.cuda.manual_seed_all(813)
    try:
        with torch.no_grad():
            teacher=model.teacher_logits(ref,x);student=model.logits(ref,x)
            torch.testing.assert_close(student,teacher,atol=0,rtol=0)
            a,_=model.generate(x,tok,baseline=True);j,_=model.generate(x,tok)
            if not torch.equal(a,j):raise ValueError('Zero-init greedy parity failed')
            bank=model.reader.bank(x);h=model.hidden(ref,x)[:,:4]
            reader=model.reader
            ra=(reader.qs(h)@bank['ks'].transpose(1,2)/math.sqrt(128)).cpu().numpy()
            rb=(reader.qo(h)@bank['ko'].transpose(1,2)/math.sqrt(128)).cpu().numpy()
            rs=(reader.qr(h)@bank['kr'].transpose(1,2)/math.sqrt(128)).cpu().numpy()
            alpha,beta,joint=attention_numpy(ra,rb,rs,bank['valid'].cpu().numpy())
            reader.output.weight.normal_(0,.01)
            for arm in ('I','J'):
                got,trace=reader.read(h,bank,arm)
                expected=[alpha,beta] if arm=='I' else [joint.sum(-1),joint.sum(-2)]
                contexts=torch.cat([torch.as_tensor(expected[0],device=device,dtype=h.dtype)@bank['e'],
                    torch.as_tensor(expected[1],device=device,dtype=h.dtype)@bank['e'],
                    torch.as_tensor(joint.reshape(len(h),h.shape[1],169),device=device,dtype=h.dtype)@bank['r']],-1)
                torch.testing.assert_close(got,reader.output(contexts),atol=1e-5,rtol=1e-4)
                torch.testing.assert_close(trace['joint'],torch.as_tensor(joint,device=device,dtype=h.dtype),atol=1e-6,rtol=1e-4)
            oi,_=reader.read(h,bank,'I',off=True);oj,_=reader.read(h,bank,'J',off=True)
            torch.testing.assert_close(oi,oj,atol=1e-5,rtol=1e-4)
            empty={**x,**{k:torch.zeros_like(x[k]) for k in ('roi','union','boxes','confidence','geometry','valid','interaction')}}
            empty['valid'][:,12]=True;null_bank=reader.bank(empty)
            for arm in ('I','J'):
                null,trace_null=reader.read(h,null_bank,arm)
                torch.testing.assert_close(null,torch.zeros_like(null),atol=0,rtol=0)
                if not trace_null['joint'][:,:,-1,-1].eq(1).all():raise ValueError('Null-only attention not normalized')
            if bank['r'][~x['interaction']].count_nonzero() or bank['e'][:,12].count_nonzero():raise ValueError('Self/null values are nonzero')
            sample,active,trace=model.generate(x,tok,sample=True,trace=True)
            rescored=model.logits(sample[:,:-1],x).log_softmax(-1).gather(2,sample[:,1:,None]).squeeze(-1)
            max_policy_error=float((trace[active]-rescored[active]).abs().max())
            torch.testing.assert_close(trace[active],rescored[active],atol=1e-5,rtol=1e-4)
        model.load_state_dict(original);model.zero_grad(set_to_none=True)
        opt=optimizer(model)
        local=tensor_batch(data,[rank*2,rank*2+1],device,relation=True)
        loss,_=model.relation_loss(local);loss.backward()
        if not model.reader.output.weight.grad.abs().sum()>0:raise ValueError('Missing initial reader output gradient')
        for name,p in model.reader.named_parameters():
            if name!='output.weight' and p.grad is not None and torch.count_nonzero(p.grad):raise ValueError('Unexpected pre-zero upstream gradient')
        sync(model);torch.nn.utils.clip_grad_norm_(parameters(model),1.,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True)
        loss,_=model.relation_loss(local);loss.backward()
        for module in ('entity','pair','qs','qo','qr','ks','ko','kr'):
            if not any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0 for p in getattr(model.reader,module).parameters()):raise ValueError('No subsequent gradient: '+module)
        if any(p.grad is not None for p in model.teacher.parameters()):raise ValueError('Frozen teacher gradient')
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise ValueError('Frozen parameter gradient')
        model.zero_grad(set_to_none=True)
        full=tensor_batch(data,[0,1,2,3],device,relation=True);loss,_=model.relation_loss(full);loss.backward()
        expected=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in parameters(model)]).detach().clone()
        model.zero_grad(set_to_none=True);loss,_=model.relation_loss(local);loss.backward();actual=sync(model)
        max_gradient_error=float((actual-expected).abs().max())
        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-4)
        if frozen_digest(model)!=frozen:raise ValueError('Frozen parameters changed')
        opt.step();same_ranks(model_digest(model))
    finally:
        model.load_state_dict(original,strict=True);model.zero_grad(set_to_none=True)
    if model_digest(model)!=initial or frozen_digest(model)!=frozen:raise ValueError('Preflight did not restore exact initial state')
    return dict(status='MODEL_PREFLIGHT_PASS',initial_model_digest=initial,frozen_digest=frozen,
        trainable_parameters=sum(p.numel() for p in parameters(model)),new_reader_parameters=sum(p.numel() for p in model.reader.parameters()),
        zero_init_logits_and_greedy_parity=True,actual_numpy_oracle_both_arms=True,
        self_null_and_padding=True,initial_then_subsequent_gradients=True,restored=True,
        max_sample_rescore_error=max_policy_error,max_serial_distributed_gradient_error=max_gradient_error)


def predict(model,data,indices,tok,device,baseline=False,off=False):
    import torch
    from captioning.scst import decode
    records=[];timing=time.monotonic();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats(device)
    for first in range(0,len(indices),8):
        ix=indices[first:first+8];x=model.inputs(tensor_batch(data,ix,device))
        seq,_=model.generate(x,tok,baseline=baseline,off=off)
        for i,text in zip(ix,decode(seq,tok)):
            r=data.meta['rows'][i];records.append(dict(image_id=r['image_id'],coco_id=r['coco_id'],caption=text))
        if first%128==0:print(f'Prediction {"A" if baseline else "off" if off else model.arm}: {min(first+8,len(indices))}/{len(indices)} local',flush=True)
    torch.cuda.synchronize();seconds=time.monotonic()-timing
    return records,dict(seconds=seconds,images=len(indices),images_per_second=len(indices)/seconds,peak_vram_bytes=torch.cuda.max_memory_allocated(device))


def worker(cfg):
    import torch
    import torch.distributed as dist
    from captioning.scst_runtime import configure_strict_runtime,model_digest,runtime_manifest
    from captioning.scst import TrainingCider,policy_loss,decode
    from captioning.object_reward import TrainingObjectReward,combine_rewards
    rank=int(os.environ['LOCAL_RANK']);device=torch.device('cuda',rank)
    if int(os.environ['WORLD_SIZE'])!=2 or torch.cuda.device_count()!=2:raise ValueError('Require T4x2')
    configure_strict_runtime(True);torch.cuda.set_device(rank);torch.set_num_threads(2)
    dist.init_process_group('nccl',timeout=timedelta(minutes=20))
    out=Path(cfg['output']);start=time.monotonic();step=0
    try:
        status=[None]
        if rank==0:
            try:
                receipt=json.loads((out/'preparation.json').read_bytes())
                if receipt['status']!='PREPARATION_COMPLETE':raise ValueError('Incomplete preparation')
                for n,h in receipt['payload_sha256'].items():
                    if Path(n).name!=n or sha(out/n)!=h:raise ValueError('Changed prepared payload '+n)
                if sha(cfg['checkpoint'])!=CHECKPOINT_SHA or sha(cfg['dataset_json'])!=DATA_SHA or sha(Path(cfg['chair'])/'chair.pkl')!=CHAIR_SHA:raise ValueError('Changed pinned input')
                status[0]=dict(error=None)
            except Exception as e:status[0]=dict(error=repr(e))
        dist.broadcast_object_list(status,src=0)
        if status[0]['error']:raise ValueError(status[0]['error'])
        if cfg['policy']!=POLICY or json.loads((out/'protocol.json').read_bytes())!=cfg:raise ValueError('Changed policy/protocol')
        data=Inputs(out);tok=tokenizer(data.meta);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        model=build(cfg,device);initial=model_digest(model);fixed=frozen_digest(model)
        save(out/f'runtime_rank{rank}.json',runtime_manifest())
        checked=preflight(model,data,rank,device);save(out/f'preflight_rank{rank}.json',checked)
        print(f'Rank{rank}: real model preflight PASS',flush=True)
        dev=data.meta['dev_indices'][rank::2];held=data.meta['held_indices'][rank::2]
        baseline,t_a=predict(model,data,dev,tok,device,baseline=True)
        save(out/f'A_rank{rank}.json',baseline)
        historical={r['image_id']:r['caption'] for r in data.meta['baseline']}
        drift=sum(r['caption']!=historical[r['image_id']] for r in baseline)
        # Recompute same-runtime A; never substitute its historical score if captions drift.
        corpus=json.loads(Path(cfg['dataset_json']).read_bytes())
        refs={i:[s['raw'] for s in r['sentences'][:5]] for i,r in enumerate(corpus['images']) if r['split'] in ('train','restval')};del corpus
        print(f'Rank{rank}: building original full-train IDF ({len(refs)} images), no heldout references.',flush=True)
        reward=TrainingCider(refs,reference_cache_size=256);del refs
        obj=TrainingObjectReward(json.loads((out/'object_targets.json').read_bytes())['data'],'aliases_v2','unique_count_v2')
        opt=optimizer(model);history=out/f'history_rank{rank}.jsonl'
        train=data.meta['train_indices'];train_start=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
        for step in range(1,501):
            if time.monotonic()-start>POLICY['gpu_seconds']:raise TimeoutError('GPU phase cap reached; incomplete')
            torch.manual_seed(420000+2*step+rank);torch.cuda.manual_seed_all(420000+2*step+rank)
            selected=schedule(len(train),8,step-1,42)[rank*4:(rank+1)*4];ix=[train[int(i)] for i in selected]
            x=model.inputs(tensor_batch(data,ix,device));sample,active=model.generate(x,tok,sample=True);greedy,_=model.generate(x,tok)
            ids=[data.meta['rows'][i]['image_id'] for i in ix];st,gt=decode(sample,tok),decode(greedy,tok)
            sr=combine_rewards(reward.score(ids,st),obj.score(ids,st)[0],.5);gr=combine_rewards(reward.score(ids,gt),obj.score(ids,gt)[0],.5)
            logits=model.logits(sample[:,:-1],x)
            rl=policy_loss(logits,sample[:,1:],active,torch.as_tensor(sr-gr,device=device))
            ref=references_batch(data,ix,step-1,device);teacher=model.teacher_logits(ref[:,:-1],x);student=model.logits(ref[:,:-1],x)
            per=(teacher.log_softmax(-1).exp()*(teacher.log_softmax(-1)-student.log_softmax(-1))).sum(-1)
            valid=ref[:,1:].ne(0);kl=((per*valid).sum(-1)/valid.sum(-1).clamp_min(1)).mean()
            ridx=schedule(256,4,step-1,142)[rank*2:(rank+1)*2]
            relation,diag=model.relation_loss(tensor_batch(data,ridx,device,relation=True))
            loss=rl+.1*kl+.2*relation
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            opt.zero_grad(set_to_none=True);loss.backward();sync(model)
            norm=torch.nn.utils.clip_grad_norm_(parameters(model),1.,error_if_nonfinite=True);opt.step()
            if step<=3 or step%25==0:
                row=dict(step=step,arm=cfg['arm'],loss=float(loss.detach()),scst=float(rl.detach()),kl=float(kl.detach()),
                    relation=float(relation.detach()),tail_gap=float(diag['tail_gap'].mean()),predicate_gap=float(diag['predicate_gap'].mean()),
                    sample_reward=float(sr.mean()),greedy_reward=float(gr.mean()),grad_norm=float(norm),seconds=time.monotonic()-start)
                with history.open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                print(f'[{cfg["arm"]} rank{rank}] '+json.dumps(row),flush=True)
        torch.cuda.synchronize();train_seconds=time.monotonic()-train_start;train_peak=torch.cuda.max_memory_allocated(device)
        if frozen_digest(model)!=fixed:raise ValueError('Frozen A/teacher tensors changed')
        final=model_digest(model);same_ranks(final)
        if rank==0:
            torch.save(dict(schema='shared_pair_readout_v1',arm=cfg['arm'],updates=500,policy=POLICY,
                base_state_dict=model.base.state_dict(),reader_state_dict=model.reader.state_dict(),
                optimizer_state_dict=opt.state_dict(),caption_word2idx=data.meta['word2idx'],
                source_A_sha256=CHECKPOINT_SHA,initial_full_digest=initial,final_full_digest=final),out/'checkpoint.bin')
        end,t_end=predict(model,data,dev,tok,device);save(out/f'end_rank{rank}.json',end)
        off,t_off=predict(model,data,dev,tok,device,off=True);save(out/f'off_rank{rank}.json',off)
        panel_a,_=predict(model,data,held,tok,device,baseline=True);save(out/f'held_A_rank{rank}.json',panel_a)
        panel_end,_=predict(model,data,held,tok,device);save(out/f'held_end_rank{rank}.json',panel_end)
        scores=[];posterior=[]
        with torch.no_grad():
            for first in range(rank*32,(rank+1)*32,4):
                indices=list(range(first,first+4));b=tensor_batch(data,indices,device,relation=True,split='heldout')
                _,d=model.relation_loss(b)
                x=model.inputs(b);_,trace=model.logits(b['ids'],x,diagnostics=True)
                a,o,t=trace['alpha'],trace['beta'],trace['joint'];ms,mo=t.sum(-1),t.sum(-2)
                tail=b['tail'];den=tail.sum().clamp_min(1)
                def average(v):return float((v*tail).sum()/den)
                posterior.append(dict(null_subject=average(ms[:,:,12]),null_object=average(mo[:,:,12]),
                    prior_posterior_l1=average((ms-a).abs().sum(-1)+(mo-o).abs().sum(-1)),
                    joint_vs_product_l1=average((t-ms[:,:,:,None]*mo[:,:,None,:]).abs().sum((-1,-2)))))
                for i,g,p in zip(indices,d['tail_gap'].cpu().tolist(),d['predicate_gap'].cpu().tolist()):
                    row=data.meta['pairs']['heldout'][i]
                    scores.append(dict(pair_id=row['pair_id'],relation=row['relation'],stratum=row['stratum'],tail_gap=g,predicate_gap=p))
        save(out/f'held_likelihood_rank{rank}.json',scores);save(out/f'posterior_rank{rank}.json',posterior)
        if time.monotonic()-start>POLICY['gpu_seconds']:raise TimeoutError('GPU evaluation exceeded fixed cap')
        save(out/f'worker_rank{rank}.json',dict(status='ENDPOINT_500_COMPLETE',updates=500,initial_digest=initial,final_digest=final,
            initial_frozen_digest=fixed,final_frozen_digest=frozen_digest(model),historical_A_caption_mismatches=drift,
            timings=dict(A=t_a,end=t_end,off=t_off,train_seconds=train_seconds,train_peak_vram_bytes=train_peak),
            elapsed_seconds=time.monotonic()-start))
    except Exception as e:
        save(out/f'failure_rank{rank}.json',dict(status='INCOMPLETE',last_step=step,error=repr(e)));raise
    finally:
        if dist.is_initialized():dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','evaluate']);p.add_argument('--config',required=True);args=p.parse_args()
    cfg=json.loads(Path(args.config).read_bytes());root=Path(__file__).resolve().parents[2]
    for name,h in cfg['source_assets'].items():
        if sha(root/name)!=h:raise ValueError('Changed pinned source '+name)
    if args.mode=='prepare':
        from prepare_readout import prepare
        prepare(cfg)
    elif args.mode=='worker':worker(cfg)
    else:
        from evaluate_readout import evaluate
        evaluate(cfg)


if __name__=='__main__':main()
