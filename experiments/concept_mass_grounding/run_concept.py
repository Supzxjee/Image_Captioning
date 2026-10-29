"""Two-GPU native C/G pilot; no model imports/execution outside user worker."""
import argparse
from datetime import timedelta
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from concept_data import *

def tokenizer(meta):
    from captioning.tokenizer import CaptionTokenizer
    tok=CaptionTokenizer([]);tok.word2idx=meta['word2idx'];tok.idx2word={i:w for w,i in tok.word2idx.items()};return tok

def build(cfg,device):
    import torch
    from captioning.models import UniversalVisionEncoder,CaptionDecoder,ImageCaptioningModel
    from captioning.scst_runtime import model_digest
    from concept_model import NativeCaptioner
    state=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)
    meta=json.loads(gzip.decompress(Path(cfg['metadata']).read_bytes()))
    if state['caption_word2idx']!=meta['word2idx']:raise ValueError('Wrong A vocabulary')
    weights=state['model_state_dict']
    if not weights or any(not k.startswith('base.') for k in weights):raise ValueError('Wrong A checkpoint wrapper')
    base=ImageCaptioningModel(UniversalVisionEncoder(load_backbone=False,visual_adapter='qformer',num_visual_queries=32,qformer_layers=2),CaptionDecoder(10000))
    base.load_state_dict({k[5:]:v for k,v in weights.items()},strict=True)
    if model_digest(base)!=BASE_SHA:raise ValueError('Wrong native A tensors')
    del weights,state
    return NativeCaptioner(base).to(device).eval()

def frozen_digest(model):
    from captioning.scst_runtime import tensor_digest
    h=hashlib.sha256()
    for n,p in list(model.named_parameters())+list(model.named_buffers()):
        if not getattr(p,'requires_grad',False):h.update(n.encode());h.update(tensor_digest(p).encode())
    return h.hexdigest()

def trainables(model):return [p for p in model.parameters() if p.requires_grad]

def sync(model):
    import torch
    import torch.distributed as dist
    ps=trainables(model);flat=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in ps])
    bad=flat.new_tensor(float(not torch.isfinite(flat).all()));dist.all_reduce(bad)
    if bad.item():raise FloatingPointError('Nonfinite rank gradient')
    dist.all_reduce(flat);flat.div_(2);offset=0
    for p in ps:p.grad=flat[offset:offset+p.numel()].view_as(p).clone();offset+=p.numel()
    return flat

def same_ranks(value):
    import torch.distributed as dist
    values=[None,None];dist.all_gather_object(values,value)
    if values[0]!=values[1]:raise ValueError('Rank states diverged')

def preflight(model,data,tok,device,rank):
    import torch
    from captioning.scst_runtime import model_digest
    from concept_model import negative_losses,global_mean
    initial=model_digest(model);fixed=frozen_digest(model);same_ranks(initial)
    for name,p in model.named_parameters():
        if p.requires_grad != name.startswith(('base.decoder.transformer_decoder.layers.1.','base.decoder.transformer_decoder.layers.2.')):raise ValueError('Wrong learning scope '+name)
    x=model.inputs(data.batch([0,1],device));seq=torch.stack([tok.encode(data.meta['rows'][i]['references'][0])[0] for i in (0,1)]).to(device)
    with torch.no_grad():
        got=model.logits(seq[:,:-1],x);want=model.logits(seq[:,:-1],x,teacher=True)
        error=float((got-want).abs().max());torch.testing.assert_close(got,want,atol=1e-5,rtol=1e-4)
        a,_=model.generate(x,tok,teacher=True);b,_=model.generate(x,tok)
        if not torch.equal(a,b):raise ValueError('Initial greedy A parity failed')
        torch.manual_seed(731+rank);sample,active,trace=model.generate(x,tok,sample=True,trace=True)
        rescored=model.logits(sample[:,:-1],x).log_softmax(-1).gather(-1,sample[:,1:,None]).squeeze(-1)
        policy_error=float((trace[active]-rescored[active]).abs().max())
        torch.testing.assert_close(trace[active],rescored[active],atol=1e-5,rtol=1e-4)
    lex=data.meta['lexicon'];categories=[c for c in ('dog','cell phone') if lex['events'][c]]
    if len(categories)!=2:raise ValueError('Missing preflight alias classes')
    events=[dict(batch=0,position=0,category=c) for c in categories]
    logits=model.logits(seq[:,:-1],x);q=model.class_logmass(seq,x,events,lex,logits)
    with torch.no_grad():
        expected=[]
        for c in categories:
            paths=lex['events'][c]
            def probability(prefix,token):
                ids=seq[:1,:1]
                if prefix:ids=torch.cat([ids,ids.new_tensor([prefix])],1)
                return float(model.logits(ids,{k:v[:1] for k,v in x.items()})[0,-1].softmax(-1)[token])
            expected.append(log_union_numpy(paths,probability))
        torch.testing.assert_close(q,torch.tensor(expected,device=device),atol=1e-5,rtol=1e-4)
    if not torch.isfinite(negative_losses(q)).all():raise FloatingPointError('Invalid event UL')
    # Real-model unequal rank event counts, compared with a serial global-event mean.
    model.zero_grad(set_to_none=True)
    all_events=[dict(batch=0,position=0,category='dog'),dict(batch=1,position=0,category='dog'),dict(batch=1,position=0,category='cell phone')]
    full=model.class_logmass(seq,x,all_events,lex,model.logits(seq[:,:-1],x));negative_losses(full).mean().backward()
    expected_grad=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in trainables(model)]).detach().clone()
    model.zero_grad(set_to_none=True);local_events=all_events[:1] if rank==0 else all_events[1:]
    local=model.class_logmass(seq,x,local_events,lex,model.logits(seq[:,:-1],x))
    loss,count=global_mean(negative_losses(local).sum(),len(local_events));assert count==3
    loss.backward();actual=sync(model);grad_error=float((actual-expected_grad).abs().max())
    torch.testing.assert_close(actual,expected_grad,atol=1e-5,rtol=1e-4)
    if not actual.abs().sum()>0:raise ValueError('No negative-loss gradient')
    if any(p.grad is not None for p in model.parameters() if not p.requires_grad):raise ValueError('Frozen tensor received gradient')
    model.zero_grad(set_to_none=True)
    zero,n=global_mean(model.logits(seq[:,:-1],x).sum()*0,0)
    if n!=0 or float(zero.detach())!=0:raise ValueError('Empty event reduction failed')
    if model_digest(model)!=initial or frozen_digest(model)!=fixed:raise ValueError('Preflight changed A state')
    return dict(status='PREFLIGHT_PASS',initial_digest=initial,frozen_digest=fixed,max_initial_logit_error=error,
        max_policy_rescore_error=policy_error,max_serial_distributed_event_gradient_error=grad_error,
        prefix_free_numpy_mass_oracle=True,empty_events=True,greedy_parity=True,restored=True,
        trainable_parameters=sum(p.numel() for p in trainables(model)),new_parameters=0)

def predict(model,data,indices,tok,device,teacher=False):
    import torch
    from captioning.scst import decode
    records=[];torch.cuda.synchronize();start=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
    for first in range(0,len(indices),8):
        ix=indices[first:first+8];seq,_=model.generate(model.inputs(data.batch(ix,device)),tok,teacher=teacher)
        records.extend(dict(image_id=data.meta['rows'][i]['image_id'],coco_id=data.meta['rows'][i]['coco_id'],caption=c) for i,c in zip(ix,decode(seq,tok)))
        if first%128==0:print(f'Prediction {"A" if teacher else "end"}: {min(first+8,len(indices))}/{len(indices)} local',flush=True)
    torch.cuda.synchronize();elapsed=time.monotonic()-start
    return records,dict(seconds=elapsed,images=len(records),peak_vram_bytes=torch.cuda.max_memory_allocated(device),scope='Common resident student+teacher wrapper; no standalone memory or feature/detector cost claim')

def worker(cfg):
    import torch
    import torch.distributed as dist
    from captioning.scst_runtime import configure_strict_runtime,model_digest,runtime_manifest
    from captioning.scst import TrainingCider,policy_loss,decode
    from captioning.object_reward import TrainingObjectReward,combine_rewards
    from concept_model import negative_losses,global_mean
    rank=int(os.environ['LOCAL_RANK']);device=torch.device('cuda',rank);out=Path(cfg['output']);step=0;data=None
    if int(os.environ['WORLD_SIZE'])!=2 or torch.cuda.device_count()!=2:raise ValueError('Require exactly2 GPUs')
    configure_strict_runtime(True);torch.cuda.set_device(rank);torch.set_num_threads(2)
    dist.init_process_group('nccl',timeout=timedelta(minutes=20));start=time.monotonic()
    try:
        status=[None]
        if rank==0:
            try:
                receipt=json.loads((out/'preparation.json').read_bytes())
                if receipt['status']!='PREPARATION_COMPLETE':raise ValueError('Incomplete prepare')
                for n,h in receipt['payload_sha256'].items():
                    if Path(n).name!=n or sha(out/n)!=h:raise ValueError('Changed payload '+n)
                if sha(cfg['checkpoint'])!=CHECKPOINT_SHA or sha(Path(cfg['chair'])/'chair.pkl')!=CHAIR_SHA:raise ValueError('Changed pinned input')
                status[0]=dict(error=None)
            except Exception as e:status[0]=dict(error=repr(e))
        dist.broadcast_object_list(status,src=0)
        if status[0]['error']:raise ValueError(status[0]['error'])
        if cfg['policy']!=POLICY or json.loads((out/'protocol.json').read_bytes())!=cfg:raise ValueError('Changed recipe')
        if POLICY['negative_policy']=='PENDING_RESEARCH_LEAD':raise RuntimeError('Negative supervision protocol not resolved; no training allowed')
        torch.manual_seed(42);torch.cuda.manual_seed_all(42);data=Inputs(out);tok=tokenizer(data.meta);lex=data.meta['lexicon']
        model=build(cfg,device);initial=model_digest(model);fixed=frozen_digest(model);save(out/f'runtime_rank{rank}.json',runtime_manifest())
        checked=preflight(model,data,tok,device,rank);save(out/f'preflight_rank{rank}.json',checked)
        dev=data.meta['dev_indices'][rank::2];a,ta=predict(model,data,dev,tok,device,teacher=True);save(out/f'A_rank{rank}.json',a)
        historical={r['image_id']:r['caption'] for r in data.meta['baseline']};drift=sum(r['caption']!=historical[r['image_id']] for r in a)
        if drift:raise ValueError('Same-runtime A greedy captions drifted; stop before training')
        corpus=json.loads(Path(cfg['dataset_json']).read_bytes())
        refs={i:[s['raw'] for s in r['sentences'][:5]] for i,r in enumerate(corpus['images']) if r['split'] in ('train','restval')};del corpus
        reward=TrainingCider(refs,reference_cache_size=256);del refs
        obj=TrainingObjectReward(json.loads((out/'object_targets.json').read_bytes())['data'],'aliases_v2','unique_count_v2')
        opt=torch.optim.AdamW(trainables(model),lr=POLICY['lr'],weight_decay=.01,betas=(.9,.999),eps=1e-8)
        exposure=[];negative_total=0;keep_total=0;negative_classes=set();selection_totals={};train_start=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
        for step in range(1,501):
            if time.monotonic()-start>POLICY['gpu_seconds']:raise TimeoutError('GPU phase cap exceeded')
            torch.manual_seed(420000+2*step+rank);torch.cuda.manual_seed_all(420000+2*step+rank)
            ix=list(range((step-1)*8+rank*4,(step-1)*8+(rank+1)*4));exposure.extend(data.meta['rows'][i]['image_id'] for i in ix)
            x=model.inputs(data.batch(ix,device));sample,active=model.generate(x,tok,sample=True);greedy,_=model.generate(x,tok)
            ids=[data.meta['rows'][i]['image_id'] for i in ix];st,gt=decode(sample,tok),decode(greedy,tok)
            sr=combine_rewards(reward.score(ids,st),obj.score(ids,st)[0],.5);gr=combine_rewards(reward.score(ids,gt),obj.score(ids,gt)[0],.5)
            logits=model.logits(sample[:,:-1],x);rl=policy_loss(logits,sample[:,1:],active,torch.as_tensor(sr-gr,device=device))
            with torch.no_grad():teacher=model.logits(sample[:,:-1],x,teacher=True).float().log_softmax(-1)
            student=logits.float().log_softmax(-1);per=(teacher.exp()*(teacher-student)).sum(-1)
            kl=((per*active).sum(-1)/active.sum(-1).clamp_min(1)).mean()
            targets=[data.meta['targets'][i] for i in ix]
            negative,selection_counts=token_events(sample.detach().cpu().tolist(),targets,lex,True)
            for k,v in selection_counts.items():selection_totals[k]=selection_totals.get(k,0)+v
            # Same event selection/monitoring for C; only G receives the negative gradient.
            if cfg['arm']=='G':logq=model.class_logmass(sample,x,negative,lex,logits);neg_sum=negative_losses(logq).sum() if negative else logits.sum()*0
            else:
                with torch.no_grad():logq=model.class_logmass(sample,x,negative,lex,logits.detach())
                neg_sum=logits.sum()*0
            monitored_negative=float(negative_losses(logq.detach()).mean()) if negative else 0.
            neg,neg_count=global_mean(neg_sum,len(negative));negative_total+=len(negative);negative_classes.update(e['category'] for e in negative)
            ref=torch.stack([tok.encode(data.meta['rows'][i]['references'][(step-1)%5])[0] for i in ix]).to(device)
            positive,_=token_events(ref.detach().cpu().tolist(),targets,lex,False)
            reference_logits=model.logits(ref[:,:-1],x);pq=model.class_logmass(ref,x,positive,lex,reference_logits)
            with torch.no_grad():aq=model.class_logmass(ref,x,positive,lex,model.logits(ref[:,:-1],x,teacher=True),teacher=True)
            keep_sum=(aq-pq).clamp_min(0).sum() if positive else reference_logits.sum()*0
            keep,keep_count=global_mean(keep_sum,len(positive));keep_total+=len(positive)
            loss=rl+.1*kl+.1*keep+(.1*neg if cfg['arm']=='G' else 0)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite objective')
            contributions={}
            if step<=3 or step%25==0:
                for name,part in [('keep',.1*keep),('negative',.1*neg if cfg['arm']=='G' else logits.sum()*0)]:
                    gs=torch.autograd.grad(part,trainables(model),retain_graph=True,allow_unused=True)
                    contributions[name]=float(torch.sqrt(sum((g.detach().square().sum() for g in gs if g is not None),logits.new_tensor(0.))))
            opt.zero_grad(set_to_none=True);loss.backward();sync(model)
            norm=torch.nn.utils.clip_grad_norm_(trainables(model),1.,error_if_nonfinite=True);opt.step()
            if step<=3 or step%25==0:
                row=dict(step=step,arm=cfg['arm'],loss=float(loss.detach()),scst=float(rl.detach()),kl=float(kl.detach()),keep=float(keep.detach()),negative=float(neg.detach()),
                    negative_monitored_local=monitored_negative,weighted_aux_gradient_norms_local=contributions,global_negative_events=neg_count,global_positive_events=keep_count,local_negative_seen=negative_total,
                    local_negative_classes=sorted(negative_classes),selection_counts=selection_counts,
                    sample_reward=float(sr.mean()),greedy_reward=float(gr.mean()),grad_norm=float(norm),seconds=time.monotonic()-start)
                with (out/f'history_rank{rank}.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                print(f'[{cfg["arm"]} rank{rank}] '+json.dumps(row),flush=True)
        torch.cuda.synchronize();train_seconds=time.monotonic()-train_start;peak=torch.cuda.max_memory_allocated(device)
        counts=torch.tensor([negative_total,keep_total],device=device);dist.all_reduce(counts)
        keys=sorted(selection_totals);totals=torch.tensor([selection_totals[k] for k in keys],device=device);dist.all_reduce(totals)
        global_selection_totals=dict(zip(keys,map(int,totals.tolist())))
        class_lists=[None,None];dist.all_gather_object(class_lists,sorted(negative_classes));global_classes=sorted(set().union(*map(set,class_lists)))
        if global_selection_totals['weak_negative']!=int(counts[0]):raise ValueError('Event exposure count mismatch')
        if counts[0].item()==0:raise RuntimeError('No usable negative events in entire run: G objective untested; do not accept a C/G comparison')
        if frozen_digest(model)!=fixed:raise ValueError('Frozen tensors drifted')
        final=model_digest(model);same_ranks(final)
        if rank==0:torch.save(dict(schema='native_concept_mass_v1',arm=cfg['arm'],updates=500,model_state_dict={'base.'+k:v for k,v in model.base.state_dict().items()},caption_word2idx=data.meta['word2idx'],policy=POLICY,source_A_sha256=CHECKPOINT_SHA),out/'checkpoint.bin')
        end,te=predict(model,data,dev,tok,device);save(out/f'end_rank{rank}.json',end)
        from diagnostic_concept import diagnostic
        diag=diagnostic(model,data,tok,device,cfg,rank);save(out/f'diagnostic_rank{rank}.json',diag)
        if time.monotonic()-start>POLICY['gpu_seconds']:raise TimeoutError('GPU diagnostic exceeded cap')
        save(out/f'exposure_rank{rank}.json',exposure)
        save(out/f'worker_rank{rank}.json',dict(status='ENDPOINT_500_COMPLETE',updates=500,initial_digest=initial,final_digest=final,
            initial_frozen_digest=fixed,final_frozen_digest=frozen_digest(model),historical_A_caption_mismatches=drift,
            global_negative_events=int(counts[0]),global_positive_events=int(counts[1]),negative_classes=global_classes,global_selection_totals=global_selection_totals,
            train_seconds=train_seconds,train_peak_vram_bytes=peak,timings=dict(A=ta,end=te),elapsed_seconds=time.monotonic()-start))
    except Exception as e:save(out/f'failure_rank{rank}.json',dict(status='INCOMPLETE',last_step=step,error=repr(e)));raise
    finally:
        if data is not None:data.close()
        if dist.is_initialized():dist.destroy_process_group()

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','worker','evaluate']);p.add_argument('--config',required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_bytes())
    for name,h in cfg['source_assets'].items():
        if sha(ROOT/name)!=h:raise ValueError('Changed source '+name)
    if a.mode=='prepare':
        from prepare_concept import prepare
        prepare(cfg)
    elif a.mode=='worker':worker(cfg)
    else:
        from evaluate_concept import evaluate
        evaluate(cfg)

if __name__=='__main__':main()
