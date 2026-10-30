"""User-run two-GPU training only; never imported by local metadata tests."""
import copy
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from kv_pilot_data import *
from kv_pilot_model import Captioner
from captioning.scst_runtime import configure_strict_runtime,model_digest,tensor_digest,runtime_manifest


def trainables(model):return [p for p in model.parameters() if p.requires_grad]


def frozen_digest(model):
    h = hashlib.sha256()
    for name,p in list(model.named_parameters())+list(model.named_buffers()):
        if not getattr(p,'requires_grad',False):h.update(name.encode());h.update(tensor_digest(p).encode())
    return h.hexdigest()


def same_ranks(value):
    values = [None,None]
    dist.all_gather_object(values,value)
    if values[0] != values[1]:raise ValueError('Rank identity differs')


def sync(model):
    parameters = trainables(model)
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in parameters])
    bad = flat.new_tensor(float(not torch.isfinite(flat).all()))
    dist.all_reduce(bad)
    if bad.item():raise FloatingPointError('Nonfinite rank gradient')
    dist.all_reduce(flat)
    flat.div_(2)
    offset = 0
    for p in parameters:
        p.grad = flat[offset:offset+p.numel()].view_as(p).clone()
        offset += p.numel()


def optimizer(model):
    return torch.optim.AdamW([
        dict(params=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('base.')],lr=5e-6),
        dict(params=list(model.reader.parameters()),lr=1e-4)],weight_decay=.01,betas=(.9,.999),eps=1e-8)


def build(cfg,data,device):
    from captioning.models import UniversalVisionEncoder,CaptionDecoder,ImageCaptioningModel
    from captioning.tokenizer import CaptionTokenizer
    state = torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)
    if state['caption_word2idx'] != data.meta['word2idx']:raise ValueError('C0 vocabulary differs')
    weights = state['model_state_dict']
    if not weights or any(not n.startswith('base.') for n in weights):raise ValueError('Wrong checkpoint wrapper')
    base = ImageCaptioningModel(UniversalVisionEncoder(load_backbone=False,visual_adapter='qformer',num_visual_queries=32,qformer_layers=2),CaptionDecoder(10000))
    base.load_state_dict({n[5:]:p for n,p in weights.items()},strict=True)
    model = Captioner(base,data.text,cfg['arm']).to(device).eval()
    tok = CaptionTokenizer([])
    tok.word2idx = data.meta['word2idx']
    tok.idx2word = {i:w for w,i in tok.word2idx.items()}
    return model,tok


def fact_scores(model,data,facts,device,effective=False):
    features = model.reader.features(data.fact_batch(facts,device))
    return model.reader.fact_scores(features,facts,effective)


def objective(model,data,tok,device,rank,step,reward,obj):
    from captioning.scst import policy_loss,decode
    from captioning.object_reward import combine_rewards
    seed = 420000+2*step+rank
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    indices = list(range((step-1)*8+rank*4,(step-1)*8+(rank+1)*4))
    x = model.inputs(data.batch(indices,device))
    sampled,active = model.generate(x,tok,sample=True)
    greedy,_ = model.generate(x,tok)
    ids = [data.meta['rows'][i]['image_id'] for i in indices]
    st,gt = decode(sampled,tok),decode(greedy,tok)
    sr = combine_rewards(reward.score(ids,st),obj.score(ids,st)[0],.5)
    gr = combine_rewards(reward.score(ids,gt),obj.score(ids,gt)[0],.5)
    logits = model.logits(sampled[:,:-1],x)
    rl = policy_loss(logits,sampled[:,1:],active,torch.as_tensor(sr-gr,device=device))
    with torch.no_grad():teacher = model.logits(sampled[:,:-1],x,teacher=True).float().log_softmax(-1)
    student = logits.float().log_softmax(-1)
    kl_tokens = (teacher.exp()*(teacher-student)).sum(-1)
    kl = ((kl_tokens*active).sum(-1)/active.sum(-1).clamp_min(1)).mean()
    aux_ids = data.schedule['bce'][step-1,rank*4:(rank+1)*4].tolist()
    facts = [data.aux['partitions']['fit'][i] for i in aux_ids]
    scores = fact_scores(model,data,facts,device)
    labels = torch.tensor([r['label'] for r in facts],device=device,dtype=torch.float32)
    bce = F.binary_cross_entropy_with_logits(scores,labels[:,None].expand_as(scores))
    group,members = data.mp_facts(step)
    z = fact_scores(model,data,members,device).mean(1)
    mp = z.logsumexp(0)-z[:len(group['positive'])].mean()
    loss = rl+.1*kl+.1*bce+.05*mp
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite total loss')
    return loss,dict(scst=float(rl.detach()),kl=float(kl.detach()),bce=float(bce.detach()),mp=float(mp.detach()),
        mp_excess=float(mp.detach())-math.log(len(group['positive'])),sample_reward=float(sr.mean()),greedy_reward=float(gr.mean()),
        caption_ids=ids,bce_fact_ids=[r['fact_id'] for r in facts],mp_group=data.schedule['mp'][step-1],
        mp_positive=len(group['positive']),mp_negative=len(group['negative']))


@torch.no_grad()
def predict(model,data,tok,indices,device,teacher=False,reader_off=False):
    from captioning.scst import decode
    start = time.monotonic()
    records = []
    for first in range(0,len(indices),8):
        batch = indices[first:first+8]
        seq,_ = model.generate(model.inputs(data.batch(batch,device)),tok,teacher=teacher,reader_off=reader_off)
        records.extend(dict(image_id=data.meta['rows'][i].get('image_id',data.meta['rows'][i]['coco_id']),
            coco_id=data.meta['rows'][i]['coco_id'],caption=c) for i,c in zip(batch,decode(seq,tok)))
        if first%128 == 0:print('Generated',min(first+8,len(indices)),'/',len(indices),'teacher',teacher,'off',reader_off,flush=True)
    torch.cuda.synchronize()
    return records,time.monotonic()-start


@torch.no_grad()
def held_diagnostics(model,data,tok,device,rank):
    rows = data.aux['partitions']['held'][rank::2]
    records = []
    for first in range(0,len(rows),8):
        facts = rows[first:first+8]
        pre = fact_scores(model,data,facts,device).mean(1)
        effective = fact_scores(model,data,facts,device,effective=True).mean(1)
        x = model.inputs(data.fact_batch(facts,device))
        likelihoods = {}
        for condition,teacher in [('C0',True),('end',False)]:
            views = []
            for view in (0,1):
                seq = torch.stack([tok.encode(r['query_views'][view],max_len=30)[0] for r in facts]).to(device)
                logits = model.logits(seq[:,:-1],x,teacher=teacher).log_softmax(-1)
                active = seq[:,1:].ne(tok.pad_idx)
                lp = logits.gather(-1,seq[:,1:,None]).squeeze(-1)
                views.append((lp*active).sum(-1)/active.sum(-1).clamp_min(1))
            likelihoods[condition] = torch.stack(views,1).mean(1).tolist()
        for i,r in enumerate(facts):
            records.append(dict(r,pre_score=float(pre[i]),effective_score=float(effective[i]),
                                C0_decoder_score=likelihoods['C0'][i],end_decoder_score=likelihoods['end'][i]))
    held_indices = [data.row[i] for i in sorted({r['image_id'] for r in rows})]
    c0,_ = predict(model,data,tok,held_indices,device,teacher=True)
    end,_ = predict(model,data,tok,held_indices,device)
    return dict(records=records,C0_captions=c0,end_captions=end,relation_caption_truth='NOT_ESTABLISHED',
                opposite_orbit_truth='NOT_MEASURED',same_class_instance_disambiguation='NOT_MEASURED')


def worker(cfg):
    from captioning.scst import TrainingCider
    from captioning.object_reward import TrainingObjectReward
    from kv_pilot_preflight import preflight
    rank = int(os.environ['LOCAL_RANK'])
    device = torch.device('cuda',rank)
    out = Path(cfg['output'])
    step = 0
    if int(os.environ['WORLD_SIZE']) != 2 or torch.cuda.device_count() != 2:raise ValueError('Require T4x2 / two GPUs')
    configure_strict_runtime(True);torch.cuda.set_device(rank);torch.set_num_threads(2)
    dist.init_process_group('nccl',timeout=timedelta(minutes=10))
    start = time.monotonic()
    try:
        if cfg['policy'] != POLICY:raise ValueError('Changed scientific policy')
        prep = json.loads((out/'preparation.json').read_bytes())
        status = [None]
        if rank == 0:
            try:
                checked_cache(cfg['prepared_root'])
                for name,h in prep['payload_sha256'].items():
                    if digest(out/name) != h:raise ValueError('Changed pilot preparation '+name)
                if digest(cfg['checkpoint']) != CHECKPOINT_SHA:raise ValueError('Changed C0 checkpoint')
                status[0] = dict(error=None)
            except Exception as exc:status[0] = dict(error=repr(exc))
        dist.broadcast_object_list(status,src=0)
        if status[0]['error']:raise ValueError(status[0]['error'])
        data = Inputs(cfg['prepared_root'],out/'held_native_prompts.npz')
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        model,tok = build(cfg,data,device)
        initial = model_digest(model);fixed = frozen_digest(model)
        same_ranks(initial)
        initial_state = {n:p.detach().cpu().clone() for n,p in model.state_dict().items()}
        if rank == 0:torch.save(initial_state,out/'common_init.bin')
        model.load_state_dict(initial_state,strict=True)
        save(out/f'runtime_rank{rank}.json',runtime_manifest())
        text_hash = tensor_digest(model.reader.text)
        save(out/f'text_identity_rank{rank}.json',dict(native_embedding_sha256=tensor_digest(model.base.decoder.embedding.weight),
            pooled_text_sha256=text_hash,token_ids_sha256=digest(out/'text_tokens.json')))
        report = preflight(model,data,tok,device,rank)
        model.load_state_dict(initial_state,strict=True);model.zero_grad(set_to_none=True)
        if model_digest(model) != initial:raise ValueError('Preflight init restoration failed')
        corpus = json.loads(Path(cfg['dataset_json']).read_bytes())
        refs = {i:[s['raw'] for s in r['sentences'][:5]] for i,r in enumerate(corpus['images']) if r['split'] in ('train','restval')}
        del corpus
        reward = TrainingCider(refs,reference_cache_size=256)
        del refs
        obj = TrainingObjectReward(json.loads((out/'object_targets.json').read_bytes())['data'],'aliases_v2','unique_count_v2')
        opt = optimizer(model)
        torch.cuda.synchronize();bench = time.monotonic()
        for bstep in range(1,21):
            opt.zero_grad(set_to_none=True)
            loss,_ = objective(model,data,tok,device,rank,bstep,reward,obj)
            loss.backward();sync(model)
            torch.nn.utils.clip_grad_norm_(trainables(model),1.,error_if_nonfinite=True);opt.step()
        torch.cuda.synchronize();step_seconds = (time.monotonic()-bench)/20
        model.load_state_dict(initial_state,strict=True);model.zero_grad(set_to_none=True)
        opt = optimizer(model)
        if model_digest(model) != initial:raise ValueError('Benchmark init restoration failed')
        dev = data.meta['dev_indices'][rank::2]
        _,student_seconds = predict(model,data,tok,dev[:32],device)
        _,teacher_seconds = predict(model,data,tok,dev[:32],device,teacher=True)
        # Includes two full dev passes, held caption/teacher diagnostic and off intervention.
        forecast = 500*step_seconds+(student_seconds+teacher_seconds)*(len(dev)+100)/32+180
        estimate = torch.tensor([forecast],device=device);dist.all_reduce(estimate,op=dist.ReduceOp.MAX)
        elapsed = torch.tensor([time.monotonic()-start],device=device);dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
        allowed = .8*(7200-float(elapsed.item()))
        report.update(benchmark_steps=20,step_seconds=step_seconds,forecast_remaining_seconds=float(estimate.item()),
            allowed_remaining_seconds=allowed,initial_digest=initial,frozen_digest=fixed,common_text_sha256=text_hash,
            reset_after_benchmark=True,quota_accounting=POLICY['quota_accounting'])
        save(out/f'preflight_rank{rank}.json',report)
        if float(estimate.item()) > allowed:raise RuntimeError('STOP_PREFLIGHT_NO_TRAINING: full endpoint forecast exceeds budget')
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        c0,c0_seconds = predict(model,data,tok,dev,device,teacher=True)
        save(out/f'C0_rank{rank}.json',c0)
        exposures,bce_exposures,mp_exposures = [],[],[]
        train_start = time.monotonic()
        for step in range(1,501):
            if time.monotonic()-start > 7200:raise TimeoutError('GPU phase cap exceeded')
            opt.zero_grad(set_to_none=True)
            loss,info = objective(model,data,tok,device,rank,step,reward,obj)
            loss.backward();sync(model)
            norm = torch.nn.utils.clip_grad_norm_(trainables(model),1.,error_if_nonfinite=True)
            gradients = {}
            if step <= 3 or step%25 == 0:
                for prefix in ('reader.f.','reader.q.','reader.k.','reader.v.','reader.o.','base.decoder.'):
                    gradients[prefix] = float(torch.sqrt(sum((p.grad.detach().square().sum() for n,p in model.named_parameters()
                        if n.startswith(prefix) and p.grad is not None),loss.new_zeros(()))))
            opt.step()
            exposures.extend(info['caption_ids']);bce_exposures.extend(info['bce_fact_ids']);mp_exposures.append(info['mp_group'])
            if step <= 3 or step%25 == 0:
                row = dict(step=step,arm=cfg['arm'],loss=float(loss.detach()),grad_norm=float(norm),
                    gradient_norms=gradients,readout_weight_norm=float(model.reader.o.weight.detach().norm()),
                    reader_statistics={k:float(v) for k,v in dict(model.reader.last_bank_stats,**model.last_read_stats).items()},
                    elapsed_seconds=time.monotonic()-start,**info)
                with (out/f'history_rank{rank}.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
                print(f'[{cfg["arm"]} rank{rank}] '+json.dumps(row),flush=True)
        train_seconds = time.monotonic()-train_start
        if frozen_digest(model) != fixed:raise ValueError('Frozen tensors drifted')
        final = model_digest(model);same_ranks(final)
        if rank == 0:
            torch.save(dict(schema='relation-kv-de-v1',arm=cfg['arm'],updates=500,policy=POLICY,
                model_state_dict=model.state_dict(),caption_word2idx=data.meta['word2idx'],source_C0_sha256=CHECKPOINT_SHA,
                initial_digest=initial,final_digest=final),out/'checkpoint.bin')
        end,end_seconds = predict(model,data,tok,dev,device)
        save(out/f'end_rank{rank}.json',end)
        selected = sorted(data.meta['dev_indices'],key=lambda i:hashlib.sha256(f'kv-off-v1:{data.meta["rows"][i]["coco_id"]}'.encode()).hexdigest())[:64]
        off,off_seconds = predict(model,data,tok,selected[rank::2],device,reader_off=True)
        save(out/f'reader_off_rank{rank}.json',off)
        from kv_pilot_preflight import intervention
        save(out/f'intervention_rank{rank}.json',intervention(model,data,tok,selected[rank::2],device))
        save(out/f'held_rank{rank}.json',held_diagnostics(model,data,tok,device,rank))
        if time.monotonic()-start > 7200:raise TimeoutError('GPU evaluation exceeded cap')
        save(out/f'exposure_rank{rank}.json',dict(caption=exposures,bce=bce_exposures,mp_groups=mp_exposures))
        save(out/f'worker_rank{rank}.json',dict(status='ENDPOINT_500_COMPLETE',updates=500,initial_digest=initial,
            final_digest=final,initial_frozen_digest=fixed,final_frozen_digest=frozen_digest(model),
            train_seconds=train_seconds,elapsed_seconds=time.monotonic()-start,timings=dict(C0=c0_seconds,end=end_seconds,reader_off=off_seconds),
            common_text_sha256=text_hash,peak_gpu_bytes=torch.cuda.max_memory_allocated(device),quota_accounting=POLICY['quota_accounting']))
    except Exception as exc:
        save(out/f'failure_rank{rank}.json',dict(status='INCOMPLETE',last_step=step,error=repr(exc),elapsed_seconds=time.monotonic()-start))
        raise
    finally:
        if dist.is_initialized():dist.destroy_process_group()
