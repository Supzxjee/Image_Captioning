"""USER-RUN: one-GPU FP32 full-cache throughput, not a quality/training run."""
import argparse
import importlib.metadata
import math
from pathlib import Path
import time
import numpy as np
from real_bank_data import HERE,read,save,sha
from pilot_data import load_banks
from full_cache_data import stage_full,collate,stress_events,target_tokens

def run(args):
    import torch
    from model import Config,CSACaptioner,beam_search
    from torch.nn import functional as F
    started=time.monotonic()
    if not torch.cuda.is_available():raise RuntimeError('Enable GPU; no CPU benchmark fallback')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    # Validate the small bank first, so a missing input cannot waste full-cache staging.
    inputs=read(HERE/'REAL_BANK_INPUT_PACKAGE.json');word_path=HERE/'lemma_correction_review_text/bank_words_corrected.json'
    if sha(word_path)!=inputs['overlay_sha256']['bank_words_corrected.json']:raise ValueError('Corrected word-bank SHA differs')
    words=read(word_path)
    banks,bank_package=load_banks(args.bank_root,words,inputs)
    data,vocab,staging=stage_full(args.full_root,args.stage)
    save(args.output/'stage_summary.json',staging)
    save(args.output/'runtime.json',dict(device=torch.cuda.get_device_name(0),used_gpu=0,visible_gpus=torch.cuda.device_count(),
        precision='FP32,noAMP,noTF32',software={k:importlib.metadata.version(k) for k in ('torch','numpy')}))
    def guard():
        if time.monotonic()-started>1650:raise TimeoutError('Internal benchmark cap; partial logs retained')
    def build():
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        model=CSACaptioner(Config(vocab=len(vocab)),*[torch.from_numpy(b) for b in banks]).cuda()
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,betas=(.9,.99),eps=1e-8,weight_decay=0)
        return model,optimizer
    def tensors(indices):return [torch.from_numpy(x).cuda() for x in collate(data,indices)]
    def update(model,optimizer,indices,step=False,inspect=False):
        model.train();optimizer.zero_grad(set_to_none=True);denom=target_tokens(data,indices);actual=0;nll=0.;shapes=[]
        if denom<=0:raise ValueError('No target tokens')
        for offset in range(0,len(indices),4):
            x,valid,seq=tensors(indices[offset:offset+4]);logits=model(x,valid,seq[:,:-1])
            loss_sum=F.cross_entropy(logits.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum')
            if not torch.isfinite(logits).all() or not torch.isfinite(loss_sum):raise ValueError('Nonfinite forward')
            (loss_sum/denom).backward();actual+=int((seq[:,1:]!=0).sum());nll+=float(loss_sum.detach())
            shapes.append(dict(batch=len(seq),regions=x.shape[1],prefix_length=seq.shape[1]-1))
        if actual!=denom:raise ValueError('Accumulation token denominator differs')
        if inspect:
            for name,param in model.named_parameters():
                if param.grad is None or not torch.isfinite(param.grad).all():raise ValueError('Missing/nonfinite gradient: '+name)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True)
        if step:optimizer.step()
        return dict(nll=nll/denom,target_tokens=denom,grad_norm=float(norm),micro_shapes=shapes)

    torch.cuda.reset_peak_memory_stats();model,optimizer=build()
    stress=stress_events(data,566747);stress_result=update(model,optimizer,stress,step=False,inspect=True)
    torch.cuda.synchronize();stress_result.update(peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,optimizer_steps=0,event_indices=stress)
    save(args.output/'stress.json',stress_result)
    del model,optimizer;torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    model,optimizer=build();order=np.random.default_rng(42).choice(566747,size=40*32,replace=False);history=[]
    for number in range(40):
        guard();indices=order[number*32:(number+1)*32];torch.cuda.synchronize();step_started=time.monotonic()
        values=update(model,optimizer,indices,step=True,inspect=number==0);torch.cuda.synchronize()
        values.update(step=number+1,timed=number>=10,seconds=time.monotonic()-step_started,event_indices=indices.tolist())
        history.append(values);save(args.output/'steps.json',history)
        print('Full-cache benchmark step',number+1,'/40',values['seconds'],'s',flush=True)
    for kind,original in zip(('nouns','verbs'),banks):
        buf=getattr(model,kind)
        if buf.requires_grad or not torch.equal(buf.cpu(),torch.from_numpy(original)):raise ValueError('Frozen bank changed')
    train_peak=dict(allocated_mib=torch.cuda.max_memory_allocated()/2**20,reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    val_images=np.arange(113287,113287+32);probe=np.flatnonzero(np.isin(data['caption_image'],val_images));model.eval()
    torch.cuda.synchronize();teacher_started=time.monotonic();teacher_sum=0.;teacher_tokens=0
    with torch.inference_mode():
        for offset in range(0,len(probe),4):
            guard();x,valid,seq=tensors(probe[offset:offset+4]);logits=model(x,valid,seq[:,:-1])
            if not torch.isfinite(logits).all():raise ValueError('Nonfinite validation forward')
            loss_sum=F.cross_entropy(logits.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum')
            if not torch.isfinite(loss_sum):raise ValueError('Nonfinite timing-probe NLL')
            teacher_sum+=float(loss_sum);teacher_tokens+=int((seq[:,1:]!=0).sum())
    torch.cuda.synchronize();teacher_seconds=time.monotonic()-teacher_started
    beam_rows=[];torch.cuda.synchronize();beam_started=time.monotonic()
    with torch.inference_mode():
        for number,image in enumerate(val_images,1):
            guard();event=int(np.searchsorted(data['caption_image'],image));x,valid,_=tensors([event])
            seq,scores=beam_search(model,*model.encode(x,valid),valid,beams=5,maximum=20)[0]
            ids=seq[0].cpu().tolist()[1:];words_count=ids.index(2) if 2 in ids else len(ids)
            beam_rows.append(dict(coco_id=int(data['ids'][image]),tokens=words_count,eos=2 in ids))
            if number%8==0:print('Validation timing beam',number,'/32',flush=True)
    torch.cuda.synchronize();beam_seconds=time.monotonic()-beam_started
    timings=np.array([x['seconds'] for x in history if x['timed']]);median=float(np.median(timings));p90=float(np.quantile(timings,.9));updates=math.ceil(566747/32)
    save(args.output/'result.json',dict(status='FULL_CACHE_XE_BENCHMARK_COMPLETE',actual_optimizer_updates=40,warmup_updates=10,timed_updates=30,
        effective_batch=32,microbatch=4,accumulation=8,precision='FP32',gpus_used=1,bank_buffers_unchanged=True,checkpoint_saved=False,
        train_caption_events=566747,val_caption_events=25010,updates_per_full_epoch=updates,train_step_seconds=dict(median=median,p90=p90,mean=float(timings.mean())),
        training_memory=train_peak,stress=stress_result,teacher_probe=dict(images=32,caption_events=len(probe),seconds=teacher_seconds,nll=teacher_sum/teacher_tokens,target_tokens=teacher_tokens),
        beam_probe=dict(images=32,seconds=beam_seconds,outputs=beam_rows,fresh_model_after_updates=40),
        projection=dict(train_only_hours_1epoch=median*updates/3600,train_only_hours_15epochs=median*updates*15/3600,
            p90_train_only_hours_15epochs=p90*updates*15/3600,teacher_seconds_per_full_val=teacher_seconds/len(probe)*25010,
            beam_seconds_per_full_val=beam_seconds/32*5000,
            limitations='Scenario only: excludes officialmetric/checkpoint/coldcache costs; maturecaptionlength and throughput may differ. No fullrun auto-start.'),
        full_cache_manifest_sha256=read(HERE/'SAVE_FULL_ROI_V134_PACKAGE.json')['files']['manifest.json'],bank_package=bank_package,
        stage_seconds=staging['seconds'],worker_seconds=time.monotonic()-started,quality_metrics=False,grounding_accuracy=False))
    print('FULL CACHE BENCHMARK COMPLETE. No checkpoint or full training launched.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--full-root',required=True);p.add_argument('--bank-root',required=True)
    p.add_argument('--stage',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);run(args)
