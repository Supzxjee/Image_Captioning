"""User-run single-GPU XE pilot only. Never import/execute locally."""
import argparse
import importlib.metadata
import math
import os
import random
import time
from pathlib import Path
import numpy as np
from real_bank_data import load_prepared,read,save,sha,HERE
from pilot_data import events,epoch_batches,learning_rate,event_batch,target_tokens,load_banks

def run(args):
    import torch
    from model import Config,CSACaptioner,beam_search
    from torch.nn import functional as F
    start=time.monotonic();deadline=start+5100
    root,inputs,a,images,vocab,words=load_prepared(args.prepared_root)
    banks,bank_package=load_banks(args.bank_root,words,inputs)
    train,val=events(images,'train'),events(images,'val')
    if len(train)!=1280 or len(val)!=320 or any(len(x['sequences'])!=5 for x in images):raise ValueError('Pilot event count differs')
    train_probe=train[:160];val_indices=[i for i,m in enumerate(images) if m['group']=='val']
    source=read(args.output/'source_receipt.json')
    identity=dict(prepared_inputs=inputs,bank_package=bank_package,source=source,policy_sha256=sha(HERE/'REAL_XE_PILOT_PROTOCOL.md'))
    save(args.output/'input_receipt.json',identity)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    if not torch.cuda.is_available():raise RuntimeError('Enable GPU; no CPU training fallback')
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    save(args.output/'runtime.json',dict(device=torch.cuda.get_device_name(0),visible_gpus=torch.cuda.device_count(),used_gpu=0,
        precision='FP32,noAMP,noTF32',software={k:importlib.metadata.version(k) for k in ('torch','numpy')},gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory))
    model=CSACaptioner(Config(vocab=len(vocab)),*[torch.from_numpy(v) for v in banks]).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=8e-6,betas=(.9,.99),eps=1e-8,weight_decay=0)
    frozen=[model.nouns.clone(),model.verbs.clone()]
    progress=dict(status='RUNNING',completed_epochs=0,updates=0,caption_events=0,target_tokens=0,history=[])
    save(args.output/'progress.json',progress)

    def guard():
        if time.monotonic()>deadline:raise TimeoutError('Pilot internal5100s cap; last completed epoch checkpoint retained')

    def tensors(items):
        return [torch.from_numpy(v).to(device) for v in event_batch(a,images,items)]

    def bank_guard():
        if model.nouns.requires_grad or model.verbs.requires_grad or not torch.equal(frozen[0],model.nouns) or not torch.equal(frozen[1],model.verbs):
            raise ValueError('Frozen bank changed')

    def evaluate(items):
        model.eval();total=0.;count=0
        with torch.inference_mode():
            for offset in range(0,len(items),4):
                guard();x,valid,seq=tensors(items[offset:offset+4]);logits=model(x,valid,seq[:,:-1])
                loss=F.cross_entropy(logits.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum')
                if not torch.isfinite(loss):raise ValueError('Nonfinite evaluation NLL')
                total+=float(loss);count+=int((seq[:,1:]!=0).sum())
        return dict(nll=total/count,target_tokens=count,caption_events=len(items))

    def generate(label):
        model.eval();rows=[]
        with torch.inference_mode():
            for number,i in enumerate(val_indices):
                guard();x,valid,_=tensors([(i,0)])
                seq,scores=beam_search(model,*model.encode(x,valid),valid,beams=5,maximum=20)[0]
                ids=seq[0].cpu().tolist()[1:];ended=2 in ids;content=ids[:ids.index(2)] if ended else ids
                rows.append(dict(coco_id=images[i]['coco_id'],filename=images[i]['filename'],ids=ids,
                    caption=' '.join(vocab[t] for t in content if t!=0),eos=ended,score=float(scores[0]),
                    references=images[i]['captions'],repeated_single_token=len(set(content))==1 if content else False))
                if (number+1)%8==0:print(label,'beam',number+1,'/64',flush=True)
        save(args.output/('captions_'+label+'.json'),rows)
        return dict(images=len(rows),eos=sum(x['eos'] for x in rows),single_token_repetition=sum(x['repeated_single_token'] for x in rows))

    def checkpoint(epoch):
        bank_guard();temp=args.output/'checkpoint.tmp'
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch,updates=progress['updates'],
            config=vars(model.c),identity=identity,progress=progress,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate()),temp)
        os.replace(temp,args.output/'checkpoint.bin')
        save(args.output/'checkpoint_receipt.json',dict(epoch=epoch,updates=progress['updates'],sha256=sha(args.output/'checkpoint.bin'),
            complete_endpoint=epoch==15,selection='fixed last epoch, no best selection'))

    initial=dict(epoch=0,val=evaluate(val),train_probe=evaluate(train_probe))
    progress['history'].append(initial);save(args.output/'progress.json',progress)
    print('INITIAL',initial,flush=True);initial_generation=generate('initial')
    training_start=time.monotonic();ten_time=None
    for epoch in range(1,16):
        epoch_start=time.monotonic();observed_sum=0.;observed_tokens=0
        for items in epoch_batches(train,epoch):
            guard();model.train();optimizer.zero_grad(set_to_none=True);step=progress['updates']+1
            lr=learning_rate(step)
            for group in optimizer.param_groups:group['lr']=lr
            denominator=target_tokens(images,items)
            if denominator<=0:raise ValueError('No XE target tokens')
            batch_sum=0.;count=0
            for offset in range(0,len(items),4):
                x,valid,seq=tensors(items[offset:offset+4]);logits=model(x,valid,seq[:,:-1])
                loss_sum=F.cross_entropy(logits.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum')
                if not torch.isfinite(logits).all() or not torch.isfinite(loss_sum):raise ValueError('Nonfinite training forward')
                (loss_sum/denominator).backward();batch_sum+=float(loss_sum.detach());count+=int((seq[:,1:]!=0).sum())
            if count!=denominator:raise ValueError('Accumulation token counts differ')
            if step==1:
                for name,param in model.named_parameters():
                    if param.grad is None or not torch.isfinite(param.grad).all():raise ValueError('Missing/nonfinite gradient: '+name)
                    if name in ('project.weight','noun_agg.wa.weight','verb_agg.wa.weight') or 'gate' in name:
                        if not param.grad.abs().sum()>0:raise ValueError('Disconnected gradient: '+name)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
            progress['updates']=step;progress['caption_events']+=len(items);progress['target_tokens']+=denominator
            observed_sum+=batch_sum;observed_tokens+=denominator
            if step==10:
                ten_time=time.monotonic()-training_start
                print('Timing first10updates:',ten_time,'s; estimated remaining TRAIN steps only:',ten_time/10*590,'s (not a guarantee)',flush=True)
            if step%10==0:
                print('TRAIN',dict(epoch=epoch,step=step,total_updates=600,lr=lr,nll=batch_sum/denominator,grad_norm=float(norm),seconds=time.monotonic()-start),flush=True)
        progress['completed_epochs']=epoch
        # Persist training progress before evaluation, so evaluation errors cannot discard the epoch.
        save(args.output/'progress.json',progress);checkpoint(epoch)
        record=dict(epoch=epoch,updates=progress['updates'],observed_train_nll=observed_sum/observed_tokens,val=evaluate(val),seconds=time.monotonic()-epoch_start)
        if epoch in (1,3,5,10,15):record['train_probe']=evaluate(train_probe)
        progress['history'].append(record);save(args.output/'progress.json',progress)
        print('EPOCH',record,flush=True)
    final_generation=generate('final');bank_guard();checkpoint(15)
    save(args.output/'result.json',dict(status='REAL_XE_PILOT_COMPLETE',epochs=15,updates=progress['updates'],caption_events=progress['caption_events'],
        train_images=256,val_images=64,test_images=0,train_caption_events_per_epoch=1280,val_caption_events=320,
        history=progress['history'],initial_generation=initial_generation,final_generation=final_generation,bank_buffers_unchanged=True,
        first10_updates_seconds=ten_time,worker_seconds=time.monotonic()-start,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20,
        official_quality_metrics=False,grounding_accuracy=False,checkpoint_selection='fixed endpoint15',checkpoint_receipt=read(args.output/'checkpoint_receipt.json')))
    print('REAL XE PILOT COMPLETE. Preserve checkpoint.bin and Dataset inputs.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared-root',required=True);p.add_argument('--bank-root',required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True);run(args)
