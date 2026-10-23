"""One frozen-A noun-evidence arm per two-GPU Kaggle notebook."""
import argparse
import json
import os
from pathlib import Path
import time
import zipfile
import numpy as np
from evidence_data import A_SHA,A_BASE_SHA,digest,save
import evidence_stage as stage


def build(cfg,tokenizer):
    import torch
    from captioning.models import UniversalVisionEncoder,CaptionDecoder,ImageCaptioningModel
    from captioning.scst_runtime import model_digest
    from evidence_model import EvidenceModel
    if digest(cfg['checkpoint'])!=A_SHA:raise ValueError('Wrong A checkpoint bytes.')
    base=ImageCaptioningModel(UniversalVisionEncoder(load_backbone=False,visual_adapter='qformer',num_visual_queries=32,qformer_layers=2),CaptionDecoder(len(tokenizer.word2idx)))
    state=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)
    if state['caption_word2idx']!=tokenizer.word2idx:raise ValueError('Checkpoint vocabulary mismatch.')
    weights=state['model_state_dict']
    if not weights or any(not k.startswith('base.') for k in weights):raise ValueError('Expected exact A wrapper state, no hidden branch.')
    base.load_state_dict({k[5:]:v for k,v in weights.items()},strict=True)
    if model_digest(base)!=A_BASE_SHA:raise ValueError('A tensor digest mismatch.')
    labels=Path(cfg['labels']);prior=json.loads((labels/'prior.json').read_text())['probability'];lexical=json.loads((labels/'lexicon.json').read_text())
    return EvidenceModel(base,prior,lexical,cfg['arm']).cuda().eval()


def distributed_start():
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from captioning.scst_runtime import configure_strict_runtime
    rank=int(os.environ['RANK']);local=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=2:raise ValueError('One arm requires two GPUs.')
    torch.cuda.set_device(local);torch.set_num_threads(2);configure_strict_runtime(True)
    dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    return rank,local


def predict(model,batches,tokenizer,condition,overrides=None):
    import torch
    from captioning.scst import rollout,decode
    from run_pilot import cuda_batch
    result=[]
    with torch.no_grad():
        for index,batch in enumerate(batches):
            batch=cuda_batch(batch)
            if overrides is not None:batch['override_d']=torch.stack([overrides[int(i)] for i in batch['image_id'].cpu().tolist()]).cuda()
            memory,padding=model.memory(batch,condition);seq,_=rollout(model,memory,padding,tokenizer,sample=False)
            result.extend(dict(image_id=i,caption=s) for i,s in zip(batch['image_id'].cpu().tolist(),decode(seq,tokenizer)))
            if index%10==0:print(f'Evaluation {condition}: {len(result)} local images',flush=True)
    return result


def preflight(protocol):
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    from captioning.scst import rollout
    from captioning.scst_runtime import model_digest,runtime_manifest
    from evidence_model import global_mean,logmeanexp
    from run_pilot import corpus,cuda_batch,seed_all
    cfg=json.loads(Path(protocol).read_text());stage.verify_sources(cfg);rank,local=distributed_start();out=Path(cfg['output'])
    try:
        seed_all(42);data,tokenizer=corpus(cfg);model=build(cfg,tokenizer)
        batch=cuda_batch(next(iter(stage.loader(cfg,data,tokenizer,'train',rank))))
        frozen=model_digest(model.base)
        with torch.no_grad():
            memory,padding=model.memory(batch);targets=batch['captions'].flatten(0,1)[:,:-1];masks=batch['caption_mask'].flatten(0,1)[:,:-1]
            if not torch.equal(model.current_d,torch.zeros_like(model.current_d)):raise AssertionError('Initial correction must be exactly zero.')
            on=model.teacher(memory,padding,targets,masks);model.condition='off';off=model.teacher(memory,padding,targets,masks)
            # Same batch and same expanded memory: no invalid cross-batch bitwise assertion.
            torch.testing.assert_close(on,off,rtol=0,atol=0)
            a,_=rollout(model,memory,padding,tokenizer,sample=False);model.condition='on';b,_=rollout(model,memory,padding,tokenizer,sample=False)
            if not torch.equal(a,b):raise AssertionError('Initial greedy must reproduce A.')
            # A constant field must pool to itself, globally and with fractional bbox weights.
            field=torch.randn(2,1,80,device='cuda').expand(2,196,80);weights=torch.rand(2,196,device='cuda')
            torch.testing.assert_close(logmeanexp(field),field[:,0],rtol=0,atol=0)
            torch.testing.assert_close(logmeanexp(field,weights),field[:,0],rtol=0,atol=2e-7)
            # Correction identity through five-reference expansion; all ineligible logits unchanged.
            model.current_d.fill_(.75);model.condition='on';shifted=model.teacher(memory,padding,targets,masks)
            keep=torch.ones(on.shape[-1],device='cuda',dtype=torch.bool);keep[model.candidate_ids]=False
            torch.testing.assert_close(shifted[:,:,keep],off[:,:,keep],rtol=0,atol=0)
            if not bool((shifted[:,:,model.candidate_ids]!=off[:,:,model.candidate_ids]).any()):raise AssertionError('No decoder correction.')
            # Exercise every compound guard with real vocab IDs. Prefix position is last input word.
            for guard in model.guard_data:
                prefix=[ids[0] for ids in reversed(guard['prefix_ids_reversed'])]
                tgt=torch.tensor([[tokenizer.bos_idx]+prefix],device='cuda').repeat(len(batch['image_id']),1)
                raw=model.base.decoder(tgt,memory,memory_key_padding_mask=padding);corrected=model.decoder(tgt,memory,memory_key_padding_mask=padding)
                ids=model.candidate_ids[torch.tensor(guard['columns'],device='cuda')]
                torch.testing.assert_close(raw[:,-1,ids],corrected[:,-1,ids],rtol=0,atol=0)
            # Teacher/prefix last-token scores use the same correction (decoder floating-point tolerance only).
            tgt=targets.reshape(len(batch['image_id']),5,-1)[:,0,:8];full=model.decoder(tgt,memory,memory_key_padding_mask=padding)
            for j in (1,4,8):
                short=model.decoder(tgt[:,:j],memory,memory_key_padding_mask=padding)
                torch.testing.assert_close(short[:,-1],full[:,j-1],rtol=1e-5,atol=1e-5)
        # Uneven count and zero-region collective regression: DDP mean equals global mean.
        for count in (rank,0):
            p=torch.tensor(2.,device='cuda',requires_grad=True);loss,total=global_mean(p*count,count);loss.backward()
            gradient=p.grad.clone();dist.all_reduce(gradient);gradient/=2
            expected=1. if total else 0.
            torch.testing.assert_close(gradient,gradient.new_tensor(expected),rtol=0,atol=0)
        # Actual CE+presence+region and sampled sequence rescoring have head gradients, frozen A has none.
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            memory,padding=model.memory(batch);seq,active=rollout(model,memory,padding,tokenizer,sample=True)
        loss,_,_=model(batch,seq,active,torch.ones(len(seq),device='cuda'),.05,tokenizer.pad_idx)
        if not torch.isfinite(loss):raise AssertionError('Nonfinite preflight loss.')
        loss.backward()
        if not bool(model.head.net[-1].weight.grad.abs().sum()>0):raise AssertionError('No head gradients.')
        if any(p.grad is not None for p in model.base.parameters()) or model_digest(model.base)!=frozen:raise AssertionError('Frozen A changed.')
        save(out/f'preflight_rank{rank}.json',dict(status='PREFLIGHT_PASS',base_sha256=frozen,trainable_parameters=108752,
            tests=['zero correction','same-batch logits','greedy A parity','compound guard','five refs','noneligible logits','rollout teacher','global uneven/zero means','CE/SCST/aux gradients','frozen A'],runtime=runtime_manifest()))
        print(f'PREFLIGHT PASS rank{rank}; next process resets state/RNG.',flush=True)
    finally:dist.destroy_process_group()


def worker(protocol):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from captioning.scst import TrainingCider,rollout,decode
    from captioning.object_reward import TrainingObjectReward,combine_rewards
    from captioning.scst_runtime import model_digest,runtime_manifest
    from run_pilot import corpus,cuda_batch,seed_all
    cfg=json.loads(Path(protocol).read_text());stage.verify_sources(cfg);rank,local=distributed_start();out=Path(cfg['output'])
    try:
        if json.loads((out/f'preflight_rank{rank}.json').read_text())['status']!='PREFLIGHT_PASS':raise ValueError('Missing preflight.')
        if digest(cfg['stage'])!=cfg['stage_sha256']:raise ValueError('Changed stage.')
        seed_all(42);dataset,tokenizer=corpus(cfg);model=build(cfg,tokenizer)
        save(out/f'runtime_rank{rank}.json',runtime_manifest())
        train=stage.loader(cfg,dataset,tokenizer,'train',rank);val=stage.loader(cfg,dataset,tokenizer,'val',rank)
        if len(train)!=1250:raise ValueError('Expected1250 updates/epoch.')
        save(out/f'origin_rank{rank}.json',predict(model,val,tokenizer,'off'))
        initial=model_digest(model.head);seed_all(42+rank)
        if digest(out/'train_object_targets.json')!=cfg['object_targets_sha256']:raise ValueError('Reward targets changed.')
        object_reward=TrainingObjectReward(json.loads((out/'train_object_targets.json').read_text())['data'],'aliases_v2','unique_count_v2')
        refs={i:[s['raw'] for s in row['sentences'][:5]] for i,row in enumerate(dataset['images']) if row['split'] in ('train','restval')}
        print(f'[rank{rank}] Building fixed TRAIN CIDEr-D IDF ({len(refs)} images)',flush=True)
        reward=TrainingCider(refs,reference_cache_size=256)
        optimizer=torch.optim.AdamW(model.head.parameters(),lr=1e-4,weight_decay=.01)
        save(out/f'optimizer_rank{rank}.json',[{k:v for k,v in g.items() if k!='params'} for g in optimizer.param_groups])
        ddp=DDP(model,device_ids=[local],output_device=local,broadcast_buffers=False,find_unused_parameters=False)
        history=[];begin=time.monotonic()
        for epoch in range(2):
            train.sampler.set_epoch(epoch)
            for batch in train:
                step=len(history)+1;batch=cuda_batch(batch);optimizer.zero_grad(set_to_none=True)
                weight=0. if step<=200 else .05;sample=active=advantage=None
                record=dict(step=step,epoch=epoch,global_images_seen=step*8,scst_weight=weight)
                if weight:
                    with torch.no_grad():
                        memory,padding=model.memory(batch);sample,active=rollout(model,memory,padding,tokenizer,sample=True);greedy,_=rollout(model,memory,padding,tokenizer,sample=False)
                    ids=batch['image_id'].cpu().tolist();sampled_text=decode(sample,tokenizer);greedy_text=decode(greedy,tokenizer)
                    sc=reward.score(ids,sampled_text);gc=reward.score(ids,greedy_text);so,_=object_reward.score(ids,sampled_text);go,_=object_reward.score(ids,greedy_text)
                    sr=combine_rewards(sc,so,.5);gr=combine_rewards(gc,go,.5);advantage=torch.from_numpy(sr-gr).cuda()
                    record.update(sample_reward=float(sr.mean()),greedy_reward=float(gr.mean()),advantage_mean=float((sr-gr).mean()))
                loss,ce,rl=ddp(batch,sample,active,advantage,weight,tokenizer.pad_idx)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training loss.')
                loss.backward();grad=torch.nn.utils.clip_grad_norm_(model.head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                record.update(model.detail);record.update(loss=float(loss.detach()),ce_loss=float(ce),scst_loss=float(rl),grad_norm=float(grad),seconds=time.monotonic()-begin);history.append(record)
                if step==1 or step%25==0:
                    save(out/f'history_rank{rank}.json',history);print(f'[{cfg["arm"]} rank{rank}] {json.dumps(record)}',flush=True)
        if len(history)!=2500 or model_digest(model.base)!=A_BASE_SHA:raise AssertionError('Update count/frozen A changed.')
        final=model_digest(model.head);all_hash=[None,None];dist.all_gather_object(all_hash,final)
        if len(set(all_hash))!=1:raise AssertionError('Rank head states differ.')
        if rank==0:
            temp=out/'head.bin.partial';torch.save(dict(format='noun_evidence_head_v1',head=model.head.state_dict(),arm=cfg['arm'],source_checkpoint_sha256=A_SHA,
                label_receipt_sha256=cfg['label_receipt_sha256'],protocol_sha256=digest(protocol),updates=2500),temp,_use_new_zipfile_serialization=False);temp.replace(out/'head.bin')
        save(out/f'on_rank{rank}.json',predict(model,val,tokenizer,'on'))
        # Frozen A needs no redundant 5000-image off decode: verify the same input batch after training.
        with torch.no_grad():
            check=cuda_batch(next(iter(val)));memory,padding=model.memory(check,'off');tgt=check['captions'][:,0,:-1];mask=check['caption_mask'][:,0,:-1]
            torch.testing.assert_close(model.decoder(tgt,memory,tgt_key_padding_mask=mask,memory_key_padding_mask=padding),model.base.decoder(tgt,memory,tgt_key_padding_mask=mask,memory_key_padding_mask=padding),rtol=0,atol=0)
        save(out/f'worker_complete_rank{rank}.json',dict(status='COMPLETE',updates=2500,initial_head_sha256=initial,final_head_sha256=final,final_base_sha256=A_BASE_SHA,
            trainable_base_parameters=0,trainable_head_parameters=108752,test_used=False,off_parity_after_training=True))
    finally:dist.destroy_process_group()


def controls(protocol):
    """No further training. Provisional S/A gate; final interpretation also requires S/G."""
    import torch
    import torch.distributed as dist
    from run_pilot import corpus,cuda_batch,seed_all
    cfg=json.loads(Path(protocol).read_text());out=Path(cfg['output']);summary=json.loads((out/'run_summary.json').read_text())
    if cfg['arm']!='S' or not summary['S_A_gate']:
        print('Controls skipped: not S or S/A gate failed.',flush=True);return
    rank,local=distributed_start()
    try:
        seed_all(42);data,tokenizer=corpus(cfg);model=build(cfg,tokenizer);state=torch.load(out/'head.bin',map_location='cpu',weights_only=False);model.head.load_state_dict(state['head'],strict=True)
        total=torch.zeros(80,device='cuda',dtype=torch.float64);count=torch.zeros((),device='cuda',dtype=torch.float64)
        with torch.no_grad():
            for b in stage.loader(cfg,data,tokenizer,'train',rank):
                d=model.head(cuda_batch(b)['visual'])[2];total+=d.double().sum(0);count+=len(d)
            dist.all_reduce(total);dist.all_reduce(count)
            if int(count)!=10000:raise ValueError('TRAIN mean count mismatch.')
            mean=(total/count).float().cpu()
            val=stage.loader(cfg,data,tokenizer,'val',rank);local_values={}
            for b in val:
                d=model.head(cuda_batch(b)['visual'])[2].cpu();local_values.update({int(i):v for i,v in zip(b['image_id'].tolist(),d)})
        parts=[None,None];dist.all_gather_object(parts,local_values);values={k:v for part in parts for k,v in part.items()}
        ids=sorted(values);donors=cfg['shuffle_donors']
        if set(ids)!={int(k) for k in donors}:raise ValueError('Shuffle cohort changed.')
        override_mean={i:mean for i in ids};override_shuffle={i:values[int(donors[str(i)])] for i in ids}
        save(out/f'train_mean_rank{rank}.json',predict(model,val,tokenizer,'train_mean',override_mean))
        save(out/f'shuffle_rank{rank}.json',predict(model,val,tokenizer,'shuffle',override_shuffle))
        save(out/f'controls_rank{rank}.json',dict(status='COMPLETE',train_images=int(count),provisional_gate='S/A only; final S/G comparison still required',mean_d=mean.tolist()))
    finally:dist.destroy_process_group()


def archive(protocol):
    cfg=json.loads(Path(protocol).read_text());out=Path(cfg['output'])
    with zipfile.ZipFile(str(out)+'_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.log','.py'):z.write(path,path.relative_to(out))
    print('SEND:',str(out)+'_review.zip','KEEP:',out/'head.bin',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','preflight','worker','evaluate','controls','archive']);p.add_argument('--config',required=True);a=p.parse_args()
    if a.mode=='prepare':stage.prepare(a.config)
    elif a.mode=='evaluate':
        from evidence_metrics import evaluate
        evaluate(a.config)
    else:globals()[a.mode](a.config)
