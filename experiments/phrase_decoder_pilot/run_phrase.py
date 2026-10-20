"""Manual Kaggle pipeline. No model execution on import; two GPUs train ONE arm."""
import argparse
from collections import Counter
from datetime import timedelta
import json
import os
from pathlib import Path
import shutil
import time
import zipfile
import numpy as np
from phrase_data import digest,identity,geometry

BASE_SHA='5f8eead3f4e960189345b1a9bf492763fe52ec71f70115b73244ffe9d03d2e75'
DATA_SHA='2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'


def save(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def load(config):
    cfg=json.loads(Path(config).read_text());manifest=json.loads(Path(cfg['manifest']).read_text())
    assert digest(cfg['manifest'])==cfg['manifest_sha256']
    content={k:v for k,v in manifest.items() if k!='manifest_content_sha256'}
    assert identity(content)==manifest['manifest_content_sha256']
    for name,expected in cfg['source_sha256'].items():assert digest(Path(__file__).parent/name)==expected,name
    assert cfg['arm'] in ('CE','CROSS')
    return cfg,manifest


def corpus(cfg):
    from captioning.tokenizer import CaptionTokenizer
    data=json.loads(Path(cfg['dataset_json']).read_text())
    tokenizer=CaptionTokenizer([s['raw'] for r in data['images'] if r['split'] in ('train','restval') for s in r['sentences'][:5]])
    assert (tokenizer.pad_idx,tokenizer.bos_idx,tokenizer.eos_idx)==(0,1,2)
    return data,tokenizer


def base_model(cfg,tokenizer):
    from captioning.models import UniversalVisionEncoder,CaptionDecoder,ImageCaptioningModel
    from captioning.checkpoints import load_checkpoint
    base=ImageCaptioningModel(UniversalVisionEncoder(load_backbone=False,visual_adapter='qformer',num_visual_queries=32,qformer_layers=2),CaptionDecoder(len(tokenizer.word2idx)))
    load_checkpoint(base,cfg['checkpoint'],tokenizer)
    assert not base.encoder.prompt_conditioned_qformer
    return base.requires_grad_(False).eval()


def encode(tokenizer,text,length):
    ids,_=tokenizer.encode(text,max_len=length)
    assert tokenizer.word2idx[tokenizer.UNK] not in ids,'Phrase contains UNK; do not silently score it'
    assert tokenizer.eos_idx in ids,'Phrase truncated; do not silently score it'
    return ids


def image_path(cfg,event):
    root=Path(cfg['coco_images']);paths=[root/event['filepath']/event['filename'],root/event['filename']]
    found=next((p for p in paths if p.is_file()),None)
    if found is None:raise FileNotFoundError(f'Raw review image missing: {paths}')
    return found


def prepare(config):
    import torch
    from captioning.visual_cache import VisualCache
    from captioning.scst_runtime import configure_strict_runtime,model_digest,tensor_digest,runtime_manifest
    cfg,m=load(config);out=Path(cfg['output'])
    if out.exists():raise FileExistsError(f'Use a fresh OUTPUT: {out}')
    assert torch.cuda.device_count()==2,'Select T4 x2; one arm uses both GPUs'
    assert digest(cfg['checkpoint'])==BASE_SHA,'Use original preserved SCST+CE+object baseline SHA5f8eead3, NOT F checkpoint/bin'
    assert digest(cfg['dataset_json'])==DATA_SHA,'COCO split/vocabulary JSON changed'
    dataset,tokenizer=corpus(cfg);by_id={r['cocoid']:r for r in dataset['images']}
    events={e['image_id']:e for g in m['groups'] for e in g['events']}
    for e in events.values():
        assert by_id[e['image_id']]['filename']==e['filename'] and by_id[e['image_id']]['split'] in ('train','restval')
        image_path(cfg,e)
    for g in m['groups']:
        for text in g['descriptions']:encode(tokenizer,text,m['max_length'])
    configure_strict_runtime(True);torch.manual_seed(42);torch.cuda.set_device(0)
    base=base_model(cfg,tokenizer).cuda();base_hash=model_digest(base)
    ids=sorted(events);patches=[];globals_=[];receipts=[]
    cache=VisualCache([cfg['visual_cache']],id_key='coco_id');cache.require_ids(ids,'phrase pilot')
    try:
        with torch.no_grad():
            for k,iid in enumerate(ids,1):
                visual=torch.from_numpy(cache.read(iid)).cuda()[None]
                # Pure visual path: NO legacy/text prompts or prompt-conditioned QFormer.
                g=base.encoder.qformer(base.encoder.relu(base.encoder.vis_projection(visual)))
                assert g.shape==(1,32,512) and torch.isfinite(g).all()
                patches.append(visual[0,1:].cpu().numpy());globals_.append(g[0].cpu().numpy())
                receipts.append(dict(image_id=iid,visual_sha256=tensor_digest(visual),global_sha256=tensor_digest(g)))
                print(f'Staged frozen visual memory: {k}/{len(ids)}',flush=True)
    finally:cache.close()
    assert model_digest(base)==base_hash
    stage=Path(cfg['stage']);stage.parent.mkdir(parents=True,exist_ok=True)
    if stage.exists():raise FileExistsError(f'Use a fresh stage: {stage}')
    np.savez(stage,image_ids=np.asarray(ids),patches=np.stack(patches),global_memory=np.stack(globals_))
    out.mkdir(parents=True)
    cfg.update(stage_sha256=digest(stage),stage_tensor_sha256=identity(receipts),base_sha256=base_hash,
        checkpoint_sha256=BASE_SHA,dataset_sha256=DATA_SHA,vocabulary_sha256=identity(tokenizer.word2idx),
        fit_groups=14,adaptation_holdout_groups=7,no_test=True,
        loss='positive mean-token CE including EOS + arm_CROSS_only scheduled0.25*softplus(negative_mean_word_logP-positive_mean_word_logP)',
        optimizer=dict(name='AdamW',evidence_lr=1e-4,adapter_lr=1e-5,weight_decay=.01,clip_norm=1.),
        train_scope='Evidence encoder and rank8 cross-attention K/V updates only; original decoder/projection/QFormer frozen; dropout OFF',
        evaluation='Before and after, all21groups on/off; tiny adaptation holdout7 only for description, no checkpoint/hyperparameter selection; no automatic semantic verdict')
    cfg['protocol_sha256']=identity(cfg)
    save(config,cfg);save(out/'protocol.json',cfg);save(out/'locked_manifest.json',m)
    save(out/'stage_receipt.json',receipts);save(out/'prepare_runtime.json',runtime_manifest())
    print('PREPARE PASS: original baseline, 38 images, no new text input; no SCST in this phrase diagnostic',flush=True)


def build(cfg,tokenizer):
    from phrase_model import PhraseModel
    base=base_model(cfg,tokenizer)
    return PhraseModel(base.decoder,base.encoder.vis_projection).cuda().eval()


def frozen_digest(model):
    from captioning.scst_runtime import tensor_digest
    return identity([(n,tensor_digest(p)) for n,p in model.named_parameters() if not p.requires_grad])


class Batches:
    def __init__(self,cfg,m,tokenizer):
        import torch
        archive=np.load(cfg['stage'],allow_pickle=False);self.index={int(i):k for k,i in enumerate(archive['image_ids'])}
        self.patches=torch.from_numpy(archive['patches']).cuda();self.global_memory=torch.from_numpy(archive['global_memory']).cuda();archive.close()
        self.groups={g['group_id']:g for g in m['groups']};self.mean=np.asarray(m['geometry_fit_mean'],dtype=np.float32);self.std=np.asarray(m['geometry_fit_std'],dtype=np.float32)
        self.encoded={g['group_id']:[encode(tokenizer,t,m['max_length']) for t in g['descriptions']] for g in m['groups']}
    def __call__(self,keys):
        import torch
        groups=[self.groups[key] for key in keys];events=[e for g in groups for e in g['events']]
        indices=torch.tensor([self.index[e['image_id']] for e in events],device='cuda')
        boxes=torch.tensor([[e['subject_box'],e['object_box']] for e in events],device='cuda')
        geo=torch.from_numpy(np.stack([(geometry(e['subject_box'],e['object_box'])-self.mean)/self.std for e in events])).cuda()
        targets=torch.stack([seq for key in keys for _ in range(2) for seq in self.encoded[key]]).cuda()
        positive=torch.tensor([g['positive_indices'] for g in groups],device='cuda')
        return (self.patches[indices],self.global_memory[indices],boxes,geo,targets,positive)


def decode(sequence,tokenizer):
    words=[]
    for token in sequence.tolist():
        if token==tokenizer.eos_idx:break
        if token not in (tokenizer.bos_idx,tokenizer.pad_idx):words.append(tokenizer.idx2word[token])
    return ' '.join(words)


def predict(model,batches,keys,tokenizer,max_length):
    import torch
    records=[]
    with torch.no_grad():
        for key in keys:
            p,g,b,geo,targets,positive=batches([key]);conditions={}
            for name,off in (('on',False),('off',True)):
                memory,padding,_=model.memory(p,g,b,geo,off=off)
                scores,nll=model.scores(memory,padding,targets)
                seq=model.greedy(memory,padding,max_length)
                scores=scores.reshape(2,2);pos=scores.gather(1,positive[0,:,None]).squeeze(1);neg=scores.gather(1,1-positive[0,:,None]).squeeze(1)
                margins=(pos-neg).cpu().tolist()
                conditions[name]=dict(scores=scores.cpu().tolist(),positive_margins=margins,
                    agreement=sum(.5 if abs(v)<=1e-7 else float(v>0) for v in margins)/2,
                    both_directions_correct=all(v>1e-7 for v in margins),
                    captions=[decode(s,tokenizer) for s in seq],truncated=[not bool(s.eq(2).any()) for s in seq],
                    positive_nll=nll.reshape(2,2).gather(1,positive[0,:,None]).mean().item())
            records.append(dict(group_id=key,conditions=conditions))
    return records


def worker(config,smoke=False):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from captioning.scst_runtime import configure_strict_runtime,model_digest,state_digest,runtime_manifest
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);configure_strict_runtime(True);torch.manual_seed(42)
    dist.init_process_group('nccl',timeout=timedelta(minutes=10));assert dist.get_world_size()==2
    cfg,m=load(config)
    if smoke:
        from test_phrase_model import synthetic
        model,batch=synthetic('cuda');model.eval();fit_keys=None
        # Different synthetic inputs per rank make the synchronization check meaningful.
        batch=(batch[0]+rank*.1,batch[1]-rank*.1,*batch[2:])
    else:
        assert digest(cfg['stage'])==cfg['stage_sha256'] and digest(cfg['checkpoint'])==BASE_SHA
        assert identity({k:v for k,v in cfg.items() if k!='protocol_sha256'})==cfg['protocol_sha256']
        _,tokenizer=corpus(cfg);assert identity(tokenizer.word2idx)==cfg['vocabulary_sha256']
        model=build(cfg,tokenizer);batches=Batches(cfg,m,tokenizer)
        fit_keys=m['partitions']['fit'];assert set(sum(m['schedule'],[]))==set(fit_keys)
    initial=model_digest(model);frozen=frozen_digest(model)
    all_initial=[None]*2;dist.all_gather_object(all_initial,initial);assert len(set(all_initial))==1
    ddp=DDP(model,device_ids=[rank],broadcast_buffers=False);ddp.eval()
    params=[p for p in model.parameters() if p.requires_grad]
    adapter=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('evidence.')]
    optimizer=torch.optim.AdamW([dict(params=model.evidence.parameters(),lr=1e-4),dict(params=adapter,lr=1e-5)],weight_decay=.01)
    if not smoke:
        all_keys=[g['group_id'] for g in m['groups']][rank::2]
        save(Path(cfg['output'])/f'before_rank{rank}.json',predict(model,batches,all_keys,tokenizer,m['max_length']))
    history=[];started=time.monotonic()
    for update in range(1,3 if smoke else m['updates']+1):
        if not smoke:
            keys=m['schedule'][update-1][rank*4:(rank+1)*4];assert set(keys)<=set(fit_keys);batch=batches(keys)
        weight=.25 if smoke else (m['cross_weight'] if cfg['arm']=='CROSS' and update>=m['cross_start_update'] else 0.)
        optimizer.zero_grad(set_to_none=True);loss,ce,cross=ddp(*batch,weight)
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params),'Missing/nonfinite trainable gradient'
        norm=torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step()
        if not smoke and update==60:
            save(Path(cfg['output'])/f'ce_warmup_rank{rank}.json',dict(update=60,model_sha256=model_digest(model),optimizer_sha256=state_digest(optimizer.state_dict())))
        if update==1 or update%25==0 or (not smoke and update==m['updates']):
            gradients={name:sum(float(p.grad.float().square().sum()) for p in module.parameters() if p.grad is not None)**.5 for name,module in [('evidence',model.evidence),('decoder_adapter',model.decoder)]}
            row=dict(update=update,CE=float(ce),cross=float(cross),cross_weight=weight,total=float(loss),grad_norm=float(norm),gradient_groups=gradients,seconds=time.monotonic()-started)
            history.append(row);print(f'[rank{rank}] '+json.dumps(row),flush=True)
    assert frozen_digest(model)==frozen and all(p.grad is None for p in model.parameters() if not p.requires_grad)
    final=model_digest(model);hashes=[None]*2;dist.all_gather_object(hashes,final);assert len(set(hashes))==1,'DDP final weights differ'
    if smoke:
        assert final!=initial
        if rank==0:print('TWO_GPU_SYNTHETIC_DDP_SMOKE_PASS',flush=True)
    else:
        assert final!=initial
        save(Path(cfg['output'])/f'after_rank{rank}.json',predict(model,batches,all_keys,tokenizer,m['max_length']))
        save(Path(cfg['output'])/f'history_rank{rank}.json',history)
        save(Path(cfg['output'])/f'complete_rank{rank}.json',dict(rank=rank,updates=m['updates'],arm=cfg['arm'],
            initial_sha256=initial,final_sha256=final,frozen_sha256=frozen,original_parameters_unchanged=True,
            protocol_sha256=cfg['protocol_sha256'],runtime=runtime_manifest(),
            trainable_parameters=sum(p.numel() for p in params),peak_gpu_bytes=torch.cuda.max_memory_allocated()))
        if rank==0:
            torch.save(dict(format='phrase_decoder_adaptation_v1',arm=cfg['arm'],source_checkpoint_sha256=BASE_SHA,
                protocol_sha256=cfg['protocol_sha256'],manifest_sha256=cfg['manifest_sha256'],vocabulary_sha256=cfg['vocabulary_sha256'],
                completed_updates=m['updates'],trainable_state={n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}),Path(cfg['output'])/'adaptation.bin')
    dist.destroy_process_group()


def metrics(records,keys):
    selected=[r for r in records if r['group_id'] in keys];assert len(selected)==len(keys)
    return {condition:dict(groups=len(selected),agreement=float(np.mean([r['conditions'][condition]['agreement'] for r in selected])),
        both_directions_correct=sum(r['conditions'][condition]['both_directions_correct'] for r in selected),
        mean_positive_margin=float(np.mean([v for r in selected for v in r['conditions'][condition]['positive_margins']])),
        positive_nll=float(np.mean([r['conditions'][condition]['positive_nll'] for r in selected])),
        captions_changed_vs_off=sum(a!=b for r in selected for a,b in zip(r['conditions'][condition]['captions'],r['conditions']['off']['captions']))) for condition in ('on','off')}


def finish(config):
    from phrase_review import render
    cfg,m=load(config);out=Path(cfg['output']);complete=[json.loads((out/f'complete_rank{r}.json').read_text()) for r in (0,1)]
    assert complete[0]['initial_sha256']==complete[1]['initial_sha256'] and complete[0]['final_sha256']==complete[1]['final_sha256']
    phases={}
    for phase in ('before','after'):
        rows=sum([json.loads((out/f'{phase}_rank{r}.json').read_text()) for r in (0,1)],[])
        rows.sort(key=lambda r:r['group_id']);assert {r['group_id'] for r in rows}=={g['group_id'] for g in m['groups']}
        phases[phase]=rows;save(out/f'{phase}_records.json',rows)
    by={g['group_id']:g for g in m['groups']};knn=[]
    std=np.asarray(m['geometry_fit_std'])
    for key in m['partitions']['adaptation_holdout']:
        group=by[key];donors=[(geometry(e['subject_box'],e['object_box']),g['positive_indices'][i]) for k in m['partitions']['fit'] for g in [by[k]] if g['family']==group['family'] for i,e in enumerate(g['events'])]
        predictions=[]
        for e in group['events']:
            query=geometry(e['subject_box'],e['object_box']);best=min(range(len(donors)),key=lambda j:float((((donors[j][0]-query)/std)**2).sum()))
            predictions.append(donors[best][1])
        knn.append(dict(group_id=key,predicted_indices=predictions,correct=[a==b for a,b in zip(predictions,group['positive_indices'])]))
    save(out/'geometry_control.json',dict(method='fit-only 1NN standardized eight-dimensional geometry within family; no image features',rows=knn))
    summary=dict(status='PHRASE_TECHNICAL_PILOT_COMPLETE',arm=cfg['arm'],updates=m['updates'],
        metrics={phase:{split:metrics(rows,keys) for split,keys in m['partitions'].items()} for phase,rows in phases.items()},
        initial_sha256=complete[0]['initial_sha256'],final_sha256=complete[0]['final_sha256'],
        stage_tensor_sha256=cfg['stage_tensor_sha256'],schedule_sha256=m['schedule_sha256'],manifest_sha256=cfg['manifest_sha256'],
        original_parameters_unchanged=True,semantic_human_evaluation_available=False,test_used=False,
        geometry_control=dict(agreement=float(np.mean([v for row in knn for v in row['correct']])),both_directions_correct=sum(all(row['correct']) for row in knn)),
        limits=['14 fit / 7 adaptation holdout groups previously reviewed; no independent benchmark or seed uncertainty',
                'Likelihood agreement and changed captions do not establish semantic/instance correctness; inspect free generation',
                'GT boxes and selected feasible groups; not end-to-end UPT or full caption evaluation',
                'Full-caption baseline A metrics cannot be compared to these phrase scores; no SCST/CIDEr/CHAIR here',
                'One bicycle group in fit only; no bicycle holdout. Tiny groups/template vocabulary can permit shortcuts.'])
    save(out/'summary.json',summary);(out/'images').mkdir()
    for e in {e['image_id']:e for g in m['groups'] for e in g['events']}.values():shutil.copy2(image_path(cfg,e),out/'images'/e['filename'])
    render(m,phases,out/'caption_review.html');print(json.dumps(summary,indent=2),flush=True)


def archive(config):
    cfg,_=load(config);out=Path(cfg['output'])
    if not out.is_dir():return
    with zipfile.ZipFile(str(out)+'_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.log','.py','.html','.jpg'):z.write(p,p.relative_to(out))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','smoke','worker','finish','archive']);p.add_argument('--config',required=True)
    a=p.parse_args()
    if a.mode in ('smoke','worker'):worker(a.config,smoke=a.mode=='smoke')
    else:globals()[a.mode](a.config)
