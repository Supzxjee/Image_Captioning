"""Frozen F, four conditioning interventions. Two independent GPU workers; no training."""
import argparse
import json
import os
from pathlib import Path
import shutil
import zipfile
import numpy as np
from probe_data import digest,identity
from oracle_data import oracle_slots,summarize
from review_html import render


def load(cfg):
    for key,expected in cfg['source_sha256'].items():assert digest(Path(__file__).parent/key)==expected,key
    assert digest(cfg['manifest'])==cfg['manifest_sha256']
    manifest=json.loads(Path(cfg['manifest']).read_text(encoding='utf-8'))
    assert identity(manifest['cases'])==manifest['cases_sha256'] and len(manifest['cases'])==20
    assert digest(cfg['checkpoint'])==manifest['checkpoint_sha256'],'Use completed F checkpoint SHA4938bfe8..., not E or original'
    for key,expected in manifest['expected_input_hashes'].items():assert digest(cfg[key])==expected,key
    return manifest


def worker(config,rank):
    import torch
    from transformers import CLIPTextModel,AutoTokenizer
    from captioning.models import UniversalVisionEncoder,CaptionDecoder,ImageCaptioningModel
    from captioning.visual_cache import VisualCache
    from captioning.semantic_prompts import align_prompt_cache
    from captioning.scst_runtime import configure_strict_runtime,model_digest,runtime_manifest
    from captioning.scst import rollout,decode
    from base_runner import corpus,CLIP,CLIP_REVISION
    from pilot_model import PilotModel
    from pilot_data import region_slots,evidence_bounds
    print(f'[GPU {rank}] Checking pinned inputs and frozen F checkpoint; no training',flush=True)
    cfg=json.loads(Path(config).read_text());manifest=load(cfg)
    torch.cuda.set_device(rank);configure_strict_runtime(True);torch.manual_seed(42)
    dataset,tokenizer=corpus(cfg);by_id={r['cocoid']:r for r in dataset['images']}
    state=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True)
    assert state['arm']=='F' and state['format']=='semantic_evidence_pilot_v1' and state['completed_updates']==1250
    assert state['protocol_sha256']==manifest['checkpoint_protocol_sha256']
    assert state['cohort_sha256']==manifest['cohort_sha256'] and state['source_checkpoint_sha256']==manifest['source_checkpoint_sha256']
    assert state['caption_word2idx']==tokenizer.word2idx
    base=ImageCaptioningModel(UniversalVisionEncoder(load_backbone=False,visual_adapter='qformer',num_visual_queries=32,qformer_layers=2),CaptionDecoder(len(tokenizer.word2idx)))
    model=PilotModel(base,'F');model.load_state_dict(state['model_state_dict'],strict=True)
    del state
    model=model.cuda().eval().requires_grad_(False);before=model_digest(model)
    cases=manifest['cases'][rank::2]
    names=[str(Path('/images')/by_id[r['coco_id']]['filepath']/r['filename']) for r in cases]
    bundle=torch.load(cfg['legacy_prompt'],map_location='cpu',weights_only=False)
    legacy=align_prompt_cache(bundle['data'],names);del bundle
    prompts=json.loads(Path(cfg['upt_prompt']).read_text())
    text_sets=[]
    for row in cases:
        text_sets+=oracle_slots(row,True)['texts']+oracle_slots(row,False)['texts']
        # Only text fields, independent of actual image values.
        text_sets+=region_slots(prompts[row['filename']],np.zeros((197,768),dtype=np.float32))[3]
    texts=sorted(set(text_sets)-{''});embedding={}
    print(f'[GPU {rank}] Loading frozen CLIP text encoder; {len(cases)} cases',flush=True)
    encoder=CLIPTextModel.from_pretrained(CLIP,revision=CLIP_REVISION).cuda().eval().requires_grad_(False)
    clip_tokenizer=AutoTokenizer.from_pretrained(CLIP,revision=CLIP_REVISION)
    with torch.inference_mode():
        for start in range(0,len(texts),64):
            batch=texts[start:start+64];tokens=clip_tokenizer(batch,padding=True,truncation=True,max_length=77,return_tensors='pt').to('cuda')
            for text,value in zip(batch,encoder(**tokens).pooler_output.float().cpu().numpy()):embedding[text]=value
    del encoder
    def text_tensor(words):return torch.tensor(np.stack([embedding[w] if w else np.zeros(512,dtype=np.float32) for w in words]),device='cuda')[None]
    cache=VisualCache([cfg['visual_cache']],id_key='coco_id');cache.require_ids([r['coco_id'] for r in cases],'diagnostic')
    records=[]
    try:
        with torch.inference_mode():
            for row,name in zip(cases,names):
                visual=cache.read(row['coco_id']);old=legacy[name];entry=prompts[row['filename']]
                _,roles,valid,words=region_slots(entry,visual)
                original=dict(visual=torch.from_numpy(visual).cuda()[None],legacy=old['tokens'].float().cuda()[None],
                    legacy_mask=old['mask'].long().cuda()[None],text=text_tensor(words),
                    roles=torch.from_numpy(roles).cuda()[None],valid=torch.from_numpy(valid).cuda()[None],
                    bounds=torch.from_numpy(evidence_bounds(entry)).cuda()[None])
                p=row['positive_index'];phrases=[row['candidates'][j]['text'].rstrip('.') for j in (p,1-p)]
                encoded=[tokenizer.encode(t) for t in phrases];target=torch.stack([v[0] for v in encoded]).cuda();mask=torch.stack([v[1] for v in encoded]).cuda()
                assert not (target==tokenizer.word2idx[tokenizer.UNK]).any(),'Contrast description contains UNK; do not interpret its likelihood'
                assert (target==tokenizer.eos_idx).any(-1).all(),'Contrast description was truncated'
                conditions={};memory_bank={};attentions={}
                for condition in manifest['conditions']:
                    data=original.copy()
                    if condition.startswith('annotation_'):
                        slots=oracle_slots(row,condition=='annotation_positive')
                        data.update(text=text_tensor(slots['texts']),roles=torch.from_numpy(slots['roles']).cuda()[None],
                            valid=torch.from_numpy(slots['valid']).cuda()[None],bounds=torch.from_numpy(slots['bounds']).cuda()[None])
                    memory,padding=model.memory(data,'off' if condition=='off' else 'on')
                    memory_bank[condition]=memory.clone();attentions[condition]=model.last_attention.clone()
                    logits=model.teacher(memory,padding,target[:,:-1],mask[:,:-1]).float()
                    assert torch.isfinite(logits).all(),'Nonfinite diagnostic logits'
                    logp=logits.log_softmax(-1).gather(-1,target[:,1:,None]).squeeze(-1);keep=target[:,1:].ne(tokenizer.pad_idx)
                    total=(logp*keep).sum(-1);means=total/keep.sum(-1)
                    seq,_=rollout(model,memory,padding,tokenizer,sample=False)
                    conditions[condition]=dict(caption=decode(seq,tokenizer)[0],target_descriptions=phrases,
                        target_logp=total.cpu().tolist(),target_mean_logp=means.cpu().tolist(),target_tokens=keep.sum(-1).cpu().tolist(),
                        positive_minus_negative_logp=float(total[0]-total[1]),positive_minus_negative_mean_logp=float(means[0]-means[1]))
                pos,neg=memory_bank['annotation_positive'],memory_bank['annotation_negative']
                for memory in memory_bank.values():torch.testing.assert_close(memory[:,:-9],pos[:,:-9],rtol=0,atol=0)
                torch.testing.assert_close(pos[:,-9:-7],neg[:,-9:-7],rtol=0,atol=0)
                # Replay R matcher scores with identical text and boxes, independent of generated captions.
                text=text_tensor([r['text'] for r in row['candidates']])[0]
                bounds=torch.tensor([c['bounds'] for c in row['candidates']],device='cuda')
                _,scores,_=model.extra.attend(original['visual'][0,1:][None].expand(2,-1,-1),text,torch.full((2,),2,dtype=torch.long,device='cuda'),bounds)
                replay_error=float((scores-torch.tensor(row['scores']['boxed'],device='cuda')).abs().max())
                assert replay_error<1e-4,('Matcher replay mismatch',row['case_id'],replay_error)
                record=dict(row,conditions=conditions,matcher_replay_max_abs_error=replay_error,
                    evidence_change=dict(R_attention_L1=float((attentions['annotation_positive'][:,2]-attentions['annotation_negative'][:,2]).abs().sum()),
                        R_token_relative_L2=float((pos[:,-7]-neg[:,-7]).norm()/pos[:,-7].norm().clamp_min(1e-12))))
                records.append(record)
                print(f'[GPU {rank}] {len(records)}/{len(cases)} {row["case_id"]}',flush=True)
    finally:cache.close()
    assert model_digest(model)==before and all(p.grad is None for p in model.parameters())
    output=dict(rank=rank,records=records,no_training=True,parameters_unchanged=True,model_sha256=before,
                runtime=runtime_manifest(),manifest_sha256=cfg['manifest_sha256'],checkpoint_sha256=manifest['checkpoint_sha256'])
    (Path(cfg['output'])/f'rank{rank}.json').write_text(json.dumps(output,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def finish(config):
    cfg=json.loads(Path(config).read_text());manifest=load(cfg);out=Path(cfg['output'])
    ranks=[json.loads((out/f'rank{rank}.json').read_text()) for rank in (0,1)]
    assert ranks[0]['model_sha256']==ranks[1]['model_sha256']
    records=[]
    for rank,data in enumerate(ranks):
        assert data['rank']==rank and data['no_training'] and data['parameters_unchanged']
        assert [r['case_id'] for r in data['records']]==[r['case_id'] for r in manifest['cases'][rank::2]]
        records+=data['records']
    records.sort(key=lambda r:r['case_id'])
    images=Path(cfg['coco_images']);(out/'images').mkdir(exist_ok=True)
    for row in manifest['cases']:
        candidates=[images/row['filepath']/row['filename'],images/row['filename']]
        source=next((p for p in candidates if p.is_file()),None)
        if source is None:raise FileNotFoundError(f'Raw image missing: {candidates}; set COCO_IMAGES to actual images root')
        shutil.copy2(source,out/'images'/row['filename'])
    render(manifest['cases'],out/'case_review_20.html',records)
    summary=dict(status='FROZEN_RELATION_CONDITIONING_DIAGNOSTIC_COMPLETE',metrics=summarize(records),
        no_training=True,test_used=False,checkpoint_sha256=manifest['checkpoint_sha256'],
        manifest_sha256=cfg['manifest_sha256'],human_semantic_verdicts_available=False,
        limits=['20 post-hoc failure images; descriptive diagnostic, not a benchmark or checkpoint selection',
               'Annotation zero does not logically contradict the alternate phrase; human plausibility review required',
               'Annotation +/- fix GT boxes and vary only R text; UPT differs in boxes and slot count, so oracle-versus-UPT is confounded',
               'Module outputs image-value tokens, not the input words. No caption change does not isolate decoder failure',
               'Likelihood preference is language-model sensitivity, not verified caption correctness or word-level attribution'])
    (out/'records.json').write_text(json.dumps(records,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False),flush=True)


def archive(config):
    cfg=json.loads(Path(config).read_text());out=Path(cfg['output'])
    with zipfile.ZipFile(str(out)+'_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix.lower() in ('.json','.py','.log','.html','.jpg'):z.write(p,p.relative_to(out))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','finish','archive']);p.add_argument('--config',required=True)
    a=p.parse_args()
    if a.mode=='worker':worker(a.config,int(os.environ['LOCAL_RANK']))
    elif a.mode=='finish':finish(a.config)
    else:archive(a.config)
