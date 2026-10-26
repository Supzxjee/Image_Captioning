"""User-run two-GPU frozen diagnostic, never an optimizer or teacher-bank builder."""
from collections import Counter
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
import numpy as np
from diagnostic_data import A_SHA,BASE_SHA,DATA_SHA,LEGACY_SHA,digest,save,paired_ci
from utility_data import POLICY,q_choice,teacher_contrasts,summarize_teacher


def prepare(cfg):
    import torch
    from captioning.semantic_prompts import align_prompt_cache
    from captioning.tokenizer import CaptionTokenizer
    from captioning.object_consistency import ALIASES
    from hidden_data import locked_groups
    from audit import spans
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    for key,sha in [('checkpoint',A_SHA),('dataset_json',DATA_SHA),('legacy_prompt',LEGACY_SHA)]:
        print('Hashing pinned input:',key,flush=True)
        if digest(cfg[key])!=sha:raise ValueError('Pinned input differs: '+key)
    if digest('cohort.json')!=cfg['cohort_sha256']:raise ValueError('Locked138 cohort metadata differs.')
    cohort=json.loads(Path('cohort.json').read_text(encoding='utf-8'))
    if cohort['policy']!=POLICY or len(cohort['rows'])!=138:raise ValueError('Wrong diagnostic cohort/policy.')
    data=json.loads(Path(cfg['dataset_json']).read_text(encoding='utf-8'));groups=locked_groups(data)
    train={r['coco_id']:r for r in groups['train']}
    tok=CaptionTokenizer([s['raw'] for r in data['images'] if r['split'] in ('train','restval') for s in r['sentences'][:5]])
    vocab_sha=hashlib.sha256(json.dumps(tok.word2idx,sort_keys=True).encode()).hexdigest()
    if vocab_sha!='427a33573c9302661282dd4553e87ccd82138f5903cea25823afc7eddb49bd3c':raise ValueError('Caption vocabulary differs.')
    names=[r['filename'] for r in cohort['rows']]
    bundle=torch.load(cfg['legacy_prompt'],map_location='cpu',weights_only=False)
    legacy=align_prompt_cache(bundle['data'],names)
    if set(legacy)!=set(names):raise ValueError('Missing unchanged prompt entries.')
    counts=Counter();rows=[]
    for r in cohort['rows']:
        source=train[r['coco_id']]
        if source['image_id']!=r['image_id'] or source['filename']!=r['filename']:raise ValueError('Wrong TRAIN image join.')
        image=data['images'][r['image_id']];targets={};caps=[];pads=[]
        for s in image['sentences'][:5]:
            ids,pad=tok.encode(s['raw']);caps.append(ids.tolist());pads.append(pad.tolist())
        for role in ('receiver','donor'):
            valid=[];category=r[role]['category']
            for s in r['targets'][role]:
                ci=s['caption_index'];text=image['sentences'][ci]['raw']
                if s['caption']!=text or s['coco_id']!=r['coco_id']:raise ValueError('Reference source changed.')
                raw,found=spans(text,ALIASES)
                first=min((start for start,end,label,surface in found if label==category),default=None)
                if first is None or s['span_words'][0]!=first:continue
                positions=s['decoder_logit_positions']
                if any(p>=29 for p in positions) or [caps[ci][p+1] for p in positions]!=s['token_ids']:raise ValueError('Wrong first-noun prediction positions.')
                if set(s['token_ids'])&{0,1,2,3}:raise ValueError('Special/unknown noun target.')
                if not any(t['caption_index']==ci for t in valid):valid.append(dict(caption_index=ci,positions=positions,token_ids=s['token_ids']))
            targets[role]=valid
            counts[role+'_with_mentions']+=bool(valid)
        entry=legacy[r['filename']]
        if list(entry['tokens'].shape)!=[20,512] or list(entry['mask'].shape)!=[20]:raise ValueError('Legacy shape differs from audited A.')
        if not torch.isfinite(entry['tokens']).all() or not set(entry['mask'].tolist())<={0,1}:raise ValueError('Invalid legacy input.')
        for role in ('receiver','donor'):
            if not all(entry['mask'][p] for p in r[role]['positions']):raise ValueError('Intervention on masked prompt token.')
        counts['both_mentions']+=bool(targets['receiver'] and targets['donor'])
        rows.append(dict(r,targets=targets,captions=caps,caption_padding=pads))
    torch.save({n:legacy[n] for n in names},out/'selected_legacy.pt')
    cfg.update(rows=rows,word2idx=tok.word2idx,policy=POLICY,train_ids=[r['image_id'] for r in groups['train']],
        prior_review_sha256=cohort['prior_review_sha256'],prior_original_text_sha256=cohort['original_text_sha256'],
        selected_legacy_sha256=digest(out/'selected_legacy.pt'),test_used=False)
    save(out/'coverage_pre_forward.json',dict(images=138,targets=dict(counts),q_selection=cohort['metadata_counts'],
        no_text_reconstruction=True,original_text_binding='Previously audited on these exact138 prompts; unchanged pinned .pt inputs'))
    save(out/'policy.json',POLICY);save(out/'protocol.json',cfg)
    print('Prepared exact138 TRAIN images; no model loaded, no training.',flush=True)


def score_nouns(model,memory,padding,row):
    import torch
    caps=torch.tensor(row['captions'],device='cuda');pads=torch.tensor(row['caption_padding'],device='cuda',dtype=torch.bool)
    logits=model.decoder(caps[:,:-1],memory.repeat_interleave(5,0),tgt_key_padding_mask=pads[:,:-1],memory_key_padding_mask=padding.repeat_interleave(5,0))
    if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite teacher logits.')
    logp=logits.float().log_softmax(-1);scores={}
    for role in ('receiver','donor'):
        values=[]
        for target in row['targets'][role]:
            ci=target['caption_index'];pos=torch.tensor(target['positions'],device='cuda');ids=torch.tensor(target['token_ids'],device='cuda')
            values.append(float(logp[ci,pos,ids].mean()))
        scores[role]=float(np.mean(values)) if values else None
    return scores


def worker(cfg):
    import h5py
    import torch
    from captioning.scst_runtime import configure_strict_runtime,model_digest,runtime_manifest
    from captioning.visual_cache import VisualCache
    from captioning.tokenizer import CaptionTokenizer
    from captioning.scst import rollout,decode
    from run_diagnostic import build,Probe
    from utility_intervention import Capture,Shift,correction_plan
    rank=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=2:raise ValueError('Select T4x2; independent workers, no DDP collectives.')
    torch.cuda.set_device(rank);torch.set_num_threads(2);configure_strict_runtime(True)
    for name,sha in cfg['implementation_sha256'].items():
        if digest(Path(__file__).parent/name)!=sha:raise ValueError('Source differs: '+name)
    out=Path(cfg['output'])
    if digest(out/'selected_legacy.pt')!=cfg['selected_legacy_sha256']:raise ValueError('Staged prompts changed.')
    legacy=torch.load(out/'selected_legacy.pt',map_location='cpu',weights_only=False)
    model=build(cfg);module=model.encoder.prompt_to_visual_attn
    tok=CaptionTokenizer([]);tok.word2idx=cfg['word2idx'];tok.idx2word={i:w for w,i in tok.word2idx.items()}
    cache=VisualCache([cfg['visual_cache']]);rows=cfg['rows'][rank::2]
    cache.require_ids([r['coco_id'] for r in rows],'A/T/W138 TRAIN');records=[];checked=set();start=time.monotonic();parity={}
    try:
        with torch.no_grad():
            for step,row in enumerate(rows):
                cid=row['coco_id'];entry=legacy[row['filename']];file_id,_=cache.index[cid]
                if file_id not in checked:
                    with h5py.File(cache.paths[file_id],'r') as f:
                        meta=json.loads(f.attrs['metadata_json'])
                        if (meta['dataset_json_sha256']!=DATA_SHA or meta['image_size']!=[224,224] or meta['preprocessing']!='bilinear'
                            or meta['model_id']!='openai/clip-vit-base-patch16' or meta['output']!='vision_model.last_hidden_state'
                            or meta['antialias'] is not True or meta['compute_dtype']!='float32' or meta['autocast'] is not False
                            or meta['mean']!=[.48145466,.4578275,.40821073] or meta['std']!=[.26862954,.26130258,.27577711]):
                            raise ValueError('Visual cache differs from square-resized A cache.')
                    checked.add(file_id)
                v=torch.from_numpy(cache.read(cid))[None].cuda();p=entry['tokens'].float()[None].cuda();mask=entry['mask'].long()[None].cuda()
                with Probe(model) as probe,Capture(module) as captured:
                    original,padding=model.build_memory(v,p,mask)
                q=probe.weights['q'].mean(1)[0].cpu().numpy();a=probe.weights['p'].mean(1)[0].cpu().numpy()
                selected=q_choice(q,row['receiver'],row['donor'])
                if (selected['q_plus'],selected['q_minus'],selected['distinguishable'])!=(row['selection']['q_plus'],row['selection']['q_minus'],row['selection']['distinguishable']):
                    raise ValueError('Q choices changed from audited frozen A; preserve review, do not reselect.')
                if step==0:
                    plain,ppad=model.build_memory(v,p,mask)
                    if not torch.equal(plain,original) or not torch.equal(ppad,padding):raise ValueError('Passive observation changed A.')
                    with Shift(module,[],None,0):zero,zpad=model.build_memory(v,p,mask)
                    bos=torch.tensor([[tok.bos_idx]],device='cuda')
                    before=model.decoder(bos,plain,memory_key_padding_mask=ppad);after=model.decoder(bos,zero,memory_key_padding_mask=zpad)
                    if not torch.equal(plain,zero) or not torch.equal(before,after):raise ValueError('Zero-rho parity failed.')
                    parity=dict(memory_max=0.,logit_max=0.,same_shape_images=1)
                pos=row['receiver']['positions'];plan=None;reason='q_indistinguishable';checks={}
                if selected['distinguishable']:
                    plan=correction_plan(module,captured.value,captured.read,pos,selected['q_plus'],selected['q_minus'])
                    reason='zero_visual_displacement' if plan is None else ''
                box=np.asarray(row['receiver']['mask']);b=q[:,1:]@box
                raw_mass=float((a[pos]@b).mean())
                if plan is not None:
                    changed_mass=raw_mass+plan['T'][1]*(float(b[selected['q_plus']])-raw_mass)
                    if changed_mass-raw_mass<=POLICY['epsilon']:plan=None;reason='no_positive_raw_mass_headroom'
                    else:checks.update(raw_mass_A=raw_mass,raw_mass_T=changed_mass,raw_mass_W=raw_mass+plan['W'][1]*(float(b[selected['q_minus']])-raw_mass))
                memories={'A':original};pre_norms={};post_norms={}
                for condition in ('T','W'):
                    if plan is None:memories[condition]=original;continue
                    delta,lam=plan[condition]
                    with Shift(module,pos,delta,lam),Capture(module) as intervention:
                        changed,changed_pad=model.build_memory(v,p,mask)
                    untouched=torch.ones(original.shape[1],dtype=torch.bool,device='cuda');untouched[pos]=False
                    if not torch.equal(changed[:,untouched],original[:,untouched]) or not torch.equal(changed_pad,padding):raise ValueError('Intervention changed Q/nonreceiver/text memory slots.')
                    pre_norms[condition]=float((intervention.read-captured.read).norm())
                    post_norms[condition]=float((changed-original).norm());memories[condition]=changed
                if plan is not None:
                    if not np.isclose(pre_norms['T'],pre_norms['W'],rtol=1e-4,atol=1e-6):raise ValueError('Pre-fusion perturbation norms differ.')
                    checks.update(lambda_T=plan['T'][1],lambda_W=plan['W'][1],pre_norms=pre_norms,post_norms=post_norms,
                        matched_norm=plan['matched_norm'],post_norm_T_over_W=post_norms['T']/post_norms['W'] if post_norms['W'] else None)
                scores={};captions={}
                for condition in ('A','T','W'):
                    if condition!='A' and plan is None:scores[condition]=scores['A'];captions[condition]=captions['A'];continue
                    scores[condition]=score_nouns(model,memories[condition],padding,row)
                    seq,_=rollout(model,memories[condition],padding,tok,sample=False)
                    captions[condition]=decode(seq,tok)[0]
                records.append(dict(image_id=row['image_id'],coco_id=cid,filename=row['filename'],intervened=plan is not None,
                    nonintervention_reason=reason,receiver=row['receiver']['category'],donor=row['donor']['category'],
                    selection=selected,manipulation=checks,scores=scores,contrasts=teacher_contrasts(scores),captions=captions))
                if step%10==0:print(f'GPU{rank}: {step+1}/{len(rows)} A/T/W images; no training',flush=True)
        if model_digest(model)!=BASE_SHA or any(x.requires_grad for x in model.parameters()):raise ValueError('Frozen A changed.')
    finally:cache.close()
    save(out/f'rows_rank{rank}.json',records)
    save(out/f'complete_rank{rank}.json',dict(status='FROZEN_ATW_COMPLETE',images=len(records),base_sha256=BASE_SHA,
        rows_sha256=digest(out/f'rows_rank{rank}.json'),parity=parity,seconds=time.monotonic()-start,
        runtime=runtime_manifest(),optimizer_used=False))


def evaluate(cfg):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    out=Path(cfg['output']);rows=[]
    for rank in (0,1):
        complete=json.loads((out/f'complete_rank{rank}.json').read_text())
        part=json.loads((out/f'rows_rank{rank}.json').read_text())
        if complete['status']!='FROZEN_ATW_COMPLETE' or complete['images']!=len(part) or complete['base_sha256']!=BASE_SHA or complete['rows_sha256']!=digest(out/f'rows_rank{rank}.json'):raise ValueError('Incomplete/wrong A/T/W worker.')
        rows+=part
    rows.sort(key=lambda r:r['image_id']);expected=sorted(r['image_id'] for r in cfg['rows'])
    if [r['image_id'] for r in rows]!=expected:raise ValueError('Wrong138 image coverage/order.')
    byid={r['image_id']:r for r in rows};data=json.loads(Path(cfg['dataset_json']).read_text(encoding='utf-8'))
    ids=sorted(cfg['train_ids']);ptb=PTBTokenizer()
    refs=ptb.tokenize({i:[dict(caption=s['raw']) for s in data['images'][i]['sentences'][:5]] for i in ids})
    cider={};metrics={};chair_root=Path(cfg['chair'])
    for condition in ('A','T','W'):
        hyp=ptb.tokenize({i:[dict(caption=byid[i]['captions'][condition] if i in byid else '')] for i in ids})
        _,values=Cider().compute_score(refs,hyp);cider[condition]={i:float(v) for i,v in zip(ids,values) if i in byid}
        predictions=[dict(image_id=r['coco_id'],caption=r['captions'][condition]) for r in rows]
        save(out/f'{condition}_predictions.json',predictions)
        chairfile=out/f'{condition}_chair.json'
        subprocess.run([sys.executable,'-u',str(chair_root/'chair.py'),'--cap_file',str(out/f'{condition}_predictions.json'),
            '--cache',str(chair_root/'chair.pkl'),'--save_path',str(chairfile)],cwd=chair_root,check=True)
        result=json.loads(chairfile.read_text());sentences=result['sentences']
        if len(sentences)!=138 or {s['image_id'] for s in sentences}!={r['coco_id'] for r in rows}:raise ValueError('CHAIR coverage differs.')
        counts=dict(images=138,hallucinated_captions=sum(bool(s['mscoco_hallucinated_words']) for s in sentences),
            hallucinated_mentions=sum(len(s['mscoco_hallucinated_words']) for s in sentences),object_mentions=sum(len(s['mscoco_generated_words']) for s in sentences),
            correct_objects=sum(len(set(s['mscoco_generated_words'])&set(s['mscoco_gt_words'])) for s in sentences),gt_objects=sum(len(set(s['mscoco_gt_words'])) for s in sentences))
        chair=dict(CHAIRs=counts['hallucinated_captions']/138,CHAIRi=counts['hallucinated_mentions']/max(1,counts['object_mentions']),Recall=counts['correct_objects']/max(1,counts['gt_objects']))
        if any(abs(chair[k]-result['overall_metrics'][k])>1e-12 for k in chair):raise ValueError('CHAIR totals differ.')
        metrics[condition]=dict(CIDEr=float(np.mean(list(cider[condition].values()))),chair=chair,counts=counts,
            mean_caption_words=float(np.mean([len(r['captions'][condition].split()) for r in rows])))
    for row in rows:row['CIDEr']={c:cider[c][row['image_id']] for c in cider}
    teacher=summarize_teacher(rows);contrasts={c:paired_ci([r['CIDEr']['T']-r['CIDEr'][c] for r in rows]) for c in ('A','W')}
    active=[r for r in rows if r['intervened']]
    secondary={c:paired_ci([r['CIDEr']['T']-r['CIDEr'][c] for r in active]) for c in ('A','W')}
    changed=sum(r['captions']['A']!=r['captions']['T'] for r in rows)
    safe=all(metrics['T']['chair'][k]<=metrics['A']['chair'][k] for k in ('CHAIRs','CHAIRi')) and metrics['T']['chair']['Recall']>=metrics['A']['chair']['Recall'] and metrics['T']['mean_caption_words']>=metrics['A']['mean_caption_words']
    go=teacher['positive_teacher_signal'] and changed>0 and all(x['mean']>0 for x in contrasts.values()) and safe
    ratios=[r['manipulation']['post_norm_T_over_W'] for r in active if r['manipulation']['post_norm_T_over_W'] is not None]
    summary=dict(status='ATW_DIAGNOSTIC_COMPLETE',decision='GO_DESIGN_SHORT_PILOT_ONLY' if go else 'NOT_YET_GO',teacher=teacher,
        metrics=metrics,T_minus=contrasts,secondary_intervened_T_minus=secondary,changed_T_vs_A=changed,
        nonintervention_reasons=dict(Counter(r['nonintervention_reason'] for r in rows if not r['intervened'])),
        post_fusion_norm_ratio=dict(median=float(np.median(ratios)) if ratios else None,min=float(min(ratios)) if ratios else None,max=float(max(ratios)) if ratios else None),
        no_metric_tradeoff_guard=safe,trained=False,test_used=False,
        limits=['GT-informed oracle control, not deployable inference or validated semantic Q identities.',
            'Reused138 restricted TRAIN images; screening only, not confirmed held-out caption gain.',
            'Positive CIDEr mean gate is not a significance claim; teacher benefit/selectivity require image-bootstrap95 lower>0.',
            '50primary images/50%active coverage and no length/recall decrease are conservative budget guards, not calibrated statistical power.',
            'Nonintervened images remain A in all138 caption metrics; do not promote changed-caption-only win rates.',
            'Pre-fusion norm matching does not guarantee matched post-gate/LN nuisance; inspect reported ratios.',
            'A negative frozen small correction does not rule out decoder co-adaptation, all alignment, or larger interventions.'])
    save(out/'scored_rows.json',rows);save(out/'result.json',summary)
    save(out/'receipt.json',dict(status='ATW_COMPLETE',images=138,decision=summary['decision'],policy=POLICY,
        payload_sha256={p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in ('receipt.json','selected_legacy.pt')},trained=False,test_used=False))
    print(json.dumps(summary,indent=2),flush=True)


def archive(cfg):
    out=Path(cfg['output'])
    with zipfile.ZipFile(str(out)+'_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.log','.py','.md'):z.write(p,p.relative_to(out))
    print('SEND:',str(out)+'_review.zip',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','worker','evaluate','archive']);parser.add_argument('--config',required=True)
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text(encoding='utf-8'));globals()[args.mode](cfg)
