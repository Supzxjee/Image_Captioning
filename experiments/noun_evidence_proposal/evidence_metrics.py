"""Matched raw-greedy val5000 metrics. Validation is reused; no test selection."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from evidence_data import A_BASE_SHA,digest,save


def paired(a,b):
    delta=np.asarray(a)-np.asarray(b);rng=np.random.default_rng(20261023);draws=[]
    for _ in range(200):draws.extend(delta[rng.integers(0,len(delta),(100,len(delta)))].mean(1).tolist())
    low,high=np.quantile(draws,[.025,.975])
    return dict(mean=float(delta.mean()),paired95=[float(low),float(high)],resamples=20000,seed=20261023)


def exposures(cfg,out):
    done=[json.loads((out/f'worker_complete_rank{r}.json').read_text()) for r in (0,1)]
    for d in done:
        if d['status']!='COMPLETE' or d['updates']!=2500 or d['final_base_sha256']!=A_BASE_SHA or d['trainable_base_parameters']!=0:raise ValueError('Incomplete/wrong frozen-base run.')
    for key in ('initial_head_sha256','final_head_sha256'):
        if done[0][key]!=done[1][key]:raise ValueError('Different rank state: '+key)
    histories=[json.loads((out/f'history_rank{r}.json').read_text()) for r in (0,1)]
    expected={r['image_id'] for r in cfg['groups']['train']}
    for h in histories:
        if len(h)!=2500 or [v['step'] for v in h]!=list(range(1,2501)):raise ValueError('Wrong update sequence.')
        if sum(v['global_regions'] for v in h)!=11204 or sum(v['global_observed_labels'] for v in h)!=1597744:raise ValueError('Wrong two-epoch region/observed exposure.')
    for epoch in (0,1):
        ids=[i for h in histories for v in h if v['epoch']==epoch for i in v['local_image_ids']]
        if len(ids)!=10000 or len(set(ids))!=10000 or set(ids)!=expected:raise ValueError('TRAIN epoch order coverage mismatch.')
    save(out/'exposure_check.json',dict(status='PASS',epochs=2,images=20000,unique_regions=5602,region_exposures=11204,observed_label_exposures=1597744))


def evaluate(protocol):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    cfg=json.loads(Path(protocol).read_text());out=Path(cfg['output']);exposures(cfg,out)
    dataset=json.loads(Path(cfg['dataset_json']).read_text());expected={r['image_id']:r for r in cfg['groups']['val']};ids=sorted(expected)
    tokenizer=PTBTokenizer();refs=tokenizer.tokenize({i:[{'caption':s['raw']} for s in dataset['images'][i]['sentences'][:5]] for i in ids})
    conditions=['origin','on']+(['train_mean','shuffle'] if (out/'controls_rank0.json').exists() and (out/'controls_rank1.json').exists() else [])
    metrics={};ciders={};predictions={}
    for condition in conditions:
        rows=sum((json.loads((out/f'{condition}_rank{r}.json').read_text()) for r in (0,1)),[]);rows.sort(key=lambda r:r['image_id'])
        if len(rows)!=5000 or [r['image_id'] for r in rows]!=ids:raise ValueError('Missing/duplicate val predictions.')
        save(out/f'{condition}_predictions.json',rows);predictions[condition]=rows
        metric_file=out/f'{condition}_metrics.json';per_file=out/f'{condition}_per_image_cider.json'
        if metric_file.exists() and per_file.exists():
            prior=json.loads(metric_file.read_text())
            if prior['predictions_sha256']!=digest(out/f'{condition}_predictions.json'):raise ValueError('Metric cache predictions changed.')
            metrics[condition]=prior;ciders[condition]=np.asarray([r['CIDEr'] for r in json.loads(per_file.read_text())]);continue
        hyp=tokenizer.tokenize({r['image_id']:[{'caption':r['caption']}] for r in rows});score,values=Cider().compute_score(refs,hyp)
        ciders[condition]=np.asarray(values);save(per_file,[dict(image_id=i,CIDEr=float(v)) for i,v in zip(ids,values)])
        ci=out/f'{condition}_chair_input.json';co=out/f'{condition}_chair_output.json'
        save(ci,[dict(image_id=expected[r['image_id']]['coco_id'],caption=r['caption']) for r in rows])
        subprocess.run([sys.executable,'-u',str(Path(cfg['chair'])/'chair.py'),'--cap_file',str(ci),'--cache',str(Path(cfg['chair'])/'chair.pkl'),'--save_path',str(co)],cwd=cfg['chair'],check=True)
        chair=json.loads(co.read_text());sentences=chair['sentences']
        if len(sentences)!=5000 or {s['image_id'] for s in sentences}!={r['coco_id'] for r in expected.values()}:raise ValueError('CHAIR ID mismatch.')
        hallucinated=sum(len(s['mscoco_hallucinated_words']) for s in sentences);mentions=sum(len(s['mscoco_generated_words']) for s in sentences)
        correct=sum(len(set(s['mscoco_generated_words'])&set(s['mscoco_gt_words'])) for s in sentences);total=sum(len(set(s['mscoco_gt_words'])) for s in sentences)
        reagg=dict(CHAIRs=sum(bool(s['mscoco_hallucinated_words']) for s in sentences)/5000,CHAIRi=hallucinated/mentions if mentions else 0.,Recall=correct/total if total else 0.)
        if any(abs(reagg[k]-chair['overall_metrics'][k])>1e-12 for k in reagg):raise ValueError('CHAIR aggregation mismatch.')
        dev=set(cfg['secondary_dev_ids']);by_coco={s['image_id']:s for s in sentences};dev_s=[by_coco[expected[i]['coco_id']] for i in ids if i in dev]
        metrics[condition]=dict(CIDEr=float(score),chair=chair['overall_metrics'],counts=dict(images=5000,hallucinated_captions=sum(bool(s['mscoco_hallucinated_words']) for s in sentences),hallucinated_mentions=hallucinated,object_mentions=mentions,recalled_unique_objects=correct,reference_unique_objects=total),
            mean_caption_words=float(np.mean([len(r['caption'].split()) for r in rows])),predictions_sha256=digest(out/f'{condition}_predictions.json'),
            secondary_dev1000=dict(CIDEr_full_val5000_IDF=float(np.mean([v for i,v in zip(ids,values) if i in dev])),CHAIRs=sum(bool(s['mscoco_hallucinated_words']) for s in dev_s)/1000,
                note='Secondary CIDEr uses val5000 IDF; do not compare its magnitude directly to historical dev1000 IDF.'))
        save(metric_file,metrics[condition])
    comparison=paired(ciders['on'],ciders['origin']);a=metrics['origin']['chair'];s=metrics['on']['chair']
    passed=comparison['mean']>=.01 and comparison['paired95'][0]>0 and s['CHAIRs']<=a['CHAIRs'] and s['CHAIRi']<=a['CHAIRi'] and s['Recall']>=a['Recall']-.005
    reference=json.loads((Path(__file__).parent/'A_secondary_reference.json').read_text());origin={r['image_id']:r['caption'] for r in predictions['origin']}
    mismatches=[r['image_id'] for r in reference if origin[r['image_id']]!=r['caption']]
    controls={c:paired(ciders['on'],ciders[c]) for c in ('train_mean','shuffle') if c in ciders}
    changed=[]
    from difflib import SequenceMatcher
    lex=json.loads((Path(cfg['labels'])/'lexicon.json').read_text());eligible=set(lex['candidate_surfaces'].values())
    for before,after in zip(predictions['origin'],predictions['on']):
        x=before['caption'].split();y=after['caption'].split();outside=0
        for tag,i,j,k,l in SequenceMatcher(a=x,b=y,autojunk=False).get_opcodes():
            if tag!='equal':outside+=sum(w not in eligible for w in x[i:j]+y[k:l])
        changed.append(dict(image_id=before['image_id'],changed=x!=y,outside_eligible_token_edits=outside))
    save(out/'caption_changes.json',changed)
    summary=dict(status='NOUN_EVIDENCE_PILOT_COMPLETE_WAIT_MATCHED_G_S',arm=cfg['arm'],metrics=metrics,on_minus_A=comparison,S_A_gate=passed,
        controls_on_minus=controls,controls_gate=all(v['paired95'][0]>0 for v in controls.values()) if len(controls)==2 else None,
        historical_A_secondary_caption_mismatches=mismatches,trainable_parameters=108752,head_sha256=digest(out/'head.bin'),protocol_sha256=digest(protocol),
        test_used=False,changed_captions=sum(r['changed'] for r in changed),limits=[cfg['validation_scope'],'One seed; S/G paired comparison is required before any bbox claim.',
        'Class-level noun evidence, not relation grounding. Outside-noun output changes can follow autoregressive prefix changes.',
        'Controls run provisionally after S/A numeric gate; final interpretation also requires S/G. No extra training/checkpoint selection.',
        'Prior-subtracted scores are not calibrated likelihood ratios. Presence negatives and noun boxes are weak supervision.'])
    save(out/'run_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
