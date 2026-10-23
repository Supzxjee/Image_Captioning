"""Reaggregate cached user-run R receipts and paired CIDEr; no torch or model execution."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def load(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(text):return hashlib.sha256(text.encode()).hexdigest()


def paired(rows,other):
    x={r['image_id']:r['CIDEr'] for r in rows};y={r['image_id']:r['CIDEr'] for r in other}
    assert len(rows)==len(other)==len(x)==len(y)==1000 and set(x)==set(y)
    delta=np.asarray([x[i]-y[i] for i in sorted(x)])
    rng=np.random.default_rng(20261023)
    means=np.concatenate([delta[rng.integers(0,1000,(250,1000))].mean(1) for _ in range(80)])
    return dict(delta=float(delta.mean()),paired95_percentile=np.quantile(means,[.025,.975]).tolist(),
        resamples=20000,seed=20261023,unit='CIDEr raw, not multiplied by100',
        higher=int((delta>1e-12).sum()),tied=int((abs(delta)<=1e-12).sum()),lower=int((delta < -1e-12).sum()),
        scope='Exploratory paired image bootstrap on reused development1000; not seed variance.')


def chair(rows):
    hallucinated=sum(len(r['mscoco_hallucinated_words']) for r in rows)
    mentions=sum(len(r['mscoco_generated_words']) for r in rows)
    recalled=sum(len(set(r['mscoco_generated_words'])&set(r['mscoco_gt_words'])) for r in rows)
    total=sum(len(set(r['mscoco_gt_words'])) for r in rows)
    return dict(CHAIRs=sum(bool(r['mscoco_hallucinated_words']) for r in rows)/1000,
        CHAIRi=hallucinated/mentions,Recall=recalled/total)


def verify_run(cached,a_cache,arm):
    r=cached['json'];a=a_cache['json']
    cfg=r['protocol.json'];summary=r['run_summary.json']
    assert cfg['arm']==arm and summary['status']=='ROUTING_PILOT_COMPLETE_WAIT_B_R_COMPARISON'
    assert cfg['noun_receipt_sha256']==load(HERE/'FULL_DATA_READY.json')['data_manifest_receipt_sha256']
    assert r['groups.json']==a['groups.json'] and r['origin_predictions.json']==a['origin_predictions.json']
    assert summary['evaluation_reference_tokens_sha256']==a['run_summary.json']['evaluation_reference_tokens_sha256']
    for name,expected in cfg['implementation_sha256'].items():
        assert sha(cached['sources']['source/'+name])==expected,name
        own=HERE/name
        if own.exists():assert sha(own.read_text(encoding='utf-8'))==expected,name
    assert sha(cached['sources']['source/run_pilot.py'].replace(
        "        if hasattr(model, 'training_diagnostics'):\n            record.update(model.training_diagnostics())\n",''))==sha(a_cache['source_loop'])
    for key in ('global_batch_size','warmup_updates','base_lr','scst_weight','seed','adamw_options','object_targets_sha256'):
        assert cfg[key]==a['protocol.json'][key],key
    assert r['optimizer_rank0.json']==r['optimizer_rank1.json']==a['optimizer_rank0.json']
    assert r['runtime_rank0.json']==r['runtime_rank1.json']==a['runtime_rank0.json']
    completions=[r[f'worker_complete_rank{rank}.json'] for rank in (0,1)]
    for done in completions:
        assert done['status']=='COMPLETE' and done['updates']==1250 and done['trainable_branch_parameters']==0
        assert done['test_used'] is False and done['initial_base_sha256']==a['worker_complete_rank0.json']['initial_base_sha256']
    for key in ('initial_full_sha256','final_full_sha256','final_base_sha256'):
        assert completions[0][key]==completions[1][key],key
    histories=[r[f'history_rank{rank}.json'] for rank in (0,1)]
    ids=[]
    for rank,h in enumerate(histories):
        assert len(h)==1250 and [s['step'] for s in h]==list(range(1,1251))
        pre=r[f'preflight_rank{rank}.json']
        assert pre['status']=='PREFLIGHT_PASS' and pre['optimizer_updates']==0
        assert pre['same_batch_logits_parity_checked'] and pre['expansion_check']
        assert sum(s['global_spans'] for s in h)==17904 and sum(s['global_person_spans'] for s in h)==4476
        assert all(s['scst_weight']==(0 if s['step']<=200 else .05) for s in h)
        assert all(np.isfinite(s[k]) for s in h for k in ('loss','ce_loss','scst_loss','aux_global_mean','grad_norm'))
        assert all(-1e-6<=s['bbox_mass']<=s['spatial_mass']+1e-6<=s['decoder_q_mass']+2e-6<=1+3e-6 for s in h)
        ids.extend(i for s in h for i in s['local_image_ids'])
    assert len(ids)==len(set(ids))==10000 and set(ids)=={i['image_id'] for i in r['groups.json']['train']}
    for x,y in zip(*histories):
        assert x['global_spans']==y['global_spans']==x['local_spans']+y['local_spans']
        assert x['global_person_spans']==y['global_person_spans']
        realized=np.mean([s['loss']-s['ce_loss']-s['scst_weight']*s['scst_loss'] for s in (x,y)])
        np.testing.assert_allclose(realized,.05*x['aux_global_mean'],rtol=0,atol=1e-6)
    for condition in ('origin','on'):
        predictions=r[condition+'_predictions.json'];scores=r[condition+'_per_image_cider.json']
        assert len(predictions)==1000 and {v['image_id'] for v in predictions}=={v['image_id'] for v in scores}
        np.testing.assert_allclose(np.mean([v['CIDEr'] for v in scores]),summary['metrics'][condition]['CIDEr'],atol=1e-12,rtol=0)
        sentences=r[condition+'_chair_output.json']['sentences']
        assert len(sentences)==1000 and {s['image_id'] for s in sentences}=={g['coco_id'] for g in r['groups.json']['val']}
        for key,value in chair(sentences).items():np.testing.assert_allclose(value,summary['metrics'][condition]['chair'][key],rtol=0,atol=1e-12)
    return r,a,cfg,summary,histories


def main():
    cached=load(HERE/'review/loaded_R.json')
    a_cache=load(ROOT/'experiments/vg_region_caption/review/loaded_A.json')
    r,a,cfg,summary,histories=verify_run(cached,a_cache,'R')
    ra=paired(r['on_per_image_cider.json'],a['on_per_image_cider.json'])
    ro=paired(r['on_per_image_cider.json'],r['origin_per_image_cider.json'])
    baseline=a['run_summary.json']['metrics']['on'];new=summary['metrics']['on']
    gates=dict(R_minus_A_at_least_001=ra['delta']>=.01,R_minus_A_paired95_lower_positive=ra['paired95_percentile'][0]>0,
        R_mean_CIDEr_above_original=ro['delta']>0,CHAIRs_not_above_A=new['chair']['CHAIRs']<=baseline['chair']['CHAIRs'],
        CHAIRi_not_above_A=new['chair']['CHAIRi']<=baseline['chair']['CHAIRi'],Recall_at_least_A_minus_001=new['chair']['Recall']>=baseline['chair']['Recall']-.01)
    hallucinated=lambda rows:{s['image_id'] for s in rows if s['mscoco_hallucinated_words']}
    old=hallucinated(a['on_chair_output.json']['sentences']);now=hallucinated(r['on_chair_output.json']['sentences'])
    result=dict(status='R_COMPLETE_VALID_MEAN_GAIN_BUT_LOCKED_GATE_FAIL_B_PENDING',archive_sha256=cached['archive_sha256'],
        arm='R',source_protocol_history_metric_audit='PASS',metrics=dict(original=summary['metrics']['origin'],A=baseline,R=new),
        R_minus_A=ra,R_minus_original=ro,gates=gates,numeric_gate_pass=all(gates.values()),
        hallucination_caption_transitions=dict(A_only_fixed=len(old-now),R_only_new=len(now-old),both=len(old&now)),
        actual_image_exposures=10000,annotation_span_exposures=17904,person_span_exposures=4476,
        clipping=dict(R_fraction_preclip_norm_above1=float(np.mean([s['grad_norm']>1 for s in histories[0]])),
            A_fraction_preclip_norm_above1=float(np.mean([s['grad_norm']>1 for s in a['history_rank0.json']]))),
        checkpoint_sha256_reported=summary['checkpoint_sha256'],checkpoint_bytes_independently_verified=False,
        next='Obtain B results; no numeric promotion or extra blind annotation yet. R-vs-B needed for spatial attribution.',
        limits=['No local model execution; checked receipts and arithmetic, not an independent GPU rerun.',
            'Historical A sampler/seed/source match; actual A order was not byte-logged.',
            'Development reused; one seed; attention path regularization is weak noun/class supervision, not causal grounding or relation truth.'])
    (HERE/'R_RESULT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','R_minus_A','R_minus_original','gates','hallucination_caption_transitions']},indent=2))


if __name__=='__main__':main()
