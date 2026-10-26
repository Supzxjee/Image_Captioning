"""Audit cached review texts and paired metrics; no ML execution or ZIP rereading."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    archives={a:json.loads((HERE/'review'/f'{a}_loaded.json').read_text(encoding='utf-8')) for a in ('C','T')}
    texts={a:d['texts'] for a,d in archives.items()}
    cfg={a:json.loads(t['protocol.json']) for a,t in texts.items()}
    summaries={a:json.loads(t['summary.json']) for a,t in texts.items()}
    allowed={'arm','output','stage','staging_seconds'}
    differences={k for k in cfg['C'] if cfg['C'][k]!=cfg['T'].get(k)}
    assert differences<=allowed,differences
    workers={}
    for arm in ('C','T'):
        t=texts[arm];c=cfg[arm]
        for name,expected in c['implementation_sha256'].items():assert hashlib.sha256(t['source/'+name].encode()).hexdigest()==expected
        workers[arm]=[json.loads(t[f'worker_rank{r}.json']) for r in (0,1)]
        for w in workers[arm]:
            assert w['status']=='COMPLETE' and w['updates']==1250 and w['aux_images_consumed']==1000 and w['anchor_unchanged']
        for field in ('initial_sha256','final_sha256','frozen_sha256'):
            assert workers[arm][0][field]==workers[arm][1][field]
        logs=[json.loads(r) for r in t['training.jsonl'].splitlines() if r.strip()]
        assert [r['step'] for r in logs]==list(range(25,1251,25))
        assert np.isfinite([[r['loss'],r['grad_norm']] for r in logs]).all()
    for field in ('initial_sha256','frozen_sha256'):assert workers['C'][0][field]==workers['T'][0][field]
    for name in ('A_predictions.json','A_per_image_cider.json','A_chair_output.json'):
        assert json.loads(texts['C'][name])==json.loads(texts['T'][name]),name
    predictions={};scores={};metrics={};sentences={}
    for name,source,condition in [('A','C','A'),('C','C','end'),('T','T','end')]:
        t=texts[source];rows=json.loads(t[condition+'_predictions.json'])
        predictions[name]={r['image_id']:r for r in rows};assert len(rows)==len(predictions[name])==1000
        expected={r['image_id']:r['coco_id'] for r in cfg[source]['groups']['val']}
        assert {r['image_id']:r['coco_id'] for r in rows}==expected
        per_image=json.loads(t[condition+'_per_image_cider.json']);scores[name]={r['image_id']:r['CIDEr'] for r in per_image}
        assert len(per_image)==len(scores[name])==1000 and set(scores[name])==set(expected)
        chair=json.loads(t[condition+'_chair_output.json']);ss=chair['sentences'];sentences[name]={r['image_id']:r for r in ss}
        assert len(ss)==len(sentences[name])==1000 and set(sentences[name])==set(expected.values())
        counts=dict(images=1000,hallucinated_captions=sum(bool(s['mscoco_hallucinated_words']) for s in ss),
            hallucinated_mentions=sum(len(s['mscoco_hallucinated_words']) for s in ss),object_mentions=sum(len(s['mscoco_generated_words']) for s in ss),
            recalled_unique_objects=sum(len(set(s['mscoco_generated_words'])&set(s['mscoco_gt_words'])) for s in ss),reference_unique_objects=sum(len(set(s['mscoco_gt_words'])) for s in ss))
        values=dict(CHAIRs=counts['hallucinated_captions']/1000,CHAIRi=counts['hallucinated_mentions']/counts['object_mentions'],Recall=counts['recalled_unique_objects']/counts['reference_unique_objects'])
        original=summaries[source]['metrics'][condition]
        assert counts==original['counts']
        assert all(abs(values[k]-original['chair'][k])<1e-12 for k in values)
        assert abs(np.mean(list(scores[name].values()))-original['CIDEr'])<1e-12
        metrics[name]=original
    ids=sorted(predictions['A']);rng=np.random.default_rng(42);draw=rng.integers(0,1000,(10000,1000));paired={}
    for a,b in [('C','A'),('T','A'),('T','C')]:
        delta=np.array([scores[a][i]-scores[b][i] for i in ids])
        paired[a+'-'+b]=dict(mean=float(delta.mean()),ci95=np.quantile(delta[draw].mean(1),[.025,.975]).tolist(),
            positive_images=int((delta>0).sum()),negative_images=int((delta<0).sum()),
            changed_captions=sum(predictions[a][i]['caption']!=predictions[b][i]['caption'] for i in ids))
    a,t=metrics['A'],metrics['T']
    gates=dict(T_minus_A_at_least_point01=paired['T-A']['mean']>=.01,
        T_minus_A_CI_positive=paired['T-A']['ci95'][0]>0,T_minus_C_CI_positive=paired['T-C']['ci95'][0]>0,
        CHAIRs_not_worse=t['chair']['CHAIRs']<=a['chair']['CHAIRs'],CHAIRi_not_worse=t['chair']['CHAIRi']<=a['chair']['CHAIRi'],
        recall_drop_at_most_point005=t['chair']['Recall']>=a['chair']['Recall']-.005)
    coco_ids=sorted(sentences['A']);transitions={}
    for first,second in [('A','T'),('C','T')]:
        before=np.array([bool(sentences[first][i]['mscoco_hallucinated_words']) for i in coco_ids]);after=np.array([bool(sentences[second][i]['mscoco_hallucinated_words']) for i in coco_ids])
        transitions[first+'->'+second]=dict(no_longer_flagged=int((before&~after).sum()),newly_flagged=int((~before&after).sum()),
            CHAIRs_delta_ci95=np.quantile((after.astype(float)-before)[draw].mean(1),[.025,.975]).tolist())
    result=dict(status='MATCHED_AUDIT_PASS_RESEARCH_GATE_FAIL' if not all(gates.values()) else 'GATE_PASS_WAIT_REVIEW',
        metrics=metrics,paired_CIDEr=paired,gates=gates,CHAIR_flag_transitions=transitions,
        bootstrap=dict(unit='paired image',resamples=10000,seed=42,scope='Single-seed reused development1000; descriptive conditional uncertainty, not final held-out test.'),
        audit=dict(A_predictions_scores_CHAIR_identical=True,protocol_only_differences=sorted(differences),sources_verified_per_arm=len(cfg['C']['implementation_sha256']),
            stage_SHA_equal=cfg['C']['stage_sha256']==cfg['T']['stage_sha256'],anchor_frozen_state_checks='Both worker receipts and source/log checks passed; source cache/checkpoint bytes not in ZIP.'),
        zip_sha256={a:d['zip_sha256'] for a,d in archives.items()},checkpoint_sha256={a:s['checkpoint_sha256'] for a,s in summaries.items()},
        timings={a:dict(staging_seconds=summaries[a]['staging_seconds'],max_worker_train_end_seconds=max(w['seconds'] for w in workers[a])) for a in workers},
        limits=['No checkpoint/HDF5 bytes in review ZIP; cannot directly verify them locally.','No post-training four-condition score probe; mechanism change unmeasured.',
            'Both C and T below A; T-C CI crosses0, so incremental stale-loss damage is not established.','CHAIR flag improvement alone does not establish semantic alignment.'],
        recommendation='Retain A; do not scale/sweep/add seeds to failed recipe. Await Astra next decision; do not reject all visual-semantic alignment approaches.',
        local_model_execution=False,test_used=False)
    (HERE/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=result['status'],paired=paired,gates=gates,transitions=transitions),indent=2))


if __name__=='__main__':main()
