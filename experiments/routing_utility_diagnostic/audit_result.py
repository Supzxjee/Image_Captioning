"""Recompute saved A/T/W summaries with NumPy; no local ML/scorer inference."""
import hashlib
import json
from pathlib import Path
import numpy as np
import sys

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/attention_geometry_diagnostic'))
from diagnostic_data import BASE_SHA,paired_ci,save
from utility_data import POLICY,teacher_contrasts,summarize_teacher


def compare(actual,expected):
    if isinstance(actual,dict):
        assert actual.keys()==expected.keys()
        for k in actual:compare(actual[k],expected[k])
    elif isinstance(actual,list):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):compare(a,b)
    elif isinstance(actual,float):np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
    else:assert actual==expected


def main():
    loaded=json.loads((HERE/'review/loaded.json').read_text(encoding='utf-8'))
    receipt=json.loads(loaded['receipt.json']);cfg=json.loads(loaded['protocol.json']);reported=json.loads(loaded['result.json'])
    assert receipt['status']=='ATW_COMPLETE' and receipt['images']==138 and receipt['policy']==POLICY==cfg['policy']
    assert not receipt['trained'] and not receipt['test_used'] and not reported['trained']
    for n,sha in receipt['payload_sha256'].items():assert hashlib.sha256(loaded[n].encode()).hexdigest()==sha,n
    for n,sha in cfg['implementation_sha256'].items():assert hashlib.sha256(loaded['source/'+n].encode()).hexdigest()==sha,n
    original=[];done=[]
    for rank in (0,1):
        d=json.loads(loaded[f'complete_rank{rank}.json']);part=json.loads(loaded[f'rows_rank{rank}.json'])
        assert d['status']=='FROZEN_ATW_COMPLETE' and d['images']==len(part)==69
        assert d['base_sha256']==BASE_SHA and not d['optimizer_used']
        assert d['parity']==dict(memory_max=0.,logit_max=0.,same_shape_images=1)
        assert d['rows_sha256']==hashlib.sha256(loaded[f'rows_rank{rank}.json'].encode()).hexdigest()
        original+=part;done.append(d)
    rows=json.loads(loaded['scored_rows.json']);original.sort(key=lambda r:r['image_id'])
    assert len(rows)==len({r['coco_id'] for r in rows})==138
    assert [r['image_id'] for r in rows]==sorted(r['image_id'] for r in cfg['rows'])
    for row,raw in zip(rows,original):
        assert {k:v for k,v in row.items() if k!='CIDEr'}==raw
        compare(teacher_contrasts(row['scores']),row['contrasts'])
        if row['intervened']:
            m=row['manipulation'];assert m['raw_mass_T']>m['raw_mass_A']
            assert 0<m['lambda_T']<=.1 and 0<m['lambda_W']<=.1
            np.testing.assert_allclose(m['pre_norms']['T'],m['pre_norms']['W'],rtol=1e-4,atol=1e-6)
            np.testing.assert_allclose(list(m['pre_norms'].values()),m['matched_norm'],rtol=1e-4,atol=1e-6)
        else:
            assert row['nonintervention_reason']=='q_indistinguishable'
            assert len(set(row['captions'].values()))==1
        for a,b in [('A','T'),('A','W'),('T','W')]:
            if row['captions'][a]==row['captions'][b]:assert row['CIDEr'][a]==row['CIDEr'][b]
    teacher=summarize_teacher(rows);compare(teacher,reported['teacher'])
    contrasts={c:paired_ci([r['CIDEr']['T']-r['CIDEr'][c] for r in rows]) for c in ('A','W')}
    compare(contrasts,reported['T_minus'])
    metrics={}
    for c in ('A','T','W'):
        chair=json.loads(loaded[f'{c}_chair.json']);sentences=chair['sentences']
        assert len(sentences)==138 and {s['image_id'] for s in sentences}=={r['coco_id'] for r in rows}
        predictions=json.loads(loaded[f'{c}_predictions.json'])
        assert predictions==[dict(image_id=r['coco_id'],caption=r['captions'][c]) for r in rows]
        counts=dict(images=138,hallucinated_captions=sum(bool(s['mscoco_hallucinated_words']) for s in sentences),
            hallucinated_mentions=sum(len(s['mscoco_hallucinated_words']) for s in sentences),object_mentions=sum(len(s['mscoco_generated_words']) for s in sentences),
            correct_objects=sum(len(set(s['mscoco_generated_words'])&set(s['mscoco_gt_words'])) for s in sentences),gt_objects=sum(len(set(s['mscoco_gt_words'])) for s in sentences))
        cm=dict(CHAIRs=counts['hallucinated_captions']/138,CHAIRi=counts['hallucinated_mentions']/counts['object_mentions'],Recall=counts['correct_objects']/counts['gt_objects'])
        metrics[c]=dict(CIDEr=float(np.mean([r['CIDEr'][c] for r in rows])),chair=cm,counts=counts,
            mean_caption_words=float(np.mean([len(r['captions'][c].split()) for r in rows])))
        compare(metrics[c],reported['metrics'][c])
        compare(cm,{k:chair['overall_metrics'][k] for k in cm})
    changes=[{k:r[k] for k in ('image_id','coco_id','receiver','donor','captions','CIDEr','contrasts')} for r in rows if len(set(r['captions'].values()))>1]
    assert len(changes)==3==reported['changed_T_vs_A']
    active=[r for r in rows if r['intervened']];assert len(active)==137
    increases=[r['manipulation']['raw_mass_T']-r['manipulation']['raw_mass_A'] for r in active]
    assert reported['decision']==receipt['decision']=='NOT_YET_GO' and not teacher['positive_teacher_signal'] and not reported['no_metric_tradeoff_guard']
    result=dict(audit_status='SAVED_ARTIFACT_AUDIT_PASS',zip_sha256='34c3070226cb88c82be4a4ecbd5332ed7a4b05cb5ec83f208bec2139468ff7f0',
        summary=reported,raw_receiver_box_mass_increase=dict(mean=float(np.mean(increases)),min=float(min(increases)),max=float(max(increases))),
        caption_changes=changes,worker_seconds=[d['seconds'] for d in done],
        evidence=['All receipt payload and embedded source hashes verified.',
            'Worker coverage/frozen tensor identity/zero parity supported by source-checked receipts.',
            'Saved teacher contrasts, primary common cohort, paired bootstrap and CHAIR counts independently reaggregated.',
            'Positive raw mass increase, <=10% convex weights and matched pre-fusion norms checked for all137 interventions.'],
        limits=['No local model execution; actual before/after memory tensors not in ZIP.',
            'CIDEr values replayed/reaggregated, not rescored with Java/PTB locally.',
            'CHAIR word lists reaggregated, not rerun from source captions locally.',
            'Frozen10%GT-informed Q-mass intervention on restricted reused TRAIN; not a test of all trainable alignment methods.'])
    save(HERE/'RESULT.json',result)
    print(json.dumps(dict(audit_status=result['audit_status'],decision=reported['decision'],teacher=teacher,
        metrics=metrics,T_minus=contrasts,raw_mass_increase=result['raw_receiver_box_mass_increase'],changed=len(changes)),indent=2))


if __name__=='__main__':main()
