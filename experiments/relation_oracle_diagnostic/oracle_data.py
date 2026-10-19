"""Fixed-box text interventions and deterministic failure-case selection; no model imports."""
import numpy as np
from probe_data import identity


def select_cases(partitions,size=20):
    eligible=[]
    for partition,rows in partitions.items():
        for row in rows:
            if row['task']!='wrong_predicate':continue
            a,b=row['candidates'];p=row['positive_index']
            assert a['bounds']==b['bounds'] and a['text']!=b['text']
            if row['scores']['boxed'][p]-row['scores']['boxed'][1-p]<=1e-7:
                eligible.append(dict(row,partition=partition))
    chosen=[];seen=set()
    for row in sorted(eligible,key=lambda r:identity(r['case_id'])):
        if row['coco_id'] not in seen:chosen.append(row);seen.add(row['coco_id'])
    if len(chosen)<size:raise ValueError('Not enough unique failure images')
    return sorted(chosen[:size],key=lambda r:r['case_id'])


def oracle_slots(row,positive):
    """One S/O/R triple. Annotation +/- differ ONLY in R text; GT boxes identical."""
    p=row['positive_index'] if positive else 1-row['positive_index']
    candidate=row['candidates'][p];s,o,u=candidate['bounds']
    boxes=np.zeros((9,3,4),dtype=np.float32)
    boxes[:3]=[[s,o,s],[s,o,o],[s,o,u]]
    texts=['person',row['object_label'],candidate['text']]+['']*6
    roles=np.ones(9,dtype=np.int64);roles[:3]=[0,1,2]
    valid=np.zeros(9,dtype=bool);valid[:3]=True
    return dict(texts=texts,bounds=boxes,roles=roles,valid=valid)


def summarize(records):
    assert len(records)==20 and len({r['case_id'] for r in records})==20
    conditions=('off','upt','annotation_positive','annotation_negative')
    result={}
    for condition in conditions:
        values=[r['conditions'][condition] for r in records]
        result[condition]=dict(mean_positive_minus_negative_logp=float(np.mean([v['positive_minus_negative_logp'] for v in values])),
            mean_positive_minus_negative_mean_logp=float(np.mean([v['positive_minus_negative_mean_logp'] for v in values])),
            annotation_positive_preferred=sum(v['positive_minus_negative_logp']>0 for v in values),
            captions_changed_vs_off=sum(r['conditions'][condition]['caption']!=r['conditions']['off']['caption'] for r in records))
    result['controlled_annotation_positive_vs_negative']=dict(
        mean_preference_shift=float(np.mean([r['conditions']['annotation_positive']['positive_minus_negative_logp']-r['conditions']['annotation_negative']['positive_minus_negative_logp'] for r in records])),
        captions_changed=sum(r['conditions']['annotation_positive']['caption']!=r['conditions']['annotation_negative']['caption'] for r in records),
        mean_R_attention_L1=float(np.mean([r['evidence_change']['R_attention_L1'] for r in records])),
        mean_R_token_relative_L2=float(np.mean([r['evidence_change']['R_token_relative_L2'] for r in records])))
    return result
