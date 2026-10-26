"""Locked, model-free A/T/W policy and metric aggregation."""
import hashlib
import numpy as np
from diagnostic_data import paired_ci

POLICY=dict(version='routing-utility-ATW-v1',images=138,rho=.10,epsilon=1e-8,
    pair='hash-first unique unordered separated-box pair; hash-fixed receiver direction',
    q='argmax RAW Q-to-patch box mass; lowest-index tie; no alternate-pair rescue',
    intervention='attended_prompt before existing gate/LN; exact learned value/output projections',
    control='same Frobenius displacement before fusion, convex coefficients <=.10',
    primary='benefit and selectivity on same valid-intervention images with both first-mention reference nouns',
    minimum_primary_images=50,maximum_nonintervened_fraction=.5,
    coverage_guard='50 primary images and <=50% nonintervened are budget guards, not power calculations',
    decode='raw greedy, original vocab, max30 incl BOS, EOS stopping, no rerank',
    cider='official PTB/CIDEr with fixed locked TRAIN10k reference IDF; score full138 and secondary intervention subset',
    gate='benefit and selectivity paired95 lower>0; whole138 T-A and T-W CIDEr means>0; no hallucination/recall/length tradeoff',
    scope='GT-informed frozen-model oracle intervention on reused TRAIN; no inference-ready method/training/novelty claim')


def key(value):return hashlib.sha256(('routing-utility-v1:'+str(value)).encode()).hexdigest()


def choose_pair(row):
    pairs=sorted({tuple(sorted((p['i'],p['j']))) for p in row['pairs']})
    i,j=min(pairs,key=lambda p:key(f"{row['coco_id']}:{p[0]}:{p[1]}"))
    return (j,i) if int(key(row['coco_id'])[-1],16)%2 else (i,j)


def q_choice(q,receiver,donor):
    q=np.asarray(q,dtype=float)
    if q.shape!=(32,197) or not np.isfinite(q).all():raise ValueError('Expected raw32x197 Q attention.')
    masses=np.stack([q[:,1:]@np.asarray(p['mask']) for p in (receiver,donor)])
    plus,minus=(int(x) for x in masses.argmax(-1))
    distinguish=plus!=minus and masses[0,plus]-masses[0,minus]>POLICY['epsilon'] and masses[1,minus]-masses[1,plus]>POLICY['epsilon']
    return dict(q_plus=plus,q_minus=minus,distinguishable=bool(distinguish),box_mass=masses[:,[plus,minus]].tolist())


def coefficients(positive_norm,negative_norm,rho=POLICY['rho']):
    if not 0<=rho<=POLICY['rho']:raise ValueError('No perturbation sweep/increase allowed.')
    if not np.isfinite([positive_norm,negative_norm]).all():raise ValueError('Nonfinite displacement.')
    if min(positive_norm,negative_norm)<=POLICY['epsilon']:return None
    h=rho*min(positive_norm,negative_norm)
    return min(rho,h/positive_norm),min(rho,h/negative_norm),h


def teacher_contrasts(scores):
    if any(scores[c][n] is None for c in ('A','T','W') for n in ('receiver','donor')):return None
    benefit=scores['T']['receiver']-scores['A']['receiver']
    contrast=scores['T']['receiver']-scores['W']['receiver']
    selectivity=contrast-(scores['T']['donor']-scores['W']['donor'])
    return dict(benefit=benefit,contrast=contrast,selectivity=selectivity)


def summarize_teacher(rows):
    primary=[r for r in rows if r['intervened'] and r['contrasts'] is not None]
    metrics={}
    for name in ('benefit','contrast','selectivity'):
        v=[r['contrasts'][name] for r in primary]
        metrics[name]=dict(paired_ci(v),median=float(np.median(v)) if v else None,positive=sum(x>0 for x in v))
    nonintervened=sum(not r['intervened'] for r in rows)
    adequate=len(primary)>=POLICY['minimum_primary_images'] and nonintervened/len(rows)<=POLICY['maximum_nonintervened_fraction']
    positive=adequate and all(metrics[m]['ci95'] is not None and metrics[m]['ci95'][0]>0 for m in ('benefit','selectivity'))
    return dict(primary_images=len(primary),nonintervened_images=nonintervened,coverage_guard_pass=adequate,
        positive_teacher_signal=positive,metrics=metrics,
        note='Identical primary image set for benefit/selectivity; degenerate images remain unchanged in full138 captions.')
