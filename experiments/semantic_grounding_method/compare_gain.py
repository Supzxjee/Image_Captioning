"""Matched end-only C/H/T comparison, operating on saved small reviews."""
import argparse
import json
from pathlib import Path
import zipfile
import numpy as np


def load(path):
    path=Path(path)
    if path.is_dir():
        read=lambda n:json.loads((path/n).read_text(encoding='utf-8'))
        return read
    z=zipfile.ZipFile(path)
    return lambda n:json.loads(z.read(n))


def compare(paths,out):
    readers={k:load(v) for k,v in paths.items()};summaries={k:r('summary.json') for k,r in readers.items()}
    for k,s in summaries.items():
        if s['status']!='COMPLETE_WAIT_MATCHED_CHT' or s['arm']!=k or s['updates']!=500 or s['test_used']:raise ValueError('Incomplete/wrong arm')
    for field in ('preparation_receipt_sha256','sources','policy'):
        if any(s[field]!=summaries['C'][field] for s in summaries.values()):raise ValueError('Unmatched '+field)
    baseline=readers['C']('A_predictions.json')
    if any(r('A_predictions.json')!=baseline for r in readers.values()):raise ValueError('A predictions differ')
    start=readers['C']('worker_rank0.json')['initial_head_sha256']
    for r in readers.values():
        if any(r(f'worker_rank{i}.json')['initial_head_sha256']!=start for i in (0,1)):raise ValueError('Initialization differs')
    values={k:{r['image_id']:r['CIDEr'] for r in rd('end_per_image_cider.json')} for k,rd in readers.items()}
    values['A']={r['image_id']:r['CIDEr'] for r in readers['C']('A_per_image_cider.json')}
    ids=sorted(values['A'])
    if len(ids)!=1000 or any(set(v)!=set(ids) for v in values.values()):raise ValueError('Incomplete per-image metrics')
    rng=np.random.default_rng(42);boots={k:[] for k in ('A','C','H')}
    differences={k:np.array([values['T'][i]-values[k][i] for i in ids]) for k in boots}
    for _ in range(20):
        indices=rng.integers(0,1000,size=(500,1000))
        for k in boots:boots[k].extend(differences[k][indices].mean(1).tolist())
    result={f'T-{k}':dict(delta=float(differences[k].mean()),ci95=np.quantile(boots[k],[.025,.975]).tolist()) for k in boots}
    t=summaries['T']['metrics']['end'];a=summaries['T']['metrics']['A']
    guards=dict(cider_delta=result['T-A']['delta']>=.01,confidence=all(v['ci95'][0]>0 for v in result.values()),
        chair_s=t['chair']['CHAIRs']<=a['chair']['CHAIRs'],chair_i=t['chair']['CHAIRi']<=a['chair']['CHAIRi'],recall=t['chair']['Recall']>=a['chair']['Recall']-.005)
    artifact=dict(status='EXPLORATORY_CONTINUATION_GATE_PASS' if all(guards.values()) else 'COMPLETED_QUALITY_NO_GO',
        comparisons=result,guards=guards,bootstrap=dict(resamples=10000,seed=42,unit='paired image',ci='percentile95'),
        limitations=['One seed,reused dev;not final test or corrected significance.','No held-out phrase grounding;no novelty/factuality claim.'])
    Path(out).write_text(json.dumps(artifact,indent=2)+'\n',encoding='utf-8');print(json.dumps(artifact,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('C','H','T'):p.add_argument('--'+k,required=True)
    p.add_argument('--out',default='grounded_gain_comparison.json');a=p.parse_args()
    compare({k:getattr(a,k) for k in ('C','H','T')},a.out)
