"""CPU-only paired fixed-stratum bootstrap. No model loads or checkpoint selection."""
import argparse
import json
from pathlib import Path
import numpy as np
from frame_data import FILES, save


def compare(pos,contrast):
    for result,arm in ((pos,'RelPos'),(contrast,'RelContrast')):
        if result['status']!='ENDPOINT_200_COMPLETE' or result['updates']!=200 or result['arm']!=arm:raise ValueError('Incomplete/wrong arm')
    if pos['initial_model_digest']!=contrast['initial_model_digest']:raise ValueError('Initializations differ')
    for key in ('seed','updates','device_gpu_hours_cap','source_commit','metadata_sha256','cache_sha256','checkpoint_sha256'):
        if pos['protocol'][key]!=contrast['protocol'][key]:raise ValueError('Protocol differs: '+key)
    if pos['protocol']['cache_sha256']!=FILES:raise ValueError('Wrong v105 dataset')
    a={r['pair_id']:r for r in pos['rows']};b={r['pair_id']:r for r in contrast['rows']}
    if set(a)!=set(b) or len(a)!=64:raise ValueError('Paired heldout IDs differ')
    keys=sorted(a);x=np.array([float(a[k]['likelihood_gap']>0) for k in keys]);y=np.array([float(b[k]['likelihood_gap']>0) for k in keys])
    strata={s:[i for i,k in enumerate(keys) if a[k]['stratum']==s] for s in ('projective_containment','contact_proximity')}
    if any(len(v)!=32 for v in strata.values()) or any(a[k]['stratum']!=b[k]['stratum'] or a[k]['relation']!=b[k]['relation'] for k in keys):raise ValueError('Strata/relations differ')
    rng=np.random.default_rng(20261028)
    draws=np.concatenate([rng.choice(v,(10000,32),replace=True) for v in strata.values()],axis=1)
    delta=(y-x)[draws].mean(1);ci=np.quantile(delta,[.025,.975]).tolist()
    passed=bool(y.mean()>=.70 and y.mean()-x.mean()>=.10-1e-12 and ci[0]>0)
    return dict(status='PRETRAIN_GATE_PASS' if passed else 'PRETRAIN_GATE_NO_GO',RelPos_accuracy=float(x.mean()),
        RelContrast_accuracy=float(y.mean()),difference=float(y.mean()-x.mean()),paired_difference_ci95=ci,
        paired_bootstrap_draws=10000,seed=20261028,stratified=True,n=64,
        strata={s:dict(n=len(v),RelPos=float(x[v].mean()),RelContrast=float(y[v].mean())) for s,v in strata.items()},
        limits='Exploratory reused adaptation-heldout images, one seed, assisted sample audit; no caption/novelty claim.',
        next_step='Caption controls Tune/RelPos/RelContrast only if PASS; no epoch/sweep rescue in this pilot.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pos',required=True);p.add_argument('--contrast',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();result=compare(json.loads(Path(args.pos).read_text()),json.loads(Path(args.contrast).read_text()))
    save(args.output,result);print(json.dumps(result,indent=2))
