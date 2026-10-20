"""Read saved artifacts only. Refuse confounded CE/CROSS comparisons; no model loading."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from phrase_data import identity
from run_phrase import metrics,BASE_SHA
from make_notebooks import HERE,BASE,sources


def read_review(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        read=lambda n:json.loads(z.read(n))
        cfg,m,summary=read('protocol.json'),read('locked_manifest.json'),read('summary.json')
        assert m==json.loads((HERE/'locked_manifest.json').read_text(encoding='utf-8')),'Unexpected groups/schedule'
        assert cfg['project_commit']==BASE
        assert hashlib.sha256(z.read('locked_manifest.json')).hexdigest()==cfg['manifest_sha256']
        assert identity({k:v for k,v in cfg.items() if k!='protocol_sha256'})==cfg['protocol_sha256']
        assert cfg['checkpoint_sha256']==BASE_SHA and summary['arm']==cfg['arm']
        expected_sources=sources()
        assert set(cfg['source_sha256'])==set(expected_sources)
        for name,source in expected_sources.items():
            raw=z.read('source/'+name)
            assert raw==source.encode() and hashlib.sha256(raw).hexdigest()==cfg['source_sha256'][name],name
        completed=[read(f'complete_rank{r}.json') for r in (0,1)]
        for rank,d in enumerate(completed):
            assert d['rank']==rank and d['updates']==300 and d['original_parameters_unchanged']
            assert d['protocol_sha256']==cfg['protocol_sha256']
        for k in ('initial_sha256','final_sha256','frozen_sha256'):assert completed[0][k]==completed[1][k]
        warm=[read(f'ce_warmup_rank{r}.json') for r in (0,1)];assert warm[0]==warm[1] and warm[0]['update']==60
        phases={}
        for phase in ('before','after'):
            rows=sorted(read(f'{phase}_rank0.json')+read(f'{phase}_rank1.json'),key=lambda x:x['group_id'])
            assert rows==read(f'{phase}_records.json')
            assert len(rows)==21 and len({r['group_id'] for r in rows})==21
            by={g['group_id']:g for g in m['groups']}
            for r in rows:
                for c in r['conditions'].values():
                    matrix=np.asarray(c['scores']);assert matrix.shape==(2,2) and np.isfinite(matrix).all()
                    v=c['positive_margins'];assert len(v)==2 and np.isfinite(v).all()
                    targets=by[r['group_id']]['positive_indices']
                    np.testing.assert_allclose(v,[matrix[i,j]-matrix[i,1-j] for i,j in enumerate(targets)],rtol=1e-6,atol=1e-6)
                    assert c['agreement']==sum(.5 if abs(x)<=1e-7 else float(x>0) for x in v)/2
                    assert c['both_directions_correct']==all(x>1e-7 for x in v)
            assert summary['metrics'][phase]=={split:metrics(rows,keys) for split,keys in m['partitions'].items()}
            phases[phase]=rows
        assert not summary['test_used'] and not summary['semantic_human_evaluation_available']
        return dict(cfg=cfg,manifest=m,summary=summary,phases=phases,warmup=warm[0],completed=completed)


def compare(a,b):
    assert a['cfg']['arm']=='CE' and b['cfg']['arm']=='CROSS'
    for k in ('checkpoint_sha256','dataset_sha256','vocabulary_sha256','stage_tensor_sha256','source_sha256','project_commit','manifest_sha256','optimizer','base_sha256'):
        assert a['cfg'][k]==b['cfg'][k],f'Confounded input/source: {k}'
    assert a['manifest']==b['manifest'] and a['warmup']==b['warmup'],'Initial CE60 steps differ; investigate before comparison'
    assert a['phases']['before']==b['phases']['before'],'Starting predictions differ'
    for rank in (0,1):assert a['completed'][rank]['runtime']==b['completed'][rank]['runtime'],'Different runtime'
    keys=a['manifest']['partitions']['adaptation_holdout'];left={r['group_id']:r for r in a['phases']['after']};right={r['group_id']:r for r in b['phases']['after']}
    delta=np.asarray([right[k]['conditions']['on']['agreement']-left[k]['conditions']['on']['agreement'] for k in keys])
    rng=np.random.default_rng(42);ci=np.percentile([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(5000)],[2.5,97.5]).tolist()
    return dict(status='MATCHED_ARTIFACT_COMPARISON_PASS',holdout_groups=len(keys),
        holdout_CROSS_minus_CE_agreement=float(delta.mean()),descriptive_paired_group_bootstrap95=ci,
        CE=a['summary'],CROSS=b['summary'],semantic_human_verdicts_available=False,
        interpretation='Only scoring and sensitivity statistics. Read generated phrases and inspect grounding/binding before any go decision.',
        limits=['7 previously reviewed adaptation groups, not an independent test; no seed variation or exhaustive near-duplicate audit',
                'Paired group bootstrap is descriptive and does not establish statistical power or full-caption benefit'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ce',required=True);p.add_argument('--cross',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=compare(read_review(a.ce),read_review(a.cross))
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8');print(result['status'])
