"""Compare saved pilot artifacts; no archived code execution or model replay."""
import argparse
import json
from pathlib import Path
import zipfile
import numpy as np


def compare(a_path, b_path, output):
    with zipfile.ZipFile(a_path) as a, zipfile.ZipFile(b_path) as b:
        read = lambda z, n: json.loads(z.read(n))
        ac, bc = read(a, 'protocol.json'), read(b, 'protocol.json')
        excluded = {'arm', 'output', 'stage'}
        assert {k:v for k,v in ac.items() if k not in excluded} == {k:v for k,v in bc.items() if k not in excluded}
        for name in ('origin_predictions.json','origin_per_image_cider.json','val_gt.json','train_object_targets.json'):
            assert read(a,name) == read(b,name)
        for rank in (0,1):
            assert read(a,f'runtime_rank{rank}.json') == read(b,f'runtime_rank{rank}.json')
            assert read(a,f'worker_complete_rank{rank}.json')['initial_base_sha256'] == read(b,f'worker_complete_rank{rank}.json')['initial_base_sha256']
        by_id = lambda rows: {r['image_id']:r for r in rows}
        expected = by_id(bc['groups']['val'])
        on = by_id(read(b,'on_predictions.json'))
        diag = by_id(sum((read(b,f'on_diagnostic_rank{rank}.json') for rank in (0,1)),[]))
        summary = read(b,'run_summary.json')
        for probe in ('off','text_swap','region_swap'):
            rows = read(b,probe+'_predictions.json')
            shards = sum((read(b,f'{probe}_rank{rank}.json') for rank in (0,1)),[])
            assert sorted(shards,key=lambda r:r['image_id']) == rows
            pred = by_id(rows)
            diagnostics = sum((read(b,f'{probe}_diagnostic_rank{rank}.json') for rank in (0,1)),[])
            d = by_id(diagnostics)
            assert len(pred)==len(d)==len(diagnostics)==1000 and pred.keys()==d.keys()==expected.keys()
            for mode, stored in summary['sensitivity'][probe].items():
                ids = [i for i,r in expected.items() if mode=='all' or r['mode']==mode]
                delta = [d[i]['reference_nll_per_token']-diag[i]['reference_nll_per_token'] for i in ids]
                assert np.isfinite(delta).all()
                assert stored['images']==len(ids)
                assert stored['changed_captions']==sum(pred[i]['caption']!=on[i]['caption'] for i in ids)
                assert abs(stored['mean_reference_nll_delta']-np.mean(delta))<1e-12
            if probe=='text_swap':
                assert pred==on and all(d[i]['reference_nll_per_token']==diag[i]['reference_nll_per_token'] for i in expected)
        def pair(left, right):
            assert [r['image_id'] for r in left]==[r['image_id'] for r in right]
            delta=np.array([r['CIDEr']-l['CIDEr'] for l,r in zip(left,right)])
            rng=np.random.default_rng(20261019)
            ci=np.percentile([delta[rng.integers(0,1000,1000)].mean() for _ in range(5000)],[2.5,97.5]).tolist()
            return {'CIDEr_delta':float(delta.mean()),'paired_fixed_IDF_bootstrap95':ci}
        result={'status':'MATCHED_ARTIFACT_COMPARISON_PASS',
                'B_minus_A':pair(read(a,'on_per_image_cider.json'),read(b,'on_per_image_cider.json')),
                'B_on_minus_off':pair(read(b,'off_per_image_cider.json'),read(b,'on_per_image_cider.json')),
                'sensitivity':summary['sensitivity'],
                'limits':['Saved scores only, no independent metric retokenization or checkpoint replay.',
                          'Single seed and reused val1000; bootstrap does not measure training-seed uncertainty.',
                          'Object-relation subgroup CIDEr is caption similarity, not predicate or instance correctness.',
                          'Swapping is sensitivity only; unchanged captions do not prove zero visual use.']}
    Path(output).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='sensitivity'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--a',required=True);p.add_argument('--b',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();compare(args.a,args.b,args.output)
