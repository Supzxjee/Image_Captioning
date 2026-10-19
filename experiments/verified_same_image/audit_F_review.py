"""Audit saved F artifacts, reuse caption audit, never execute ZIP code or load tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
from make_notebook import HERE,ROOT,sources
sys.path[:0]=[str(ROOT/'experiments/semantic_evidence_pilot'),str(ROOT/'experiments/semantic_evidence_probe')]
from audit_review import audit as caption_audit
from audit_training_diagnostics import audit as training_audit
from probe_data import identity,summarize
from verified_data import training_schedule,geometry_summary


def pair(left,right):
    assert [r['image_id'] for r in left]==[r['image_id'] for r in right]
    delta=np.array([r['CIDEr']-l['CIDEr'] for l,r in zip(left,right)])
    rng=np.random.default_rng(20261019)
    ci=np.percentile([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(5000)],[2.5,97.5]).tolist()
    return dict(CIDEr_delta=float(delta.mean()),paired_saved_fixed_IDF_bootstrap95=ci)


def audit(source,a_path,e_path,dataset,output):
    with zipfile.ZipFile(source) as z:
        read=lambda n:json.loads(z.read(n))
        cfg=read('protocol.json');m=read('verified_manifest.json');summary=read('run_summary.json')
        assert m==json.loads((HERE/'verified_manifest.json').read_text(encoding='utf-8'))
        assert hashlib.sha256(z.read('verified_manifest.json')).hexdigest()==cfg['verified_manifest_sha256']
        assert cfg['arm']=='F' and cfg['alignment_weight']==.1 and cfg['alignment_margin']==.2
        assert m['schedule']==training_schedule(m['splits']['fit']) and all(0<=i<503 for i in m['schedule'])
        ids={k:{r['coco_id'] for r in rows} for k,rows in m['splits'].items()}
        assert not ids['fit']&ids['calibration'] and not ids['fit']&ids['probe'] and not ids['calibration']&ids['probe']
        assert all(identity(rows)==m['split_sha256'][key] for key,rows in m['splits'].items())
        assert not {r['coco_id'] for r in cfg['groups']['val']}&set.union(*ids.values())
        assert not {r['coco_id'] for r in cfg['groups']['train']}&ids['probe']
        matcher={}
        for phase in ('before','after'):
            matcher[phase]={}
            for split in ('calibration','probe'):
                merged=[]
                for rank in (0,1):
                    rows=read(f'matcher_{phase}_{split}_rank{rank}.json')
                    assert {r['coco_id'] for r in rows}==set(sorted(ids[split])[rank::2])
                    merged.extend(rows)
                merged.sort(key=lambda r:(r['task'],r['case_id']))
                assert merged==read(f'matcher_{phase}_{split}.json')
                assert [{k:v for k,v in r.items() if k!='scores'} for r in merged]==m['splits'][split]
                matcher[phase][split]=summarize(merged)
        matcher['geometry_controls']={split:geometry_summary(m['splits'][split]) for split in ('calibration','probe')}
        assert matcher==read('verified_matcher_summary.json')==summary['verified_matcher']
        histories=[read(f'history_rank{r}.json') for r in (0,1)]
        assert all(x['alignment']['pairs']==4 for h in histories for x in h)
        on={r['image_id']:r for r in read('on_predictions.json')}
        diagnostics={r['image_id']:r for rank in (0,1) for r in read(f'on_diagnostic_rank{rank}.json')}
        for condition in ('off','text_swap','region_swap'):
            pred=read(condition+'_predictions.json')
            shards=sum((read(f'{condition}_rank{r}.json') for r in (0,1)),[])
            assert sorted(shards,key=lambda r:r['image_id'])==pred
            drows=sum((read(f'{condition}_diagnostic_rank{r}.json') for r in (0,1)),[])
            p={r['image_id']:r for r in pred};d={r['image_id']:r for r in drows}
            assert len(p)==len(d)==len(drows)==1000 and p.keys()==d.keys()==on.keys()
            for mode,stored in summary['sensitivity'][condition].items():
                chosen=[r['image_id'] for r in cfg['groups']['val'] if mode=='all' or mode==r['mode']]
                assert stored['images']==len(chosen)
                assert stored['changed_captions']==sum(p[i]['caption']!=on[i]['caption'] for i in chosen)
                assert abs(stored['mean_reference_nll_delta']-np.mean([d[i]['reference_nll_per_token']-diagnostics[i]['reference_nll_per_token'] for i in chosen]))<1e-12
        comparisons={}
        for arm,path in (('A',a_path),('E',e_path)):
            with zipfile.ZipFile(path) as other:
                other_read=lambda n:json.loads(other.read(n))
                ocfg=other_read('protocol.json')
                for key in ('groups','expected_hashes','source_checkpoint_sha256','object_targets_sha256','global_batch_size','warmup_updates','base_lr','branch_lr','scst_weight','seed','cohort_sha256'):
                    assert cfg[key]==ocfg[key],key
                for name in ('origin_predictions.json','origin_per_image_cider.json','val_gt.json','train_object_targets.json'):
                    assert read(name)==other_read(name),name
                for rank in (0,1):
                    assert read(f'runtime_rank{rank}.json')==other_read(f'runtime_rank{rank}.json')
                    assert read(f'worker_complete_rank{rank}.json')['initial_base_sha256']==other_read(f'worker_complete_rank{rank}.json')['initial_base_sha256']
                comparisons['F_minus_'+arm]=pair(other_read('on_per_image_cider.json'),read('on_per_image_cider.json'))
                modes={g['image_id']:g['mode'] for g in cfg['groups']['val']}
                comparisons['F_minus_'+arm]['by_mode']={mode:pair(
                    [r for r in other_read('on_per_image_cider.json') if modes[r['image_id']]==mode],
                    [r for r in read('on_per_image_cider.json') if modes[r['image_id']]==mode]) for mode in cfg['counts_by_mode']['val']}
        comparisons['F_on_minus_off']=pair(read('off_per_image_cider.json'),read('on_per_image_cider.json'))
        sensitivity=summary['sensitivity']
    caption_audit(source,dataset,output,expected_sources=sources())
    result=json.loads(Path(output).read_text(encoding='utf-8'))
    result.update(status='F_VERIFIED_ARTIFACT_AUDIT_PASS',matcher=matcher,comparisons=comparisons,sensitivity=sensitivity,
                  verified_counts=m['counts'],decision='DO_NOT_SCALE_F: CIDEr below matched A; relation discrimination not improved; instance matcher below geometry controls.')
    result['checks']+=['Exact fit/calibration/probe manifest, source hashes and fit-only schedule; disjoint IDs and no auxiliary caption-val images',
                       'Every pair score and bootstrap summary recomputed; matched origin/reference/reward/runtime between A/E/F; sensitivity recounted']
    result['limits']+=['F adds verified labels/image exposures and compute, so A/E comparison is not equal supervision budget.',
                       'Matcher before is freshly initialized F, not trained E. Calibration/probe small and historically studied; no independent unseen test.',
                       'Zero/global instance comparisons have identical inputs and tie; cannot establish image-conditioned semantic understanding.']
    predicate_pairs=[r for r in m['splits']['fit'] if r['task']=='wrong_predicate']
    result['label_validity_caution']=dict(
        fit_predicate_pairs=len(predicate_pairs),
        holding_carrying_pairs=sum({r['source_positive']['action'],r['source_negative']['action']}=={'carry','hold'} for r in predicate_pairs),
        note='Explicit VCOCO actor/action zero is an annotation negative, not logical contradiction of a natural-language phrase. Actions can co-occur. Matcher accuracy is annotation agreement; visual plausibility review is required. No caption metrics changed.')
    Path(output).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    training_audit(source,HERE/'F_training_diagnostics.json')
    print(json.dumps(dict(status=result['status'],comparisons=comparisons,decision=result['decision']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--a',required=True);p.add_argument('--e',required=True);p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.source,a.a,a.e,a.dataset,a.output)
