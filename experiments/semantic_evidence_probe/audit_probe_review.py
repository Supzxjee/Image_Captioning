"""Audit saved probe scores; never execute archived source or load pickle/checkpoints."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from probe_data import digest, identity, summarize


def geometry(candidate):
    s, o = np.asarray(candidate['bounds'][:2])
    inter = np.maximum(0, np.minimum(s[2:],o[2:])-np.maximum(s[:2],o[:2])).prod()
    union = (s[2:]-s[:2]).prod()+(o[2:]-o[:2]).prod()-inter
    return {'box_iou':float(inter/union),
            'negative_center_distance':-float(np.linalg.norm((s[:2]+s[2:]-o[:2]-o[2:])/2))}


def audit(source, output):
    here = Path(__file__).parent
    with zipfile.ZipFile(source) as z:
        assert z.testzip() is None
        read = lambda n: json.loads(z.read(n))
        cfg, manifest, summary, rows = (read(n) for n in ('config.json','probe_manifest.json','summary.json','scores.json'))
        assert manifest == json.loads((here/'probe_manifest.json').read_text(encoding='utf-8'))
        assert identity(manifest['rows']) == manifest['rows_sha256']
        assert hashlib.sha256(z.read('probe_manifest.json')).hexdigest() == cfg['manifest_sha256'] == summary['manifest_sha256']
        assert summary['checkpoint_sha256'] == manifest['checkpoint_sha256'] == '651f411f589c5dd7beb0fae98fafaa7d9ea06a7fd7f2b49400ac878f60227b18'
        for name, expected in cfg['source_sha256'].items():
            assert hashlib.sha256(z.read('source/'+name)).hexdigest() == expected
            path = here.parent/'semantic_evidence_pilot/pilot_model.py' if name=='pilot_model.py' else here/name
            assert hashlib.sha256(path.read_text(encoding='utf-8').encode()).hexdigest() == expected
        rank_rows, states = [], []
        image_ids = sorted({r['coco_id'] for r in manifest['rows']})
        for rank in (0,1):
            shard = read(f'rank{rank}.json')
            assert shard['rank']==rank and shard['no_training'] and shard['parameters_unchanged']
            assert shard['runtime']['worker_gpu']=={'rank':rank,'name':'Tesla T4'}
            assert shard['runtime']['deterministic_algorithms'] and not shard['runtime']['warn_only']
            assert {r['coco_id'] for r in shard['records']}==set(image_ids[rank::2])
            rank_rows.extend(shard['records']); states.append(shard['module_state_sha256'])
        assert states[0]==states[1]
        assert sorted(rank_rows,key=lambda r:(r['task'],r['case_id']))==rows
        assert [{k:v for k,v in r.items() if k!='scores'} for r in rows]==manifest['rows']
        assert len(rows)==62 and len({r['coco_id'] for r in rows})==59
        assert summary['metrics']==summarize(rows)
        assert summary['status']=='FROZEN_MATCHER_DIAGNOSTIC_COMPLETE'
        assert summary['no_training'] and summary['parameters_unchanged'] and not summary['test_used']
        controls, details = {}, {}
        for task in summary['metrics']:
            selected = [r for r in rows if r['task']==task]
            controls[task], details[task] = {}, {}
            for method in ('boxed','global_view','zero_visual'):
                details[task][method] = {}
                for predicate in ('holding','riding'):
                    subset=[r for r in selected if r['predicate']==predicate]
                    margins=[r['scores'][method][r['positive_index']]-r['scores'][method][1-r['positive_index']] for r in subset]
                    details[task][method][predicate]=dict(pairs=len(subset),wins=sum(d>1e-7 for d in margins),ties=sum(abs(d)<=1e-7 for d in margins))
            for method in ('box_iou','negative_center_distance'):
                per_image = {}
                for row in selected:
                    scores=[geometry(c)[method] for c in row['candidates']]
                    delta=scores[row['positive_index']]-scores[1-row['positive_index']]
                    per_image.setdefault(row['coco_id'],[]).append(.5 if abs(delta)<=1e-7 else float(delta>0))
                controls[task][method]=dict(pair_accuracy=float(np.mean([v for group in per_image.values() for v in group])),
                    image_macro_pair_accuracy=float(np.mean([np.mean(v) for v in per_image.values()])))
        counts=Counter((r['object_label'],r['predicate']) for r in rows if r['task']=='wrong_predicate')
        for obj in {k[0] for k in counts}: assert counts[obj,'holding']==counts[obj,'riding']
    result=dict(status='PROBE_ARTIFACT_AUDIT_PASS',source_zip_sha256=digest(source),
        metrics=summary['metrics'],counts_by_predicate=details,posthoc_geometry_controls=controls,
        recorded_module_state_sha256=states[0],no_training=True,test_used=False,
        checks=['ZIP CRC; exact locked manifest and source hashes; checkpoint identity in saved metadata',
                'Two complete Tesla T4 inference shards with unique image partition and equal recorded module digests',
                'All scores finite; 62 pairs/59 images; exact candidate labels; recomputed tie-aware accuracies and image-cluster intervals',
                'Predicate positives balanced within each object class'],
        limits=summary['limitations']+['No independent checkpoint/visual-cache inference replay. Unchanged parameters are recorded in worker metadata, not verified from archived tensors.',
        'Geometry controls are post-hoc descriptive comparators; no fitted parameters or thresholds. They do not establish semantic understanding or a trained baseline improvement.'])
    Path(output).write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],posthoc_geometry_controls=controls),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.source,a.output)
