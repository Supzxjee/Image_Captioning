"""Audit cached review text and recompute paired statistics; no model execution."""
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluate_readout import paired_ci
from readout_data import POLICY, panel_ids

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'experiments/relation_frames_inventory/review/shared_readout_v2'

def main():
    loaded = {a: json.loads((CACHE / f'{a}_loaded.json').read_text(encoding='utf-8')) for a in ('I', 'J')}
    summaries, scores = {}, {}
    for arm, cached in loaded.items():
        texts = cached['texts']
        read = lambda name: json.loads(texts[name])
        s = read('summary.json'); summaries[arm] = s
        assert s['arm'] == arm and s['updates'] == 500 and s['policy'] == POLICY
        assert read('execution_status.json')['success'] and not s['test_used']
        for name, expected in s['source_assets'].items():
            if name.endswith('.py'):
                raw = texts['source/' + name].encode('utf-8')
                assert hashlib.sha256(raw).hexdigest() == expected, name
                assert raw == subprocess.check_output(['git', 'show', s['source_commit'] + ':' + name], cwd=ROOT), name
        prep = read('preparation.json'); meta = read('metadata.json')
        assert prep['images'] == len(meta['rows']) == 1612
        assert [len(meta[k]) for k in ('train_indices','held_indices','dev_indices')] == [484,128,1000]
        assert len(meta['word2idx']) == 10000
        assert [len(meta['pairs'][k]) for k in ('train','heldout')] == [256,64]
        assert set(meta['semantic_panel_pair_ids']) == set(panel_ids(meta['pairs']['heldout']))
        for name in ('metadata.json','object_targets.json'):
            assert hashlib.sha256(texts[name].encode('utf-8')).hexdigest() == prep['payload_sha256'][name]
        for rank in (0,1):
            w=read(f'worker_rank{rank}.json'); p=read(f'preflight_rank{rank}.json')
            assert w['status']=='ENDPOINT_500_COMPLETE' and w['updates']==500
            assert w['initial_digest']==s['initial_digest'] and w['final_digest']==s['final_digest']
            assert w['initial_frozen_digest']==w['final_frozen_digest']==s['frozen_digest']
            assert p['status']=='MODEL_PREFLIGHT_PASS'
            hist=[json.loads(line) for line in texts[f'history_rank{rank}.jsonl'].splitlines()]
            assert [h['step'] for h in hist] == [1,2,3]+list(range(25,501,25))
            for h in hist:
                assert all(np.isfinite(h[k]) for k in ('loss','scst','kl','relation','grad_norm'))
                assert abs(h['loss']-(h['scst']+.1*h['kl']+.2*h['relation'])) < 1e-5
        expected={meta['rows'][i]['image_id']:meta['rows'][i]['coco_id'] for i in meta['dev_indices']}
        scores[arm]={}
        for condition in ('A','end','off'):
            pred=read(condition+'_predictions.json'); per=read(condition+'_per_image_cider.json')
            assert len(pred)==len(per)==1000
            assert [r['image_id'] for r in pred] == sorted(expected)
            assert {r['image_id']:r['coco_id'] for r in pred} == expected
            assert [(r['image_id'],r['coco_id']) for r in per] == [(r['image_id'],r['coco_id']) for r in pred]
            scores[arm][condition]=np.array([r['CIDEr'] for r in per])
            assert abs(scores[arm][condition].mean()-s['metrics'][condition]['CIDEr'])<1e-12
            chair=read(condition+'_chair_output.json')['sentences']
            assert len(chair)==1000 and {r['image_id'] for r in chair}==set(expected.values())
            count=sum(bool(r['mscoco_hallucinated_words']) for r in chair)
            assert count==s['metrics'][condition]['counts']['hallucinated_captions']
        assert {r['image_id']:r['caption'] for r in read('A_predictions.json')} == {r['image_id']:r['caption'] for r in meta['baseline']}
        assert len(read('held_likelihood.json'))==64
    for key in ('source_commit','policy','initial_digest','frozen_digest','tensor_content_sha256','metadata_sha256','object_targets_sha256'):
        assert summaries['I'][key]==summaries['J'][key],key
    for name in ('metadata.json','object_targets.json','A_predictions.json'):
        assert loaded['I']['texts'][name]==loaded['J']['texts'][name], name
    contrasts={}
    for name, delta in (
        ('J_minus_I',scores['J']['end']-scores['I']['end']),
        ('J_minus_A',scores['J']['end']-scores['J']['A']),
        ('I_minus_A',scores['I']['end']-scores['I']['A']),
        ('J_on_minus_off',scores['J']['end']-scores['J']['off']),
        ('I_on_minus_off',scores['I']['end']-scores['I']['off'])):
        contrasts[name]={'mean':float(delta.mean()),'paired95_ci':paired_ci(delta)}
    result={'status':'MATCHED_I_J_REVIEW_VERIFIED_BASELINE_NOT_IMPROVED','source_commit':summaries['I']['source_commit'],
        'zip_sha256':{a:d['zip_sha256'] for a,d in loaded.items()},'metrics':{a:s['metrics'] for a,s in summaries.items()},
        'contrasts':contrasts,'ranking':{a:{k:s[k] for k in ('held_tail_ranking','held_predicate_ranking')} for a,s in summaries.items()},
        'semantic_panel':'Pending blind review; ranking is not free-caption relation correctness',
        'limits':['One seed; reused development and adaptation-heldout data; bootstrap does not measure seed uncertainty.',
                  'No checkpoint or complete HDF5 payload in review ZIP: full tensor/checkpoint hashes remain runtime reports.',
                  'Off simultaneously zeros pair scores and pair values; not an isolated intervention on interaction appearance.',
                  'Metrics were aggregated from exported outputs, not rerun with neural models or external metric engines.'],
        'recommendation':'Keep A. J beats I in this pilot but does not improve A. Ask research lead before additional training.'}
    out=Path(__file__).parent
    (out/'RESULT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    lines=['# Shared pair readout I/J: matched 500-update pilot','',
      'Both user-run v2 jobs completed. Source, initialization, frozen state, input identities and fixed A captions match. Exported code matches the pinned Git blobs. No local model execution.','',
      '| Model | CIDEr | CHAIRs | Hallucinated captions / 1000 | Object recall |','|---|---:|---:|---:|---:|']
    for label,s,k in [('A',summaries['I'],'A'),('I',summaries['I'],'end'),('J',summaries['J'],'end')]:
        m=s['metrics'][k];lines.append(f"| {label} | {m['CIDEr']:.6f} | {m['chair']['CHAIRs']:.1%} | {m['counts']['hallucinated_captions']} | {m['chair']['Recall']:.2%} |")
    lines+=['','## Paired CIDEr differences','', '| Contrast | Mean | Bootstrap 95% interval |','|---|---:|---|']
    for name,d in contrasts.items():lines.append(f"| {name} | {d['mean']:+.6f} | [{d['paired95_ci'][0]:+.6f}, {d['paired95_ci'][1]:+.6f}] |")
    lines+=['','Bootstrap: 10,000 paired image resamples, seed 20261028. Intervals are conditional on this run and development set.','',
      'J ranks the positive image above its negative on 46/64 whole-tail pairs versus I 44/64; predicate-only ranking is 43/64 for both. These are likelihood tests, not evidence that generated captions express correct relations. The locked semantic panel is not scored yet.','',
      'J improves on the matched independent reader, but both degrade the stronger A baseline. Turning the reader off also improves CIDEr within each trained model. This does not establish the cause: off changes pair scores and values together, and training also updates two decoder layers.','',
      'Recommendation: preserve A as the best model. Do not treat this as a successful caption improvement or automatically extend training. Send the evidence to Astra for the next decision.','', '## Limits','']+['- '+x for x in result['limits']]
    (out/'RESULT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'contrasts':contrasts},indent=2))

if __name__=='__main__':main()
