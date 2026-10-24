"""Audit the once-loaded review; no model inference or metric dependency install."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from detail_metrics import paired
from detail_data import A_SHA, BASE_SHA, schedule

HERE = Path(__file__).resolve().parent


def audit(loaded):
    f = loaded['files']
    obj = {n: json.loads(v) for n, v in f.items() if n.endswith('.json')}
    cfg, summary = obj['protocol.json'], obj['run_summary.json']
    assert summary['status'] == 'PILOT_COMPLETE' and not summary['test_used']
    assert hashlib.sha256(f['protocol.json'].encode()).hexdigest() == summary['protocol_sha256']
    for name, expected in cfg['implementation_sha256'].items():
        assert hashlib.sha256(f['source/' + name].encode()).hexdigest() == expected, name
    assert cfg['input_sha256']['checkpoint'] == A_SHA == summary['source_A_sha256']
    train, val = cfg['groups']['train'], cfg['groups']['val']
    assert len(train) == 10000 and len(val) == 5000
    assert not {r['coco_id'] for r in train} & {r['coco_id'] for r in val}
    assert cfg['schedule'] == schedule(train)
    by_id = {r['image_id']: i for i, r in enumerate(train)}
    counts, epochs = Counter(), [[], [], [], []]
    for rank in (0, 1):
        done = obj[f'train_complete_rank{rank}.json']
        assert done['status'] == 'COMPLETE' and done['updates'] == 5000
        assert done['final_base_sha256'] == BASE_SHA and done['trainable_base_parameters'] == 0
        assert done['final_head_sha256'] == summary['final_head_sha256']
        assert obj[f'worker_complete_rank{rank}.json']['final_head_sha256'] == done['final_head_sha256']
        assert obj[f'profile_rank{rank}.json']['status'] == 'PASS'
        history = obj[f'history_rank{rank}.json']
        assert [r['step'] for r in history] == list(range(1, 5001))
        for row in history:
            assert len(row['image_ids']) == len(row['conditions']) == 4
            assert np.isfinite(row['CE_global']) and np.isfinite(row['grad_norm'])
            for iid, condition in zip(row['image_ids'], row['conditions']):
                assert condition == cfg['schedule'][row['epoch']][by_id[iid]]
                counts[iid, condition] += 1
                epochs[row['epoch']].append(iid)
    assert all(counts[i, c] == 2 for i in by_id for c in (0, 1))
    assert all(len(ids) == len(set(ids)) == 10000 and set(ids) == set(by_id) for ids in epochs)
    expected_ids = sorted(r['image_id'] for r in val)
    coco = {r['image_id']: r['coco_id'] for r in val}
    scores, captions, flagged = {}, {}, {}
    for condition in ('A', 'L', 'H'):
        pred = obj[f'{condition}_predictions.json']
        per = obj[f'{condition}_per_image_cider.json']
        assert [r['image_id'] for r in pred] == [r['image_id'] for r in per] == expected_ids
        scores[condition] = [r['CIDEr'] for r in per]
        assert abs(np.mean(scores[condition]) - summary['metrics'][condition]['CIDEr']) < 1e-12
        captions[condition] = {coco[r['image_id']]: r['caption'] for r in pred}
        sentences = obj[f'{condition}_chair_output.json']['sentences']
        assert len(sentences) == 5000 and {r['image_id'] for r in sentences} == set(coco.values())
        assert all(s['caption'] == captions[condition][s['image_id']] for s in sentences)
        flagged[condition] = {s['image_id'] for s in sentences if s['mscoco_hallucinated_words']}
        mentions = sum(len(s['mscoco_generated_words']) for s in sentences)
        metrics = dict(CHAIRs=len(flagged[condition])/5000,
                       CHAIRi=sum(len(s['mscoco_hallucinated_words']) for s in sentences)/mentions,
                       Recall=sum(len(set(s['mscoco_generated_words']) & set(s['mscoco_gt_words'])) for s in sentences)/sum(len(set(s['mscoco_gt_words'])) for s in sentences))
        assert all(abs(v-summary['metrics'][condition][k]) < 1e-12 for k,v in metrics.items())
    for other in ('A', 'L'):
        actual = paired(scores['H'], scores[other])
        assert actual == summary['paired_CIDEr']['H_minus_' + other]
    assert not summary['investment_gate']
    return dict(status='AUDIT_PASS', zip_sha256=loaded['zip_sha256'],
                metrics=summary['metrics'], paired_CIDEr=summary['paired_CIDEr'],
                flagged_counts={c:len(v) for c,v in flagged.items()},
                H_vs_A_chair=dict(removed=len(flagged['A']-flagged['H']), added=len(flagged['H']-flagged['A'])),
                H_L_identical_captions=sum(captions['H'][i]==captions['L'][i] for i in captions['H']),
                updates=5000, image_exposures=40000, investment_gate=False,
                limits=['Saved head/actual HDF5 payloads absent from review: rely on Kaggle receipts for those checks.',
                        'CIDEr scorer not rerun locally; per-image aggregation and paired bootstrap reproduced.',
                        'One seed and reused validation; negative result applies to this adapter/recipe only.'])


if __name__ == '__main__':
    result = audit(json.loads((HERE/'review/loaded.json').read_text(encoding='utf-8')))
    (HERE/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
