"""Reference-aware candidate diagnosis; no learned selector or model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np


def run(source, output):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from captioning.object_consistency import extract_caption_objects
    out = Path(output)
    if out.exists(): raise FileExistsError('Use a fresh diagnostic output directory')
    out.mkdir(parents=True)
    with zipfile.ZipFile(source) as z:
        assert z.testzip() is None
        read = lambda name: json.loads(z.read(name))
        cfg, summary, gt = read('protocol.json'), read('run_summary.json'), read('val_gt.json')
        assert not cfg['test_used'] and cfg['object_reward_weight'] == .5
        ids = [r['image_id'] for r in cfg['groups']['val']]
        assert len(ids) == len(set(ids)) == 1000
        refs = {i: [] for i in ids}
        for r in gt['annotations']: refs[r['image_id']].append({'caption': r['caption']})
        assert all(len(r) == 5 for r in refs.values())
        tokenizer = PTBTokenizer()
        token_refs = tokenizer.tokenize(refs)
        results, cases = {}, []
        for arm in ('original', 'scst_ce', 'scst_ce_object'):
            rows = read(arm+'/clip_beams.json')['data']
            assert [r['image_id'] for r in rows] == ids
            assert all(len(r['candidates']) == 5 for r in rows)
            tokens = tokenizer.tokenize({r['image_id']: [{'caption': c['caption']} for c in r['candidates']] for r in rows})
            scores = np.asarray([Cider().compute_score(token_refs, {i: [tokens[i][rank]] for i in ids})[1]
                                 for rank in range(5)]).T
            fixed = read(arm+'/fixed_predictions.json')
            assert [r['image_id'] for r in fixed] == ids
            choice = np.asarray([next(k for k, c in enumerate(r['candidates']) if c['caption'] == p['caption'])
                                 for r, p in zip(rows, fixed)])
            clip = np.asarray([np.argmax([c['clipscore'] for c in r['candidates']]) for r in rows])
            oracle = scores.argmax(1)
            get = lambda selected: scores[np.arange(len(ids)), selected]
            baseline = get(choice)
            assert abs(scores[:, 0].mean()-summary['metrics'][arm+'_rank0']['caption']['CIDEr']) < 1e-10
            assert abs(baseline.mean()-summary['metrics'][arm+'_fixed']['caption']['CIDEr']) < 1e-10
            gap = get(oracle)-baseline
            normalized_unique = [len(set(tokens[i])) for i in ids]
            objects = [[frozenset(extract_caption_objects(c['caption'])) for c in r['candidates']] for r in rows]
            # Exact per-image adaptation of choosing one of the other four candidates.
            top2 = np.sort(scores, axis=1)[:, -2]
            result = {'CIDEr': {'rank0': float(scores[:, 0].mean()), 'fixed': float(baseline.mean()),
                                'clip_only': float(get(clip).mean()), 'reference_oracle': float(get(oracle).mean())},
                      'gap_to_fixed': float(gap.mean()), 'images_with_better_candidate': int((gap > 1e-10).sum()),
                      'images_with_gain_at_least_point1': int((gap >= .1).sum()),
                      'mean_unique_PTB_candidates': float(np.mean(normalized_unique)),
                      'images_all_candidates_PTB_identical': sum(n == 1 for n in normalized_unique),
                      'images_different_object_sets': sum(len(set(v)) > 1 for v in objects),
                      'oracle_fixed_different_object_sets': sum(v[int(a)] != v[int(b)] for v, a, b in zip(objects, choice, oracle)),
                      'oracle_best_second_gap_mean': float((get(oracle)-top2).mean())}
            results[arm] = result
            for j, r in enumerate(rows):
                cases.append({'arm': arm, 'image_id': r['image_id'], 'coco_id': r['coco_id'],
                              'fixed_rank': int(choice[j]), 'oracle_rank': int(oracle[j]),
                              'fixed_caption': r['candidates'][choice[j]]['caption'],
                              'oracle_caption': r['candidates'][oracle[j]]['caption'],
                              'CIDEr_gap': float(gap[j]),
                              'object_set_changed': objects[j][choice[j]] != objects[j][oracle[j]]})
            np.savez_compressed(out/(arm+'_scores.npz'), image_ids=ids, scores=scores,
                                fixed=choice, clip=clip, oracle=oracle)
        result = {'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                  'images': 1000, 'candidates_per_arm': 5000, 'results': results,
                  'limits': ['Oracle sees references; not deployable and not a model-quality improvement.',
                             'CIDEr headroom may reflect phrasing, not factual/relational correctness.',
                             'Lexical object-set differences are partial; not visual truth labels.',
                             'No CHAIR labels for all candidates; no claim oracle reduces hallucination.',
                             'Existing reused validation; no training, inference, notebook launch or test.']}
        (out/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        (out/'oracle_cases.json').write_text(json.dumps(sorted(cases, key=lambda x: -x['CIDEr_gap']), indent=2), encoding='utf-8')
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); run(args.source, args.output)
