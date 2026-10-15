"""Audit a user-run CE/SCST review ZIP without executing its contents."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile
import numpy as np


def run(source, output, prior_review=None):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.rouge.rouge import Rouge
    source, output = Path(source), Path(output)
    with zipfile.ZipFile(source) as z:
        def read(name): return json.loads(z.read(name))
        cfg, summary = read('protocol.json'), read('run_summary.json')
        protocol_sha = hashlib.sha256(z.read('protocol.json')).hexdigest()
        assert summary['protocol_sha256'] == protocol_sha
        anchored = cfg.get('ce_anchor_comparison', False)
        assert cfg['code_commit'] == ('2e20864b046d381c27e54af3b9176ab0effaba6c' if anchored
                                     else 'dabaca2b66640f7113733f16fd30597fefeab5f7')
        arms = ('scst', 'scst_ce') if anchored else ('ce', 'scst')
        treatment, control = ('scst_ce', 'scst') if anchored else ('scst', 'ce')
        if anchored:
            assert cfg['scst_weight'] == .05 and cfg['ce_weight'] == 1.
            assert summary['variant'] == cfg['variant'] == 'scst_ce_anchor_v1'
        identity = json.loads((Path(__file__).parent/'audit/hoi_validation_input_identity.json').read_text())['input_sha256']
        assert all(v == identity[k] for k, v in cfg['input_sha256'].items())
        assert [len(cfg['groups'][k]) for k in ('train', 'val')] == [20000, 1000]
        assert len({r['coco_id'] for rows in cfg['groups'].values() for r in rows}) == 21000
        assert len({r['image_id'] for rows in cfg['groups'].values() for r in rows}) == 21000
        assert not cfg['test_used'] and not summary['test_used']
        gt = read('val_gt.json'); rows = cfg['groups']['val']
        ids = [r['image_id'] for r in rows]
        refs = {i: [] for i in ids}
        for r in gt['annotations']: refs[r['image_id']].append({'caption': r['caption']})
        assert all(len(v) == 5 for v in refs.values())
        tokenizer = PTBTokenizer(); refs = tokenizer.tokenize(refs)
        training = {}
        for arm in arms:
            history, trained = read(f'{arm}/history.json'), read(f'{arm}/training_summary.json')
            assert len(history) == trained['steps'] == 2500
            assert [r['step'] for r in history] == list(range(1, 2501))
            assert all(r['images_seen'] == r['step']*8 for r in history)
            assert all(math.isfinite(v) for r in history for v in r.values())
            assert trained['protocol_sha256'] == protocol_sha
            assert trained['source_checkpoint_sha256'] == cfg['input_sha256']['checkpoint']
            assert trained['dropout'] is False and trained['precision'] == 'fp32'
            if anchored:
                assert trained['scst_weight'] == .05
                assert trained['ce_weight'] == (1. if arm == 'scst_ce' else 0.)
                assert all(abs(r['loss']-(.05*r['scst_loss']+r.get('ce_loss', 0.))) < 1e-5 for r in history)
            training[arm] = {**trained, 'first250_mean': {k: float(np.mean([r[k] for r in history[:250]]))
                for k in history[0] if k not in ('step', 'images_seen', 'seconds')},
                'last250_mean': {k: float(np.mean([r[k] for r in history[-250:]]))
                for k in history[0] if k not in ('step', 'images_seen', 'seconds')}}
        results, per_image, captions, chair_rows = {}, {}, {}, {}
        for arm in ('original', *arms):
            beams, clipped = read(f'{arm}/beams.json'), read(f'{arm}/clip_beams.json')
            assert [r['image_id'] for r in beams['data']] == ids
            assert [r['image_id'] for r in clipped['data']] == ids
            for b, c in zip(beams['data'], clipped['data']):
                assert b == {**c, 'candidates': [{k: v for k, v in x.items() if k != 'clipscore'} for x in c['candidates']]}
                assert all(math.isfinite(x['clipscore']) for x in c['candidates'])
            expected_sha = cfg['input_sha256']['checkpoint'] if arm == 'original' else training[arm]['checkpoint_sha256']
            for rank in range(2):
                journal = [json.loads(line) for line in z.read(f'{arm}/beams_{rank}.jsonl').decode().splitlines()]
                assert journal == beams['data'][rank::2]
                worker = read(f'{arm}/beams_worker_config.json')
                assert worker['checkpoint_sha256'] == expected_sha
            for policy in ('rank0', 'fixed'):
                name = arm+'_'+policy; predictions = read(f'{arm}/{policy}_predictions.json')
                assert [r['image_id'] for r in predictions] == ids
                if policy == 'rank0':
                    assert [r['caption'] for r in predictions] == [r['candidates'][0]['caption'] for r in beams['data']]
                else:
                    assert all(p['caption'] in {c['caption'] for c in b['candidates']} for p, b in zip(predictions, beams['data']))
                tokens = tokenizer.tokenize({p['image_id']: [{'caption': p['caption']}] for p in predictions})
                cider, scores = Cider().compute_score(refs, tokens)
                bleu, _ = Bleu(4).compute_score(refs, tokens); rouge, _ = Rouge().compute_score(refs, tokens)
                saved = read(f'{arm}/{policy}_metrics.json')
                for key, value in {'CIDEr': cider, 'ROUGE_L': rouge, **{f'Bleu_{i+1}': b for i, b in enumerate(bleu)}}.items():
                    assert abs(saved[key]-value) < 1e-10, (name, key, saved[key], value)
                assert saved == summary['metrics'][name]['caption']
                chair = read(f'{arm}/{policy}_chair_output.json'); sentences = chair['sentences']
                assert len(sentences) == 1000
                lookup = {r['image_id']: r for r in sentences}
                assert set(lookup) == {r['coco_id'] for r in rows}
                assert all(lookup[r['coco_id']]['caption'] == p['caption'] for r, p in zip(rows, predictions))
                hallucinated = sum(len(r['mscoco_hallucinated_words']) for r in sentences)
                mentions = sum(len(r['mscoco_generated_words']) for r in sentences)
                correct = sum(len(set(r['mscoco_generated_words']) & set(r['mscoco_gt_words'])) for r in sentences)
                gt_objects = sum(len(set(r['mscoco_gt_words'])) for r in sentences)
                bad = sum(bool(r['mscoco_hallucinated_words']) for r in sentences)
                aggregate = {'CHAIRs': bad/1000, 'CHAIRi': hallucinated/mentions, 'Recall': correct/gt_objects}
                assert all(abs(aggregate[k]-chair['overall_metrics'][k]) < 1e-12 for k in aggregate)
                assert chair['overall_metrics'] == summary['metrics'][name]['chair']
                results[name] = {'caption': saved, 'chair': aggregate, 'hallucinated_captions': bad,
                    'hallucinated_mentions': hallucinated, 'object_mentions': mentions,
                    'correct_unique_objects': correct, 'reference_unique_objects': gt_objects,
                    'mean_words': float(np.mean([len(r['caption'].split()) for r in predictions]))}
                per_image[name] = np.asarray(scores); captions[name] = predictions
                chair_rows[name] = lookup
        paired = {}
        for policy in ('rank0', 'fixed'):
            for baseline in ('original', control):
                left, right = baseline+'_'+policy, treatment+'_'+policy
                delta = per_image[right]-per_image[left]
                rng = np.random.default_rng(20261015)
                ci = np.percentile([delta[rng.integers(0, 1000, 1000)].mean() for _ in range(5000)], [2.5, 97.5])
                fixed = introduced = 0
                for image in rows:
                    a = bool(chair_rows[left][image['coco_id']]['mscoco_hallucinated_words'])
                    b = bool(chair_rows[right][image['coco_id']]['mscoco_hallucinated_words'])
                    fixed += a and not b; introduced += b and not a
                paired[right+'_vs_'+left] = {'CIDEr_delta': float(delta.mean()), 'conditional_bootstrap_95': ci.tolist(),
                    'hallucination_removed': int(fixed), 'hallucination_introduced': int(introduced),
                    'changed_captions': sum(a['caption'] != b['caption'] for a, b in zip(captions[left], captions[right]))}
        raw, base, reference = (results[k+'_rank0'] for k in (treatment, control, 'original'))
        gate = {'raw_CIDEr_beats_both': raw['caption']['CIDEr'] > max(base['caption']['CIDEr'], reference['caption']['CIDEr']),
                'raw_CHAIR_not_worse_than_both': all(raw['chair'][k] <= min(base['chair'][k], reference['chair'][k]) for k in ('CHAIRs', 'CHAIRi')),
                'raw_recall_within_1pp_of_best': raw['chair']['Recall'] >= max(base['chair']['Recall'], reference['chair']['Recall'])-.01}
        if anchored:
            f, b = results[treatment+'_fixed'], results['original_fixed']
            gate['fixed_not_worse_than_original'] = (f['caption']['CIDEr'] >= b['caption']['CIDEr'] and
                  all(f['chair'][k] <= b['chair'][k] for k in ('CHAIRs', 'CHAIRi')) and f['chair']['Recall'] >= b['chair']['Recall']-.01)
        assert bool(summary['expansion_gate_passed']) == all(gate.values())
        if anchored:
            prior_path = Path(__file__).parent/'audit/scst_comparison_train20k_val1000_result.json'
            prior = json.loads(prior_path.read_text())
            assert all(results[k]['caption'] == prior['metrics'][k]['caption'] and results[k]['chair'] == prior['metrics'][k]['chair']
                       for k in ('original_rank0', 'original_fixed'))
            if prior_review:
                prior_review = Path(prior_review)
                assert hashlib.sha256(prior_review.read_bytes()).hexdigest() == prior['source_zip_sha256']
                with zipfile.ZipFile(prior_review) as previous:
                    before = json.loads(previous.read('protocol.json'))
                    assert before['groups'] == cfg['groups'] and before['input_sha256'] == cfg['input_sha256']
                    for policy in ('rank0', 'fixed'):
                        assert json.loads(previous.read(f'original/{policy}_predictions.json')) == read(f'original/{policy}_predictions.json')
        result = {'status': 'AUDIT_PASS', 'variant': cfg.get('variant', 'ce_scst_comparison'),
                  'source_zip_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                  'source_zip_bytes': source.stat().st_size, 'protocol_sha256': protocol_sha,
                  'training': training, 'metrics': results, 'paired_comparisons': paired,
                  'gate_components': gate, 'expansion_gate_passed': all(gate.values()), 'test_used': False,
                  'previous_review_cohort_inputs_and_original_captions_exact_match': bool(anchored and prior_review),
                  'verification_limits': ['Model weights absent: actual model load/parameter hashes not independently replayed.',
                    'Full source dataset and detector cache absent: split provenance and fixed rerank formula not independently replayed.',
                    'CHAIR reaggregated from supplied object annotations, not retokenized against external ground truth.',
                    'METEOR cross-checked across reports, not recomputed.',
                    'Bootstrap resamples paired per-image CIDEr with validation IDF fixed; one pool/checkpoint/seed, not a multi-seed confidence interval.']}
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
        print(json.dumps({'status': result['status'], 'paired': paired, 'metrics': results}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', required=True); p.add_argument('--output', required=True)
    p.add_argument('--prior-review')
    a = p.parse_args(); run(a.source, a.output, a.prior_review)
