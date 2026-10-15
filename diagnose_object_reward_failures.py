"""Read-only diagnosis of observed errors; no images, weights or training invoked."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from captioning.object_reward import recognized_objects, TrainingObjectReward


def diagnose(source, previous, output):
    with zipfile.ZipFile(source) as z, zipfile.ZipFile(previous) as prior:
        def read(n): return json.loads(z.read(n))
        current_history = read('scst_ce/history.json')
        previous_history = json.loads(prior.read('scst_ce/history.json'))
        assert read('protocol.json')['groups'] == json.loads(prior.read('protocol.json'))['groups']
        divergences = {}
        for key in ('ce_loss', 'scst_loss', 'grad_norm', 'sample_reward', 'greedy_reward'):
            difference = next(((i+1, a[key], b[key]) for i, (a, b) in enumerate(zip(previous_history, current_history))
                               if a[key] != b[key]), None)
            divergences[key] = None if difference is None else {'step': difference[0], 'previous': difference[1],
                'current': difference[2], 'absolute_delta': abs(difference[2]-difference[1])}
        common = set(previous_history[0]) & set(current_history[0]) - {'seconds'}
        first_step_equal = all(previous_history[0][k] == current_history[0][k] for k in common)
        parser_checks, chairs = {}, {}
        for arm in ('scst_ce', 'scst_ce_object'):
            records = read(arm+'/rank0_chair_output.json')['sentences']; chairs[arm] = {r['image_id']: r for r in records}
            misses, total = [], 0
            for r in records:
                labels = recognized_objects(r['caption'])
                for word, label in r['mscoco_hallucinated_words']:
                    total += 1
                    if label not in labels:
                        misses.append({'coco_id': r['image_id'], 'caption': r['caption'], 'word': word,
                            'chair_label': label, 'recognized_legacy': sorted(labels),
                            'recognized_aliases_v2': sorted(recognized_objects(r['caption'], 'aliases_v2')),
                            'category': 'verified_alias_gap' if word in ('turkey', 'desk') else 'context_ambiguous_evaluator_mapping'})
            parser_checks[arm] = {'chair_hallucinated_mentions': total, 'caught_by_legacy_parser': total-len(misses),
                                  'missed_mentions': misses}
        before, after = chairs['scst_ce'], chairs['scst_ce_object']
        introduced = [{'coco_id': i, 'before': before[i]['caption'], 'after': after[i]['caption'],
                       'chair_hallucinated_words': after[i]['mscoco_hallucinated_words']}
                      for i in before if not before[i]['mscoco_hallucinated_words'] and after[i]['mscoco_hallucinated_words']]
        targets = read('train_object_targets.json')['data']
        toy = TrainingObjectReward([{'image_id': 0, 'supported': ['person', 'sink'], 'salient': ['person'], 'detector_neutral': []}])
        texts = ['a person next to a motorcycle', 'a person next to a sink and a motorcycle']
        scores, terms = toy.score([0, 0], texts)
        result = {'source_sha256': hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            'previous_sha256': hashlib.sha256(Path(previous).read_bytes()).hexdigest(),
            'first_step_common_numeric_fields_equal': first_step_equal, 'first_exact_divergences': divergences,
            'runtime_root_cause': 'Unconfirmed. Seed alone is insufficient; kernels/environments not captured in old ZIPs.',
            'parser_alignment': parser_checks, 'new_raw_chair_errors': introduced,
            'target_state_counts': {'train_images': len(targets),
                'detector_neutral_images': sum(bool(r['detector_neutral']) for r in targets),
                'detector_neutral_labels': sum(len(r['detector_neutral']) for r in targets),
                'empty_salience_images': sum(not r['salient'] for r in targets),
                'support_with_non_salient_classes': sum(bool(set(r['supported'])-set(r['salient'])) for r in targets)},
            'fraction_penalty_counterexample': {'type': 'Design limitation, not a newly observed training behavior',
                'captions': texts, 'scores': scores.tolist(), 'terms': terms,
                'explanation': 'The same unsupported motorcycle remains; adding a supported non-salient sink dilutes the fraction penalty.'},
            'official_metrics_changed': False, 'new_training_run': False,
            'limits': ['No image-level human adjudication of mother/baby zebra; lexical false-positive suspicion only.',
                       'Alias gaps account for two of forty treatment CHAIR mentions; they do not explain all remaining errors.',
                       'CPU tests cannot confirm same-GPU bitwise reproducibility.',
                       'No conclusion that annotation neutral exemptions, reward scale or dilution caused the observed validation regression.']}
        Path(output).write_text(json.dumps(result, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--previous',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();diagnose(a.source,a.previous,a.output)
