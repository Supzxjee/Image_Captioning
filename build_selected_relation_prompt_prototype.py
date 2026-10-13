"""Offline prototype using scored matched pairs; no label-based selection or retuning."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from captioning.selected_relation_prompts import build_selected_prompts, entity_id
from captioning.supervised_relations import RELATIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--evaluation-summary', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-relations', type=int, default=3)
    args = parser.parse_args()
    source = Path(args.predictions)
    rows = json.loads(source.read_text())
    evaluation = json.loads(Path(args.evaluation_summary).read_text())
    if evaluation['variant'] != 'frozen_residual_detector_matched_evaluation_v1':
        raise ValueError('Expected scored detector-box evaluation.')
    thresholds = evaluation['residual']['thresholds']
    # Deliberately omit labels, GT IDs/boxes, match IoU, caption references from inference.
    allowed = ('filename', 'subject', 'object', 'subject_box', 'object_box', 'scores',
               'subject_detector_confidence', 'object_detector_confidence')
    prompts = build_selected_prompts([{k: r[k] for k in allowed if k in r} for r in rows],
                                      thresholds, args.max_relations)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'selected_prompt_cache.json').write_text(json.dumps(prompts, indent=2), encoding='utf-8')
    # Annotation lookup exists ONLY in a separate audit after selection.
    lookup = {}
    for row in rows:
        sid = entity_id(row['filename'], row['subject'], row['subject_box'])
        oid = entity_id(row['filename'], row['object'], row['object_box'])
        for c, predicate in enumerate(RELATIONS):
            key = (row['filename'], sid, predicate, oid)
            if key in lookup and lookup[key] != row['labels'][c]:
                raise ValueError('Inconsistent audit annotation for detector pair.')
            lookup[key] = row['labels'][c]
    audit = []
    for filename, entry in prompts.items():
        for claim in entry['relations']:
            label = lookup[(filename, claim['subject_instance'], claim['relation'], claim['object_instance'])]
            audit.append({'filename': filename, **claim, 'annotation': {1: 'known_positive',
                            0: 'known_negative', -1: 'unknown'}[label],
                          'human_status': '', 'human_notes': ''})
    fields = ['filename', 'subject_instance', 'subject', 'relation', 'object_instance', 'object',
              'score', 'threshold', 'score_margin', 'prompt', 'annotation', 'human_status', 'human_notes']
    with (output / 'selected_claim_review.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit)
    counts = Counter(row['annotation'] for row in audit)
    summary = {'variant': 'selected_detector_relation_prompt_prototype_v1',
               'images': len(prompts), 'scored_pairs': len(rows), 'max_relations': args.max_relations,
               'modes': dict(Counter(e['mode'] for e in prompts.values())),
               'selected_claims': len(audit), 'annotation_counts_after_selection': dict(counts),
               'capped_claims': sum(e['capped_claims'] for e in prompts.values()),
               'thresholds': thresholds, 'source_predictions_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
               'model_checkpoint_sha256': evaluation['protocol']['checkpoint_sha256'],
               'notes': 'Prototype on already evaluated GT-referenced matched detector pairs only. '
                        'Object inventory also limited to observed scored pairs, not full detector cache. '
                        'Not full inference prompt cache; labels used only for separate audit. '
                        'Scores/margins are not calibrated confidence; no caption improvement claim.'}
    (output / 'prompt_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
