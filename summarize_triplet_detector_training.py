"""Compare matched GT/detector heads on identical detector dev features.

Known full100 assistant assessments are a regression cross-check, not accuracy.
"""
import argparse
import json
from pathlib import Path

import torch
from captioning.actor_compatibility import pair_geometry, actor_edge_metrics
from captioning.triplet_verifier import TripletVerifier
from evaluate_frozen_residual import sha
from run_actor_compatibility_cached import read_archive
from run_instance_alignment_diagnostic import metrics


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'gt-head', 'detector-head', 'gt-transfer',
                 'detector-transfer', 'visual-review', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    rows, views, text, lengths = read_archive(args.source, ['train_pairs.json', 'val_pairs.json'])
    dev = rows[lengths[0]:]
    views, text = views[lengths[0]:], text[lengths[0]:]
    labels = torch.tensor([row['labels'] for row in dev])
    review = json.loads(Path(args.visual_review).read_text(encoding='utf-8'))
    result = {'source_sha256': sha(args.source), 'common_detector_dev_pairs': len(dev), 'runs': {},
              'notes': 'Single fixed seed42 development audit. Identical detector dev features for both heads; '
                       'thresholds fitted on each training condition only. Full100 review is method-visible '
                       'assistant inspection of selected claims, not independent accuracy or exhaustive recall.'}
    for condition, head_path, transfer_path in (
        ('gt_control', args.gt_head, args.gt_transfer),
        ('detector', args.detector_head, args.detector_transfer),
    ):
        directory = Path(head_path)
        summary = json.loads((directory / 'triplet_summary.json').read_text())
        if summary['sources']['detector_training'] != result['source_sha256']:
            raise ValueError('Different matched training archive.')
        if summary['training_condition'] != condition or summary['seed'] != 42 or summary['epochs'] != 30:
            raise ValueError('Unexpected training condition/configuration.')
        cp = torch.load(directory / 'triplet_head.pt', map_location='cpu', weights_only=True)
        model = TripletVerifier()
        model.load_state_dict(cp['model'])
        model.eval().requires_grad_(False)
        with torch.no_grad():
            logits = model(views, pair_geometry(dev), text)
        prompts = json.loads((directory / 'triplet_prompt_cache.json').read_text())
        selected = {(filename, r['subject_instance'], r['relation'], r['object_instance'])
                    for filename, entry in prompts.items() for r in entry['relations']}
        crosscheck = []
        for claim in review['claims']:
            key = tuple(claim[k] for k in ('filename', 'subject_instance', 'relation', 'object_instance'))
            crosscheck.append({**claim, 'selected_by_new_head': key in selected})
        transfer = json.loads((Path(transfer_path) / 'transfer_summary.json').read_text())
        if transfer['head_sha256'] != sha(directory / 'triplet_head.pt') or transfer['threshold_refit']:
            raise ValueError('Transfer head changed or thresholds refitted.')
        result['runs'][condition] = {
            'head_sha256': transfer['head_sha256'],
            'common_detector_dev_metrics': metrics(logits.sigmoid(), labels, cp['thresholds']),
            'common_detector_dev_actor_edges': actor_edge_metrics(logits, dev, exhaustive=True),
            'transfer': transfer,
            'full100_claims': len(selected),
            'full100_relation_images': sum(bool(e['relations']) for e in prompts.values()),
            'known_assistant_review_crosscheck': crosscheck,
        }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'paired_detector_training_summary.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    for condition, run in result['runs'].items():
        print(condition, json.dumps({
            'common_detector_dev_AP': run['common_detector_dev_metrics']['macro_AP_on_known_labels'],
            'actor': run['common_detector_dev_actor_edges'],
            'transfer': run['transfer']['after'],
            'full100_claims': run['full100_claims'],
            'selected_known_reviewed': [c for c in run['known_assistant_review_crosscheck'] if c['selected_by_new_head']],
        }, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
