"""Frozen subject intervention and ambiguous-region ablation on cached features."""
import argparse
import io
import json
import zipfile
from pathlib import Path
from collections import Counter
import torch
from captioning.instance_contrast_alignment import ResidualInstanceAlignment, PairOnlyResidualAlignment
from captioning.instance_evidence_audit import ambiguous_regions, subject_swap_indices
from captioning.selected_relation_prompts import build_selected_prompts
from captioning.supervised_relations import RELATIONS
from evaluate_frozen_residual import sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--scored', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--comparison-head', help='Optional newly trained pair-only development head; never replaces frozen baseline.')
    args = parser.parse_args()
    previous = json.loads((Path(args.scored)/'full_candidate_audit_summary.json').read_text())
    if sha(args.checkpoint) != previous['checkpoint_sha256'] or sha(args.source) != previous['source_zip_sha256']:
        raise ValueError('Source/head changed since validated scoring.')
    with zipfile.ZipFile(args.source) as archive:
        def read(name):
            matches = [n for n in archive.namelist() if n.endswith(name)]
            if len(matches) != 1:
                raise ValueError(name)
            return archive.read(matches[0])
        inventory = json.loads(read('data/object_inventory.json'))
        rows = json.loads(read('data/evaluation_pairs.json'))
        records = sum([torch.load(io.BytesIO(read(f'features_{i}.pt')), map_location='cpu', weights_only=True)
                       for i in range(2)], [])
    records.sort(key=lambda r: r['index'])
    if [r['index'] for r in records] != list(range(len(rows))):
        raise ValueError('Feature order mismatch.')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = ResidualInstanceAlignment(scale=checkpoint['scale'])
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval().requires_grad_(False)
    torch.set_num_threads(2)
    views = torch.stack([r['views'] for r in records])
    text = torch.stack([r['text'] for r in records])
    with torch.inference_mode():
        scores, _, _ = model(views, text)
    predictions = json.loads((Path(args.scored)/'all_candidate_predictions.json').read_text())
    if len(predictions) != len(rows) or any(any(p[k] != r[k] for k in r) for p, r in zip(predictions, rows)):
        raise ValueError('Scored pair order changed.')
    if not torch.allclose(scores, torch.tensor([r['scores'] for r in predictions]), atol=1e-6):
        raise ValueError('Frozen scores differ.')
    swaps = subject_swap_indices(rows)
    interventions = []
    for start in range(0, len(swaps), 256):
        batch = swaps[start:start+256]
        source, replacement = zip(*batch)
        changed = views[list(source)].clone()
        changed[:, 0] = views[list(replacement), 0]
        with torch.inference_mode():
            values, _, _ = model(changed, text[list(source)])
        for k, (i, j) in enumerate(batch):
            interventions.append({'pair_index': i, 'replacement_pair_index': j,
                'filename': rows[i]['filename'], 'subject': rows[i]['subject_id'],
                'replacement_subject': rows[j]['subject_id'], 'object': rows[i]['object_id'],
                'score_change': (values[k]-scores[i]).tolist()})
    original = build_selected_prompts(predictions, checkpoint['thresholds'], object_inventory=inventory)
    ambiguous = ambiguous_regions(inventory)
    excluded = {(r['filename'], identity) for r in ambiguous for identity in (r['a'], r['b'])}
    # Diagnostic conservative abstention: inventory labels remain explicit alternatives.
    # We neither choose a label from GT nor merge different actors into one instance.
    filtered = [r for r in predictions if (r['filename'], r['subject_id']) not in excluded
                and (r['filename'], r['object_id']) not in excluded]
    alternative = build_selected_prompts(filtered, checkpoint['thresholds'], object_inventory=inventory)
    selected = []
    lookup = {(r['filename'], r['subject_id'], r['object_id']): i for i, r in enumerate(rows)}
    for filename, entry in original.items():
        for claim in entry['relations']:
            i = lookup[(filename, claim['subject_instance'], claim['object_instance'])]
            c = RELATIONS.index(claim['relation'])
            competitors = [j for j, r in enumerate(rows) if r['filename'] == filename
                           and r['object_id'] == rows[i]['object_id'] and r['subject_id'] != rows[i]['subject_id']]
            changes = [r['score_change'][c] for r in interventions if r['pair_index'] == i]
            selected.append({**claim, 'filename': filename, 'pair_index': i,
                'other_actor_count': len(competitors),
                'gap_to_best_other_actual_pair': float(scores[i,c]-scores[competitors,c].max()) if competitors else None,
                'actor_only_swap_min_change': min(changes) if changes else None,
                'actor_only_swap_max_change': max(changes) if changes else None,
                'ambiguous_region': (filename, rows[i]['object_id']) in excluded or (filename, rows[i]['subject_id']) in excluded})
    delta = torch.tensor([r['score_change'] for r in interventions])
    summary = {'variant': 'frozen_instance_evidence_development_audit_v1',
        'images': len(inventory), 'pairs': len(rows), 'actor_interventions': len(swaps),
        'mean_abs_actor_only_score_change': float(delta.abs().mean()) if len(swaps) else None,
        'mean_abs_actor_only_change_by_relation': dict(zip(RELATIONS, delta.abs().mean(0).tolist())) if len(swaps) else {},
        'ambiguous_region_pairs': len(ambiguous), 'ambiguity_iou_threshold': .9,
        'original_selected_claims': sum(len(e['relations']) for e in original.values()),
        'ablation_selected_claims': sum(len(e['relations']) for e in alternative.values()),
        'ablation_modes': dict(Counter(e['mode'] for e in alternative.values())),
        'source_sha256': sha(args.source), 'checkpoint_sha256': sha(args.checkpoint),
        'notes': 'Development diagnostic only. Subject crop alone is changed; union/object/text remain fixed. '
                 'Counterfactual features may be off-distribution and are not labelled negatives. '
                 'No training, threshold fit, actor-winner filter or benchmark improvement claim. '
                 'IoU0.9 ambiguity abstention is a new exploratory rule, not frozen R50 protocol. '
                 'Object inventory retains conflicting labels; physical-label resolution remains unsolved.'}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [('evidence_summary.json', summary), ('selected_claim_evidence.json', selected),
                        ('subject_interventions.json', interventions), ('ambiguous_regions.json', ambiguous),
                        ('ambiguity_ablation_prompt_cache.json', alternative)]:
        (output/name).write_text(json.dumps(value, indent=2), encoding='utf-8')
    if args.comparison_head:
        trial = torch.load(args.comparison_head, map_location='cpu', weights_only=True)
        if trial.get('architecture') != 'pair_only' or trial['relations'] != RELATIONS:
            raise ValueError('Expected explicit pair-only comparison architecture.')
        comparison = PairOnlyResidualAlignment(scale=trial['scale'])
        comparison.load_state_dict(trial['model'], strict=True)
        comparison.eval().requires_grad_(False)
        with torch.inference_mode():
            new_scores, new_base, new_delta = comparison(views, text)
        new_rows = [{**row, 'scores': new_scores[i].tolist(), 'clip': new_base[i].tolist(),
                     'correction': new_delta[i].tolist()} for i, row in enumerate(rows)]
        new_prompts = build_selected_prompts(new_rows, trial['thresholds'], object_inventory=inventory)
        trial_result = {'architecture': 'pair_only', 'checkpoint_sha256': sha(args.comparison_head),
                        'thresholds': trial['thresholds'],
                        'selected_claims': sum(len(e['relations']) for e in new_prompts.values()),
                        'modes': dict(Counter(e['mode'] for e in new_prompts.values())),
                        'notes': 'New development model and its train-fit thresholds; not the frozen baseline protocol.'}
        for name, value in [('pair_only_comparison_summary.json', trial_result),
                            ('pair_only_predictions.json', new_rows), ('pair_only_prompt_cache.json', new_prompts)]:
            (output/name).write_text(json.dumps(value, indent=2), encoding='utf-8')
        print('PAIR-ONLY COMPARISON', json.dumps(trial_result), flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(json.dumps(selected, indent=2), flush=True)


if __name__ == '__main__':
    main()
