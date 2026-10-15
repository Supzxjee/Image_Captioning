"""Nested image-group development CV for separate action and target heads."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from captioning.actor_compatibility import pair_geometry
from captioning.evidence_routing import masked_verification_loss, fit_temperature, fit_selective_policy, selective_decisions
from captioning.factorized_evidence import (actor_examples, ActionSupportHead, target_features,
    ConditionalTargetHead, reviewed_labels, target_objective, image_folds, combine_decisions, relative_target_prior)
from captioning.reviewed_targets import validate_target_cases, target_metrics, target_changes
from captioning.supervised_relations import RELATIONS
from import_target_annotations import filter_annotations
from run_actor_compatibility_cached import read_archive
from run_reviewed_target_evidence import capped_selection


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shared_target_policy(logits, labels):
    known = labels >= 0
    temperature = fit_temperature(logits, labels)
    # Small calibration fold: a shared conditional-target threshold, not six
    # separately tuned rules or a guarantee of per-predicate calibration.
    policy = fit_selective_policy((logits[known] / temperature).sigmoid().reshape(-1, 1),
                                  labels[known].reshape(-1, 1), precision=.9, min_known=5)
    expanded = {**policy, 'positive': policy['positive'] * 6, 'negative': policy['negative'] * 6}
    return temperature, policy, expanded


def run(args):
    torch.set_num_threads(2)
    output = Path(args.output); package = Path(args.package)
    output.mkdir(parents=True, exist_ok=True)
    base_protocol_path = Path(args.r99_directory) / 'protocol.json'
    base = json.loads(base_protocol_path.read_text())
    prior_result = json.loads(Path('audit/reviewed_target_evidence_v1_result.json').read_text())
    if sha(base_protocol_path) != prior_result['protocol_sha256'] or sha(args.source) != base['source_zip_sha256'] or sha(args.review) != base['review_sha256']:
        raise ValueError('Frozen R99 source/protocol/review changed.')
    mapping = json.loads((package / 'private_mapping.json').read_text())
    if sha(package / 'cases.json') != mapping['manifest_sha256'] or sha('audit/interaction_region_train_pairs.json') != mapping['source_pairs_sha256']:
        raise ValueError('Annotation source changed.')
    filtered = filter_annotations(json.loads(Path(args.review).read_text(encoding='utf-8-sig')), mapping)
    cases = filtered['eligible_contrasts']
    rows, views, text, lengths = read_archive(args.source, ['train_pairs.json', 'val_pairs.json'])
    n = lengths[0]; rows, views, text = rows[:n], views[:n], text[:n]
    if rows != json.loads(Path('audit/interaction_region_train_pairs.json').read_text()):
        raise ValueError('Human-review row identities differ.')
    validate_target_cases(rows, cases)
    folds = image_folds(rows, cases, count=5, seed=42)
    actor_views, actor_labels, actor_keys, row_actor = actor_examples(rows, views)
    blocked = set(base['blocked_images'])
    actual_blocked = {rows[c[k]]['image_id'] for c in mapping['cases'] for k in ('positive_row_index', 'alternate_row_index')}
    if actual_blocked != blocked:
        raise ValueError('Review image exclusion pool differs.')
    afit = torch.tensor([i for i, k in enumerate(actor_keys) if k[0] in base['fit_images']])
    acal = torch.tensor([i for i, k in enumerate(actor_keys) if k[0] in base['calibration_images']])
    if not len(afit) or not len(acal) or any(actor_keys[int(i)][0] in blocked for i in torch.cat([afit, acal])):
        raise ValueError('Review images leaked into action fit/calibration.')
    relative = args.relative_targets
    protocol = {'variant': 'factorized_action_target_relative_nested_cv_v2' if relative else 'factorized_action_target_nested_cv_v1', 'seed': 42, 'folds': folds,
        'source_sha256': sha(args.source), 'review_sha256': sha(args.review), 'r99_protocol_sha256': sha(base_protocol_path),
        'action_fit_actors': len(afit), 'action_calibration_actors': len(acal), 'all61_review_images_excluded_from_action_fit_calibration': True,
        'action_epochs': 30, 'action_lr': .001, 'target_epochs': 30, 'target_lr': .01,
        'weight_decay': .01, 'target_correction_bound': .5, 'ranking_weight': .2,
        'threshold_precision': .9, 'minimum_known_tail': 5,
        'target_threshold': 'global across predicates, fit only on each inner calibration image fold',
        'conditions': ['clip_prior', 'conditional_bce', 'conditional_bce_rank'],
        'notes': 'Fixed pilot, no best-fold/epoch/seed selection. Same target architecture/data/calibration for both learned arms; '
                 'only rank-loss weight differs. Reused single-reviewer development; cross-validation, not independent benchmark.'}
    if relative:
        protocol['target_reference'] = 'Subtract frozen CLIP mean across unique targets of same detector subject/predicate before adding learned correction. No labels, no forced single target.'
    protocol_path = output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Changed protocol; use a new output.')
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    torch.manual_seed(42); action = ActionSupportHead()
    optimizer = torch.optim.AdamW(action.parameters(), lr=.001, weight_decay=.01)
    action_history = []
    for epoch in range(30):
        optimizer.zero_grad(); loss = masked_verification_loss(action(actor_views[afit]), actor_labels[afit])
        if not torch.isfinite(loss): raise ValueError('Nonfinite actor loss.')
        loss.backward(); torch.nn.utils.clip_grad_norm_(action.parameters(), 1); optimizer.step()
        action_history.append(float(loss.detach()))
    action.eval().requires_grad_(False)
    with torch.no_grad(): actor_logits = action(actor_views)
    action_temperature = fit_temperature(actor_logits[acal], actor_labels[acal])
    actor_probabilities = (actor_logits / action_temperature).sigmoid()
    action_policy = fit_selective_policy(actor_probabilities[acal], actor_labels[acal])
    actor_decisions = selective_decisions(actor_probabilities, torch.ones_like(actor_probabilities).bool(), action_policy)
    action_prob, action_dec = actor_probabilities[row_actor], actor_decisions[row_actor]
    torch.save({'model': action.state_dict(), 'relations': RELATIONS, 'temperature': action_temperature,
                'policy': action_policy, 'fit_actor_keys': [actor_keys[int(i)] for i in afit],
                'calibration_actor_keys': [actor_keys[int(i)] for i in acal], 'protocol_sha256': sha(protocol_path)}, output / 'action_head.pt')
    features = target_features(views, pair_geometry(rows), text)
    frozen_prior = 10 * features[:, :, 2] - 3
    target_reference = relative_target_prior(frozen_prior, rows) - frozen_prior if relative else torch.zeros_like(frozen_prior)
    actor_truth = {(row_actor[c['positive_row_index']].item(), c['relation_index']) for c in cases}
    actor_review_summary = {'known_true_actor_actions': len(actor_truth),
        'supported': sum(int(actor_decisions[i, c]) == 1 for i, c in actor_truth),
        'contradicted_prediction': sum(int(actor_decisions[i, c]) == 0 for i, c in actor_truth),
        'abstained': sum(int(actor_decisions[i, c]) == -1 for i, c in actor_truth),
        'notes': 'Reviewed contrasts confirm action-positive actors only; this is not action precision.'}
    conditions = protocol['conditions']
    oof = {name: {'scores': torch.zeros(n, 6), 'target': torch.full((n, 6), -1),
                  'joint': torch.full((n, 6), -1), 'capped': torch.zeros(n, 6).bool()} for name in conditions}
    coverage = torch.zeros(n).long(); fold_results = []; histories = {}
    for fold in folds:
        train_images, cal_images, test_images = (set(fold[k]) for k in ('fit_images', 'calibration_images', 'evaluation_images'))
        split_cases = {name: [c for c in cases if rows[c['positive_row_index']]['image_id'] in images]
                       for name, images in [('fit', train_images), ('cal', cal_images), ('test', test_images)]}
        cal_labels = reviewed_labels(rows, split_cases['cal'])
        test_rows = torch.tensor([i for i, r in enumerate(rows) if r['image_id'] in test_images])
        coverage[test_rows] += 1
        for name in conditions:
            torch.manual_seed(42); target = ConditionalTargetHead()
            history = []
            if name != 'clip_prior':
                optimizer = torch.optim.AdamW(target.parameters(), lr=.01, weight_decay=.01)
                for epoch in range(30):
                    optimizer.zero_grad(); logits, _, _ = target(features)
                    logits = logits + target_reference
                    loss, bce, ranking = target_objective(logits, rows, split_cases['fit'], ranking_weight=.2 if name.endswith('_rank') else 0)
                    if not torch.isfinite(loss): raise ValueError('Nonfinite target loss.')
                    loss.backward(); torch.nn.utils.clip_grad_norm_(target.parameters(), 1); optimizer.step()
                    history.append({'epoch': epoch + 1, 'loss': float(loss.detach()), 'bce': float(bce.detach()), 'rank': float(ranking.detach())})
            target.eval().requires_grad_(False)
            with torch.no_grad(): logits, _, corrections = target(features)
            logits = logits + target_reference
            temperature, scalar_policy, policy = shared_target_policy(logits, cal_labels)
            target_prob = (logits / temperature).sigmoid()
            target_dec = selective_decisions(target_prob, torch.ones_like(target_prob).bool(), policy)
            joint = combine_decisions(action_dec, target_dec)
            joint_prob = action_prob * target_prob
            # Explicit joint decision precedes capping; cap priority is the product
            # score, not a fake six-predicate calibrated probability guarantee.
            capped = capped_selection(rows, joint_prob, joint, {'positive': [0.] * 6})
            for key, value in [('scores', logits), ('target', target_dec), ('joint', joint), ('capped', capped)]:
                oof[name][key][test_rows] = value[test_rows]
            fold_metric, _ = target_metrics(logits, joint, rows, split_cases['test'], capped)
            fold_results.append({'fold': fold['fold'], 'condition': name,
                'fit_cases': len(split_cases['fit']), 'calibration_cases': len(split_cases['cal']), 'evaluation_cases': len(split_cases['test']),
                'target_temperature': temperature, 'target_policy': scalar_policy,
                'mean_abs_test_correction': float(corrections[test_rows].abs().mean()), 'evaluation': fold_metric})
            histories[f"fold{fold['fold']}_{name}"] = history
            torch.save({'model': target.state_dict(), 'relations': RELATIONS, 'correction_bound': .5,
                        'temperature': temperature, 'policy': policy, 'fold': fold, 'condition': name,
                        'relative_targets': relative,
                        'protocol_sha256': sha(protocol_path)}, output / f"target_fold{fold['fold']}_{name}.pt")
            print(f"fold {fold['fold']} {name}: {fold_metric['all']['ranking_wins']}/{len(split_cases['test'])} target rankings", flush=True)
    evaluation_rows = [i for i, r in enumerate(rows) if any(r['image_id'] in set(f['evaluation_images']) for f in folds)]
    if not (coverage[evaluation_rows] == 1).all(): raise ValueError('Out-of-fold coverage differs.')
    results, details = {}, {}
    for name, values in oof.items():
        target_only, _ = target_metrics(values['scores'], values['target'], rows, cases)
        joint, detail = target_metrics(values['scores'], values['joint'], rows, cases, values['capped'])
        results[name] = {'target_only': target_only, 'action_and_target': joint}
        details[name] = detail
    changes, private_changes = target_changes(details['clip_prior'], details['conditional_bce_rank'])
    reviewed_truth = reviewed_labels(rows, cases)
    source_labels = torch.tensor([r['labels'] for r in rows])
    candidate_counts = {}
    for name, values in oof.items():
        mask = values['capped']; unreviewed = mask & (reviewed_truth < 0)
        candidate_counts[name] = {'all_selected_claims': int(mask.sum()),
            'reviewed_supported': int((mask & (reviewed_truth == 1)).sum()),
            'reviewed_contradicted': int((mask & (reviewed_truth == 0)).sum()),
            'unreviewed_source_positive': int((unreviewed & (source_labels == 1)).sum()),
            'unreviewed_source_zero': int((unreviewed & (source_labels == 0)).sum()),
            'unreviewed_source_unknown': int((unreviewed & (source_labels < 0)).sum())}
    result = {'variant': protocol['variant'], 'protocol_sha256': sha(protocol_path), 'eligible_cases': len(cases),
        'eligible_images': len({rows[c['positive_row_index']]['image_id'] for c in cases}), 'review_filter_counts': filtered['counts'],
        'within_each_fold_fit_calibration_evaluation_image_overlap': 0, 'each_reviewed_case_evaluated_out_of_fold_once': True,
        'actor_review_summary': actor_review_summary, 'action_policy': action_policy, 'action_temperature': action_temperature,
        'conditions': results, 'clip_vs_rank_paired_changes': changes, 'production_promotion': False,
        'candidate_pool_selected_claim_counts': candidate_counts,
        'limitations': ['Reused single-reviewer22-image development pool; nested CV is not independent scientific confirmation.',
            'Target thresholds have very few calibration claims; no per-predicate calibration guarantee.',
            'Known correct actor actions provide no reviewed action-negative precision estimate.',
            'Classification control and ranking arm both receive human target labels; label benefit and separation itself are not isolated by CLIP comparison.',
            'Multiple targets may be supported; unknown alternatives are not negatives.',
            'No decoder adaptation, reranking, caption metrics or novelty claim.']}
    for name, data in [('factorized_evidence_summary.json', result), ('fold_results.json', fold_results),
                       ('private_case_predictions.json', details), ('private_target_changes.json', private_changes),
                       ('history.json', {'action': action_history, 'target': histories})]:
        (output / name).write_text(json.dumps(data, indent=2), encoding='utf-8')
    torch.save(oof, output / 'private_out_of_fold_tensors.pt')
    print(json.dumps({'actor': actor_review_summary, 'results': {
        name: {'rank': x['action_and_target']['all'], 'same_class': x['action_and_target']['same_class'],
               'capped': x['action_and_target']['unique_claims_after_cap']} for name, x in results.items()}}, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('source', 'package', 'review', 'r99-directory', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--relative-targets', action='store_true')
    run(parser.parse_args())
