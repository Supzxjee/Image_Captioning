"""Fixed R101 folds: actual anatomical evidence vs matched box-anchor control."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import torch
from torch import nn
from captioning.pose_evidence import pair_features, subject_key, FEATURE_NAMES
from captioning.factorized_evidence import actor_examples, ActionSupportHead, reviewed_labels, relative_target_prior, target_objective, combine_decisions
from captioning.reviewed_targets import target_metrics, target_changes
from captioning.supervised_relations import RELATIONS
from captioning.evidence_routing import selective_decisions
from run_factorized_evidence_cv import shared_target_policy
from run_reviewed_target_evidence import capped_selection
from run_actor_compatibility_cached import read_archive
from import_target_annotations import filter_annotations


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PoseCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(len(FEATURE_NAMES), 2)
        nn.init.zeros_(self.linear.weight); nn.init.zeros_(self.linear.bias)

    def forward(self, features, available):
        value = .5 * self.linear(features.detach()).tanh() * available[:, None]
        output = value.new_zeros((len(features), 6)); output[:, 1:3] = value
        return output


def run(args):
    torch.set_num_threads(2)
    package, pose_dir, reference, output = map(Path, (args.package, args.poses, args.r101_directory, args.output))
    output.mkdir(parents=True, exist_ok=True)
    source = json.loads((reference/'protocol.json').read_text())
    reference_summary = json.loads((reference/'factorized_evidence_summary.json').read_text())
    if sha(reference/'protocol.json') != reference_summary['protocol_sha256'] or sha(args.source) != source['source_sha256'] or sha(args.review) != source['review_sha256']:
        raise ValueError('Frozen R101 source/folds/review changed.')
    mapping = json.loads((package/'private_mapping.json').read_text())
    if sha(package/'cases.json') != mapping['manifest_sha256'] or sha('audit/interaction_region_train_pairs.json') != mapping['source_pairs_sha256']:
        raise ValueError('Annotation source changed.')
    cases = filter_annotations(json.loads(Path(args.review).read_text(encoding='utf-8-sig')), mapping)['eligible_contrasts']
    focus = [c for c in cases if c['relation_index'] in (1, 2)]
    extraction = json.loads((pose_dir/'extraction_summary.json').read_text())
    if sha(pose_dir/'private_subject_poses.json') != extraction['pose_cache_sha256'] or sha(pose_dir/'extraction_protocol.json') != extraction['protocol_sha256']:
        raise ValueError('Pose cache/protocol changed.')
    poses = json.loads((pose_dir/'private_subject_poses.json').read_text())
    pose_protocol = json.loads((pose_dir/'extraction_protocol.json').read_text())
    if pose_protocol['source_pairs_sha256'] != mapping['source_pairs_sha256']:
        raise ValueError('Pose/annotation source mismatch.')
    rows, views, text, lengths = read_archive(args.source, ['train_pairs.json', 'val_pairs.json'])
    n = lengths[0]; rows, views, text = rows[:n], views[:n], text[:n]
    if rows != json.loads(Path('audit/interaction_region_train_pairs.json').read_text()):
        raise ValueError('Pose evaluation indices changed.')
    true_features, proxy_features, availability = [], [], []
    for row in rows:
        key = subject_key(row['filename'], row['subject_box'])
        entry = poses.get(key, {'status': 'unavailable'})
        real, valid = pair_features(entry, row['subject_box'], row['object_box'])
        proxy, proxy_valid = pair_features(entry, row['subject_box'], row['object_box'], proxy=True)
        if valid != proxy_valid: raise ValueError('Proxy and real coverage differ.')
        true_features.append(real); proxy_features.append(proxy); availability.append(valid)
    true_features, proxy_features = torch.tensor(true_features), torch.tensor(proxy_features)
    available = torch.tensor(availability).float()
    if any(availability[c['positive_row_index']] != availability[c['alternate_row_index']] for c in focus):
        raise ValueError('Pose availability must hold subject fixed.')
    raw_prior = 10 * torch.einsum('bd,bkd->bk', views[:, 2], text) - 3
    prior = relative_target_prior(raw_prior, rows)
    actor_features, _, _, row_actor = actor_examples(rows, views)
    cp = torch.load(reference/'action_head.pt', weights_only=True, map_location='cpu')
    action = ActionSupportHead(); action.load_state_dict(cp['model'], strict=True); action.eval().requires_grad_(False)
    with torch.no_grad(): actor_probs = (action(actor_features)/cp['temperature']).sigmoid()
    actor_decisions = selective_decisions(actor_probs, torch.ones_like(actor_probs).bool(), cp['policy'])[row_actor]
    actor_probs = actor_probs[row_actor]
    protocol = {'variant': 'pose_target_nested_cv_v1', 'folds': source['folds'], 'source_sha256': sha(args.source),
        'review_sha256': sha(args.review), 'r101_protocol_sha256': sha(reference/'protocol.json'),
        'pose_cache_sha256': extraction['pose_cache_sha256'], 'pose_model_sha256': pose_protocol['model_sha256'],
        'action_checkpoint_sha256': sha(reference/'action_head.pt'), 'focus_relations': ['riding', 'sitting_on'],
        'feature_names': FEATURE_NAMES, 'epochs': 30, 'learning_rate': .01, 'weight_decay': .01,
        'ranking_weight': .2, 'correction_bound': .5,
        'calibration': 'All original inner calibration cases; same90% known-precision/5-known tail rule as R101.',
        'controls': 'Fixed detector-box anchors use same availability, landmark visibility and matchingIoU as true poses.',
        'missing': 'Exactly zero correction; retain prior; no missing-pose negative label.'}
    protocol = json.loads(json.dumps(protocol))
    path = output/'protocol.json'
    if path.exists() and json.loads(path.read_text()) != protocol: raise ValueError('Changed protocol; use new output.')
    path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    conditions = ('clip_prior', 'box_anchor_control', 'pose_evidence')
    oof = {name: {'scores': torch.zeros(n, 6), 'target': torch.full((n, 6), -1),
        'joint': torch.full((n, 6), -1), 'capped': torch.zeros(n, 6).bool()} for name in conditions}
    fold_results, histories = [], {}
    coverage = torch.zeros(n).long()
    for fold in source['folds']:
        fit_images, cal_images, test_images = (set(fold[k]) for k in ('fit_images', 'calibration_images', 'evaluation_images'))
        if fit_images & cal_images or (fit_images | cal_images) & test_images: raise ValueError('Image leakage.')
        fit_cases = [c for c in focus if rows[c['positive_row_index']]['image_id'] in fit_images]
        cal_cases = [c for c in cases if rows[c['positive_row_index']]['image_id'] in cal_images]
        test_rows = torch.tensor([i for i, r in enumerate(rows) if r['image_id'] in test_images])
        coverage[test_rows] += 1
        for name in conditions:
            torch.manual_seed(42); head = PoseCorrection(); history = []
            features = true_features if name == 'pose_evidence' else proxy_features
            if name != 'clip_prior' and fit_cases:
                optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.01)
                for epoch in range(30):
                    optimizer.zero_grad(); scores = prior + head(features, available)
                    loss, bce, rank = target_objective(scores, rows, fit_cases, ranking_weight=.2)
                    if not torch.isfinite(loss): raise ValueError('Nonfinite pose loss.')
                    loss.backward(); torch.nn.utils.clip_grad_norm_(head.parameters(), 1); optimizer.step()
                    history.append({'epoch': epoch+1, 'loss': float(loss.detach()), 'bce': float(bce.detach()), 'rank': float(rank.detach())})
            head.eval().requires_grad_(False)
            with torch.no_grad(): correction = head(features, available); scores = prior + correction
            if not torch.equal(scores[~available.bool()], prior[~available.bool()]):
                raise ValueError('Missing pose changed the prior.')
            temperature, scalar_policy, policy = shared_target_policy(scores, reviewed_labels(rows, cal_cases))
            probabilities = (scores/temperature).sigmoid()
            decisions = selective_decisions(probabilities, torch.ones_like(probabilities).bool(), policy)
            joint = combine_decisions(actor_decisions, decisions)
            capped = capped_selection(rows, probabilities*actor_probs, joint, {'positive': [0.] * 6})
            for key, value in [('scores', scores), ('target', decisions), ('joint', joint), ('capped', capped)]:
                oof[name][key][test_rows] = value[test_rows]
            torch.save({'model': head.state_dict(), 'condition': name, 'fold': fold, 'temperature': temperature,
                        'policy': policy, 'protocol_sha256': sha(path)}, output/f"pose_fold{fold['fold']}_{name}.pt")
            fold_results.append({'fold': fold['fold'], 'condition': name, 'fit_focus_cases': len(fit_cases),
                'target_temperature': temperature, 'target_policy': scalar_policy})
            histories[f"{fold['fold']}_{name}"] = history
        print(f"Pose fold {fold['fold']} complete; {len(fit_cases)} focus training contrasts", flush=True)
    eval_images = {rows[c['positive_row_index']]['image_id'] for c in cases}
    if not all(int(coverage[i]) == 1 for i, r in enumerate(rows) if r['image_id'] in eval_images): raise ValueError('OOF coverage differs.')
    results, details = {}, {}
    known = reviewed_labels(rows, cases)
    for name, values in oof.items():
        full, detail = target_metrics(values['scores'], values['joint'], rows, cases, values['capped'])
        focussed, _ = target_metrics(values['scores'], values['joint'], rows, focus, values['capped'])
        target_only, _ = target_metrics(values['scores'], values['target'], rows, focus)
        results[name] = {'focus': focussed, 'all_reviewed': full, 'focus_target_only': target_only,
                        'total_candidate_claims_selected': int(values['capped'].sum()),
                        'selected_without_human_label': int((values['capped'] & (known < 0)).sum())}
        details[name] = detail
    changes, private_changes = target_changes(details['box_anchor_control'], details['pose_evidence'])
    subjects = {subject_key(rows[c['positive_row_index']]['filename'], rows[c['positive_row_index']]['subject_box']) for c in focus}
    result = {'variant': protocol['variant'], 'protocol_sha256': sha(path), 'focus_cases': len(focus),
        'focus_images': len({rows[c['positive_row_index']]['image_id'] for c in focus}), 'focus_subjects': len(subjects),
        'focus_pose_statuses': dict(Counter(poses[k]['status'] for k in subjects)),
        'focus_cases_with_pose': sum(availability[c['positive_row_index']] for c in focus),
        'conditions': results, 'box_vs_pose_ranking_changes': changes, 'production_promotion': False,
        'limitations': ['Fixed reused development folds and one human reviewer; not an independent benchmark.',
            'Pose is a predicted geometry cue, not contact/action truth. Anatomical matching can still fail.',
            'Proxy and pose share availability/visibility/matching quality; comparison isolates coordinates/angles given those masks.',
            'Only riding/sitting_on are trained; other predicates retain the frozen prior.',
            'Same frozen action gate as R101; actor support and target attachment must be reported separately.',
            'No decoder adaptation, caption metric improvement or novelty claim.']}
    for name, data in [('pose_target_summary.json', result), ('private_case_predictions.json', details),
                       ('private_target_changes.json', private_changes), ('fold_results.json', fold_results), ('history.json', histories)]:
        (output/name).write_text(json.dumps(data, indent=2), encoding='utf-8')
    torch.save(oof, output/'private_out_of_fold_tensors.pt')
    print(json.dumps({k: v for k, v in result.items() if k != 'conditions'}, indent=2), flush=True)
    for name, value in results.items(): print(name, json.dumps(value['focus'], indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('source', 'package', 'review', 'poses', 'r101-directory', 'output'): parser.add_argument('--'+name, required=True)
    run(parser.parse_args())
