"""Frozen R98 audit and fresh refit excluding the entire human-review image pool."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from captioning.actor_compatibility import pair_geometry
from captioning.evidence_routing import (EvidenceRelationVerifier, masked_verification_loss,
    fit_temperature, fit_selective_policy, selective_decisions)
from captioning.reviewed_targets import validate_target_cases, target_metrics, target_changes
from captioning.selected_relation_prompts import build_selected_prompts, entity_id
from captioning.supervised_relations import RELATIONS
from import_target_annotations import filter_annotations
from run_actor_compatibility_cached import read_archive


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def capped_selection(rows, probabilities, decisions, policy):
    scored = probabilities.clone(); scored[decisions != 1] = -2.
    prompts = build_selected_prompts([{**r, 'scores': scored[i].tolist()} for i, r in enumerate(rows)], policy['positive'])
    selected_keys = {(filename, claim['subject_instance'], claim['relation'], claim['object_instance'])
                     for filename, entry in prompts.items() for claim in entry['relations']}
    mask = torch.zeros_like(decisions, dtype=torch.bool)
    for i, row in enumerate(rows):
        subject = entity_id(row['filename'], row['subject'], row['subject_box'])
        target = entity_id(row['filename'], row['object'], row['object_box'])
        for c, relation in enumerate(RELATIONS):
            mask[i, c] = (row['filename'], subject, relation, target) in selected_keys
    return mask


def run(args):
    torch.set_num_threads(2)
    package = Path(args.package); original = Path(args.r98_directory); output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    mapping = json.loads((package / 'private_mapping.json').read_text(encoding='utf-8'))
    if sha(package / 'cases.json') != mapping['manifest_sha256'] or sha('audit/interaction_region_train_pairs.json') != mapping['source_pairs_sha256']:
        raise ValueError('Human-review manifest or pair source changed.')
    filtered = filter_annotations(json.loads(Path(args.review).read_text(encoding='utf-8-sig')), mapping)
    cases = filtered['eligible_contrasts']
    source_hash = sha(args.source)
    expected = json.loads(Path('audit/interaction_region_protocol.json').read_text())['splits']['train']['source_sha256']
    if source_hash != expected:
        raise ValueError('Expected the exact feature archive used by the annotation manifest and R98.')
    rows, views, text, lengths = read_archive(args.source, ['train_pairs.json', 'val_pairs.json'])
    n = lengths[0]; rows, views, text = rows[:n], views[:n], text[:n]
    if rows != json.loads(Path('audit/interaction_region_train_pairs.json').read_text()):
        raise ValueError('Review indices do not refer to these exact rows.')
    validate_target_cases(rows, cases)
    # Exclude every image in the125-case manifest, not just the22 eligible images.
    blocked = {rows[c[key]]['image_id'] for c in mapping['cases'] for key in ('positive_row_index', 'alternate_row_index')}
    summary = json.loads((original / 'evidence_routing_summary.json').read_text())
    manifest = json.loads((original / 'verified_evidence_manifest.json').read_text())
    if sha(original / 'evidence_verifier.pt') != manifest['files']['evidence_verifier.pt']:
        raise ValueError('Original verifier changed.')
    checkpoint = torch.load(original / 'evidence_verifier.pt', map_location='cpu', weights_only=True)
    if checkpoint['source_members_sha256'] != summary['source_members_sha256'] or checkpoint['relations'] != RELATIONS:
        raise ValueError('Original feature source/predicate provenance differs.')
    if checkpoint['fit_images'] != summary['fit_images'] or checkpoint['calibration_images'] != summary['calibration_images']:
        raise ValueError('Original checkpoint split provenance differs.')
    if summary['seed'] != 42 or summary['epochs'] != 30:
        raise ValueError('This comparison requires the fixed R98 schedule.')
    train_ids, cal_ids = set(summary['fit_images']), set(summary['calibration_images'])
    if train_ids & cal_ids or train_ids | cal_ids != {r['image_id'] for r in rows}:
        raise ValueError('Original fit/calibration partition differs.')
    fit = torch.tensor([i for i, r in enumerate(rows) if r['image_id'] in train_ids - blocked])
    cal = torch.tensor([i for i, r in enumerate(rows) if r['image_id'] in cal_ids - blocked])
    if not len(fit) or not len(cal):
        raise ValueError('Insufficient images after excluding review pool.')
    if any(rows[int(i)]['image_id'] in blocked for i in torch.cat([fit, cal])):
        raise ValueError('Human evaluation image leaked into refit/calibration.')
    geometry = pair_geometry(rows); labels = torch.tensor([r['labels'] for r in rows])
    protocol = {'variant': 'reviewed_target_evidence_check_v1', 'seed': 42, 'epochs': 30,
        'learning_rate': .001, 'weight_decay': .01, 'human_labels_used_for_training_or_calibration': 0,
        'review_sha256': sha(args.review), 'manifest_sha256': mapping['manifest_sha256'],
        'source_zip_sha256': source_hash, 'r98_checkpoint_sha256': sha(original / 'evidence_verifier.pt'),
        'blocked_images': sorted(blocked), 'fit_images': sorted(train_ids - blocked),
        'calibration_images': sorted(cal_ids - blocked), 'fit_pairs': len(fit), 'calibration_pairs': len(cal),
        'notes': 'Fixed R98 architecture/schedule. All review-manifest images excluded, including uncompleted/conflicting cases. '
                 'Eligible human cases evaluation-only; development reuse, not independent benchmark.'}
    protocol_path = output / 'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise ValueError('Protocol changed; use a separate output.')
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    frozen = EvidenceRelationVerifier(); frozen.load_state_dict(checkpoint['model'], strict=True)
    frozen.eval().requires_grad_(False)
    with torch.no_grad(): familiar, _ = frozen(views, geometry, text)
    torch.manual_seed(42); isolated = EvidenceRelationVerifier()
    optimizer = torch.optim.AdamW(isolated.parameters(), lr=.001, weight_decay=.01)
    history = []
    for epoch in range(30):
        isolated.train(); optimizer.zero_grad()
        logits, _ = isolated(views[fit], geometry[fit], text[fit])
        loss = masked_verification_loss(logits, labels[fit])
        if not torch.isfinite(loss): raise ValueError('Nonfinite refit loss.')
        loss.backward(); torch.nn.utils.clip_grad_norm_(isolated.parameters(), 1); optimizer.step()
        history.append({'epoch': epoch + 1, 'known_bce': float(loss.detach())})
        print(history[-1], flush=True)
    isolated.eval().requires_grad_(False)
    with torch.no_grad(): isolated_logits, _ = isolated(views, geometry, text)
    baseline = 10 * torch.einsum('bd,bkd->bk', views[:, 2], text) - 3
    results, per_case = {}, {}
    conditions = [('r98_familiar_verifier', familiar, checkpoint['temperature'], checkpoint['policy'])]
    policies = {}
    for name, logits in [('isolated_clip', baseline), ('isolated_verifier', isolated_logits)]:
        temperature = fit_temperature(logits[cal], labels[cal])
        policy = fit_selective_policy((logits[cal] / temperature).sigmoid(), labels[cal])
        policies[name] = {'temperature': temperature, 'policy': policy}
        conditions.append((name, logits, temperature, policy))
    for name, logits, temperature, policy in conditions:
        probabilities = (logits / temperature).sigmoid()
        decisions = selective_decisions(probabilities, torch.ones_like(probabilities, dtype=torch.bool), policy)
        selected = capped_selection(rows, probabilities, decisions, policy)
        results[name], per_case[name] = target_metrics(logits, decisions, rows, cases, selected)
    reviewed_ids = {rows[c['positive_row_index']]['image_id'] for c in cases}
    paired_changes, private_changes = target_changes(per_case['isolated_clip'], per_case['isolated_verifier'])
    result = {'variant': protocol['variant'], 'protocol_sha256': sha(protocol_path),
        'eligible_cases': len(cases), 'eligible_images': len(reviewed_ids), 'review_filter_counts': filtered['counts'],
        'manifest_images_excluded': len(blocked), 'fit_pairs': len(fit), 'calibration_pairs': len(cal),
        'eligible_overlap_with_original_r98': {'fit_images': len(reviewed_ids & train_ids), 'calibration_images': len(reviewed_ids & cal_ids)},
        'eligible_overlap_with_isolated_fit_or_calibration': 0,
        'human_labels_used_for_training_or_calibration': 0, 'conditions': results,
        'paired_target_ranking_changes': paired_changes,
        'promotion_allowed': False,
        'limitations': ['Single self-reported human reviewer; source-positive conflicts/both-supported/unclear cases excluded.',
            'R98 familiar condition saw original positive image labels; report separately, not heldout accuracy.',
            'Refit excludes all61 manifest images but still reuses a previously inspected development pool.',
            'Known-label coverage/selection conditional on existing matched detector candidates.',
            'No decoder adaptation, new caption metrics, independent benchmark or novelty claim.']}
    for name, value in [('reviewed_target_summary.json', result), ('history.json', history), ('policies.json', policies),
                        ('private_case_predictions.json', per_case), ('private_target_changes.json', private_changes)]:
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    torch.save({'architecture': 'EvidenceRelationVerifier', 'dim': 512, 'hidden': 64,
                'model': isolated.state_dict(), **policies['isolated_verifier'], 'relations': RELATIONS,
                'fit_images': protocol['fit_images'], 'calibration_images': protocol['calibration_images'],
                'protocol_sha256': sha(protocol_path)}, output / 'isolated_evidence_verifier.pt')
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('source', 'package', 'review', 'r98-directory', 'output'):
        parser.add_argument('--' + name, required=True)
    run(parser.parse_args())
