"""Train/calibrate the first evidence-routing stage on already extracted features."""
import argparse
import hashlib
import json
import random
from pathlib import Path
import torch
from torch.nn import functional as F
from captioning.actor_compatibility import pair_geometry, actor_edge_metrics
from captioning.supervised_relations import RELATIONS
from captioning.evidence_routing import (EvidenceRelationVerifier, masked_verification_loss,
    fit_temperature, fit_selective_policy, selective_decisions, verified_prompt_memory)
from run_actor_compatibility_cached import read_archive
from captioning.feature_archive import open_feature_archive


def summarize(probabilities, labels, decisions):
    known = labels >= 0
    positive, negative = decisions == 1, decisions == 0
    tp, fp = int((positive & (labels == 1)).sum()), int((positive & (labels == 0)).sum())
    tn, fn = int((negative & (labels == 0)).sum()), int((negative & (labels == 1)).sum())
    return {'known_labels': int(known.sum()), 'known_positive': int((labels == 1).sum()),
        'supported_tp': tp, 'supported_fp': fp, 'supported_known_precision': tp / (tp + fp) if tp + fp else None,
        'supported_known_recall': tp / int((labels == 1).sum()) if (labels == 1).any() else None,
        'contradicted_tn': tn, 'contradicted_fn': fn,
        'known_decision_coverage': float(((decisions >= 0) & known).sum() / known.sum()) if known.any() else None,
        'unknown_selected_supported': int((positive & ~known).sum()),
        'unknown_selected_contradicted': int((negative & ~known).sum()),
        'known_brier': float((probabilities[known] - labels[known].float()).square().mean()) if known.any() else None,
        'known_nll': float(F.binary_cross_entropy(probabilities[known].clamp(1e-7, 1-1e-7), labels[known].float())) if known.any() else None}


def run(source, output, epochs=30, seed=42):
    if epochs < 1:
        raise ValueError('Epochs must be positive.')
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rows, views, text, lengths = read_archive(source, ['train_pairs.json', 'val_pairs.json'])
    n = lengths[0]
    if not n or n == len(rows):
        raise ValueError('Need source train and evaluation rows.')
    if any(r['caption_split'] not in ('train', 'restval') for r in rows):
        raise ValueError('Do not train this development prototype on caption val/test.')
    train_ids, dev_ids = {r['image_id'] for r in rows[:n]}, {r['image_id'] for r in rows[n:]}
    if train_ids & dev_ids:
        raise ValueError('Source train/evaluation images overlap.')
    ids = sorted(train_ids)
    if len(ids) < 8:
        raise ValueError('At least eight source training images required.')
    random.Random(seed).shuffle(ids)
    cal_ids = set(ids[:max(2, len(ids) // 4)])
    fit = torch.tensor([i for i in range(n) if rows[i]['image_id'] not in cal_ids])
    cal = torch.tensor([i for i in range(n) if rows[i]['image_id'] in cal_ids])
    dev = torch.arange(n, len(rows))
    labels, geometry = torch.tensor([r['labels'] for r in rows]), pair_geometry(rows)
    model = EvidenceRelationVerifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    history = []
    for epoch in range(epochs):
        model.train(); optimizer.zero_grad()
        logits, _ = model(views[fit], geometry[fit], text[fit])
        loss = masked_verification_loss(logits, labels[fit])
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss.')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
        history.append({'epoch': epoch + 1, 'known_bce': float(loss.detach())})
        print(history[-1], flush=True)
    model.eval().requires_grad_(False)
    with torch.no_grad():
        logits, features = model(views, geometry, text)
    base = 10 * torch.einsum('bd,bkd->bk', views[:, 2], text) - 3
    comparisons = {}; saved = None
    for name, values in [('frozen_clip', base), ('instance_evidence', logits)]:
        temperature = fit_temperature(values[cal], labels[cal])
        probabilities = (values / temperature).sigmoid()
        policy = fit_selective_policy(probabilities[cal], labels[cal], precision=.9, min_known=5)
        decisions = selective_decisions(probabilities, torch.ones_like(labels, dtype=torch.bool), policy)
        comparisons[name] = {'temperature': temperature, 'policy': policy,
            'calibration': summarize(probabilities[cal], labels[cal], decisions[cal]),
            'evaluation': summarize(probabilities[dev], labels[dev], decisions[dev]),
            'evaluation_actor_contrasts': actor_edge_metrics(values[dev], rows[n:], exhaustive=True)}
        comparisons[name]['evaluation_by_predicate'] = {
            relation: summarize(probabilities[dev, c:c+1], labels[dev, c:c+1], decisions[dev, c:c+1])
            for c, relation in enumerate(RELATIONS)}
        selected_prompts, _ = verified_prompt_memory(rows[n:], features[n:], probabilities[n:], decisions[n:], policy)
        from captioning.selected_relation_prompts import entity_id
        selected_keys = {(filename, claim['subject_instance'], claim['relation'], claim['object_instance'])
                         for filename, entry in selected_prompts.items() for claim in entry['relations']}
        capped = torch.full_like(decisions[dev], -1)
        for i, row in enumerate(rows[n:]):
            s = entity_id(row['filename'], row['subject'], row['subject_box'])
            o = entity_id(row['filename'], row['object'], row['object_box'])
            for c, relation in enumerate(RELATIONS):
                if (row['filename'], s, relation, o) in selected_keys:
                    capped[i, c] = 1
        comparisons[name]['after_shared_three_claim_cap'] = summarize(probabilities[dev], labels[dev], capped)
        if name == 'instance_evidence':
            saved = probabilities, decisions, temperature, policy
    probabilities, decisions, temperature, policy = saved
    prompts, memory = verified_prompt_memory(rows[n:], features[n:], probabilities[n:], decisions[n:], policy)
    # Stable manifest for ZIP and Kaggle's already-extracted input directory.
    with open_feature_archive(source) as archive:
        source_members = {}
        for name in ('manifest.json', 'features_0.pt', 'features_1.pt', 'data/train_pairs.json',
                     'data/val_pairs.json', 'data/pair_summary.json'):
            matches = [member for member in archive.namelist() if member == name or member.endswith('/' + name)]
            if len(matches) != 1:
                raise ValueError('Ambiguous or missing source member: ' + name)
            source_members[name] = hashlib.sha256(archive.read(matches[0])).hexdigest()
    source_hash = hashlib.sha256(json.dumps(source_members, sort_keys=True).encode()).hexdigest()
    result = {'variant': 'selective_instance_evidence_routing_v1', 'stage': 'verifier_calibration_and_prompt_memory',
        'source_members_sha256': source_hash, 'source_members': source_members,
        'seed': seed, 'epochs': epochs,
        'fit_pairs': len(fit), 'calibration_pairs': len(cal), 'evaluation_pairs': len(dev),
        'fit_images': sorted(train_ids - cal_ids), 'calibration_images': sorted(cal_ids), 'evaluation_images': sorted(dev_ids),
        'image_overlap': 0, 'comparisons': comparisons,
        'exported_claims': sum(len(e['relations']) for e in prompts.values()),
        'exported_images_with_claims': sum(bool(e['relations']) for e in prompts.values()),
        'decoder_router_trained': False, 'caption_metrics_measured': False,
        'limitations': ['Previously inspected development pools; image separation is not a new independent benchmark.',
            'Localization is conditional on existing class/IoU-matched detector boxes; no localization confidence calibration.',
            'Unknown labels are masked; selected unknown claims have no measured truth value.',
            'Calibration thresholds meet empirical known-label precision only, not guaranteed precision.',
            'No decoder adaptation, candidate reranker or caption-quality gain demonstrated.']}
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    for name, value in [('evidence_routing_summary.json', result), ('history.json', history),
                        ('verified_prompt_cache.json', prompts), ('evaluation_predictions.json', [
        {**r, 'probabilities': probabilities[n+i].tolist(), 'decisions': decisions[n+i].tolist()}
        for i, r in enumerate(rows[n:])])]:
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    torch.save({'architecture': 'EvidenceRelationVerifier', 'dim': 512, 'hidden': 64,
        'model': model.state_dict(), 'temperature': temperature, 'policy': policy, 'relations': RELATIONS,
        'source_members_sha256': source_hash, 'fit_images': sorted(train_ids-cal_ids), 'calibration_images': sorted(cal_ids)},
        output / 'evidence_verifier.pt')
    torch.save(memory, output / 'verified_decoder_evidence.pt')
    manifest = {'variant': result['variant'], 'relations': RELATIONS, 'temperature': temperature, 'policy': policy,
                'source_members_sha256': source_hash, 'files': {
        name: hashlib.sha256((output / name).read_bytes()).hexdigest()
        for name in ('evidence_verifier.pt', 'verified_decoder_evidence.pt', 'verified_prompt_cache.json')},
                'decoder_router_trained': False, 'evidence_feature_dim': 64}
    (output / 'verified_evidence_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if not k.endswith('_images')}, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run(args.source, args.output, args.epochs, args.seed)
