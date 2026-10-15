"""Replay cache provenance, native scores, cal-only policy and development metrics."""
import argparse
import json
from pathlib import Path
import torch
from captioning.hoi_acceptance import score_inventory, acceptance_metrics
from captioning.hoi_transfer_eval import match_inventory
from captioning.evidence_routing import fit_selective_policy, selective_decisions
from run_actor_compatibility_cached import read_archive
from import_target_annotations import filter_annotations
from extract_pair_query_patches import sha
from run_pretrained_hoi_acceptance import write


def run(a):
    torch.set_num_threads(1)
    output = Path(a.output)
    summary = json.loads((output/'acceptance_summary.json').read_text())
    assert sha(output/'protocol.json') == summary['protocol_sha256']
    protocol = json.loads((output/'protocol.json').read_text())
    assert sha(a.source) == protocol['source_sha256']
    rows, views, text, lengths = read_archive(a.source, ['train_pairs.json', 'val_pairs.json'])
    state = torch.load(output/'private_scores.pt', map_location='cpu', weights_only=True)
    selected = state['row_indices']
    metadata = json.loads(Path(a.metadata).read_text())
    assert sha(a.metadata) == protocol['metadata_sha256']
    correspondence = [[] for _ in metadata['objects']]
    for _, obj, verb in metadata['correspondence']: correspondence[obj].append(verb)
    official_cells = 0
    hashes = json.loads((output/'cache_sha256.json').read_text())
    for name, digest in hashes.items():
        assert sha(output/name) == digest
        prediction = torch.load(output/name, map_location='cpu', weights_only=True)
        prior = prediction['prior'].prod(0)
        x, y = torch.nonzero(prior).unbind(1)
        assert torch.equal(prediction['logits'][x, y].sigmoid()*prior[x, y], prediction['official_output']['scores'])
        assert torch.equal(y, prediction['official_output']['labels'])
        assert torch.equal(torch.stack([prediction['subject_indices'][x], prediction['object_indices'][x]]), prediction['official_output']['pairing'])
        official_cells += len(x)
        image_id = int(name.removeprefix('private_image_').removesuffix('.pt'))
        positions = [j for j, i in enumerate(selected) if rows[i]['image_id'] == image_id]
        mapping, _ = match_inventory([r for r in rows if r['image_id'] == image_id], prediction, metadata['objects'])
        raw, native, valid, _ = score_inventory([rows[selected[j]] for j in positions], prediction, mapping,
                                                metadata['verbs'], correspondence)
        assert torch.equal(raw, state['raw'][positions])
        assert torch.equal(native, state['native'][positions])
        assert torch.equal(valid, state['available'][positions])
    assert len(hashes) == len(protocol['calibration_images']) + len(protocol['development_images'])
    cal = state['calibration']; dev = ~cal; labels = state['labels']
    assert torch.equal(labels, torch.tensor([rows[i]['labels'] for i in selected]))
    assert cal.tolist() == [rows[i]['image_id'] in protocol['calibration_images'] for i in selected]
    clip_all = (10*torch.einsum('bd,bkd->bk', views[:, 2], text)-3).sigmoid()
    assert torch.equal(clip_all[selected], state['clip'])
    for name in summary['policies']:
        scores = state['clip' if name.startswith('clip') else name.removeprefix('upt_')]
        valid = torch.ones_like(state['available']) if name == 'clip_all' else state['available']
        masked = labels[cal].clone(); masked[~valid[cal]] = -1
        policy = fit_selective_policy(scores[cal], masked, precision=.9, min_known=5)
        assert policy == summary['policies'][name]
        decisions = selective_decisions(scores, valid, policy)
        assert torch.equal(decisions, state['decisions'][name])
        metrics = acceptance_metrics(scores[dev], labels[dev], valid[dev], decisions[dev])
        assert all(summary['development'][name][key] == value for key, value in metrics.items())
    verification = {'cache_files_verified': len(hashes), 'official_scores_bit_exact': official_cells,
        'all_pair_scores_and_availability_replayed': True, 'calibration_only_policies_exact': True,
        'development_counts_and_ap_exact': True, 'protocol_sha256': sha(output/'protocol.json')}
    write(output/'verification.json', verification)

    # Apply these source-calibrated thresholds unchanged to existing human contrasts.
    # This extension is a separate, exploratory transfer diagnostic, not a new policy.
    package = Path(a.package); old = Path(a.review_output)
    mapping = json.loads((package/'private_mapping.json').read_text())
    assert sha(package/'cases.json') == mapping['manifest_sha256']
    assert sha('audit/interaction_region_train_pairs.json') == mapping['source_pairs_sha256']
    review = package/'human_batch1_review.json'
    assert sha(review) == '5ffdcedb7f9b72565f052b2e10860562c53ff9e1d4e386e6851e15291444507f'
    cases = filter_annotations(json.loads(review.read_text(encoding='utf-8-sig')), mapping)['eligible_contrasts']
    cells = {}
    for case in cases:
        for field, label in [('positive_row_index', 1), ('alternate_row_index', 0)]:
            key = (case[field], case['relation_index'])
            if key in cells: assert cells[key] == label
            cells[key] = label
    indices = sorted({i for i, _ in cells}); position = {i: j for j, i in enumerate(indices)}
    human_labels = torch.full((len(indices), 6), -1)
    for (i, c), label in cells.items(): human_labels[position[i], c] = label
    raw = torch.zeros_like(human_labels, dtype=torch.float); native = torch.zeros_like(raw)
    valid = torch.zeros_like(human_labels, dtype=torch.bool)
    old_summary = json.loads((old/'pretrained_hoi_summary.json').read_text())
    assert sha(old/'protocol.json') == old_summary['protocol_sha256']
    old_protocol = json.loads((old/'protocol.json').read_text())
    assert old_protocol['checkpoint_sha256'] == protocol['checkpoint_sha256']
    assert old_protocol['metadata_sha256'] == protocol['metadata_sha256']
    for image_id in sorted({rows[i]['image_id'] for i in indices}):
        assert image_id not in protocol['calibration_images'] + protocol['development_images']
        cache = old/f'private_image_{image_id}.pt'
        assert sha(cache) == old_summary['cache_sha256'][cache.name]
        prediction = torch.load(cache, map_location='cpu', weights_only=True)
        mapping, _ = match_inventory([r for r in rows[:lengths[0]] if r['image_id'] == image_id], prediction, metadata['objects'])
        positions = [j for j, i in enumerate(indices) if rows[i]['image_id'] == image_id]
        rr, nn, vv, _ = score_inventory([rows[indices[j]] for j in positions], prediction, mapping, metadata['verbs'], correspondence)
        raw[positions], native[positions], valid[positions] = rr, nn, vv
    human_results = {}
    for name, (scores, mask) in {'clip_all': (clip_all[indices], torch.ones_like(valid)),
        'clip_teacher_mask': (clip_all[indices], valid), 'upt_raw': (raw, valid), 'upt_native': (native, valid)}.items():
        decisions = selective_decisions(scores, mask, summary['policies'][name])
        human_results[name] = acceptance_metrics(scores, human_labels, mask, decisions)
        human_results[name]['both_contrast_endpoints_accepted'] = sum(
            decisions[position[c['positive_row_index']], c['relation_index']] == 1 and
            decisions[position[c['alternate_row_index']], c['relation_index']] == 1 for c in cases)
        human_results[name]['both_contrast_endpoints_accepted'] = int(human_results[name]['both_contrast_endpoints_accepted'])
    write(output/'human_threshold_transfer.json', {'review_sha256': sha(review), 'contrasts': len(cases),
        'unique_labeled_cells': len(cells), 'images': len({rows[i]['image_id'] for i in indices}),
        'thresholds_changed': False, 'status': 'Exploratory acceptance transfer on reused single-reviewer contrasts; no fitting on human labels.',
        'results': human_results})
    print(json.dumps(verification, indent=2), flush=True)
    print(json.dumps(human_results, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('output', 'source', 'metadata', 'package', 'review-output'): p.add_argument('--'+name, required=True)
    run(p.parse_args())
