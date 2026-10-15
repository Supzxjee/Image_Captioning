"""Frozen UPT acceptance audit, fixed source calibration/development image split."""
import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
import torch
from PIL import Image
from captioning.upt_reference import load_reference, prepare_image, infer, UPT_REVISION, POCKET_REVISION
from captioning.hoi_transfer_eval import match_inventory
from captioning.hoi_acceptance import score_inventory, acceptance_metrics
from captioning.evidence_routing import fit_selective_policy, selective_decisions
from run_actor_compatibility_cached import read_archive
from extract_pair_query_patches import sha


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def run(a):
    torch.set_num_threads(4)
    output = Path(a.output); output.mkdir(parents=True, exist_ok=True)
    root = Path(a.upt)
    revisions = {root: UPT_REVISION, Path(a.pocket): POCKET_REVISION,
                 root/'detr': '820d17abc697e8dd5f49d4cd1fdbde412545bac5',
                 root/'hicodet': '9dd59c1f6ee2ddc011567765d0bc644858391fea'}
    for directory, revision in revisions.items():
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=directory, text=True).strip() == revision
        assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=directory, text=True).strip()
    assert sha(a.checkpoint) == '12f7f84c7c8d5bcc38f53470f57c5cc30ff2f025726e8bcc27addeb0578a258c'
    assert sha(a.source) == 'c3fcf5d65a462392891f4efa8adbd33a8d5359598ad0d0e80e4c91e863ee449f'
    assert sha(a.partition) == '3e7c3368245bf4c50b9a422630fb6db7488fcb570202c299f2178824120e8b2d'
    partition = json.loads(Path(a.partition).read_text())['partition']
    calibration = set(partition['calibration_images']); development = set(partition['development_images'])
    assert not calibration & development
    assert not (calibration | development) & set(partition['excluded_manifest_images_present'])
    rows, views, text, lengths = read_archive(a.source, ['train_pairs.json', 'val_pairs.json'])
    for split, subset in [('train', rows[:lengths[0]]), ('val', rows[lengths[0]:])]:
        assert subset == json.loads(Path(f'audit/interaction_region_{split}_pairs.json').read_text())
    selected = [i for i, r in enumerate(rows) if r['image_id'] in calibration | development]
    assert len(selected) == 497
    provenance_path = Path(a.images).parent/'image_provenance.json'
    provenance = {r['filename']: r for r in json.loads(provenance_path.read_text())}
    metadata_path = root/'hicodet/instances_train2015.json'
    metadata = json.loads(metadata_path.read_text())
    correspondence = [[] for _ in metadata['objects']]
    for _, obj, verb in metadata['correspondence']: correspondence[obj].append(verb)
    protocol = {'variant': 'pretrained_upt_acceptance_v1', 'source_sha256': sha(a.source),
        'partition_sha256': sha(a.partition), 'image_manifest_sha256': sha(provenance_path),
        'checkpoint_sha256': sha(a.checkpoint), 'metadata_sha256': sha(metadata_path),
        'reference_revisions': {str(k): v for k, v in revisions.items()},
        'calibration_images': sorted(calibration), 'development_images': sorted(development),
        'selection': 'Reuse R106 split; exclude all61 human manifest images. No local model fitting.',
        'calibration': 'Per-predicate observed high tail >=90% known precision, >=5 known accepted, both outcomes required. No temperature fit.',
        'matching': 'Class-consistent one-to-one Hungarian IoU>=0.5, no relation labels.',
        'arms': ['clip_all', 'clip_teacher_mask', 'upt_raw', 'upt_native'],
        'primary': 'Development accepted true/false and full-pool positive recall; same-mask known AP.',
        'unknown': 'Never negative. Missing/unsupported abstains and stays in full-pool positive denominator.',
        'limitations': ['Reused development pool, not fresh validation.', 'Source actor-absent labels do not prove every wrong-target relation false.',
            'HICO pretraining and COCO detector pretraining; dataset overlap not exhaustively audited.',
            'Previous 80% review-coverage gate failed and remains failed. Larger diagnostic only.',
            'No caption metrics or scientific novelty established. Empirical precision is not guaranteed.'],
        'training_performed': False, 'caption_decoder_changed': False}
    pp = output/'protocol.json'
    if pp.exists(): assert json.loads(pp.read_text()) == protocol
    else: write(pp, protocol)
    hashes_path = output/'cache_sha256.json'
    hashes = json.loads(hashes_path.read_text()) if hashes_path.exists() else {}
    model, runtime = load_reference(a.upt, a.pocket, a.checkpoint, correspondence)
    raw = torch.zeros((len(selected), 6)); native = torch.zeros_like(raw); available = torch.zeros_like(raw, dtype=torch.bool)
    reason_counts = Counter(); started = time.monotonic()
    image_ids = sorted(calibration | development)
    for number, image_id in enumerate(image_ids, 1):
        positions = [j for j, i in enumerate(selected) if rows[i]['image_id'] == image_id]
        candidates = [rows[selected[j]] for j in positions]
        filename = candidates[0]['filename']; image_path = Path(a.images)/filename
        assert sha(image_path) == provenance[filename]['sha256']
        cache = output/f'private_image_{image_id}.pt'
        if cache.name in hashes:
            assert sha(cache) == hashes[cache.name]
            prediction = torch.load(cache, map_location='cpu', weights_only=True)
        else:
            with Image.open(image_path) as image:
                assert list(image.size) == provenance[filename]['image_size']
                prediction = infer(model, prepare_image(image))
            assert torch.isfinite(prediction['logits']).all()
            torch.save(prediction, cache); hashes[cache.name] = sha(cache); write(hashes_path, hashes)
        # Inventory includes all source candidate instances from that image.
        mapping, _ = match_inventory([r for r in rows if r['image_id'] == image_id], prediction, metadata['objects'])
        rr, nn, valid, reasons = score_inventory(candidates, prediction, mapping, metadata['verbs'], correspondence)
        raw[positions], native[positions], available[positions] = rr, nn, valid
        reason_counts.update(reason for row in reasons for reason in row)
        print(f'UPT acceptance {number}/{len(image_ids)} | {time.monotonic()-started:.1f}s', flush=True)
    labels = torch.tensor([rows[i]['labels'] for i in selected])
    clip = (10*torch.einsum('bd,bkd->bk', views[:, 2], text)-3)[selected].sigmoid()
    cal = torch.tensor([rows[i]['image_id'] in calibration for i in selected])
    dev = ~cal
    policies, results = {}, {}
    arms = {'clip_all': (clip, torch.ones_like(available)), 'clip_teacher_mask': (clip, available),
            'upt_raw': (raw, available), 'upt_native': (native, available)}
    decisions_record = {}
    for name, (probabilities, valid) in arms.items():
        masked_labels = labels[cal].clone(); masked_labels[~valid[cal]] = -1
        policy = fit_selective_policy(probabilities[cal], masked_labels, precision=.9, min_known=5)
        decisions = selective_decisions(probabilities, valid, policy)
        policies[name] = policy
        results[name] = acceptance_metrics(probabilities[dev], labels[dev], valid[dev], decisions[dev])
        results[name]['by_predicate'] = {metadata['verbs'][metadata['verbs'].index(v)]: {
            'known_positive': int((labels[dev, c] == 1).sum()),
            'available_positive': int(((labels[dev, c] == 1) & valid[dev, c]).sum()),
            'accepted_true': int(((labels[dev, c] == 1) & (decisions[dev, c] == 1)).sum()),
            'accepted_false': int(((labels[dev, c] == 0) & (decisions[dev, c] == 1)).sum())}
            for c, v in enumerate(['hold', 'ride', 'sit_on', 'carry', 'eat', 'talk_on'])}
        decisions_record[name] = decisions
    torch.save({'row_indices': selected, 'calibration': cal, 'labels': labels, 'clip': clip, 'raw': raw,
                'native': native, 'available': available, 'decisions': decisions_record}, output/'private_scores.pt')
    summary = {'variant': protocol['variant'], 'protocol_sha256': sha(pp), 'runtime': runtime,
        'calibration_images': len(calibration), 'development_images': len(development),
        'calibration_pairs': int(cal.sum()), 'development_pairs': int(dev.sum()),
        'availability_reasons_all_cells': dict(reason_counts), 'policies': policies, 'development': results,
        'seconds': time.monotonic()-started, 'caption_metrics_measured': False, 'production_promotion': False}
    write(output/'acceptance_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('upt', 'pocket', 'checkpoint', 'source', 'partition', 'images', 'output'):
        p.add_argument('--'+name, required=True)
    run(p.parse_args())
