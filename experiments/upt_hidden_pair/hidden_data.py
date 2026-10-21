"""Locked UPT hidden-pair cohort, selection and numeric cache validation; no torch."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np

from pilot_data import cohorts, digest, identity, save

WIDTH, SLOTS = 1024, 8
BOX_THRESHOLD = .2
SCHEMA = 'upt_hidden_pair_caption_cache_v1'
COHORT_SHA = '0c65467f3907644a8674fe3eff71455d65bde944c30e13a71914617243cf988d'
DATASET_SHA = '2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
UPT_SHA = '12f7f84c7c8d5bcc38f53470f57c5cc30ff2f025726e8bcc27addeb0578a258c'
INPUT_HASHES = dict(checkpoint='5f8eead3f4e960189345b1a9bf492763fe52ec71f70115b73244ffe9d03d2e75',
    legacy_prompt='d7910e9bf87e3aad86d806c77ec336b34793e3bef68a3f662fdcb2e9456ca38d',
    dataset_json=DATASET_SHA, detections='498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c')


def locked_groups(dataset):
    groups = cohorts(dataset)
    if identity(groups) != COHORT_SHA:
        raise ValueError('Different train10k/val1000 cohort; do not substitute another dataset.')
    return groups


def select_pairs(labels, scores, subjects, objects, features):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float32)
    subjects, objects = np.asarray(subjects, dtype=np.int64), np.asarray(objects, dtype=np.int64)
    features = np.asarray(features, dtype=np.float32)
    if scores.shape != labels.shape or subjects.shape != objects.shape or subjects.ndim != 1:
        raise ValueError('Detection/pair columns differ.')
    if features.shape != (len(subjects), WIDTH) or not np.isfinite(features).all():
        raise ValueError('Hidden features must be finite native-pair rows x1024.')
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError('Invalid detection confidences.')
    if any(s == o or min(s, o) < 0 or max(s, o) >= len(labels) or labels[s] != 0
           for s, o in zip(subjects, objects)):
        raise ValueError('Invalid human pair endpoint.')
    if len(set(zip(subjects.tolist(), objects.tolist()))) != len(subjects):
        raise ValueError('Duplicate native pair indices.')
    eligible = [i for i, o in enumerate(objects) if labels[o] != 0
                and scores[subjects[i]] > BOX_THRESHOLD and scores[o] > BOX_THRESHOLD]
    # All native pair interactions are computed before this memory-only cap.
    order = sorted(eligible, key=lambda i: (-float(scores[subjects[i]]) * float(scores[objects[i]]),
                                          int(subjects[i]), int(objects[i]), i))[:SLOTS]
    quantized = features[order].astype(np.float16)
    if not np.isfinite(quantized).all():
        raise ValueError('FP16 feature overflow.')
    return np.asarray(order, dtype=np.int64), quantized, len(eligible)


def validate_row(features, valid, pairs, native_rows, labels, confidence, boxes):
    if features.shape != (SLOTS, WIDTH) or features.dtype != np.float16 or not np.isfinite(features).all():
        raise ValueError('Invalid padded features.')
    if valid.shape != (SLOTS,) or valid.dtype != np.bool_:
        raise ValueError('Invalid pair mask.')
    n = int(valid.sum())
    if not np.array_equal(valid, np.arange(SLOTS) < n):
        raise ValueError('Expected contiguous selected slots.')
    for a, shape in [(pairs, (SLOTS, 2)), (native_rows, (SLOTS,)), (labels, (SLOTS, 2)),
                     (confidence, (SLOTS, 2)), (boxes, (SLOTS, 2, 4))]:
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError('Invalid metadata shape/value.')
    if (features[n:] != 0).any() or (pairs[n:] != -1).any() or (native_rows[n:] != -1).any():
        raise ValueError('Padded features/identities are not blank.')
    if (labels[n:] != -1).any() or (confidence[n:] != 0).any() or (boxes[n:] != 0).any():
        raise ValueError('Padded metadata is not blank.')
    if n and ((pairs[:n] < 0).any() or (pairs[:n, 0] == pairs[:n, 1]).any()
              or (labels[:n, 0] != 0).any() or (labels[:n, 1] == 0).any()
              or (native_rows[:n] < 0).any() or len(set(native_rows[:n].tolist())) != n):
        raise ValueError('Selected rows lose human/nonhuman/native identity.')
    if n and ((confidence[:n] < 0).any() or (confidence[:n] > 1).any()
              or (confidence[:n] <= BOX_THRESHOLD).any()
              or (boxes[:n, :, 2:] <= boxes[:n, :, :2]).any()):
        raise ValueError('Invalid confidence/box extent.')


def check_cache(folder):
    """Hash and identity checks only; actual HDF5 shape checks happen when staging."""
    folder = Path(folder)
    receipt = json.loads((folder / 'cache_receipt.json').read_text())
    if receipt['status'] != 'CACHE_COMPLETE' or receipt['schema'] != SCHEMA:
        raise ValueError('Incomplete/unknown hidden-pair cache.')
    if receipt['cohort_sha256'] != COHORT_SHA or receipt['upt_checkpoint_sha256'] != UPT_SHA:
        raise ValueError('Cache checkpoint/cohort differs.')
    if (receipt['slots'], receipt['width']) != (SLOTS, WIDTH):
        raise ValueError('Cache dimensions differ.')
    required = {'cohort.json', 'features_rank0.h5', 'features_rank1.h5', 'pairs_rank0.jsonl', 'pairs_rank1.jsonl'}
    if set(receipt['files']) != required:
        raise ValueError('Cache receipt file inventory differs.')
    for name, sha in receipt['files'].items():
        if Path(name).name != name or digest(folder / name) != sha:
            raise ValueError('Cache file hash/name differs: ' + name)
    groups = json.loads((folder / 'cohort.json').read_text())
    if identity(groups) != COHORT_SHA:
        raise ValueError('Cohort file differs.')
    if receipt['selection_box_threshold'] != BOX_THRESHOLD:
        raise ValueError('Hidden-pair confidence policy differs.')
    return receipt, groups


def human_sample(rows, counts, excluded, count=32):
    available = sorted((r for r in rows if r['coco_id'] not in set(excluded)),
                       key=lambda r: hashlib.sha256(('upt-hidden-caption-human-v1:' + r['filename']).encode()).hexdigest())
    no_pairs = [r for r in available if counts[r['coco_id']] == 0][:count // 2]
    paired = [r for r in available if counts[r['coco_id']] > 0][:count - len(no_pairs)]
    chosen = no_pairs + paired
    selected = {r['coco_id'] for r in chosen}
    chosen += [r for r in available if r['coco_id'] not in selected][:count - len(chosen)]
    if len(chosen) != count:
        raise ValueError('Not enough nonexcluded review images.')
    return chosen
