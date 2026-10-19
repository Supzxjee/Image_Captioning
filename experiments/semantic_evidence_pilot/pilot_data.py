"""Deterministic image cohorts and box pooling; no ML/runtime side effects."""
import hashlib
import json
from pathlib import Path

import numpy as np

SEED = 42
SLOTS = 9
INPUT_DIM = 768 * 3 + 8
TRAIN_COUNT = 10000
VAL_COUNT = 1000


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def cohorts(dataset, train_count=TRAIN_COUNT, val_count=VAL_COUNT):
    result = {}
    for split, count in [('train', train_count), ('val', val_count)]:
        rows = []
        for i, row in enumerate(dataset['images']):
            selected = row['split'] in ('train', 'restval') if split == 'train' else row['split'] == 'val'
            if selected:
                rows.append({'image_id': i, 'coco_id': int(Path(row['filename']).stem.rsplit('_', 1)[1]),
                             'filename': row['filename'], 'split': split})
        if not 1 < count <= len(rows):
            raise ValueError(f'Insufficient {split} images.')
        rows.sort(key=lambda r: hashlib.sha256(('upt-alignment-pilot-v1:' + split + ':' + r['filename']).encode()).hexdigest())
        result[split] = sorted(rows[:count], key=lambda r: r['image_id'])
    all_rows = result['train'] + result['val']
    if len({r['coco_id'] for r in all_rows}) != len(all_rows):
        raise ValueError('Overlapping/duplicate COCO IDs.')
    return result


def box(value):
    b = np.asarray(value, dtype=np.float32)
    if b.shape != (4,) or not np.isfinite(b).all() or (b < 0).any() or (b > 1).any() or (b[2:] <= b[:2]).any():
        raise ValueError('Expected a positive-area normalized xyxy box.')
    return b


def pool(patches, bounds):
    if patches.shape != (196, 768):
        raise ValueError('Expected 14x14 spatial CLIP patch tokens, excluding CLS.')
    b = box(bounds)
    y, x = np.meshgrid((np.arange(14) + .5) / 14, (np.arange(14) + .5) / 14, indexing='ij')
    x, y = x.ravel(), y.ravel()
    mask = (x >= b[0]) & (x <= b[2]) & (y >= b[1]) & (y <= b[3])
    if mask.any():
        return patches[mask].mean(0)
    nearest = np.argmin((x - (b[0] + b[2]) / 2)**2 + (y - (b[1] + b[3]) / 2)**2)
    return patches[nearest].copy()


def geometry(s, o):
    sw, sh = np.maximum(s[2:] - s[:2], 1e-6)
    ow, oh = np.maximum(o[2:] - o[:2], 1e-6)
    sc, oc = (s[:2] + s[2:]) / 2, (o[:2] + o[2:]) / 2
    inter = np.maximum(np.minimum(s[2:], o[2:]) - np.maximum(s[:2], o[:2]), 0).prod()
    return np.asarray([(oc[0]-sc[0])/sw, (oc[1]-sc[1])/sh, np.log(ow/sw), np.log(oh/sh),
                       inter/max(sw*sh+ow*oh-inter, 1e-6), inter/(sw*sh), inter/(ow*oh),
                       np.linalg.norm(oc-sc)], dtype=np.float32)


def region_slots(entry, visual):
    """Keep every selected triplet, with separate S/O/R slots; object fallback otherwise."""
    if visual.shape != (197, 768) or not np.isfinite(visual).all():
        raise ValueError('Invalid visual cache entry.')
    features = np.zeros((SLOTS, INPUT_DIM), dtype=np.float32)
    roles = np.ones(SLOTS, dtype=np.int64)
    valid = np.zeros(SLOTS, dtype=np.bool_)
    texts = [''] * SLOTS
    objects = {o['index']: o for o in entry['objects']}
    relations = entry['relations']
    if len(relations) > 3:
        raise ValueError('Expected at most 3 selected relations.')
    if relations:
        for i, relation in enumerate(relations):
            if relation['score'] < relation['threshold']:
                raise ValueError('A selected relation is below its threshold.')
            subject, obj = (objects[relation[key + '_index']] for key in ('subject', 'object'))
            if subject['instance_id'] != relation['subject_instance'] or obj['instance_id'] != relation['object_instance']:
                raise ValueError('Instance-link mismatch.')
            s, o = box(subject['bbox']), box(obj['bbox'])
            union = np.r_[np.minimum(s[:2], o[:2]), np.maximum(s[2:], o[2:])]
            vs, vo, vu = [pool(visual[1:], b) for b in (s, o, union)]
            n = i * 3
            features[n, :768], features[n+1, :768] = vs, vo
            features[n+2] = np.r_[vs, vo, vu, geometry(s, o)]
            roles[n:n+3], valid[n:n+3] = (0, 1, 2), True
            texts[n:n+3] = (subject['label'], obj['label'], relation['text'])
    else:
        accepted = sorted((o for o in objects.values() if o['confidence'] >= .5),
                          key=lambda o: (-o['confidence'], o['index']))[:8]
        for i, obj in enumerate(accepted):
            features[i, :768] = pool(visual[1:], obj['bbox'])
            valid[i], texts[i] = True, obj['label']
    if bool(valid.any()) != (entry['mode'] != 'no_objects'):
        raise ValueError('Prompt mode and region slots disagree.')
    return features, roles, valid, texts


def evidence_bounds(entry):
    """Normalized subject/object/allowed boxes, in exactly region_slots order."""
    result = np.zeros((SLOTS, 3, 4), dtype=np.float32)
    objects = {o['index']:o for o in entry['objects']}
    if entry['relations']:
        for i,r in enumerate(entry['relations']):
            s,o = (box(objects[r[k+'_index']]['bbox']) for k in ('subject','object'))
            union=np.r_[np.minimum(s[:2],o[:2]),np.maximum(s[2:],o[2:])]
            result[i*3]=np.stack((s,o,s));result[i*3+1]=np.stack((s,o,o))
            result[i*3+2]=np.stack((s,o,union))
    else:
        accepted=sorted((o for o in objects.values() if o['confidence']>=.5),key=lambda o:(-o['confidence'],o['index']))[:8]
        for i,o in enumerate(accepted): result[i]=np.tile(box(o['bbox']),(3,1))
    return result


def phrase_identity(text):
    return int(hashlib.sha256(text.strip().casefold().encode()).hexdigest()[:15],16) if text else 0
