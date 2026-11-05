"""NumPy-only instance joins for the region-linked CSA prompt pilot."""
from collections import Counter
import hashlib
from pathlib import Path
import numpy as np

PROMPT_SHA = '24e7861a4aa0e7e731d67689dce195688096232bcbef1b4264034cf520d6957d'
OBJECT_CAP, RELATION_CAP, IOU = 8, 3, .5


def label(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Missing object label')
    return ' '.join(value.lower().replace('_', ' ').split())


def box(value):
    b = np.asarray(value, dtype=np.float64)
    if b.shape != (4,) or not np.isfinite(b).all() or (b < 0).any() or (b > 1).any() or (b[2:] <= b[:2]).any():
        raise ValueError('Expected normalized positive xyxy box')
    return b


def overlap(a, boxes):
    a = box(a)
    if not len(boxes):
        return np.empty(0)
    boxes = np.stack([box(b) for b in boxes])
    intersection = np.maximum(0, np.minimum(a[2:], boxes[:, 2:]) - np.maximum(a[:2], boxes[:, :2])).prod(1)
    return intersection / ((a[2:] - a[:2]).prod() + (boxes[:, 2:] - boxes[:, :2]).prod(1) - intersection)


def detector_regions(objects, width, height, saved_boxes, kinds):
    """Reconstruct the existing cache's confidence-sort/clip/cap before label joins."""
    if width <= 0 or height <= 0 or len(kinds) != len(saved_boxes):
        raise ValueError('Invalid ROI dimensions')
    ordered = sorted(enumerate(objects), key=lambda x: (-float(x[1]['conf']), x[0]))[:150]
    if not ordered:
        if kinds != ['full_frame_fallback'] or not np.allclose(saved_boxes, [[0, 0, width, height]], rtol=0, atol=1e-4):
            raise ValueError('Changed fallback ROI')
        return []
    if kinds != ['predicted_roi'] * len(ordered):
        raise ValueError('Detector and ROI counts/kinds disagree')
    result = []
    for i, (_, obj) in enumerate(ordered):
        score = float(obj['conf'])
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid detector confidence')
        b = np.clip(np.asarray(obj['bbox'], np.float32), [0, 0, 0, 0], [width, height, width, height]).astype(np.float32)
        if b.shape != (4,) or not np.isfinite(b).all() or not np.allclose(b, saved_boxes[i], atol=1e-4, rtol=0):
            raise ValueError('Detector order/box differs from saved ROI')
        result.append(dict(roi=i, label=label(obj['label']), bbox=box(b / [width, height, width, height]).tolist()))
    return result


def geometry(subject, obj):
    s, o = box(subject), box(obj)
    sw, sh = s[2:] - s[:2]; ow, oh = o[2:] - o[:2]
    sc, oc = (s[:2] + s[2:]) / 2, (o[:2] + o[2:]) / 2
    intersection = np.maximum(0, np.minimum(s[2:], o[2:]) - np.maximum(s[:2], o[:2])).prod()
    return [float(v) for v in ((oc[0]-sc[0])/sw, (oc[1]-sc[1])/sh, np.log(ow/sw), np.log(oh/sh),
            intersection/(sw*sh+ow*oh-intersection), intersection/(sw*sh), intersection/(ow*oh), np.linalg.norm(oc-sc))]


def prompt_text(name):
    # Use the established UPT article handling, including plural skis/scissors.
    from captioning.upt_prompts import noun
    return noun(name)


def join_entry(entry, regions):
    """Only selected objects/relations are consumed; previews/refs/GT are ignored."""
    if entry['mode'] not in ('object_relation', 'object_only', 'no_objects'):
        raise ValueError('Unknown prompt mode')
    objects = entry['objects']; relations = entry['relations']
    if not isinstance(objects, list) or not isinstance(relations, list) or len(relations) > RELATION_CAP:
        raise ValueError('Malformed or unapproved prompt caps')
    by_index = {}; instances = set(); mapped = {}; stats = Counter(objects=len(objects), relations=len(relations))
    for obj in objects:
        index = obj['index']; instance = obj['instance_id']; name = label(obj['label']); b = box(obj['bbox'])
        score = float(obj['confidence'])
        if type(index) != int or index in by_index or not isinstance(instance, str) or not instance or instance in instances:
            raise ValueError('Duplicate/invalid object identity')
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid UPT object score')
        by_index[index] = obj; instances.add(instance)
        candidates = [r for r in regions if r['label'] == name]
        values = overlap(b, [r['bbox'] for r in candidates])
        if not len(values) or values.max() < IOU:
            stats['unmatched_objects'] += 1; continue
        best = np.flatnonzero(np.isclose(values, values.max(), rtol=0, atol=1e-8))
        if len(best) != 1:
            stats['ambiguous_objects'] += 1; continue
        j = int(best[0]); r = candidates[j]
        mapped[index] = dict(instance_id=instance, upt_index=index, roi=r['roi'], label=name, bbox=b.tolist(),
            roi_bbox=r['bbox'], IoU=float(values[j]), confidence=score, text=prompt_text(name))
    collisions = Counter(r['roi'] for r in mapped.values())
    for index in list(mapped):
        if collisions[mapped[index]['roi']] > 1:
            del mapped[index]; stats['colliding_objects'] += 1
    endpoints = []; valid_relations = []; seen_pairs = set()
    for r in relations:
        s, o = r['subject_index'], r['object_index']
        if type(s) != int or type(o) != int or s == o or s not in by_index or o not in by_index:
            raise ValueError('Broken relation endpoints')
        if r['subject_instance'] != by_index[s]['instance_id'] or r['object_instance'] != by_index[o]['instance_id']:
            raise ValueError('Relation instance IDs differ')
        if label(r['subject']) != 'person' or label(by_index[s]['label']) != 'person' or label(r['object']) != label(by_index[o]['label']):
            raise ValueError('Wrong subject/object role labels')
        score, threshold = float(r['score']), float(r['threshold'])
        if not np.isfinite([score, threshold]).all() or not 0 <= score <= 1 or not 0 <= threshold <= 1.01 or score < threshold:
            raise ValueError('Rejected/nonfinite relation in selected list')
        from captioning.upt_prompts import relation_text
        expected = relation_text(r['predicate'], label(r['object']))
        if r['text'] != expected:
            raise ValueError('Relation text differs from locked UPT template')
        key = (s, r['predicate'], o)
        if key in seen_pairs:
            raise ValueError('Duplicate relation')
        seen_pairs.add(key)
        if s not in mapped or o not in mapped:
            stats['unmatched_relations'] += 1; continue
        endpoints.extend((s, o))
        valid_relations.append(dict(subject_instance=r['subject_instance'], object_instance=r['object_instance'],
            subject_roi=mapped[s]['roi'], object_roi=mapped[o]['roi'], text=expected, predicate=r['predicate'],
            score=score, threshold=threshold, geometry=geometry(by_index[s]['bbox'], by_index[o]['bbox'])))
    order = list(dict.fromkeys(endpoints))
    order += [i for i in sorted(mapped, key=lambda i: (-mapped[i]['confidence'], i)) if i not in order]
    selected = [mapped[i] for i in order[:OBJECT_CAP]]
    stats.update(mapped_objects=len(mapped), kept_objects=len(selected), mapped_relations=len(valid_relations),
                 capped_objects=max(0, len(mapped)-OBJECT_CAP))
    return dict(source_mode=entry['mode'], objects=selected, relations=valid_relations, counts=dict(stats))


def cohort(ids):
    ids = [int(i) for i in ids]
    if len(ids) != len(set(ids)) or len(ids) < 10000:
        raise ValueError('Need unique Karpathy TRAIN IDs')
    return sorted(ids, key=lambda i: hashlib.sha256(f'42:{i}'.encode()).hexdigest())[:10000]


def inference_record(meta, linked):
    """Do not copy captions, sequences or any supervision into prompt payloads."""
    return dict(coco_id=int(meta['coco_id']), filename=meta['filename'], group=meta['group'],
                roi_count=len(meta['region_kinds']), **linked)


def tensor_batch(records, embeddings, texts):
    """Data-only packing shared by the future worker and neural smoke tests."""
    lookup = {s: i for i, s in enumerate(texts)}
    if len(lookup) != len(texts) or embeddings.shape != (len(texts), 512) or not np.isfinite(embeddings).all():
        raise ValueError('Invalid prompt embedding cache')
    n = len(records)
    out = dict(object_text=np.zeros((n,8,512),np.float32), object_roi=np.zeros((n,8),np.int64),
        object_box=np.zeros((n,8,4),np.float32), object_valid=np.zeros((n,8),bool),
        relation_text=np.zeros((n,3,512),np.float32), relation_pair=np.zeros((n,3,2),np.int64),
        relation_geometry=np.zeros((n,3,8),np.float32), relation_valid=np.zeros((n,3),bool))
    for i, row in enumerate(records):
        if len(row['objects']) > 8 or len(row['relations']) > 3:
            raise ValueError('Prompt slot overflow')
        for j, obj in enumerate(row['objects']):
            if not 0 <= obj['roi'] < row['roi_count']:
                raise ValueError('Object ROI out of bounds')
            out['object_text'][i,j]=embeddings[lookup[obj['text']]];out['object_roi'][i,j]=obj['roi']
            out['object_box'][i,j]=box(obj['bbox']);out['object_valid'][i,j]=True
        for j, rel in enumerate(row['relations']):
            pair=[rel['subject_roi'],rel['object_roi']]
            if pair[0] == pair[1] or any(not 0 <= k < row['roi_count'] for k in pair):
                raise ValueError('Invalid relation pair')
            g=np.asarray(rel['geometry'],np.float32)
            if g.shape!=(8,) or not np.isfinite(g).all():raise ValueError('Invalid pair geometry')
            out['relation_text'][i,j]=embeddings[lookup[rel['text']]];out['relation_pair'][i,j]=pair
            out['relation_geometry'][i,j]=g;out['relation_valid'][i,j]=True
    return out
