"""Metadata-only K/V preparation policy. No Torch, encoders, or HDF5 imports."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from experiments.relation_frames_inventory.inventory import parse_assertion, unanimous

PREDICATES = ('contains', 'inside', 'touching')
INVERSE = {'contains': 'inside', 'inside': 'contains', 'touching': 'touching'}
POLICY = dict(version='relation-kv-cpu-v1', seed=42, maximum_objects=6,
              iou_minimum=.5, validators_minimum=3, disagreement_allowed=0,
              predicates=list(PREDICATES), inverse=INVERSE,
              meaning='literal physical containment/contact only; not agency or part-of',
              source_scope='official VSR random TRAIN and Karpathy train/restval',
              min_fit_facts=128, min_held_facts=32,
              min_fit_each_truth_per_family=8, min_held_each_truth_per_family=4,
              min_fit_mp_groups=8, min_held_mp_groups=2)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def identity(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value) + b'\n')


def image_id(row):
    name = Path(row['image']).name
    if re.fullmatch(r'\d{12}\.jpg', name) is None:
        raise ValueError('Unexpected VSR COCO filename: ' + name)
    return int(name[:-4])


def template(subject, predicate, obj):
    if predicate not in INVERSE:
        raise ValueError('Unknown predicate')
    verb = predicate if predicate == 'contains' else 'is ' + predicate
    return f'the {subject} {verb} the {obj}'


def candidates(rows, caption_splits, excluded, categories):
    result, rejected = [], Counter()
    for source_row, row in enumerate(rows):
        iid = image_id(row)
        if caption_splits.get(iid) not in ('train', 'restval') or iid in excluded:
            rejected['image_split_or_exclusion'] += 1
            continue
        if row['relation'] not in PREDICATES:
            rejected['unsupported_semantics'] += 1
            continue
        if not unanimous(row):
            rejected['validator_policy'] += 1
            continue
        parsed = parse_assertion(row, categories)
        if parsed is None:
            rejected['category_template_or_same_class'] += 1
            continue
        statement, subject, obj = parsed
        result.append(dict(image_id=iid, subject=subject, object=obj,
                           predicate=row['relation'], label=int(row['label']),
                           source_row=source_row, statement=statement,
                           reference_frame=row.get('reference_frame')))
    return result, dict(rejected)


def clipped(box, width, height, xywh=False):
    if len(box) != 4 or not all(math.isfinite(float(x)) for x in box):
        return None
    x1, y1, x2, y2 = map(float, box)
    if xywh:
        x2, y2 = x1 + x2, y1 + y2
    b = [max(0., x1), max(0., y1), min(float(width), x2), min(float(height), y2)]
    return b if b[2] > b[0] and b[3] > b[1] else None


def iou(a, b):
    intersection = max(0., min(a[2], b[2]) - max(a[0], b[0])) * max(
        0., min(a[3], b[3]) - max(a[1], b[1]))
    aa, ab = (a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1])
    return intersection / (aa + ab - intersection) if aa + ab > intersection else 0.


def bank(objects, width, height):
    if width <= 0 or height <= 0:
        raise ValueError('Invalid image dimensions')
    ordered = []
    for index, obj in enumerate(objects):
        score = float(obj['conf'])
        box = clipped(obj['bbox'], width, height)
        if not math.isfinite(score) or not 0 <= score <= 1 or box is None:
            raise ValueError('Malformed detector record')
        ordered.append(dict(original_index=index, confidence=score, bbox_xyxy=box))
    ordered.sort(key=lambda x: (-x['confidence'], x['original_index']))
    # Detector class/name is intentionally not an input to the relation branch.
    return ordered[:POLICY['maximum_objects']]


def match_record(row, image, annotations, proposals):
    targets = [annotations.get(row[role], []) for role in ('subject', 'object')]
    if any(len(v) != 1 or v[0].get('iscrowd', 0) for v in targets):
        return None, 'nonunique_missing_or_crowd_gt'
    boxes = [clipped(v[0]['bbox'], image['width'], image['height'], xywh=True)
             for v in targets]
    if any(b is None for b in boxes):
        return None, 'invalid_gt'
    eligible = []
    for i, a in enumerate(proposals):
        for j, b in enumerate(proposals):
            if i == j:
                continue
            u, v = iou(boxes[0], a['bbox_xyxy']), iou(boxes[1], b['bbox_xyxy'])
            if min(u, v) >= POLICY['iou_minimum']:
                eligible.append((min(u, v), u+v, -i, -j, i, j, u, v))
    if not eligible:
        return None, 'no_distinct_predicted_pair'
    _, _, _, _, i, j, u, v = max(eligible)
    return dict(row, subject_slot=i, object_slot=j, subject_iou=u, object_iou=v,
                subject_gt_id=targets[0][0]['id'], object_gt_id=targets[1][0]['id']), None


def fact_key(row):
    a = (row['subject_slot'], row['predicate'], row['object_slot'])
    b = (row['object_slot'], INVERSE[row['predicate']], row['subject_slot'])
    return row['image_id'], min(a, b)


def merge_facts(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[fact_key(row)].append(row)
    result, quarantined = [], []
    for key, observed in sorted(groups.items()):
        entities = set()
        for r in observed:
            forward = (r['subject_slot'], r['predicate'], r['object_slot']) == key[1]
            entity = (r['subject_gt_id'], r['object_gt_id'], r['subject'], r['object'])
            entities.add(entity if forward else (entity[1], entity[0], entity[3], entity[2]))
        reason = ('observed_equivalent_truth_conflict' if len({r['label'] for r in observed}) != 1
                  else 'ambiguous_gt_entities_for_proposal_fact' if len(entities) != 1 else None)
        if reason:
            quarantined.append(dict(fact_id=identity(key), source_rows=sorted(
                r['source_row'] for r in observed), reason=reason))
            continue
        row = min(observed, key=lambda r: r['source_row'])
        i, p, j = key[1]
        swap = (i, p, j) != (row['subject_slot'], row['predicate'], row['object_slot'])
        subject, obj = ((row['object'], row['subject']) if swap else
                        (row['subject'], row['object']))
        views = [template(subject, p, obj), template(obj, INVERSE[p], subject)]
        lexical_group = min((subject, p, obj), (obj, INVERSE[p], subject))
        result.append(dict(fact_id=identity(key), image_id=key[0], subject_slot=i,
                           object_slot=j, predicate=p, subject=subject, object=obj,
                           label=row['label'], query_views=views,
                           lexical_group=list(lexical_group),
                           family='contact' if p == 'touching' else 'containment',
                           source_rows=sorted({r['source_row'] for r in observed}),
                           transported_inverse_is_independent_observation=False))
    return result, quarantined


def partition(facts, caption_train_ids):
    result = {'fit': [], 'held': []}
    for row in facts:
        iid = row['image_id']
        held = iid not in caption_train_ids and int(identity(['kv-aux-image-v1', iid]), 16) % 5 == 0
        result['held' if held else 'fit'].append(row)
    assert not ({r['image_id'] for r in result['fit']} &
                {r['image_id'] for r in result['held']})
    assert not ({r['image_id'] for r in result['held']} & set(caption_train_ids))
    return result


def multipositive_groups(rows):
    groups = defaultdict(lambda: {0: [], 1: []})
    for row in rows:
        groups[tuple(row['lexical_group'])][row['label']].append(row['fact_id'])
    return [dict(lexical_group=list(key), positive=sorted(v[1]), negative=sorted(v[0]))
            for key, v in sorted(groups.items()) if v[0] and v[1]]


def readiness(partitions):
    failures, counts = [], {}
    for split in ('fit', 'held'):
        rows = partitions[split]
        by_family = Counter((r['family'], r['label']) for r in rows)
        mp = multipositive_groups(rows)
        counts[split] = dict(facts=len(rows), images=len({r['image_id'] for r in rows}),
                             by_family_truth={f'{k[0]}:{k[1]}': v for k, v in sorted(by_family.items())},
                             multipositive_groups=len(mp))
        if len(rows) < POLICY[f'min_{split}_facts']:
            failures.append(split + ': insufficient original facts')
        for family in ('contact', 'containment'):
            for truth in (0, 1):
                if by_family[family, truth] < POLICY[f'min_{split}_each_truth_per_family']:
                    failures.append(f'{split}: insufficient {family} truth={truth}')
        if len(mp) < POLICY[f'min_{split}_mp_groups']:
            failures.append(split + ': insufficient observed true/false lexical groups')
    return dict(passes_operational_gates=not failures, failures=failures, counts=counts,
                not_statistical_power_or_semantic_success=True)
