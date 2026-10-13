"""V-COCO pair supervision. Missing links are UNKNOWN, never invented negatives."""
import math
import random
from collections import Counter, defaultdict

RELATION_SPECS = (
    ('hold', 'obj', 'holding'), ('ride', 'instr', 'riding'),
    ('sit', 'instr', 'sitting_on'), ('carry', 'obj', 'carrying'),
    ('eat', 'obj', 'eating'), ('talk_on_phone', 'instr', 'talking_on'),
)
RELATIONS = [spec[2] for spec in RELATION_SPECS]


def decode_action(action):
    """Official serialization stores all agents, then each role column."""
    n = len(action['ann_id'])
    roles = action['role_name']
    flat = action['role_object_id']
    if n == 0 or len(flat) != n * len(roles):
        raise ValueError('Invalid V-COCO role array.')
    if len(action['image_id']) != n or len(action['label']) != n:
        raise ValueError('V-COCO column lengths differ.')
    if roles[0] != 'agent' or flat[:n] != action['ann_id']:
        raise ValueError('Agent column does not match annotation IDs.')
    rows = {}
    for i, sid in enumerate(action['ann_id']):
        if sid in rows or action['label'][i] not in (-1, 0, 1):
            raise ValueError('Duplicate actor or invalid action label.')
        rows[sid] = {'image_id': action['image_id'][i], 'label': action['label'][i],
                     'roles': {role: flat[j * n + i] for j, role in enumerate(roles)}}
    return rows


def normalized_box(annotation, image):
    x, y, w, h = annotation['bbox']
    width, height = image['width'], image['height']
    if not all(math.isfinite(v) for v in (x, y, w, h, width, height)):
        raise ValueError('Non-finite box.')
    if min(w, h, width, height) <= 0:
        raise ValueError('Empty box/image.')
    box = [max(0, x / width), max(0, y / height),
           min(1, (x + w) / width), min(1, (y + h) / height)]
    if not (box[0] < box[2] and box[1] < box[3]):
        raise ValueError('Box outside image.')
    return box


def build_pairs(actions, instances, caption_images, max_objects=12):
    if max_objects < 1:
        raise ValueError('max_objects must be positive.')
    by_action = {a['action_name']: decode_action(a) for a in actions}
    if any(name not in by_action for name, _, _ in RELATION_SPECS):
        raise ValueError('Required V-COCO action missing.')
    images = {x['id']: x for x in instances['images']}
    anns = {x['id']: x for x in instances['annotations']}
    categories = {x['id']: x['name'] for x in instances['categories']}
    caption = {x['cocoid']: x for x in caption_images}
    if len(caption) != len(caption_images):
        raise ValueError('Duplicate caption image ID.')
    by_image = defaultdict(list)
    boxes = {}
    for aid, ann in anns.items():
        if ann.get('iscrowd', 0):
            continue
        boxes[aid] = normalized_box(ann, images[ann['image_id']])
        by_image[ann['image_id']].append(aid)
    subjects = sorted(set(s for a in by_action.values() for s in a))
    rows, stats = [], Counter()
    for sid in subjects:
        if sid not in anns:
            raise ValueError(f'Missing actor annotation {sid}.')
        ann = anns[sid]
        iid = ann['image_id']
        if iid not in caption:
            raise ValueError(f'Image {iid} missing from caption split metadata.')
        if caption[iid]['split'] not in {'train', 'restval'}:
            stats['excluded_caption_heldout_subjects'] += 1
            continue
        if sid not in boxes:
            stats['excluded_crowd_subjects'] += 1
            continue
        if categories[ann['category_id']] != 'person':
            raise ValueError('V-COCO agent must be a person.')
        positives = set()
        for name, role, _ in RELATION_SPECS:
            item = by_action[name].get(sid)
            if item and item['image_id'] != iid:
                raise ValueError('Actor/image ID mismatch.')
            if item and item['label'] == 1:
                oid = item['roles'].get(role, 0)
                if oid:
                    if oid not in anns or anns[oid]['image_id'] != iid or oid == sid:
                        raise ValueError('Invalid positive role object.')
                    if oid in boxes:
                        positives.add(oid)
                    else:
                        stats['unusable_positive_box'] += 1
                else:
                    stats['positive_action_without_visible_role'] += 1
        # Keep every positive; fill with deterministic same-image candidates.
        candidates = sorted(set(by_image[iid]) - {sid} - positives)
        candidates = sorted(positives) + candidates[:max(0, max_objects - len(positives))]
        for oid in candidates:
            labels = []
            for name, role, _ in RELATION_SPECS:
                item = by_action[name].get(sid)
                label = -1
                if item and item['label'] == 0:
                    label = 0  # Annotated absence of this actor action implies absence for any B.
                elif item and item['label'] == 1 and item['roles'].get(role, 0) == oid:
                    label = 1
                # label=1 with another B, missing role, or label=-1 remains UNKNOWN.
                labels.append(label)
            if all(v == -1 for v in labels):
                continue
            image = caption[iid]
            rows.append({'image_id': iid, 'filename': image['filename'],
                         'filepath': image.get('filepath', ''), 'subject_id': sid,
                         'object_id': oid, 'subject': 'person',
                         'object': categories[anns[oid]['category_id']],
                         'subject_box': boxes[sid], 'object_box': boxes[oid],
                         'labels': labels, 'caption_split': image['split']})
    return rows, dict(stats)


def diagnostic_subset(rows, limit, seed=42):
    if limit < 1:
        raise ValueError('Pair limit must be positive.')
    rng = random.Random(seed)
    pos = [r for r in rows if 1 in r['labels']]
    neg = [r for r in rows if 1 not in r['labels']]
    rng.shuffle(pos)
    rng.shuffle(neg)
    chosen = pos[:min(len(pos), limit // 2)]
    used = {(r['subject_id'], r['object_id']) for r in chosen}
    by_target = defaultdict(list)
    for row in rows:
        by_target[(row['image_id'], row['object_id'])].append(row)
    for anchor in list(chosen):
        candidates = by_target[(anchor['image_id'], anchor['object_id'])]
        for other in candidates:
            key = (other['subject_id'], other['object_id'])
            if (key not in used and other['subject_id'] != anchor['subject_id']
                    and any(a == 1 and b == 0 for a, b in zip(anchor['labels'], other['labels']))):
                chosen.append(other)
                used.add(key)
                break
        if len(chosen) >= limit:
            break
    # Fill remaining slots with all-zero/unknown pairs before more positives.
    # Positive-first fill previously skewed the intended diagnostic composition.
    remaining = [r for r in neg + pos if (r['subject_id'], r['object_id']) not in used]
    chosen += remaining[:max(0, limit - len(chosen))]
    return sorted(chosen, key=lambda r: (r['image_id'], r['subject_id'], r['object_id']))


def contrast_edges(rows, max_per_positive=4):
    """Only annotated-zero claims can be negative; no role swaps inferred false."""
    by_target = defaultdict(list)
    for i, row in enumerate(rows):
        by_target[(row['image_id'], row['object_id'])].append(i)
    edges = []
    for i, row in enumerate(rows):
        for r, label in enumerate(row['labels']):
            if label != 1:
                continue
            wrong_pred = [(i, c, 'predicate') for c, v in enumerate(row['labels']) if v == 0]
            wrong_actor = [(j, r, 'actor') for j in by_target[(row['image_id'], row['object_id'])]
                           if rows[j]['subject_id'] != row['subject_id'] and rows[j]['labels'][r] == 0]
            # Alternate types so a predicate-heavy anchor does not exclude all actor edges.
            negatives = []
            while wrong_pred or wrong_actor:
                if wrong_actor:
                    negatives.append(wrong_actor.pop(0))
                if wrong_pred:
                    negatives.append(wrong_pred.pop(0))
            edges += [(i, r, j, c, kind) for j, c, kind in negatives[:max_per_positive]]
    return edges


def pair_statistics(rows):
    return {'pairs': len(rows), 'images': len({r['image_id'] for r in rows}),
            'labels': {name: dict(Counter(str(r['labels'][c]) for r in rows))
                       for c, name in enumerate(RELATIONS)},
            'contrast_edges': dict(Counter(e[-1] for e in contrast_edges(rows)))}
