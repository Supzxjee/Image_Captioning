"""Semantic-aware relation prompts derived only from saved detector boxes.

The generator keeps instance identities while reasoning, emits one canonical
relation per instance pair, and only collapses to class names after resolving
multi-instance conflicts.  It never consults COCO captions.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import PurePosixPath


PERSON = 'person'
SUPPORTS = {'dining table', 'bed', 'couch', 'bench', 'chair'}
WEARABLE = {'tie', 'backpack'}
CARRIED = {'handbag', 'suitcase'}
HOLDABLE = {
    'sports ball', 'baseball bat', 'baseball glove', 'tennis racket', 'frisbee',
    'bottle', 'cup', 'cell phone', 'remote', 'umbrella', 'apple', 'banana',
    'book', 'scissors', 'fork', 'knife', 'spoon', 'sandwich', 'hot dog', 'donut',
}
RIDEABLE = {'bicycle', 'motorcycle', 'horse', 'skateboard', 'snowboard', 'skis', 'surfboard'}
VEHICLE_CONTAINERS = {'car', 'truck', 'bus', 'train', 'boat', 'airplane'}
VEHICLE_PASSENGERS = {'person', 'dog', 'cat', 'cow', 'horse'}
FOOD_CONTAINERS = {'bowl', 'cup'}
FOOD_CONTENTS = {
    'broccoli', 'carrot', 'apple', 'orange', 'banana', 'sandwich', 'hot dog',
    'pizza', 'donut', 'cake',
}

RELATION_PRIORITY = {
    'wearing': 1.00,
    'carrying': 0.99,
    'holding': 0.98,
    'riding': 0.97,
    'on': 0.94,
    'inside': 0.92,
    'overlapping': 0.78,
    'left_of': 0.66,
    'right_of': 0.66,
    'above': 0.64,
    'below': 0.64,
    'near': 0.50,
}


@dataclass(frozen=True)
class Detection:
    instance_id: str
    label: str
    confidence: float
    box: tuple[float, float, float, float]

    @property
    def width(self):
        return self.box[2] - self.box[0]

    @property
    def height(self):
        return self.box[3] - self.box[1]

    @property
    def area(self):
        return self.width * self.height

    @property
    def center(self):
        return ((self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2)


def _clean_label(value):
    return ' '.join(value.lower().strip().split()) if isinstance(value, str) else ''


def _intersection(a, b):
    width = max(0.0, min(a.box[2], b.box[2]) - max(a.box[0], b.box[0]))
    height = max(0.0, min(a.box[3], b.box[3]) - max(a.box[1], b.box[1]))
    return width * height


def _iou(a, b):
    intersection = _intersection(a, b)
    union = a.area + b.area - intersection
    return intersection / union if union > 0 else 0.0


def _inside_fraction(inner, outer):
    return _intersection(inner, outer) / inner.area if inner.area > 0 else 0.0


def _axis_overlap(a1, a2, b1, b2):
    intersection = max(0.0, min(a2, b2) - max(a1, b1))
    return intersection / max(1e-6, min(a2 - a1, b2 - b1))


def _normalized_distance(a, b):
    ax, ay = a.center
    bx, by = b.center
    scale = max(1.0, math.sqrt(a.area) + math.sqrt(b.area))
    return math.hypot(ax - bx, ay - by) / scale


def select_detections(entry, objects_field='objects', name_key='label', bbox_key='bbox',
                      confidence_key='conf', min_confidence=0.5, max_objects=10,
                      duplicate_iou=0.8):
    """Validate, rank and same-class-NMS detector boxes while retaining instances."""
    objects = entry if isinstance(entry, list) else entry.get(objects_field)
    if not isinstance(objects, list):
        raise ValueError(f'Expected a detection list in {objects_field!r}.')
    candidates = []
    for source_index, item in enumerate(objects):
        if not isinstance(item, dict):
            continue
        label = _clean_label(item.get(name_key))
        confidence = item.get(confidence_key)
        box = item.get(bbox_key)
        if (not label or not isinstance(confidence, (int, float)) or
                isinstance(confidence, bool) or not math.isfinite(confidence) or
                confidence < min_confidence or not isinstance(box, (list, tuple)) or
                len(box) != 4 or not all(isinstance(v, (int, float)) and
                                        not isinstance(v, bool) and math.isfinite(v)
                                        for v in box)):
            continue
        x1, y1, x2, y2 = map(float, box)
        if x2 <= x1 or y2 <= y1:
            continue
        candidates.append((float(confidence), source_index, label, (x1, y1, x2, y2)))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    kept = []
    label_counts = {}
    for confidence, _, label, box in candidates:
        provisional = Detection('', label, confidence, box)
        if any(old.label == label and _iou(provisional, old) >= duplicate_iou for old in kept):
            continue
        label_counts[label] = label_counts.get(label, 0) + 1
        instance_id = f'{label.replace(" ", "_")}#{label_counts[label]}'
        kept.append(Detection(instance_id, label, confidence, box))
        if len(kept) == max_objects:
            break
    return kept


def _semantic_relation(a, b):
    """Return canonical (subject, relation, object, evidence) or ``None``."""
    # Person interactions take precedence over raw box containment.
    if a.label == PERSON or b.label == PERSON:
        person, item = (a, b) if a.label == PERSON else (b, a)
        containment = _inside_fraction(item, person)
        x_overlap = _axis_overlap(person.box[0], person.box[2], item.box[0], item.box[2])
        person_bottom, item_bottom = person.box[3], item.box[3]
        if item.label in RIDEABLE and x_overlap >= 0.25 and item.center[1] >= person.center[1]:
            if person_bottom <= item_bottom + 0.30 * item.height:
                return person, 'riding', item, 'person-object class and support geometry'
        if containment >= 0.55:
            if item.label in WEARABLE:
                return person, 'wearing', item, 'person contains wearable object'
            if item.label in CARRIED:
                return person, 'carrying', item, 'person contains carried object'
            if item.label in HOLDABLE:
                return person, 'holding', item, 'person contains handheld object'

    # Objects nested in furniture are normally on the visible support surface.
    if a.label in SUPPORTS or b.label in SUPPORTS:
        support, item = (a, b) if a.label in SUPPORTS else (b, a)
        if item.label != PERSON:
            horizontal = _axis_overlap(item.box[0], item.box[2], support.box[0], support.box[2])
            nested = _inside_fraction(item, support)
            surface_band = support.box[1] - 0.15 * support.height <= item.box[3] <= (
                support.box[1] + 0.65 * support.height)
            if horizontal >= 0.35 and (nested >= 0.45 or surface_band):
                return item, 'on', support, 'support-surface geometry'

    # Inside is restricted to class pairs that can be true containers.
    for inner, outer in ((a, b), (b, a)):
        allowed = ((outer.label in VEHICLE_CONTAINERS and inner.label in VEHICLE_PASSENGERS) or
                   (outer.label in FOOD_CONTAINERS and inner.label in FOOD_CONTENTS))
        if allowed and _inside_fraction(inner, outer) >= 0.70:
            return inner, 'inside', outer, 'semantic container whitelist'
    return None


def _geometric_relation(a, b):
    overlap = _iou(a, b)
    if overlap >= 0.15:
        return a, 'overlapping', b, 'box IoU'

    ax, ay = a.center
    bx, by = b.center
    dx, dy = bx - ax, by - ay
    horizontal_scale = max(a.width, b.width, 1.0)
    vertical_scale = max(a.height, b.height, 1.0)
    distance = _normalized_distance(a, b)
    if abs(dx) / horizontal_scale >= 0.30 and abs(dx) / horizontal_scale >= abs(dy) / vertical_scale:
        return (a, 'left_of', b, 'dominant horizontal separation') if dx > 0 else (
            b, 'left_of', a, 'dominant horizontal separation')
    if abs(dy) / vertical_scale >= 0.30:
        return (a, 'above', b, 'dominant vertical separation') if dy > 0 else (
            b, 'above', a, 'dominant vertical separation')
    if distance <= 1.25:
        return a, 'near', b, 'normalized center distance'
    return None


def infer_relations(detections, max_relations=3):
    """Infer diverse, non-conflicting relations from selected instances."""
    candidates = []
    max_area = max((item.area for item in detections), default=1.0)
    for index, first in enumerate(detections):
        for second in detections[index + 1:]:
            if first.label == second.label:
                continue
            relation = _semantic_relation(first, second) or _geometric_relation(first, second)
            if relation is None:
                continue
            subject, predicate, obj, evidence = relation
            confidence = min(subject.confidence, obj.confidence)
            salience = min(1.0, math.sqrt(subject.area * obj.area) / max_area)
            distance_bonus = max(0.0, 1.0 - _normalized_distance(subject, obj))
            score = RELATION_PRIORITY[predicate] + 0.12 * confidence + 0.03 * salience + 0.02 * distance_bonus
            candidates.append((score, subject, predicate, obj, evidence))

    candidates.sort(key=lambda item: (-item[0], item[1].instance_id, item[3].instance_id))
    selected, used_label_pairs = [], set()
    for score, subject, predicate, obj, evidence in candidates:
        # Class-level text cannot distinguish two instances. Keep the strongest
        # relation for a class pair and retain instance IDs in metadata.
        label_pair = tuple(sorted((subject.label, obj.label)))
        if label_pair in used_label_pairs:
            continue
        used_label_pairs.add(label_pair)
        selected.append({
            'subject': subject.label,
            'relation': predicate,
            'object': obj.label,
            'subject_id': subject.instance_id,
            'object_id': obj.instance_id,
            'score': round(score, 6),
            'evidence': evidence,
        })
        if len(selected) == max_relations:
            break
    return selected


def relation_prompt(triplets, detections):
    clauses = [f"{item['subject']} {item['relation'].replace('_', ' ')} {item['object']}"
               for item in triplets]
    if clauses:
        return 'a photo showing ' + ', '.join(clauses)
    labels = []
    for detection in detections:
        if detection.label not in labels:
            labels.append(detection.label)
    return 'a photo of ' + ', '.join(labels) if labels else ''


def build_relation_entry(entry, **kwargs):
    detections = select_detections(entry, **{key: value for key, value in kwargs.items()
                                             if key != 'max_relations'})
    triplets = infer_relations(detections, kwargs.get('max_relations', 3))
    return {
        'objects': [
            {'instance_id': item.instance_id, 'label': item.label,
             'conf': item.confidence, 'bbox': list(item.box)}
            for item in detections
        ],
        'triplets': triplets,
        'prompt': relation_prompt(triplets, detections),
        'heuristic_version': 'H1_2_semantic_instance_aware_v2',
    }


def build_relation_cache(source, **kwargs):
    if isinstance(source, dict) and isinstance(source.get('data'), dict):
        source = source['data']
    if not isinstance(source, dict):
        raise ValueError('Detection source must map image paths or filenames to entries.')
    result = {}
    seen = set()
    for key, entry in source.items():
        filename = PurePosixPath(str(key).replace('\\', '/')).name
        if filename in seen:
            raise ValueError(f'Ambiguous source filename: {filename}')
        seen.add(filename)
        result[str(key)] = build_relation_entry(entry, **kwargs)
    return result
