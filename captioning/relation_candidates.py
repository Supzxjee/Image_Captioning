"""Candidate vocabulary and conservative selection; scores require visual evidence."""
import math

RIDE_OBJECTS = frozenset({'skateboard', 'snowboard', 'surfboard', 'bicycle', 'motorcycle', 'horse'})
WEAR_OBJECTS = frozenset({'tie', 'backpack', 'handbag', 'baseball glove', 'hat', 'shirt', 'shoe'})
FOOD_OBJECTS = frozenset({'donut', 'hot dog', 'sandwich', 'pizza', 'banana', 'apple', 'orange', 'cake'})
NEUTRAL = 'near'


def candidate_phrases(subject, obj):
    """Classes determine alternatives, never the assigned action. Alternatives can coexist."""
    if subject != 'person':
        raise ValueError('This action audit currently supports person-object pairs only.')
    predicates = ['holding', 'carrying', 'standing_beside', 'sitting_beside', 'kneeling_beside']
    if obj in RIDE_OBJECTS:
        predicates += ['riding', 'sitting_on', 'standing_on', 'kneeling_on']
    if obj in WEAR_OBJECTS:
        predicates += ['wearing', 'adjusting']
    if obj in FOOD_OBJECTS:
        predicates += ['eating']
    if obj == 'cell phone':
        predicates += ['talking_on', 'looking_at']
    if obj == 'remote':
        predicates += ['using']
    predicates.append(NEUTRAL)
    article = 'an' if obj[:1].lower() in 'aeiou' else 'a'
    return [{'predicate': p, 'text': f'a photo of a person {p.replace("_", " ")} {article} {obj}'}
            for p in predicates]


def select_candidate(candidates, crop_scores, full_scores, min_gap=0.01):
    """Conservative agreement over cosine ranks. This is not calibrated correctness."""
    if len(candidates) < 2 or len(candidates) != len(crop_scores) or len(candidates) != len(full_scores):
        raise ValueError('Candidate and score lengths must match, with at least two candidates.')
    if any(not math.isfinite(x) for x in list(crop_scores) + list(full_scores)):
        raise ValueError('Non-finite candidate score.')
    if min_gap < 0:
        raise ValueError('min_gap must be nonnegative.')
    crop_order = sorted(range(len(candidates)), key=lambda i: crop_scores[i], reverse=True)
    full_order = sorted(range(len(candidates)), key=lambda i: full_scores[i], reverse=True)
    winner = crop_order[0]
    crop_gap = crop_scores[winner] - crop_scores[crop_order[1]]
    full_gap = full_scores[full_order[0]] - full_scores[full_order[1]]
    reason = ('view_disagreement' if winner != full_order[0] else
              'neutral_winner' if candidates[winner]['predicate'] == NEUTRAL else
              'ambiguous_scores' if min(crop_gap, full_gap) < min_gap else 'candidate_for_review')
    return {'proposed_predicate': candidates[winner]['predicate'] if reason == 'candidate_for_review' else None,
            'crop_top': candidates[winner]['predicate'],
            'full_top': candidates[full_order[0]]['predicate'],
            'crop_gap': crop_gap, 'full_gap': full_gap, 'decision': reason}


def union_crop_box(subject_box, object_box, width, height, padding=0.15):
    """Pixel crop from normalized xyxy boxes, preserving both complete boxes."""
    for box in (subject_box, object_box):
        if len(box) != 4 or any(not math.isfinite(x) for x in box):
            raise ValueError('Invalid normalized box.')
        if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            raise ValueError('Box must be nonempty normalized xyxy.')
    if width <= 0 or height <= 0 or padding < 0:
        raise ValueError('Invalid crop dimensions/padding.')
    x1, y1 = min(subject_box[0], object_box[0]), min(subject_box[1], object_box[1])
    x2, y2 = max(subject_box[2], object_box[2]), max(subject_box[3], object_box[3])
    px, py = (x2 - x1) * padding, (y2 - y1) * padding
    return (max(0, math.floor((x1 - px) * width)), max(0, math.floor((y1 - py) * height)),
            min(width, math.ceil((x2 + px) * width)), min(height, math.ceil((y2 + py) * height)))
