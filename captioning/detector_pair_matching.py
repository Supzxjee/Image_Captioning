"""Deterministic class-aware one-to-one matching for conditional box diagnostics."""
import math


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection/union if union > 0 else 0


def normalized_detection_box(box, width, height, coordinate_format='pixel_xyxy'):
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError('Invalid detection box.')
    if coordinate_format == 'pixel_xyxy':
        box = [box[0]/width, box[1]/height, box[2]/width, box[3]/height]
    elif coordinate_format != 'normalized_xyxy':
        raise ValueError('Unknown coordinate format.')
    box = [max(0, box[0]), max(0, box[1]), min(1, box[2]), min(1, box[3])]
    if not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1:
        raise ValueError('Empty/out-of-bounds detection.')
    return box


def match_instances(ground_truth, detections, min_iou=.5):
    if not 0 < min_iou <= 1:
        raise ValueError('IoU threshold must be in (0,1].')
    edges = []
    for aid, gt in ground_truth.items():
        for index, detection in enumerate(detections):
            if gt['label'] != detection['label']:
                continue
            iou = box_iou(gt['box'], detection['box'])
            if iou >= min_iou:
                edges.append((-iou, -detection['confidence'], aid, index))
    assigned, used = {}, set()
    for negative_iou, _, aid, index in sorted(edges):
        if aid not in assigned and index not in used:
            assigned[aid] = {'index': index, 'iou': -negative_iou,
                              'box': detections[index]['box'], 'confidence': detections[index]['confidence']}
            used.add(index)
    return assigned


def matched_pairs(rows, assignments):
    result, missing = [], {'subject': 0, 'object': 0, 'both': 0}
    for row in rows:
        subject, obj = assignments.get(row['subject_id']), assignments.get(row['object_id'])
        if subject is None or obj is None:
            key = 'both' if subject is None and obj is None else 'subject' if subject is None else 'object'
            missing[key] += 1
            continue
        result.append({**row, 'gt_subject_box': row['subject_box'], 'gt_object_box': row['object_box'],
                       'subject_box': subject['box'], 'object_box': obj['box'],
                       'subject_match_iou': subject['iou'], 'object_match_iou': obj['iou'],
                       'subject_detector_confidence': subject['confidence'],
                       'object_detector_confidence': obj['confidence']})
    return result, missing
