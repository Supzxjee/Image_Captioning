"""Evaluate native HOI proposals without pretending unlinked targets are false."""
from collections import Counter
import math

from .detector_pair_matching import match_instances
from .supervised_relations import normalized_box

PREDICATES = ('holding', 'riding')


def audit_image(record, prediction, annotations, image, categories):
    """prediction contains only detected boxes/classes, pairs and scores, no GT IDs."""
    detections = prediction['detections']
    for item in detections:
        box = item['box']
        if len(box) != 4 or not all(math.isfinite(x) for x in box) or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            raise ValueError('Invalid normalized predicted box.')
    gt = {}
    for ann in annotations:
        if ann['image_id'] != record['image_id']:
            raise ValueError('Wrong image annotation.')
        if ann.get('iscrowd', 0):
            continue
        try:
            box = normalized_box(ann, image)
        except ValueError:
            continue
        gt[ann['id']] = {'label': categories[ann['category_id']], 'box': box}
    assignments = match_instances(gt, detections, .5)
    reverse = {value['index']: aid for aid, value in assignments.items()}
    actions = {(row['subject_id'], row['predicate']): row['label'] for row in record['actor_action_labels']}
    positives = {(row['subject_id'], row['predicate'], row['object_id']) for row in record['relations']}
    candidates, seen = [], set()
    for row in prediction['claims']:
        s, o, predicate, score = row['subject'], row['object'], row['predicate'], row['score']
        key = s, predicate, o
        if key in seen or predicate not in PREDICATES or s == o or not 0 <= s < len(detections) or not 0 <= o < len(detections):
            raise ValueError('Duplicate or invalid native claim.')
        if detections[s]['label'] != 'person' or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid native person/score.')
        seen.add(key)
        sid, oid = reverse.get(s), reverse.get(o)
        target = (sid, predicate, oid)
        label = 1 if target in positives else 0 if actions.get((sid, predicate), -1) == 0 else -1
        candidates.append({**row, 'label': label, 'matched_subject_id': sid, 'matched_object_id': oid})
    # Reference denominator includes every positive link, regardless of availability.
    available = {(c['matched_subject_id'], c['predicate'], c['matched_object_id']) for c in candidates if c['label'] == 1}
    localized = set()
    for sid, p, oid in positives:
        if sid in assignments and oid in assignments:
            localized.add((sid, p, oid))
    return {'image_id': record['image_id'], 'filename': record['filename'], 'candidates': candidates,
            'positive_links': [dict(subject_id=s, predicate=p, object_id=o,
                                    localized=(s, p, o) in localized, available=(s, p, o) in available)
                               for s, p, o in sorted(positives)],
            'detections': len(detections), 'native_pairs': prediction.get('native_pairs', 0)}


def fit_thresholds(records, precision=.9, min_known=5):
    policies = {}
    for predicate in PREDICATES:
        known = [c for r in records for c in r['candidates'] if c['predicate'] == predicate and c['label'] >= 0]
        counts = Counter(c['label'] for c in known)
        threshold, support = 1.01, 0
        if counts[0] and counts[1]:
            # Group ties and sweep once; lowest qualifying score keeps maximum coverage.
            tp = total = 0
            grouped = {}
            for c in known:
                grouped.setdefault(c['score'], []).append(c['label'])
            for score in sorted(grouped, reverse=True):
                total += len(grouped[score]); tp += sum(grouped[score])
                if total >= min_known and tp / total >= precision:
                    threshold, support = score, total
        policies[predicate] = {'threshold': threshold, 'known': len(known), 'positive': counts[1],
                               'negative': counts[0], 'selected_known_calibration': support}
    return {'predicates': policies, 'precision_target': precision, 'minimum_known': min_known,
            'warning': 'Known-label calibration precision is not precision on all proposals; unknowns excluded.'}


def evaluate(records, policy):
    result = {}
    for predicate in PREDICATES:
        threshold = policy['predicates'][predicate]['threshold']
        positive = [p for r in records for p in r['positive_links'] if p['predicate'] == predicate]
        candidates = [c for r in records for c in r['candidates'] if c['predicate'] == predicate]
        selected = [c for c in candidates if c['score'] >= threshold]
        labels = Counter(c['label'] for c in selected)
        tp, fp, unknown = labels[1], labels[0], labels[-1]
        result[predicate] = {'annotated_positive_links': len(positive),
                              'localized_positive_links': sum(p['localized'] for p in positive),
                              'available_positive_links': sum(p['available'] for p in positive),
                              'all_candidate_claims': len(candidates),
                              'known_candidate_claims': sum(c['label'] >= 0 for c in candidates),
                              'accepted_true': tp, 'accepted_false_known': fp, 'accepted_unknown': unknown,
                              'known_precision': tp / (tp + fp) if tp + fp else None,
                              'full_positive_recall': tp / len(positive) if positive else None,
                              'images_with_accepted_claims': sum(any(c['predicate'] == predicate and c['score'] >= threshold
                                                                    for c in r['candidates']) for r in records)}
    return {'images': len(records), 'images_without_native_pairs': sum(r['native_pairs'] == 0 for r in records),
            'by_predicate': result}
