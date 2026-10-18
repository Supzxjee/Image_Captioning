"""UPT predictions -> instance-linked text; no annotations or reference captions."""
import math

from .selected_relation_prompts import entity_id, phrase

METHOD = 'upt_object_relation_prompt_v1'
PREDICATES = ('holding', 'riding')


def noun(label):
    return label if label in ('skis', 'scissors') else phrase(label)


def relation_text(predicate, obj):
    if predicate not in PREDICATES:
        raise ValueError('Unsupported prompt predicate.')
    # Do not infer surfing/skiing from object class alone. Keep UPT's broad ride.
    if predicate == 'riding' and obj == 'skis':
        return 'a person on skis'
    return f'a person {predicate} {noun(obj)}'


def from_audit(row):
    """Allowlist inference fields; deliberately discard every GT/matched ID/label."""
    detections = [{k: d[k] for k in ('label', 'confidence', 'box')} for d in row['detection_details']]
    claims = [{k: c[k] for k in ('subject', 'object', 'predicate', 'score')} for c in row['candidates']]
    return detections, claims


def from_cache(row, vocabulary):
    h, w = row['resized_size_hw']
    if h <= 0 or w <= 0 or len(row['boxes_resized_xyxy']) != len(row['labels']) or len(row['labels']) != len(row['object_scores']):
        raise ValueError('Invalid cache detection dimensions.')
    detections = []
    for box, label, score in zip(row['boxes_resized_xyxy'], row['labels'], row['object_scores']):
        if not 0 <= label < len(vocabulary['objects']):
            raise ValueError('Invalid cache object label.')
        detections.append({'label': vocabulary['objects'][label].replace('_', ' '), 'confidence': score,
                           'box': [max(0., min(1., x / scale)) for x, scale in zip(box, (w, h, w, h))]})
    if len(row['pairs']) != len(row['native_scores']):
        raise ValueError('Invalid native score dimensions.')
    claims = []
    for (s, o), scores in zip(row['pairs'], row['native_scores']):
        if not 0 <= s < len(detections) or not 0 <= o < len(detections):
            raise ValueError('Invalid cache pair indices.')
        verbs = vocabulary['verbs_by_object'][row['labels'][o]]
        if len(scores) != len(verbs):
            raise ValueError('Sparse native score row differs from ontology.')
        for verb, score in zip(verbs, scores):
            name = vocabulary['verbs'][verb]
            if name in ('hold', 'ride'):
                claims.append({'subject': s, 'object': o, 'predicate': 'holding' if name == 'hold' else 'riding',
                               'score': score})
    return detections, claims


def build_prompt(filename, detections, claims, policy, max_relations=3, object_threshold=.5, max_object_names=8):
    if max_relations < 1 or max_object_names < 1 or not math.isfinite(object_threshold) or not 0 <= object_threshold <= 1:
        raise ValueError('Invalid prompt options.')
    thresholds = {p: policy['predicates'][p]['threshold'] for p in PREDICATES}
    if any(not math.isfinite(t) or not 0 <= t <= 1.01 for t in thresholds.values()):
        raise ValueError('Invalid locked policy threshold.')
    objects = []
    for i, d in enumerate(detections):
        label, confidence, box = d['label'], d['confidence'], d['box']
        if not isinstance(label, str) or not label.strip() or label != label.strip():
            raise ValueError('Invalid object name.')
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Invalid object confidence.')
        if len(box) != 4 or not all(math.isfinite(v) for v in box) or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1:
            raise ValueError('Invalid normalized predicted box.')
        objects.append({'index': i, 'instance_id': entity_id(filename, label, box),
                        'label': label, 'confidence': confidence, 'bbox': box})
    unique = {}
    for c in claims:
        s, o, p, score = (c[k] for k in ('subject', 'object', 'predicate', 'score'))
        if p not in PREDICATES or s == o or not 0 <= s < len(objects) or not 0 <= o < len(objects):
            raise ValueError('Invalid relation pair/predicate.')
        if objects[s]['label'] != 'person' or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Invalid relation actor/score.')
        key = (s, p, o)
        if key not in unique or score > unique[key]['score']:
            unique[key] = {'subject_index': s, 'object_index': o, 'subject_instance': objects[s]['instance_id'],
                           'object_instance': objects[o]['instance_id'], 'subject': 'person', 'object': objects[o]['label'],
                           'predicate': p, 'score': score, 'threshold': thresholds[p],
                           'text': relation_text(p, objects[o]['label'])}
    # Margin is only a fixed cap ordering; not calibrated cross-predicate confidence.
    accepted = sorted((c for c in unique.values() if c['score'] >= c['threshold']),
                      key=lambda c: (-(c['score'] - c['threshold']), c['subject_index'], c['predicate'], c['object_index']))
    selected = accepted[:max_relations]
    endpoints = {c[k] for c in selected for k in ('subject_index', 'object_index')}
    kept = [d for d in objects if d['confidence'] >= object_threshold or d['index'] in endpoints]
    names = list(dict.fromkeys(d['label'] for d in sorted(kept, key=lambda d: (-d['confidence'], d['index']))))[:max_object_names]
    object_prompt = 'a photo containing ' + ', '.join(noun(n) for n in names) + '.' if names else ''
    texts = list(dict.fromkeys(c['text'] for c in selected))
    relation_prompt = 'a photo showing ' + '; '.join(texts) + '.' if texts else ''
    previews = sorted(unique.values(), key=lambda c: (-c['score'], c['subject_index'], c['predicate'], c['object_index']))[:3]
    return {'objects': kept, 'relations': selected, 'object_prompt': object_prompt,
            'relation_prompt': relation_prompt, 'prompt': relation_prompt or object_prompt or 'a photo.',
            'mode': 'object_relation' if selected else 'object_only' if names else 'no_objects',
            'above_threshold_claims': len(accepted), 'capped_claims': len(accepted) - len(selected),
            'candidate_preview': [{**c, 'passes_locked_threshold': c['score'] >= c['threshold']} for c in previews]}
