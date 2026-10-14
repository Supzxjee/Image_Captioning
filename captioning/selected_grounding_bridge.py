"""Convert selected detector claims into normalized, instance-aware relation tensors."""
import math
import torch


# Contrast phrases only: none of these swaps is a verified negative label.
CONTRAST_PHRASES = {'holding': 'wearing', 'riding': 'holding',
                    'sitting_on': 'standing beside', 'carrying': 'looking at',
                    'eating': 'holding', 'talking_on': 'holding'}


def phrase(subject, predicate, obj):
    return f'a photo showing {subject} {predicate.replace("_", " ")} {obj}'


def prepare_selected(entry, max_relations=3):
    if max_relations < 1:
        raise ValueError('max_relations must be positive')
    objects = {}
    for entity in entry['objects']:
        identity, box = entity['instance_id'], entity['bbox']
        if identity in objects:
            raise ValueError('Duplicate instance ID')
        if (len(box) != 4 or not all(math.isfinite(v) for v in box) or
                not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
            raise ValueError('Expected normalized xyxy boxes; pixel boxes not supported')
        objects[identity] = entity
    if len(entry['relations']) > max_relations:
        raise ValueError('Already-selected claims exceed cache capacity; do not silently recap')
    prepared = []
    for claim in entry['relations']:
        subject = objects[claim['subject_instance']]
        obj = objects[claim['object_instance']]
        predicate = claim['relation']
        if subject['instance_id'] == obj['instance_id']:
            raise ValueError('Subject and object must be distinct')
        if subject['label'] != claim['subject'] or obj['label'] != claim['object']:
            raise ValueError('Claim labels disagree with selected instance labels')
        if predicate not in CONTRAST_PHRASES:
            raise ValueError(f'Unsupported predicate: {predicate}')
        confidence = min(subject['confidence'], obj['confidence'])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Expected detector confidence in [0,1]')
        prepared.append(dict(subject_box=list(subject['bbox']), object_box=list(obj['bbox']),
            confidence=confidence, positive=phrase(subject['label'], predicate, obj['label']),
            negative=phrase(subject['label'], CONTRAST_PHRASES[predicate], obj['label']),
            relation=predicate, subject=subject['label'], object=obj['label'],
            subject_instance=subject['instance_id'], object_instance=obj['instance_id']))
    return prepared


def pack_selected(prepared, embeddings, max_relations=3):
    if len(prepared) > max_relations:
        raise ValueError('Too many claims')
    result = dict(tokens=torch.zeros(max_relations, 512, dtype=torch.float16),
                  negative_tokens=torch.zeros(max_relations, 512, dtype=torch.float16),
                  subject_boxes=torch.zeros(max_relations, 4), object_boxes=torch.zeros(max_relations, 4),
                  confidences=torch.zeros(max_relations), mask=torch.zeros(max_relations, dtype=torch.long),
                  relations=[])
    for i, item in enumerate(prepared):
        for key, source in [('tokens', 'positive'), ('negative_tokens', 'negative')]:
            vector = embeddings[item[source]]
            if vector.shape != (512,) or not torch.isfinite(vector).all():
                raise ValueError('Expected finite CLIP EOS vector (512,)')
            result[key][i] = vector
        for key, source in [('subject_boxes', 'subject_box'), ('object_boxes', 'object_box')]:
            result[key][i] = torch.tensor(item[source])
        result['confidences'][i] = item['confidence']
        result['mask'][i] = 1
        result['relations'].append({k: v for k, v in item.items()
                                    if k not in {'subject_box', 'object_box', 'confidence'}})
    return result
