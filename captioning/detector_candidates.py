"""Ordered person-object candidates from detector inventory only, no GT match."""
from captioning.selected_relation_prompts import entity_id


def person_object_pairs(image, objects):
    pairs = []
    for subject in objects:
        if subject['label'] != 'person':
            continue
        sid = entity_id(image['filename'], subject['label'], subject['bbox'])
        for obj in objects:
            oid = entity_id(image['filename'], obj['label'], obj['bbox'])
            if sid == oid:
                continue
            pairs.append({'image_id': image['image_id'], 'filename': image['filename'],
                'filepath': image['filepath'], 'caption_split': image['caption_split'],
                'subject_id': sid, 'object_id': oid, 'subject': 'person', 'object': obj['label'],
                'subject_box': subject['bbox'], 'object_box': obj['bbox'],
                'subject_detector_confidence': subject['confidence'],
                'object_detector_confidence': obj['confidence']})
    return pairs
