"""Rule-based COCO object consistency scoring for caption candidates."""
import re
from pathlib import PurePosixPath


COCO_LABELS = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush',
)

SPECIAL_ALIASES = {
    'person': ('people', 'man', 'men', 'woman', 'women', 'boy', 'boys', 'girl',
               'girls', 'child', 'children', 'kid', 'kids', 'rider', 'riders'),
    'bicycle': ('bike', 'bikes'),
    'car': ('cars', 'vehicle', 'vehicles', 'automobile', 'automobiles'),
    'motorcycle': ('motorbike', 'motorbikes'),
    'airplane': ('plane', 'planes', 'aircraft', 'jet', 'jets'),
    'bus': ('buses',),
    'boat': ('ship', 'ships'),
    'traffic light': ('traffic signal', 'traffic signals'),
    'handbag': ('purse', 'purses'),
    'suitcase': ('luggage',),
    'sports ball': ('ball', 'balls'),
    'baseball glove': ('glove', 'gloves'),
    'tennis racket': ('racket', 'rackets', 'racquet', 'racquets'),
    'couch': ('sofa', 'sofas'),
    'potted plant': ('plant', 'plants'),
    'dining table': ('table', 'tables'),
    'tv': ('television', 'televisions', 'monitor', 'monitors'),
    'cell phone': ('phone', 'phones', 'mobile phone', 'mobile phones'),
    'refrigerator': ('fridge', 'fridges'),
    'donut': ('doughnut', 'doughnuts'),
    'teddy bear': ('stuffed bear', 'stuffed bears'),
    'hair drier': ('hair dryer', 'hair dryers'),
}


def _pluralize(phrase):
    words = phrase.split()
    last = words[-1]
    if last.endswith('y') and len(last) > 1 and last[-2] not in 'aeiou':
        words[-1] = last[:-1] + 'ies'
    elif last.endswith(('s', 'x', 'ch', 'sh')):
        words[-1] = last + 'es'
    else:
        words[-1] = last + 's'
    return ' '.join(words)


def _alias_table():
    aliases = []
    for label in COCO_LABELS:
        values = {label, _pluralize(label), *SPECIAL_ALIASES.get(label, ())}
        for value in values:
            aliases.append((tuple(value.split()), label))
    return sorted(aliases, key=lambda item: (-len(item[0]), -len(' '.join(item[0]))))


ALIASES = _alias_table()


def extract_caption_objects(caption):
    """Return non-overlapping COCO classes explicitly mentioned in a caption."""
    tokens = re.findall(r"[a-z0-9]+", caption.lower())
    occupied, labels = set(), []
    for alias, label in ALIASES:
        width = len(alias)
        for start in range(len(tokens) - width + 1):
            positions = set(range(start, start + width))
            if positions & occupied or tuple(tokens[start:start + width]) != alias:
                continue
            occupied.update(positions)
            if label not in labels:
                labels.append(label)
    return labels


def index_detections(source, objects_field='objects', name_key='label', confidence_key='conf',
                     min_confidence=0.5):
    """Index maximum accepted detector confidence by filename and COCO label."""
    if not isinstance(source, dict):
        raise ValueError('Detection source must be a filename-to-entry mapping.')
    indexed = {}
    for key, entry in source.items():
        filename = PurePosixPath(str(key).replace('\\', '/')).name
        if filename in indexed:
            raise ValueError(f'Ambiguous detection filename: {filename}')
        objects = entry if isinstance(entry, list) else entry.get(objects_field)
        if not isinstance(objects, list):
            raise ValueError(f'{filename}: expected object list in {objects_field!r}')
        accepted = {}
        for detection in objects:
            if not isinstance(detection, dict):
                continue
            label = detection.get(name_key)
            confidence = detection.get(confidence_key)
            if not isinstance(label, str) or not isinstance(confidence, (int, float)):
                continue
            label = ' '.join(label.lower().strip().split())
            if label in COCO_LABELS and confidence >= min_confidence:
                accepted[label] = max(accepted.get(label, 0.0), float(confidence))
        indexed[filename] = accepted
    return indexed


def object_consistency(caption, detected, hallucination_penalty=1.0):
    """Score supported mentions and penalize unsupported COCO-class mentions."""
    mentioned = extract_caption_objects(caption)
    supported = [label for label in mentioned if label in detected]
    hallucinated = [label for label in mentioned if label not in detected]
    if mentioned:
        support = sum(detected[label] for label in supported) / len(mentioned)
        hallucination_rate = len(hallucinated) / len(mentioned)
        score = support - hallucination_penalty * hallucination_rate
    else:
        support, hallucination_rate, score = 0.0, 0.0, 0.0
    return {
        'score': float(score),
        'mentioned': mentioned,
        'supported': supported,
        'hallucinated': hallucinated,
        'support': float(support),
        'hallucination_rate': float(hallucination_rate),
    }


def rerank_record(record, detected, weight, hallucination_penalty=1.0):
    candidates = record.get('candidates', [])
    if not candidates:
        raise ValueError(f'Image {record.get("image_id")}: no candidates')
    language = [float(candidate['logprob']) for candidate in candidates]
    low, high = min(language), max(language)
    enriched = []
    for rank, (candidate, score) in enumerate(zip(candidates, language)):
        language_score = 1.0 if high == low else (score - low) / (high - low)
        consistency = object_consistency(
            candidate['caption'], detected, hallucination_penalty)
        object_score = ((consistency['score'] + hallucination_penalty) /
                        (1.0 + hallucination_penalty))
        combined = (1.0 - weight) * language_score + weight * object_score
        enriched.append({
            **candidate,
            'original_rank': rank,
            'language_score': float(language_score),
            'object_score': float(object_score),
            'combined_score': float(combined),
            'object_consistency': consistency,
        })
    selected = max(enriched, key=lambda item: (item['combined_score'],
                                                -item['original_rank']))
    return selected, enriched
