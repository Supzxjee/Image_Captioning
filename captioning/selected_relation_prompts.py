"""Select scored detector claims without consulting annotation IDs or GT labels."""
import hashlib
import json
import math
from collections import defaultdict
from captioning.supervised_relations import RELATIONS


def entity_id(filename, label, box):
    payload = json.dumps([filename, label, box], separators=(',', ':'))
    return 'det_' + hashlib.sha256(payload.encode()).hexdigest()[:12]


def phrase(label):
    article = 'an' if label[:1].lower() in 'aeiou' else 'a'
    return f'{article} {label}'


def build_selected_prompts(scored_pairs, thresholds, max_relations=3):
    if len(thresholds) != len(RELATIONS) or max_relations < 1:
        raise ValueError('Invalid thresholds/relation limit.')
    if not all(math.isfinite(x) for x in thresholds):
        raise ValueError('Thresholds must be finite.')
    groups = defaultdict(list)
    for row in scored_pairs:
        if len(row['scores']) != len(RELATIONS) or not all(math.isfinite(v) for v in row['scores']):
            raise ValueError('Invalid scores.')
        for role in ('subject', 'object'):
            box = row[role+'_box']
            if (len(box) != 4 or not all(math.isfinite(v) for v in box)
                    or not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1):
                raise ValueError('Expected normalized nonempty detector boxes.')
        groups[row['filename']].append(row)
    result = {}
    for filename, rows in sorted(groups.items()):
        objects, candidates = {}, {}
        for row in rows:
            ids = []
            for role in ('subject', 'object'):
                box, label = row[role+'_box'], row[role]
                identity = entity_id(filename, label, box)
                ids.append(identity)
                objects[identity] = {'instance_id': identity, 'label': label, 'bbox': box,
                    'confidence': row.get(role+'_detector_confidence'), 'prompt': 'a photo of '+phrase(label)}
            for c, relation in enumerate(RELATIONS):
                score = row['scores'][c]
                if score < thresholds[c] or ids[0] == ids[1]:
                    continue
                key = (ids[0], relation, ids[1])
                claim = {'subject_instance': ids[0], 'subject': row['subject'],
                         'object_instance': ids[1], 'object': row['object'], 'relation': relation,
                         'score': score, 'threshold': thresholds[c], 'score_margin': score-thresholds[c],
                         'prompt': f'a photo of {phrase(row["subject"])} {relation.replace("_", " ")} {phrase(row["object"])}'}
                if key not in candidates or score > candidates[key]['score']:
                    candidates[key] = claim
        # Margin is a deterministic cap priority, NOT calibrated cross-class confidence.
        ordered = sorted(candidates.values(), key=lambda c: (-c['score_margin'], c['subject_instance'],
                                                            c['relation'], c['object_instance']))
        selected = ordered[:max_relations]
        labels = sorted({entity['label'] for entity in objects.values()})
        object_text = 'a photo of '+', '.join(phrase(label) for label in labels)
        relation_text = ('a photo showing '+', '.join(
            f'{phrase(c["subject"])} {c["relation"].replace("_", " ")} {phrase(c["object"])}' for c in selected))
        result[filename] = {'objects': list(objects.values()), 'relations': selected,
                            'object_prompt': object_text,
                            'relation_prompt': relation_text if selected else '',
                            'prompt': relation_text if selected else object_text,
                            'mode': 'object_relation' if selected else 'object_only',
                            'above_threshold_claims': len(ordered), 'capped_claims': len(ordered)-len(selected)}
    return result
