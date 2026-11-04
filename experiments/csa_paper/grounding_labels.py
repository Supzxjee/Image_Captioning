"""CPU-only weak category/word labels; no neural or NLP model imports."""
from collections import defaultdict
import hashlib
import unicodedata
import numpy as np


def cohort(images, count=10000):
    rows = [r for r in images if r['split'] in ('train', 'restval')]
    rows.sort(key=lambda r: hashlib.sha256(('csa-grounded-attention-v1/42/'+r['filename']).encode()).hexdigest())
    selected = rows[:count]
    if len(selected) != count or len({r['cocoid'] for r in selected}) != count:
        raise ValueError('Insufficient/duplicate TRAIN IDs')
    return selected


def lexicon(categories):
    irregular = {'person': 'people', 'mouse': 'mice', 'knife': 'knives',
                 'bus': 'buses', 'sheep': 'sheep', 'skis': 'skis', 'scissors': 'scissors'}
    result = {}
    for c in categories:
        name = c['name']; words = name.split(); last = words[-1]
        plural = irregular.get(last, last[:-1]+'ies' if last.endswith('y') and last[-2] not in 'aeiou' else
                               last+'es' if last.endswith(('s', 'ch', 'sh', 'x')) else last+'s')
        forms = {name, ' '.join(words[:-1]+[plural])}
        if name == 'person':
            forms |= {'man','men','woman','women','boy','boys','girl','girls','child','children'}
        for form in forms:
            if form in result and result[form] != c['id']:
                raise ValueError('Ambiguous lexicon')
            result[form] = c['id']
    return result


def mentions(tokens, lookup):
    words = [unicodedata.normalize('NFKC', t).lower() for t in tokens]
    maximum = max(len(k.split()) for k in lookup)
    i = 0
    while i < len(words):
        match = next(((' '.join(words[i:i+n]), n) for n in range(min(maximum,len(words)-i),0,-1)
                      if ' '.join(words[i:i+n]) in lookup), None)
        if match:
            phrase, width = match
            yield dict(phrase=phrase, category_id=lookup[phrase], start=i, end=i+width,
                       target_position=i+width-1)
            i += width
        else:
            i += 1


def positive_set(boxes, kinds, annotations, category):
    """Pixel xyxy proposals, COCO xywh GT. Never infer absence from proposals."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1,4)
    if len(kinds) != len(boxes) or not np.isfinite(boxes).all() or (boxes[:,2:] <= boxes[:,:2]).any():
        raise ValueError('Corrupt proposal boxes/kinds')
    gt = [a for a in annotations if a['category_id'] == category]
    if any(a['iscrowd'] for a in gt): return [], 'crowd', []
    if not gt: return [], 'missing_gt', []
    xywh = np.asarray([a['bbox'] for a in gt],dtype=np.float64)
    if not np.isfinite(xywh).all(): raise ValueError('Nonfinite GT')
    target = np.concatenate((xywh[:,:2],xywh[:,:2]+xywh[:,2:]),axis=1)
    # Official annotations can contain unusable boxes. Skip this category mention,
    # not just the bad instance: missing an instance could create false negatives.
    if (xywh[:,2:]<=0).any(): return [], 'invalid_gt', target.tolist()
    area = np.maximum(0,np.minimum(boxes[:,None,2:],target[None,:,2:])-
                      np.maximum(boxes[:,None,:2],target[None,:,:2])).prod(-1)
    union = (boxes[:,2:]-boxes[:,:2]).prod(-1)[:,None]+xywh[:,2:].prod(-1)[None,:]-area
    iou = area/union
    positives = [int(i) for i in np.flatnonzero((iou.max(1)>=.5) & np.asarray([k=='predicted_roi' for k in kinds]))]
    return positives, 'eligible' if positives else 'no_positive', target.tolist()


def caption_labels(tokens, sequence, vocabulary, boxes, kinds, annotations, lookup):
    word2id = {w:i for i,w in enumerate(vocabulary)}
    for row in mentions(tokens,lookup):
        t = row['target_position']
        normalized = unicodedata.normalize('NFKC', tokens[t]).lower()
        expected = word2id.get(normalized,3)
        # Decoder logit t (input BOS then prefix) predicts original content token t.
        if t >= len(sequence)-2:
            row.update(status='truncated', positive_regions=[], gt_boxes=[])
        elif sequence[t+1] != expected:
            raise ValueError('Target/sequence alignment differs')
        elif any(word2id.get(unicodedata.normalize('NFKC', w).lower(),3)<4 for w in tokens[row['start']:row['end']]):
            row.update(status='unknown_token', positive_regions=[], gt_boxes=[])
        else:
            p, status, gt = positive_set(boxes,kinds,annotations,row['category_id'])
            row.update(status=status, positive_regions=p, gt_boxes=gt)
            if status == 'invalid_gt':
                row['invalid_gt_annotations'] = [dict(id=a.get('id'),bbox=a['bbox']) for a in annotations
                    if a['category_id']==row['category_id'] and (a['bbox'][2]<=0 or a['bbox'][3]<=0)]
        row['target_id'] = expected
        yield row
