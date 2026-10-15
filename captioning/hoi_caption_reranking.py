"""Conservative lexical HOI claims and existential frozen-UPT caption evidence.

Captions rarely identify a unique person instance. Pool over actual detected
person/object pairs of the mentioned class; do not claim unique instance binding.
Unknown or absent evidence is neutral, never a contradiction label.
"""
import math
import re
from .object_consistency import ALIASES
from .multiscore_reranking import rerank_multiscore

METHOD = 'upt_raw_class_contrast_lexical_v1'
VERBS = ('hold', 'ride', 'sit_on', 'carry', 'eat', 'talk_on')
PATTERNS = {
    'hold': (('holding',), ('holds',), ('hold',)),
    'ride': (('riding',), ('rides',), ('ride',)),
    'sit_on': (('sitting','on'), ('sits','on'), ('seated','on'), ('sitting','atop')),
    'carry': (('carrying',), ('carries',), ('carry',)),
    'eat': (('eating',), ('eats',), ('eat',)),
    'talk_on': (('talking','on'), ('speaking','on')),
}


def extract_claims(caption):
    tokens = re.findall(r'[a-z0-9]+', caption.lower())
    spans, occupied = [], set()
    for alias, label in ALIASES:
        for start in range(len(tokens)-len(alias)+1):
            positions = set(range(start, start+len(alias)))
            if tuple(tokens[start:start+len(alias)]) == alias and not positions & occupied:
                spans.append((start, start+len(alias), label)); occupied.update(positions)
    spans.sort()
    actions = []
    for verb, patterns in PATTERNS.items():
        for pattern in patterns:
            for start in range(len(tokens)-len(pattern)+1):
                if tuple(tokens[start:start+len(pattern)]) == pattern:
                    actions.append((start, start+len(pattern), verb))
    actions.sort()
    claims = []
    for start, end, verb in actions:
        previous = [s for s in spans if s[1] <= start]
        if not previous or previous[-1][2] != 'person' or start-previous[-1][1] > 6:
            continue
        if set(tokens[max(0,previous[-1][0]-1):end+1]) & {'not','never','no','without'}:
            continue
        next_action = min([a[0] for a in actions if a[0] > start] + [len(tokens)])
        targets = [s for s in spans if end <= s[0] < min(end+8,next_action)]
        if not targets or targets[0][2] == 'person': continue
        target = targets[0]
        # Stop at a separate clause instead of attaching its object to this verb.
        if set(tokens[end:target[0]]) & {'and','while','but','who','which'}: continue
        if verb == 'talk_on' and target[2] != 'cell phone': continue
        claim = {'subject':'person','verb':verb,'object':target[2]}
        if claim not in claims: claims.append(claim)
    return claims


def evidence_inventory(prediction, objects, verbs, correspondence):
    """Maximum raw verb probability over real, trained detected pairs per class."""
    best = {}
    probabilities = prediction['logits'].sigmoid()
    for row, (s, o) in enumerate(zip(prediction['subject_indices'], prediction['object_indices'])):
        s, o = int(s), int(o)
        if objects[int(prediction['labels'][s])].replace('_',' ') != 'person': continue
        obj = int(prediction['labels'][o]); label = objects[obj].replace('_',' ')
        for verb in VERBS:
            column = verbs.index(verb)
            if column not in correspondence[obj]: continue
            score = float(probabilities[row,column])
            if not math.isfinite(score): raise ValueError('Nonfinite UPT score.')
            key = (verb,label)
            if key not in best or score > best[key]['probability']:
                h,w = prediction['size'].tolist()
                scale = prediction['boxes'].new_tensor([w,h,w,h])
                best[key] = {'probability':score,'raw_logit':float(prediction['logits'][row,column]),'subject_index':s,'object_index':o,
                    'subject_box':(prediction['boxes'][s]/scale).tolist(),
                    'object_box':(prediction['boxes'][o]/scale).tolist()}
    for (verb,label), entry in best.items():
        competitors=[(other['raw_logit'],other_label) for (other_verb,other_label),other in best.items()
                     if other_verb==verb and other_label!=label]
        if competitors:
            other_logit,other_label=max(competitors)
            entry.update(contrast=math.tanh((entry['raw_logit']-other_logit)/2),
                         competitor_object=other_label,competitor_logit=other_logit)
        else:
            entry.update(contrast=0.,competitor_object=None,competitor_logit=None)
    return best


def candidate_evidence(caption, inventory):
    claims = []
    for claim in extract_claims(caption):
        match = inventory.get((claim['verb'],claim['object']))
        claims.append({**claim,'status':'available' if match is not None else 'unknown',
                       **(match or {})})
    supported = [r for r in claims if r['status']=='available']
    # Contrast within the same predicate/image avoids interpreting low HICO
    # sigmoid values (e.g. sit_on) as calibrated negatives. No competing trained
    # object class -> neutral. Unknown claims contribute zero; no minmax boost.
    feature = sum(r['contrast'] for r in supported)/len(claims) if claims else 0.
    return {'method':METHOD,'feature':feature,'claims':claims,'available_claims':len(supported)}


def rerank_with_hoi(record, detections, weight):
    if not math.isfinite(weight) or weight < 0: raise ValueError('Invalid UPT weight.')
    _, enriched = rerank_multiscore(record,detections,.4,.3)
    for candidate in enriched:
        evidence = candidate.get('hoi_evidence')
        if not evidence or evidence.get('method') != METHOD: raise ValueError('Missing/mismatched HOI evidence.')
        feature = evidence['feature']
        if not isinstance(feature,(int,float)) or not math.isfinite(feature) or not -1 <= feature <= 1:
            raise ValueError('Invalid HOI feature.')
        candidate['fixed_rerank_score'] = candidate['combined_score']
        candidate['combined_score'] += weight*feature
    return max(enriched,key=lambda c:(c['combined_score'],-c['original_rank'])),enriched
