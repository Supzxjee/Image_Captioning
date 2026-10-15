"""Instance-preserving positive prompts; no claim of exclusive action targets.

Calibration labels choose acceptance gates only. Missing detections, unsupported
verbs and rejected gates abstain; none are converted into negative relations.
"""
import math
from .hoi_caption_reranking import VERBS

METHOD = 'instance_local_selective_prompts_v1'
MARGIN_GRID = (None, -2., -1., 0., .5, 1., 2.)
TEXT = {'hold':'holding', 'ride':'riding', 'sit_on':'sitting on',
        'carry':'carrying', 'eat':'eating', 'talk_on':'talking on'}


def pair_inventory(prediction, objects, verbs, correspondence):
    """Keep every trained person/object/verb cell, with its exact two boxes."""
    h,w=prediction['size'].tolist()
    result=[]
    for row,(s,o) in enumerate(zip(prediction['subject_indices'],prediction['object_indices'])):
        s,o=int(s),int(o)
        if s==o or objects[int(prediction['labels'][s])].replace('_',' ')!='person': continue
        obj=int(prediction['labels'][o])
        for verb in VERBS:
            v=verbs.index(verb)
            if v not in correspondence[obj]: continue
            logit=float(prediction['logits'][row,v])
            if not math.isfinite(logit): raise ValueError('Nonfinite pair logit.')
            result.append({'subject_index':s,'object_index':o,'subject':'person',
                'object':objects[obj].replace('_',' '),'verb':verb,'raw_logit':logit,
                'probability':1/(1+math.exp(-logit)),
                'subject_box':[float(x)/d for x,d in zip(prediction['boxes'][s],[w,h,w,h])],
                'object_box':[float(x)/d for x,d in zip(prediction['boxes'][o],[w,h,w,h])]})
    for r in result:
        # Same-class alternatives count, but another person's objects never do.
        other=[x for x in result if x['subject_index']==r['subject_index']
               and x['verb']==r['verb'] and x['object_index']!=r['object_index']]
        competitor=max(other,key=lambda x:(x['raw_logit'],-x['object_index'])) if other else None
        r['target_margin']=r['raw_logit']-competitor['raw_logit'] if competitor else None
        r['competitor_object_index']=competitor['object_index'] if competitor else None
    return result


def passes(cell, policy):
    if cell is None or policy is None: return False
    margin=cell['target_margin'];gate=policy['minimum_margin']
    # Singleton targets can pass the absolute score gate. No fake negative peer.
    return cell['probability']>=policy['minimum_probability'] and (
        gate is None or margin is None or margin>=gate)


def fit_gates(cells, labels, *, local_margin, precision=.9, minimum_known=5):
    """Maximize calibration TP subject to empirical precision/minimum support.

    Unknown labels are excluded from fitting, not interpreted as false.
    The no-margin policy wins ties, avoiding unnecessary complexity.
    """
    if len(cells)!=len(labels): raise ValueError('Cell/label length differs.')
    known=[(c,y) for c,y in zip(cells,labels) if c is not None and y in (0,1)]
    if {y for _,y in known}!={0,1}: return None
    best=None;best_key=None
    for gate in MARGIN_GRID if local_margin else (None,):
        for threshold in sorted({c['probability'] for c,_ in known}):
            policy={'minimum_probability':threshold,'minimum_margin':gate}
            accepted=[y for c,y in known if passes(c,policy)]
            n=len(accepted);tp=sum(accepted);fp=n-tp
            if n<minimum_known or tp/n<precision: continue
            key=(tp,-fp,gate is None,-threshold)
            if best_key is None or key>best_key:
                best_key=key;best={**policy,'calibration_true':tp,'calibration_false':fp,
                                  'calibration_precision':tp/n,'calibration_known':len(known)}
    return best


def emit_prompts(inventory, policies):
    """No GT labels or source instance matches enter prompt generation."""
    prompts=[]
    for cell in inventory:
        if passes(cell,policies.get(cell['verb'])):
            prompts.append({**cell,'text':f"person {TEXT[cell['verb']]} {cell['object']}",
                            'status':'accepted_by_calibration_gate',
                            'binding':'detected_instance_pair; caption instance not yet resolved'})
    return prompts


def metrics(cells, labels, policies):
    tp=fp=unknown=positives=available_positive=0
    for c,y in zip(cells,labels):
        positives+=y==1;available_positive+=y==1 and c is not None
        if c is not None and passes(c,policies.get(c['verb'])):
            tp+=y==1;fp+=y==0;unknown+=y not in (0,1)
    return {'accepted_true':tp,'accepted_false':fp,'accepted_unknown':unknown,
            'known_precision':tp/(tp+fp) if tp+fp else None,
            'known_positives':positives,'available_positives':available_positive,
            'full_pool_positive_recall':tp/positives if positives else None}
