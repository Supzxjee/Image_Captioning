"""Explicit same-image contrasts. Missing roles/actions remain unknown."""
from collections import defaultdict, Counter
import numpy as np
from probe_data import bounds, identity

SPECS = (('hold','obj','holding'),('ride','instr','riding'),('sit','instr','sitting_on'),
         ('carry','obj','carrying'),('eat','obj','eating'),('talk_on_phone','instr','talking_on'))
PROMPT_NAMES = dict(hold='holding',ride='riding',sit='sit_on',carry='carry',eat='eat',talk_on_phone='talk_on')


def iou(candidate):
    s,o=np.asarray(candidate['bounds'][:2])
    inter=np.maximum(0,np.minimum(s[2:],o[2:])-np.maximum(s[:2],o[:2])).prod()
    return float(inter/((s[2:]-s[:2]).prod()+(o[2:]-o[:2]).prod()-inter))


def construct(actions, instances, captions, allowed_ids, forbidden_ids, compatibility, decode_action, normalized_box, relation_text):
    """One actor contrast per positive edge; harder geometric distractor first, never invented labels."""
    decoded={row['action_name']:decode_action(row) for row in actions}
    images={r['id']:r for r in instances['images']}
    anns={r['id']:r for r in instances['annotations']}
    categories={r['id']:r['name'] for r in instances['categories']}
    allowed=set(allowed_ids)-set(forbidden_ids)
    boxes={}
    def box(aid, iid):
        if aid not in anns or anns[aid]['image_id']!=iid or anns[aid].get('iscrowd',0): return None
        if aid not in boxes:
            try: boxes[aid]=normalized_box(anns[aid],images[iid])
            except ValueError: return None
        return boxes[aid]
    positives=[]
    for action,role,predicate in SPECS:
        for sid,row in decoded[action].items():
            iid,oid=row['image_id'],row['roles'].get(role,0)
            if iid not in allowed or row['label']!=1 or oid==sid or oid not in anns: continue
            if iid not in captions or captions[iid]['split'] not in ('train','restval'): raise ValueError('Caption heldout image in relation fitting')
            s,o=box(sid,iid),box(oid,iid)
            if s is None or o is None: continue
            assert categories[anns[sid]['category_id']]=='person'
            positives.append((action,predicate,sid,oid,iid,categories[anns[oid]['category_id']],s,o))
    # Compatibility is learned ONLY from fit positives, never calibration/probe.
    if compatibility is None: compatibility={(obj,action) for action,_,_,_,_,obj,_,_ in positives}
    by_image_action=defaultdict(list)
    for action,_,_ in SPECS:
        for sid,row in decoded[action].items():
            if row['image_id'] in allowed and row['label']==0:
                if sid in anns and categories[anns[sid]['category_id']]=='person': by_image_action[row['image_id'],action].append(sid)
    verb_candidates=defaultdict(lambda:defaultdict(list)); actor_candidates=[]
    for action,predicate,sid,oid,iid,obj,s,o in positives:
        positive=dict(text=relation_text(PROMPT_NAMES[action],obj),bounds=bounds(s,o))
        common=dict(filename=captions[iid]['filename'],coco_id=iid,caption_split=captions[iid]['split'],
                    predicate=predicate,object_label=obj,positive_index=0,
                    source_positive=dict(subject_id=sid,object_id=oid,action=action,label=1),label_source='vcoco_explicit_role_positive_and_actor_zero')
        for alternate,_,_ in SPECS:
            wrong=decoded[alternate].get(sid)
            if alternate==action or (obj,alternate) not in compatibility or wrong is None or wrong['label']!=0: continue
            negative=dict(text=relation_text(PROMPT_NAMES[alternate],obj),bounds=positive['bounds'])
            pair=tuple(sorted((action,alternate)))
            item=dict(common,case_id=f'verb_{iid}_{sid}_{oid}_{action}_{alternate}',task='wrong_predicate',
                      source_negative=dict(subject_id=sid,object_id=oid,action=alternate,label=0),candidates=[positive,negative])
            verb_candidates[obj,pair][action].append(item)
        possible=[]
        for other in by_image_action[iid,action]:
            if other==sid: continue
            other_box=box(other,iid)
            if other_box is None: continue
            wrong=dict(text=positive['text'],bounds=bounds(other_box,o))
            item=dict(common,case_id=f'actor_{iid}_{sid}_{oid}_{other}_{action}',task='wrong_subject_instance',
                      source_negative=dict(subject_id=other,object_id=oid,action=action,label=0),candidates=[positive,wrong])
            possible.append(item)
        if possible:
            # Annotation-confirmed zero first; geometry only selects difficulty, not truth.
            possible.sort(key=lambda r:(-iou(r['candidates'][1]),identity(r['case_id'])))
            actor_candidates.append(possible[0])
    verbs=[]
    for (obj,pair),directions in sorted(verb_candidates.items()):
        groups=[sorted(directions[a],key=lambda r:identity(r['case_id'])) for a in pair]
        count=min(len(g) for g in groups)
        verbs.extend(row for group in groups for row in group[:count])
    # Stable orientation prevents a positive-side implementation shortcut.
    rows=verbs+actor_candidates
    for row in rows:
        if int(identity(row['case_id'])[:2],16)%2:
            row['candidates']=row['candidates'][::-1];row['positive_index']=1
    assert len({r['case_id'] for r in rows})==len(rows)
    return sorted(rows,key=lambda r:(r['task'],r['case_id'])),compatibility


def counts(rows):
    return dict(pairs=len(rows),images=len({r['coco_id'] for r in rows}),tasks=dict(Counter(r['task'] for r in rows)),
                predicates=dict(Counter(r['predicate'] for r in rows)))


def training_schedule(rows, size=10000):
    """Equal task exposure; deterministic cycle over every pair, no score-based mining."""
    groups=[[i for i,r in enumerate(rows) if r['task']==task] for task in ('wrong_predicate','wrong_subject_instance')]
    if any(not group for group in groups): raise ValueError('Both verified contrast tasks are required')
    return [groups[i%2][(i//2)%len(groups[i%2])] for i in range(size)]


def geometry_summary(rows):
    """Untrained geometric controls; ties count half, report both pair and image means."""
    result={}
    for task in sorted({r['task'] for r in rows}):
        result[task]={}
        for method in ('box_iou','negative_center_distance'):
            by_image=defaultdict(list)
            for row in rows:
                if row['task']!=task:continue
                scores=[]
                for candidate in row['candidates']:
                    s,o=np.asarray(candidate['bounds'][:2])
                    scores.append(iou(candidate) if method=='box_iou' else -float(np.linalg.norm((s[:2]+s[2:]-o[:2]-o[2:])/2)))
                margin=scores[row['positive_index']]-scores[1-row['positive_index']]
                by_image[row['coco_id']].append(.5 if abs(margin)<=1e-7 else float(margin>0))
            result[task][method]=dict(pair_accuracy=float(np.mean([v for group in by_image.values() for v in group])),
                image_macro_pair_accuracy=float(np.mean([np.mean(v) for v in by_image.values()])))
    return result
