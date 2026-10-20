"""Candidate mining only: annotation hints never become semantic truth labels."""
from collections import Counter,defaultdict
import hashlib
import json
import math


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


# Texts describe visible configurations; the source action does NOT prove these phrasings.
FAMILIES=(
    ('bicycle_posture','bicycle',('ride','instr'),('hold','obj'),
     'a person sitting on a bicycle','a person standing beside a bicycle'),
    ('horse_posture','horse',('ride','instr'),('hold','obj'),
     'a person sitting on a horse','a person standing beside a horse'),
    ('phone_position','cell phone',('talk_on_phone','instr'),('look','obj'),
     'a person holding a cell phone near their ear','a person looking at a cell phone in their hands'),
    ('surfboard_support','surfboard',('surf','instr'),('carry','obj'),
     'a person standing on a surfboard','a person carrying a surfboard'),
    ('skateboard_support','skateboard',('skateboard','instr'),('carry','obj'),
     'a person standing on a skateboard','a person carrying a skateboard'),
    ('chair_posture','chair',('sit','instr'),('hold','obj'),
     'a person sitting on a chair','a person standing beside a chair'))


def patch_support(box):
    assert len(box)==4 and all(math.isfinite(v) for v in box)
    assert 0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1
    centers=[(i+.5)/14 for i in range(14)]
    return sum(box[0]<=x<=box[2] and box[1]<=y<=box[3] for x in centers for y in centers)


def iou(a,b):
    area=lambda r:max(0,r[2]-r[0])*max(0,r[3]-r[1])
    inter=area([max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3])])
    return inter/max(area(a)+area(b)-inter,1e-12)


def collect(actions,instances,captions,excluded,decode_action,normalized_box):
    anns={r['id']:r for r in instances['annotations']};images={r['id']:r for r in instances['images']}
    cats={r['id']:r['name'] for r in instances['categories']}
    decoded={}
    for source,rows in actions.items():
        for action in rows:
            for sid,row in decode_action(action).items():
                key=action['action_name'],sid
                assert key not in decoded,'Overlapping VCOCO train/val actors'
                decoded[key]=dict(row,annotation_file=source)
    pools={};filtered=Counter()
    def label(action,sid):return decoded.get((action,sid),{}).get('label',-1)
    for family,obj,a,b,c1,c2 in FAMILIES:
        sides=[]
        for side,(action,role) in enumerate((a,b)):
            events=[]
            for (name,sid),row in decoded.items():
                iid,oid=row['image_id'],row['roles'].get(role,0)
                if name!=action or row['label']!=1 or iid in excluded:continue
                if iid not in captions or captions[iid]['split'] not in ('train','restval'):continue
                if sid not in anns or oid not in anns or oid==sid:continue
                sa,oa=anns[sid],anns[oid]
                assert sa['image_id']==oa['image_id']==iid and cats[sa['category_id']]=='person'
                if cats[oa['category_id']]!=obj or sa.get('iscrowd',0) or oa.get('iscrowd',0):continue
                # These are mining hints only; even an explicit action zero is NOT a human negative.
                if family.endswith('_posture'):
                    if side==0 and label('sit',sid)!=1:continue
                    if side==1 and (label('stand',sid)!=1 or label('sit',sid)!=0):continue
                if family=='phone_position' and side==1 and label('talk_on_phone',sid)!=0:continue
                if family.endswith('_support') and side==1 and label(a[0],sid)!=0:continue
                s,o=normalized_box(sa,images[iid]),normalized_box(oa,images[iid])
                event=dict(event_id=f'{iid}_{sid}_{oid}',image_id=iid,filename=captions[iid]['filename'],
                    filepath=captions[iid]['filepath'],caption_split=captions[iid]['split'],
                    width=images[iid]['width'],height=images[iid]['height'],subject_id=sid,object_id=oid,
                    classes=['person',obj],subject_box=s,object_box=o,
                    patch_centers=dict(subject=patch_support(s),object=patch_support(o)),
                    source_hint=dict(file=row['annotation_file'],action=action,role=role,label=1),
                    references=[r['raw'] for r in captions[iid]['sentences']],
                    previous_exposure='Caption train/restval; not certified unseen by backbone or all historical experiments')
                if min(event['patch_centers'].values())==0:filtered['zero_patch_events']+=1
                events.append(event)
            sides.append(sorted(events,key=lambda x:identity(x['event_id'])))
        pools[family]=dict(object=obj,descriptions=[c1,c2],sides=sides)
    return pools,dict(filtered)


def choose(pools,size=30):
    """Stable stratification; prefer same-image contrasts, never reuse any image between groups."""
    candidates={};available={}
    for family,pool in pools.items():
        left,right=pool['sides'];pairs=[]
        # shortcut: Cartesian enumeration is bounded by this small VCOCO mining pool, not full COCO.
        for a in left:
            for b in right:
                if a['event_id']==b['event_id']:continue
                if a['image_id']==b['image_id'] and (a['subject_id']==b['subject_id'] or a['object_id']==b['object_id']):continue
                key=identity([family,a['event_id'],b['event_id']])
                pairs.append((a['image_id']!=b['image_id'],key,a,b))
        pairs.sort(key=lambda x:(x[0],x[1]));candidates[family]=pairs
        available[family]=dict(side_events=list(map(len,pool['sides'])),candidate_pairs=len(pairs),same_image_pairs=sum(not p[0] for p in pairs))
    selected=[];used=set();by_family=Counter()
    while len(selected)<size:
        progressed=False
        for family in sorted(candidates,key=lambda k:(by_family[k],identity(k))):
            match=next((p for p in candidates[family] if not ({p[2]['image_id'],p[3]['image_id']}&used)),None)
            if match is None:continue
            _,key,a,b=match;events=[a,b]
            if int(key[0],16)%2:events.reverse()
            group=dict(group_id='candidate_'+key[:16],component_id='audit_'+key[:16],family=family,
                split='feasibility_review_only',same_image=a['image_id']==b['image_id'],events=events,
                descriptions=pools[family]['descriptions'],human_truth_matrix=[['unreviewed']*2 for _ in range(2)],
                truth_status='CANDIDATE_ONLY_NOT_GOLD',training_eligible=False)
            selected.append(group);used|={a['image_id'],b['image_id']};by_family[family]+=1;progressed=True
            if len(selected)==size:break
        if not progressed:break
    return selected,available


def proposal_match(event,entry):
    """Coverage of selected JSON objects only, not raw/pre-filter UPT recall."""
    objects=entry.get('objects',[])
    people=[x for x in objects if x.get('label')=='person'];targets=[x for x in objects if x.get('label')==event['classes'][1]]
    pairs=[(min(iou(s['bbox'],event['subject_box']),iou(o['bbox'],event['object_box'])),s,o)
           for s in people for o in targets if s['instance_id']!=o['instance_id']]
    if not pairs:return dict(pair_found_at_iou05=False,best_pair=None,scope='selected_prompt_objects_only')
    score,s,o=max(pairs,key=lambda x:x[0])
    return dict(pair_found_at_iou05=score>=.5,best_pair=dict(subject_id=s['instance_id'],object_id=o['instance_id'],
        subject_iou=iou(s['bbox'],event['subject_box']),object_iou=iou(o['bbox'],event['object_box'])),scope='selected_prompt_objects_only')
