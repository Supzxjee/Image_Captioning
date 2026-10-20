"""Locked adaptation split/schedule and fit-only geometry statistics; no model imports."""
import hashlib
import json
import random
import numpy as np


def identity(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def geometry(s,o):
    s,o=np.asarray(s),np.asarray(o);sw,sh=np.maximum(s[2:]-s[:2],1e-6);ow,oh=np.maximum(o[2:]-o[:2],1e-6)
    sc,oc=(s[:2]+s[2:])/2,(o[:2]+o[2:])/2
    inter=np.maximum(np.minimum(s[2:],o[2:])-np.maximum(s[:2],o[:2]),0).prod()
    return np.asarray([(oc[0]-sc[0])/sw,(oc[1]-sc[1])/sh,np.log(ow/sw),np.log(oh/sh),
        inter/max(sw*sh+ow*oh-inter,1e-6),inter/(sw*sh),inter/(ow*oh),np.linalg.norm(oc-sc)],dtype=np.float32)


def build_manifest(bundle):
    assert identity(bundle['groups'])==bundle['groups_sha256'] and len(bundle['groups'])==21
    groups=[]
    for g in bundle['groups']:
        assert set(g['target_description_indices'])=={0,1} and g['human_detail_224']==['yes','yes']
        events=[{k:e[k] for k in ('event_id','image_id','filename','filepath','caption_split','subject_box','object_box','width','height')} for e in g['events']]
        assert all(e['caption_split'] in ('train','restval') for e in events)
        groups.append(dict(group_id=g['group_id'],family=g['family'],same_image=g['same_image'],events=events,
            descriptions=g['descriptions'],positive_indices=g['target_description_indices']))
    # Fixed family quotas; preserve at least one same-image group in fit and holdout.
    quotas={'bicycle_posture':0,'skateboard_support':2,'surfboard_support':2,'phone_position':3}
    held=[]
    for family,n in quotas.items():
        rows=sorted([g for g in groups if g['family']==family],key=lambda g:identity(['phrase-pilot-v1',g['group_id']]))
        # In the skateboard family two same-image groups exist: allocate exactly one to each partition.
        if family in ('skateboard_support','surfboard_support'):
            same=[g for g in rows if g['same_image']];other=[g for g in rows if not g['same_image']]
            assert len(same)==(2 if family=='skateboard_support' else 1);selected=same[:1]+other[:n-1]
        else:selected=rows[:n]
        held += [g['group_id'] for g in selected]
    fit=[g['group_id'] for g in groups if g['group_id'] not in held];assert len(fit)==14 and len(held)==7
    by={g['group_id']:g for g in groups}
    ids=lambda keys:{e['image_id'] for key in keys for e in by[key]['events']}
    assert not ids(fit)&ids(held)
    raw=np.stack([geometry(e['subject_box'],e['object_box']) for key in fit for e in by[key]['events']])
    rng=random.Random(42);schedule=[]
    while len(schedule)<300*8:
        cycle=fit.copy();rng.shuffle(cycle);schedule+=cycle
    schedule=[schedule[i:i+8] for i in range(0,300*8,8)]
    return dict(schema_version=1,groups=groups,partitions=dict(fit=fit,adaptation_holdout=held),
        geometry_fit_mean=raw.mean(0).tolist(),geometry_fit_std=np.maximum(raw.std(0),1e-3).tolist(),
        schedule=schedule,schedule_sha256=identity(schedule),human_review_sha256=bundle['review_sha256'],
        parent_groups_sha256=bundle['groups_sha256'],seed=42,updates=300,global_batch_groups=8,
        cross_weight=.25,cross_start_update=61,max_length=20,test_used=False,
        scope='Tiny adaptation holdout of previously reviewed caption-train/restval images; not an independent test, no hyperparameter/checkpoint search',
        split_policy='Fixed hash/family quotas; one skateboard and one surfboard same-image group held out, remaining same-image groups fit; no performance selection; near-duplicate independence NOT certified',
        forbidden_model_inputs=['predicate text','UPT scores/classes','legacy prompts','references','human labels except decoder targets'])
