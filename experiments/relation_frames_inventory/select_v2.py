"""Constrained metadata selection for relational v2. No neural libraries or scores."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys
import time

from inventory import HERE, POLICY, sha
from check_box_coverage import pair_rows

PROJECTIVE={'on','under','above','below','on top of','beneath','in','inside','within','outside','contains'}
CONTACT={'touching','next to','near','far from','far away from'}
META_SHA='62f43f771aec9feba31a2fbbb6fe7db2878f259979faac378b74a754822c7b67'
V2_POLICY=dict(version='relation-frames-pair-first-v2',heldout_pairs=64,train_pairs=256,
    strata=dict(projective_containment=32,contact_proximity=32),
    heldout_unique_images=128,exclude_caption_adapter_train=True,
    priority='maximize held-out relation coverage, then minimize sum of fixed hash ranks',
    tie_rule='deterministic HiGHS on sorted variables/constraints; no seed search',
    train_selection='remove all held-out image records, hash re-pair; one per available predicate then hash-fill',
    solver_seconds_per_stage=90,existing_labels_votes_bbox_policy_unchanged=True)


def stratum(relation):
    if relation in PROJECTIVE:return 'projective_containment'
    if relation in CONTACT:return 'contact_proximity'
    raise ValueError('Unknown relation')


def solve(pairs,excluded,per_stratum,train_count):
    import numpy as np
    from scipy.optimize import milp, Bounds, LinearConstraint
    from scipy.sparse import coo_matrix
    n=len(pairs);relations=sorted({p['relation'] for p in pairs});m=len(relations)
    nv=2*n+m;ri={r:2*n+i for i,r in enumerate(relations)}
    incidents=defaultdict(list);byrelation=defaultdict(list)
    for e,p in enumerate(pairs):
        for role in ('positive','negative'):incidents[p[role]['image_id']].append(e)
        byrelation[p['relation']].append(e)
    cols=[];rows=[];vals=[];lower=[];upper=[]
    def add(coeff,lo,hi):
        row=len(lower)
        for col,value in coeff.items():rows.append(row);cols.append(col);vals.append(value)
        lower.append(lo);upper.append(hi)
    for s in sorted(per_stratum):
        add({e:1 for e,p in enumerate(pairs) if stratum(p['relation'])==s},per_stratum[s],per_stratum[s])
    for image,edges in sorted(incidents.items()):add({e:1 for e in edges},0,1)
    add({n+e:1 for e in range(n)},train_count,train_count)
    for e,p in enumerate(pairs):
        for role in ('positive','negative'):
            # Any selected eval edge touching this endpoint forbids this train witness.
            add({**{f:1 for f in incidents[p[role]['image_id']]},n+e:1},0,1)
    total=sum(per_stratum.values())
    for r,edges in sorted(byrelation.items()):
        add({**{e:1 for e in edges},ri[r]:-1},0,np.inf)
        add({**{e:1 for e in edges},ri[r]:-total},-np.inf,0)
        add({**{n+e:1 for e in edges},ri[r]:-1},0,np.inf)
    ub=np.ones(nv)
    for e,p in enumerate(pairs):
        if any(p[k]['image_id'] in excluded for k in ('positive','negative')):ub[e]=0
    def run(cost,extra=None):
        rr=list(rows);cc=list(cols);vv=list(vals);lo=list(lower);hi=list(upper)
        if extra is not None:
            for col in ri.values():rr.append(len(lo));cc.append(col);vv.append(1)
            lo.append(extra);hi.append(extra)
        a=coo_matrix((vv,(rr,cc)),shape=(len(lo),nv)).tocsc()
        return milp(cost,integrality=np.ones(nv),bounds=Bounds(np.zeros(nv),ub),
                    constraints=LinearConstraint(a,lo,hi),
                    options=dict(time_limit=90,mip_rel_gap=0))
    cost=np.zeros(nv)
    for col in ri.values():cost[col]=-1
    first=run(cost)
    log=[dict(stage='maximum_relation_coverage',status=int(first.status),message=first.message)]
    if first.status!=0:return None,log
    coverage=round(sum(first.x[col] for col in ri.values()))
    cost=np.zeros(nv);cost[:n]=np.arange(1,n+1)
    second=run(cost,coverage)
    log.append(dict(stage='minimum_hash_rank',status=int(second.status),message=second.message,
                    maximum_relation_coverage=coverage))
    if second.status!=0:return None,log
    vector=second.x
    if np.max(np.abs(vector-np.rint(vector)))>1e-6:raise ValueError('Nonintegral solution')
    # Verify all integer constraints independently after rounding.
    vector=np.rint(vector)
    a=coo_matrix((vals,(rows,cols)),shape=(len(lower),nv)).tocsc()
    value=a@vector
    if (value<np.array(lower)-1e-6).any() or (value>np.array(upper)+1e-6).any():raise ValueError('Invalid solution')
    heldout=[p for e,p in enumerate(pairs) if vector[e]]
    return heldout,log


def final_train(records,heldout,required):
    ids={p[k]['image_id'] for p in heldout for k in ('positive','negative')}
    remaining=[r for r in records if r['image_id'] not in ids]
    pairs=pair_rows(remaining)
    relations=sorted({p['relation'] for p in pairs},key=lambda r:sha(('vsr-frame-quota-v1:'+r).encode()))
    chosen={}
    for relation in relations:
        p=next(p for p in pairs if p['relation']==relation);chosen[p['pair_id']]=p
    for p in pairs:
        if len(chosen)>=required:break
        chosen[p['pair_id']]=p
    if len(chosen)!=required:raise ValueError('Train capacity or predicate coverage incompatible')
    return sorted(chosen.values(),key=lambda p:p['pair_id']),remaining,len(pairs)


def validate(train,heldout,excluded,expected_train,expected_strata):
    images=[p[k]['image_id'] for p in heldout for k in ('positive','negative')]
    if len(images)!=len(set(images)):raise ValueError('Repeated held-out image')
    train_images={p[k]['image_id'] for p in train for k in ('positive','negative')}
    if set(images)&(train_images|excluded):raise ValueError('Held-out leakage')
    if len(train)!=expected_train or Counter(stratum(p['relation']) for p in heldout)!=expected_strata:
        raise ValueError('Wrong counts')
    if not {p['relation'] for p in heldout}<={p['relation'] for p in train}:raise ValueError('Unseen eval predicate')
    for split in (train,heldout):
        records=[]
        for p in split:
            pos,neg=p['positive'],p['negative']
            if (pos['label'],neg['label'])!=(1,0) or pos['statement']!=neg['statement'] or p['statement']!=pos['statement']:
                raise ValueError('False pair')
            if pos['subject']!=neg['subject'] or pos['object']!=neg['object']:raise ValueError('Concepts differ')
            for r in (pos,neg):
                records.append((r['image_id'],r['statement'],r['label']))
                for role in ('subject','object'):
                    if r['predicted'][role]['iou']<.5 or r['predicted'][role]['label']!=r[role]:raise ValueError('Bad box')
        if len(records)!=len(set(records)):raise ValueError('Repeated record in pair cohort')


def counts(pairs):
    return dict(pairs=len(pairs),images=len({p[k]['image_id'] for p in pairs for k in ('positive','negative')}),
                statements=len({p['statement'] for p in pairs}),
                by_relation=dict(sorted(Counter(p['relation'] for p in pairs).items())),
                by_stratum=dict(Counter(stratum(p['relation']) for p in pairs)))


def self_check():
    # A shared endpoint must never enter both eval edges; witnesses preserve training.
    def edge(i,j,r):
        q=f'the cat is {r} the dog'
        def record(image,label):return dict(image_id=image,label=label,statement=q,relation=r,subject='cat',object='dog',
            predicted={k:dict(iou=.8,label=v) for k,v in (('subject','cat'),('object','dog'))})
        return dict(pair_id=f'{i:03}-{j:03}',statement=q,relation=r,positive=record(i,1),negative=record(j,0))
    p=[edge(1,2,'on'),edge(1,3,'touching'),edge(4,5,'touching'),edge(6,7,'on'),edge(8,9,'touching'),edge(10,11,'on')]
    heldout,log=solve(p,{6},dict(projective_containment=1,contact_proximity=1),2)
    assert heldout is not None and all(x['status']==0 for x in log)
    assert len({q[k]['image_id'] for q in heldout for k in ('positive','negative')})==4
    assert 6 not in {q[k]['image_id'] for q in heldout for k in ('positive','negative')}
    bad=solve(p[:1],set(),dict(projective_containment=1,contact_proximity=1),1)
    assert bad[0] is None and bad[1][0]['status']==2
    try:validate(p[:2],p[:2],set(),2,dict(projective_containment=1,contact_proximity=1))
    except ValueError:pass
    else:raise AssertionError('Repeated held-out images accepted')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--solver-path',type=Path)
    parser.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.solver_path:sys.path.insert(0,str(args.solver_path))
    import scipy,numpy
    self_check()
    if args.self_check:print('Constrained-selection and leakage self-checks PASS');return
    started=time.monotonic()
    source=HERE/'detector_matched_records.json';raw=source.read_bytes();old=json.loads(raw)
    records=[r for rows in old.values() for r in rows]
    if sha(raw)!='3e3a1ef97d7f0a14a33f90f41ae7372fce8ad56e6732a0cbb01673f894e136e6' or len(records)!=3358 or len({(r['image_id'],r['statement'],r['label']) for r in records})!=3358:
        raise ValueError('Wrong matched master cohort')
    meta_path=HERE.parent/'semantic_grounding_method/data/gain_metadata.json.gz';mr=meta_path.read_bytes()
    if sha(mr)!=META_SHA:raise ValueError('Wrong adapter metadata')
    meta=json.loads(gzip.decompress(mr));excluded={r['coco_id'] for r in meta['groups']['train']}
    if len(excluded)!=2000:raise ValueError('Wrong adapter TRAIN2k')
    pairs=pair_rows(records)
    print(f'V2 fixed master: {len(records)} records, {len(pairs)} paired edges; solving one locked protocol.',flush=True)
    heldout,log=solve(pairs,excluded,V2_POLICY['strata'],256)
    report=dict(policy=V2_POLICY,master_sha256=sha(raw),adapter_metadata_sha256=META_SHA,
        parent_coverage_report_sha256=sha((HERE/'BOX_COVERAGE_RESULT.json').read_bytes()),
        candidates=counts(pairs),solver=dict(scipy=scipy.__version__,numpy=numpy.__version__,stages=log),
        local_neural_execution=False)
    if heldout is None:
        report['status']='DATA_V2_NO_GO' if log[-1]['status']==2 else 'DATA_V2_SOLVER_INCOMPLETE'
    else:
        train,remaining,capacity=final_train(records,heldout,256)
        validate(train,heldout,excluded,256,V2_POLICY['strata'])
        hold_ids={p[k]['image_id'] for p in heldout for k in ('positive','negative')}
        if any(r['image_id'] in hold_ids for r in remaining):raise ValueError('Nonselected statement leaked')
        report.update(status='DATA_V2_PASS_PENDING_AUDIT_FEATURES',train=counts(train),heldout=counts(heldout),
                      remaining_train_pair_capacity=capacity,remaining_records=len(remaining),
                      excluded_all_records_on_heldout_images=len(records)-len(remaining),
                      checks=dict(heldout_images_unique=True,relational_train_image_disjoint=True,
                                  caption_train2k_image_disjoint=True,heldout_predicates_seen_in_train=True,
                                  pair_records_unique=True,bbox_iou_and_classes_valid=True))
        manifest=dict(schema='vsr-relation-frame-v2',policy=V2_POLICY,source_master_sha256=sha(raw),
            adapter_metadata_sha256=META_SHA,train=train,heldout=heldout)
        payload=(json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode()
        compressed=gzip.compress(payload,mtime=0)
        target=HERE/'V2_MANIFEST.json.gz'
        if target.exists() and target.read_bytes()!=compressed:
            raise ValueError('Frozen manifest would change; use a separately reviewed protocol version')
        target.write_bytes(compressed)
        report['manifest_sha256']=sha(compressed)
        report['not_checked']=['32-record label audit','RGB/frozen feature/probe prompt availability',
                              'Novelty, auxiliary accuracy, free-caption improvement, neural execution']
    report['elapsed_seconds']=time.monotonic()-started
    (HERE/'V2_RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
