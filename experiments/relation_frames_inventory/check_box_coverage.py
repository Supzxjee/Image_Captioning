"""Join pinned official GT and existing detector boxes; metadata only, no inference."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

from inventory import HERE, POLICY, partition, sha, summarize

ARCHIVE_SHA='031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009'
DETECTOR_SHA='498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c'
MEMBERS={'train':'777ef504abfeba349930485bb32ecd5b6529129a3e010f7852e1437cb049b732',
         'val':'e1b4d1bc54906827b1a029981b1c77f2088bd4462569b7615db7929070c3ab47'}
BOX_POLICY=dict(top_boxes=12,sort='confidence descending, original index tie break',
                iou_min=.5,allow_crowd=False,pool_cap=dict(train=512,heldout=128),
                within_split_image_reuse='allowed across different statements; report unique images',
                pair_rule='one-to-one hashed positive/negative image lists per exact statement',
                scope='Existing pinned detector only; no feature extraction or caption scoring')


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def clipped(box,width,height,xywh=False):
    if len(box)!=4 or not all(math.isfinite(float(x)) for x in box):return None
    x,y,a,b=map(float,box)
    right,bottom=(x+a,y+b) if xywh else (a,b)
    out=[max(0.,x),max(0.,y),min(float(width),right),min(float(height),bottom)]
    return out if out[2]>out[0] and out[3]>out[1] else None


def iou(a,b):
    overlap=max(0.,min(a[2],b[2])-max(a[0],b[0]))*max(0.,min(a[3],b[3])-max(a[1],b[1]))
    area=lambda c:(c[2]-c[0])*(c[3]-c[1])
    return overlap/(area(a)+area(b)-overlap)


def pair_rows(rows):
    groups=defaultdict(lambda:{0:[],1:[]})
    for r in rows:groups[r['statement']][r['label']].append(r)
    pairs=[]
    for statement,labels in groups.items():
        if {r['image_id'] for r in labels[0]} & {r['image_id'] for r in labels[1]}:
            raise ValueError('Conflicting image/statement labels')
        ordered={k:sorted(v,key=lambda r:sha(f"vsr-frame-pair-v1:{statement}:{r['image_id']}".encode())) for k,v in labels.items()}
        for negative,positive in zip(ordered[0],ordered[1]):
            assert negative['image_id']!=positive['image_id']
            key=f"{statement}:{positive['image_id']}:{negative['image_id']}"
            pairs.append(dict(pair_id=sha(key.encode()),statement=statement,relation=positive['relation'],
                              positive=positive,negative=negative))
    return sorted(pairs,key=lambda p:p['pair_id'])


def pick_subset(pairs,relations,split):
    required=POLICY['required_pairs'][split]
    quota=POLICY['per_relation'][split]
    chosen={}
    for relation in relations:
        matches=[p for p in pairs if p['relation']==relation]
        if len(matches)<quota:return []
        for p in matches[:quota]:chosen[p['pair_id']]=p
    for p in pairs:
        if len(chosen)>=required:break
        chosen[p['pair_id']]=p
    return sorted(chosen.values(),key=lambda p:p['pair_id']) if len(chosen)==required else []


def self_check():
    assert iou([0,0,2,2],[1,1,3,3])==1/7
    assert clipped([0,0,-1,2],10,10,xywh=True) is None
    assert clipped([-1,-2,12,14],10,10)==[0.,0.,10.,10.]
    rows=[dict(statement='q',relation='on',image_id=i,label=y) for i,y in [(1,1),(2,0),(3,0)]]
    pairs=pair_rows(rows)
    assert len(pairs)==1 and pairs[0]['positive']['label']==1 and pairs[0]['negative']['label']==0
    try:pair_rows(rows+[dict(rows[0],label=0)])
    except ValueError:pass
    else:raise AssertionError('Conflicting labels accepted')
    assert pick_subset([],['on'],'train')==[]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--detections',type=Path,required=True)
    parser.add_argument('--ijson-path',type=Path)
    args=parser.parse_args()
    if args.ijson_path:sys.path.insert(0,str(args.ijson_path))
    import ijson
    self_check()
    candidates=HERE/'metadata_candidates.json'
    raw=candidates.read_bytes();cohorts=json.loads(raw)
    records=[r for rows in cohorts.values() for r in rows]
    if sha(raw)!='c0ee3d2250f927f821acad6159a3cea2482434a87a7196a73456e4e5ce22fc01' or len(records)!=4399 or len({(r['image_id'],r['statement'],r['label']) for r in records})!=len(records):
        raise ValueError('Candidate metadata changed; review before changing cohort')
    for split,rows in cohorts.items():
        if any(partition(r['image_id'])!=split for r in rows):raise ValueError('Partition changed')
    print('Verifying existing annotation archive and detector cache hashes.',flush=True)
    if file_sha(args.archive)!=ARCHIVE_SHA or file_sha(args.detections)!=DETECTOR_SHA:
        raise ValueError('Pinned input differs')
    wanted={r['image_id'] for r in records}
    images={};annotations=defaultdict(lambda:defaultdict(list));categories=None
    with zipfile.ZipFile(args.archive) as z:
        for split,expected in MEMBERS.items():
            name=f'annotations/instances_{split}2014.json'
            print('Verifying official member and streaming candidate instances:',name,flush=True)
            with z.open(name) as f:
                h=hashlib.sha256()
                for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
                if h.hexdigest()!=expected:raise ValueError('Official member differs')
            with z.open(name) as f:
                cats={r['id']:r['name'] for r in ijson.items(f,'categories.item',use_float=True)}
            if categories is not None and cats!=categories:raise ValueError('Categories differ')
            categories=cats
            with z.open(name) as f:
                for r in ijson.items(f,'images.item',use_float=True):
                    if r['id'] in wanted:
                        if r['id'] in images:raise ValueError('Duplicate GT image')
                        images[r['id']]={k:r[k] for k in ('id','file_name','width','height')}
            with z.open(name) as f:
                for r in ijson.items(f,'annotations.item',use_float=True):
                    if r['image_id'] in wanted:
                        annotations[r['image_id']][categories[r['category_id']]].append({k:r[k] for k in ('id','bbox','iscrowd')})
    detector={}
    for filename,value in json.loads(args.detections.read_bytes()).items():
        iid=int(Path(filename).stem.rsplit('_',1)[-1])
        if iid in wanted:
            if iid in detector:raise ValueError('Duplicate detector image')
            detector[iid]=value
    unique={s:[] for s in cohorts};matched={s:[] for s in cohorts};rejections=Counter()
    for split,rows in cohorts.items():
        for r in rows:
            iid=r['image_id'];im=images.get(iid)
            if im is None:rejections['gt_image_unavailable']+=1;continue
            instances=[annotations[iid][r[k]] for k in ('subject','object')]
            if any(len(v)!=1 for v in instances):rejections['nonunique_or_missing_gt_entity']+=1;continue
            if any(v[0]['iscrowd'] for v in instances):rejections['crowd_entity']+=1;continue
            boxes=[clipped(v[0]['bbox'],im['width'],im['height'],xywh=True) for v in instances]
            if any(b is None for b in boxes):rejections['invalid_gt_box']+=1;continue
            record=dict(r,gt=dict(subject=instances[0][0],object=instances[1][0]),image=im)
            unique[split].append(record)
            if iid not in detector:rejections['detector_image_unavailable']+=1;continue
            objects=detector[iid]['objects']
            if any(not math.isfinite(float(o['conf'])) for o in objects):raise ValueError('Invalid confidence')
            top=sorted(enumerate(objects),key=lambda p:(-float(p[1]['conf']),p[0]))[:12]
            selected=[]
            for role,box in zip(('subject','object'),boxes):
                matches=[]
                for index,obj in top:
                    candidate=clipped(obj['bbox'],im['width'],im['height'])
                    if obj['label']==r[role] and candidate is not None:
                        matches.append((iou(box,candidate),-index,index,candidate,float(obj['conf'])))
                if not matches or max(matches)[0]<.5:break
                value=max(matches)
                selected.append(dict(index=value[2],bbox_xyxy=value[3],iou=value[0],confidence=value[4],label=r[role]))
            if len(selected)!=2:rejections['class_or_iou_match_failed_top12']+=1;continue
            matched[split].append(dict(record,predicted=dict(subject=selected[0],object=selected[1])))
    # Lock pools before testing detector coverage, not according to detector success.
    pools={s:pair_rows(rows)[:BOX_POLICY['pool_cap'][s]] for s,rows in unique.items()}
    lookup={(s,r['image_id'],r['statement'],r['label']):r for s,rows in matched.items() for r in rows}
    valid={s:[] for s in cohorts}
    for s,pairs in pools.items():
        for p in pairs:
            keys=[(s,p[k]['image_id'],p[k]['statement'],p[k]['label']) for k in ('positive','negative')]
            if all(k in lookup for k in keys):
                valid[s].append(dict(p,positive=lookup[keys[0]],negative=lookup[keys[1]]))
    counts={s:Counter(p['relation'] for p in pairs) for s,pairs in valid.items()}
    common=[r for r in POLICY['relations'] if all(counts[s][r]>=POLICY['per_relation'][s] for s in cohorts)]
    common.sort(key=lambda r:sha(('vsr-frame-quota-v1:'+r).encode()))
    selected={s:pick_subset(pairs,common[:4],s) if len(common)>=4 else [] for s,pairs in valid.items()}
    passed=all(len(selected[s])==POLICY['required_pairs'][s] for s in cohorts) and len(common)>=4
    result=dict(status='BOX_COVERAGE_PASS_PENDING_HUMAN_AUDIT_FEATURES' if passed else 'DATA_COVERAGE_NO_GO',
        policy=BOX_POLICY,metadata_policy=POLICY,candidate_sha256=sha(raw),archive_sha256=ARCHIVE_SHA,
        detector_sha256=DETECTOR_SHA,official_member_sha256=MEMBERS,ijson_version=ijson.__version__,
        gt_images_found=len(images),detector_images_found=len(detector),candidate_images=len(wanted),
        gt_unique={s:summarize(rows) for s,rows in unique.items()},
        detector_matched_records={s:summarize(rows) for s,rows in matched.items()},
        fixed_gt_pool_pairs={s:len(pairs) for s,pairs in pools.items()},
        detector_valid_pool_pairs={s:len(pairs) for s,pairs in valid.items()},
        detector_valid_pool_relations={s:dict(c) for s,c in counts.items()},
        common_quota_relations=common,selected_pairs={s:len(pairs) for s,pairs in selected.items()},
        rejections=dict(rejections),not_checked=['Human32-record audit','RGB and feature availability',
            'Original prompt builder on new probe images','Novelty or model performance'],
        local_neural_execution=False)
    (HERE/'BOX_COVERAGE_RESULT.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    (HERE/'box_candidate_pairs.json').write_text(json.dumps(valid,indent=2)+'\n',encoding='utf-8')
    if passed:(HERE/'selected_pairs.json').write_text(json.dumps(selected,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
