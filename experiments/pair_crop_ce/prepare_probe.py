"""Mine UPT hypotheses only; no caption predictions, models or automatic truth labels."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/phrase_pair_feasibility'))
from data import identity,iou,patch_support


def mine(cache,records,excluded,images,annotations):
    pools={};result=[];used=set()
    for family,obj in [('skateboard_support','skateboard'),('surfboard_support','surfboard')]:
        candidates=[]
        for filename,entry in cache.items():
            r=records.get(filename)
            if not r or r['split'] not in ('train','restval') or r['cocoid'] in excluded:continue
            iid=r['cocoid'];im=images.get(iid,dict(width=None,height=None))
            rels=[v for v in entry.get('relations',[]) if v.get('subject')=='person' and v.get('object')==obj]
            objects={v['instance_id']:v for v in entry.get('objects',[])}
            for a in rels:
                if a['predicate'] not in ('stand_on','riding'):continue
                for b in rels:
                    if b['predicate'] not in ('carry','holding'):continue
                    if a['subject_instance']==b['subject_instance'] or a['object_instance']==b['object_instance']:continue
                    events=[]
                    for relation in (a,b):
                        boxes=[];ids=[]
                        for key,label in [('subject_instance','person'),('object_instance',obj)]:
                            detection=objects.get(relation[key]);matches=[v for v in annotations.get(iid,[]) if v['label']==label]
                            if detection is None:break
                            best=max(matches,key=lambda v:iou(v['box'],detection['bbox'])) if matches else None
                            if best and iou(best['box'],detection['bbox'])>=.5:
                                boxes.append(best['box']);ids.append(best['id'])
                            else:
                                boxes.append(detection['bbox']);ids.append(detection['instance_id'])
                        if len(boxes)!=2:break
                        s,o=boxes
                        events.append(dict(event_id=f'{iid}_{ids[0]}_{ids[1]}',image_id=iid,filename=filename,filepath=r['filepath'],caption_split=r['split'],
                            width=im['width'],height=im['height'],subject_id=ids[0],object_id=ids[1],classes=['person',obj],subject_box=s,object_box=o,
                            patch_centers=dict(subject=patch_support(s),object=patch_support(o)),
                            source_hint=dict(method='UPT pair proposal; annotation box if matched IoU>=.5, otherwise detector box; ALL boxes require human confirmation',predicate=relation['predicate'],box_ids=ids)))
                    if len(events)!=2 or events[0]['subject_id']==events[1]['subject_id'] or events[0]['object_id']==events[1]['object_id']:continue
                    key=identity(['crop-probe-v1',family,sorted(v['event_id'] for v in events)])
                    if int(key[0],16)%2:events.reverse()
                    candidates.append((key,events))
        chosen=[]
        for key,events in sorted(candidates,key=lambda x:x[0]):
            iid=events[0]['image_id']
            if iid in used:continue
            used.add(iid);chosen.append((key,events))
            if len(chosen)==10:break
        pools[family]=dict(candidate_pairs=len(candidates),distinct_candidate_images=len({e[0]['image_id'] for _,e in candidates}),selected=len(chosen))
        for key,events in chosen:
            result.append(dict(group_id='crop_probe_'+key[:16],family=family,same_image=True,events=events,descriptions=[f'a person standing on a {obj}',f'a person carrying a {obj}'],
                human_truth_matrix=[['unreviewed']*2 for _ in range(2)],truth_status='CANDIDATE_ONLY_NOT_GOLD',training_eligible=False,split='new_adaptation_probe_candidate'))
    return result,pools


def main():
    base=Path('C:/KhoaLuanTotNghiep/Code/output/vcoco_alignment_week1/sources')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    old=read(ROOT/'experiments/phrase_pair_feasibility/candidate_manifest.json')
    excluded=set(old['excluded_known_review_image_ids'])|{e['image_id'] for g in old['groups'] for e in g['events']}
    # shortcut: exclude all image IDs recorded in local historical manifests/results, not unknown external logs.
    def ids(value):
        if isinstance(value,dict):
            for k,v in value.items():
                if k in ('image_id','coco_id','cocoid') and isinstance(v,int):excluded.add(v)
                else:ids(v)
        elif isinstance(value,list):
            for v in value:ids(v)
    for path in (ROOT/'experiments').rglob('*.json'):
        if 'review' in path.parts or HERE in path.parents or path.stat().st_size>12_000_000:continue
        ids(read(path))
    dataset=read(base/'dataset_coco.json');instances=read(base/'instances_vcoco_trainval2014.json')
    images={v['id']:v for v in instances['images']};cats={v['id']:v['name'] for v in instances['categories']};annotations={}
    for a in instances['annotations']:
        if a.get('iscrowd'):continue
        im=images[a['image_id']];x,y,w,h=a['bbox'];W,H=im['width'],im['height']
        b=[max(0,x/W),max(0,y/H),min(1,(x+w)/W),min(1,(y+h)/H)]
        if b[2]<=b[0] or b[3]<=b[1]:continue
        annotations.setdefault(a['image_id'],[]).append(dict(id=a['id'],label=cats[a['category_id']],box=b))
    source=Path('C:/Users/leduc/Downloads/upt_prompts_full_all_actions_v2_review.zip')
    with zipfile.ZipFile(source) as z:cache=json.loads(z.read('prompt_cache.json'))
    groups,pools=mine(cache,{r['filename']:r for r in dataset['images']},excluded,images,annotations)
    oldfit=read(ROOT/'experiments/phrase_decoder_pilot/locked_manifest.json')
    confirm=next(g for g in oldfit['groups'] if any(e['image_id']==235799 for e in g['events']))
    confirm=dict(confirm,group_id='fit_reconfirm_235799',split='fit_label_reconfirmation_not_probe',truth_status='UNREVIEWED_RECONFIRMATION',training_eligible=False)
    for e in confirm['events']:e['patch_centers']=dict(subject=patch_support(e['subject_box']),object=patch_support(e['object_box']))
    manifest=dict(schema_version=1,status='CANDIDATES_ONLY_TRAIN_BLOCKED',groups=groups+[confirm],groups_sha256=identity(groups+[confirm]),
        pools=pools,excluded_known_ids=sorted(excluded),test_used=False,no_training=True,
        selection='Fixed hash order, max10 unique images/family; UPT proposes; annotation boxes when matched, otherwise UPT boxes requiring human confirmation; no caption output used',
        eligibility='Human 2x2 contrast and clear boxes on original pixels; detail224 is descriptive only, NOT eligibility',
        pending=['Human labels','Image perceptual duplicate audit','HDF5 metadata and GPU feature compatibility'],
        provenance_limit='Known IDs in local experiment JSONs under12MB excluded; other historical exposures not certified; backbone may have seen train/restval')
    out=HERE/'review';out.mkdir(exist_ok=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    public={k:v for k,v in manifest.items() if k!='groups'}
    public['candidate_groups']=len(groups);public['fit_reconfirmation_groups']=1
    (HERE/'PREPARATION.json').write_text(json.dumps(public,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(pools));print('probe candidates:',len(groups),'plus1fit confirmation; NO semantic labels or training')


if __name__=='__main__':main()
