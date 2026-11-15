from pathlib import Path
import hashlib
import json
import math
from collections import Counter

ROOT=Path('/kaggle/input/datasets/ducanh2403')
OUTPUT=Path('/kaggle/working/cider_roi_cohort_candidates.json')

def checked(path,pin):
    with path.open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=pin:
            raise ValueError(f'Wrong input: {path}')
        stream.seek(0)
        return json.load(stream)

def geometry(a,b):
    area=lambda x:(x[2]-x[0])*(x[3]-x[1])
    aa,bb=area(a),area(b)
    overlap=math.prod(max(0,min(a[i+2],b[i+2])-max(a[i],b[i])) for i in (0,1))
    distance=math.hypot(*[(a[i]+a[i+2]-b[i]-b[i+2])/2 for i in (0,1)])
    return max(aa,bb)/min(aa,bb),distance,overlap/(aa+bb-overlap)

def pair_pass(a,b):
    ratio,distance,iou=geometry(a,b)
    return ratio<=2 and distance<=.35 and iou<=.10

def main():
    meta={}
    for name,pin in VAL_PINS.items():
        path,=(ROOT/'csa-full-roi-karpathy-v134').rglob(name)
        for row in checked(path,pin)['images']:
            if row['coco_id'] in ARCHIVED:
                if row['group']!='val' or row['coco_id'] in meta:
                    raise ValueError('Invalid VALID metadata')
                meta[row['coco_id']]=row
    assert meta.keys()==ARCHIVED.keys()
    links,=(ROOT/'csa-prompt-full-karpathy-v175').rglob('prompt_links.jsonl')
    texts=checked(links.parent/'prompt_texts.json','35f28b5f9f6cbbc2eaa6a0fc3ea816f8722013a5804814da6555edcfe97f04c2')
    digest=hashlib.sha256()
    prompts={}
    with links.open('rb') as f:
        for line in f:
            digest.update(line)
            row=json.loads(line)
            if row['group']=='val' and row['coco_id'] in meta:
                if row['coco_id'] in prompts:
                    raise ValueError('Duplicate VALID prompt row')
                prompts[row['coco_id']]=row
    assert digest.hexdigest()=='e90c79b3b04540630b4bd2d279a4070ef05da7f4b65970c952851c37e2423bf0'
    assert prompts.keys()==meta.keys()
    data=checked(ROOT/'objectdetectionecache/objectdetectioncache.json','498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c')
    data=data.get('data',data)
    by_name={Path(k).name:v for k,v in data.items()}
    assert len(by_name)==len(data)
    candidates=[]
    for cid,row in meta.items():
        saved=prompts[cid]
        entry=ARCHIVED[cid]
        objects=by_name[row['filename']]
        if isinstance(objects,dict):
            objects=objects['objects']
        ordered=sorted(enumerate(objects),key=lambda item:(-float(item[1]['conf']),item[0]))[:150]
        if not ordered or len(saved['objects'])>=8:
            continue
        assert len(ordered)==saved['roi_count']==len(row['region_kinds'])
        scales=(row['width'],row['height'])*2
        rois=[]
        for idx,(_,obj) in enumerate(ordered):
            b=[min(max(float(x),0),s)/s for x,s in zip(obj['bbox'],scales,strict=True)]
            assert b[0]<b[2] and b[1]<b[3]
            label=' '.join(obj['label'].lower().replace('_',' ').split())
            rois.append(dict(roi=idx,label=label,bbox=b,confidence=float(obj['conf'])))
        for obj in saved['objects']:
            r=rois[obj['roi']]
            assert r['label']==obj['label'] and max(abs(a-b) for a,b in zip(r['bbox'],obj['roi_bbox'],strict=True))<=1e-6
        used={o['roi'] for o in saved['objects']}
        existing={o['label'] for o in saved['objects']}
        available=[r for r in rois if r['roi'] not in used and r['confidence']>=.5]
        for a in available:
            target=a['label']
            phrase=('an ' if target[0] in 'aeiou' else 'a ')+target
            if target not in entry['targets'] or target in existing or phrase not in texts:
                continue
            for b in available:
                if b['label'] not in entry['foils'] or not pair_pass(a['bbox'],b['bbox']):
                    continue
                if (target[0] in 'aeiou') != (b['label'][0] in 'aeiou'):
                    continue
                ratio,distance,iou=geometry(a['bbox'],b['bbox'])
                candidates.append(dict(coco_id=cid,filename=row['filename'],image_sha256=entry['image_sha256'],
                    caption=entry['caption'],prompt=phrase,matched_roi=a,wrong_roi=b,area_ratio=ratio,
                    center_distance=distance,iou=iou,rank_quality=distance+math.log(ratio)/2))
    count=Counter()
    seen=set()
    shortlist=[]
    for pair in sorted(candidates,key=lambda r:(r['rank_quality'],r['coco_id'],r['matched_roi']['roi'],r['wrong_roi']['roi'])):
        label=pair['matched_roi']['label']
        if pair['coco_id'] in seen or count[label]>=6:
            continue
        seen.add(pair['coco_id'])
        count[label]+=1
        shortlist.append(pair)
        if len(shortlist)==60:
            break
    OUTPUT.write_text(json.dumps(dict(split='val',screened_images=len(meta),eligible_pairs=len(candidates),
                          shortlisted_images=len(shortlist),rows=shortlist),indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(OUTPUT, len(shortlist), 'candidates before visual review')

if __name__=='__main__':
    main()
