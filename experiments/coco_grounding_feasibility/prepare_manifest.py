"""Build locked noun/box supervision from official TRAIN metadata; never load a model."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import runpy
import zipfile
import numpy as np
from audit import spans
from manifest_policy import POLICY,context_exclusion

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
DATA=Path('C:/KhoaLuanTotNghiep/Code/output/vcoco_alignment_week1/sources/dataset_coco.json')
OUT=HERE/'review/noun_box_train10k_v1'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def hash_order(prefix,value):return sha((prefix+str(value)).encode())


def rectangle_mask(bbox,width,height):
    if len(bbox)!=4 or not all(np.isfinite(bbox)) or min(bbox[2:])<=0 or min(width,height)<=0:
        return None
    x,y,w,h=map(float,bbox)
    x1,y1=max(0.,x/width),max(0.,y/height)
    x2,y2=min(1.,(x+w)/width),min(1.,(y+h)/height)
    if x2<=x1 or y2<=y1:return None
    axis=np.arange(14)/14
    horizontal=(np.minimum(axis+1/14,x2)-np.maximum(axis,x1)).clip(0)*14
    vertical=(np.minimum(axis+1/14,y2)-np.maximum(axis,y1)).clip(0)*14
    mask=np.outer(vertical,horizontal).clip(0,1)
    area=(x2-x1)*(y2-y1)*196
    assert abs(float(mask.sum())-area)<1e-8
    return dict(normalized_xyxy=[x1,y1,x2,y2],patch_area_equivalents=area,
        image_fraction=area/196,patch_area_fraction=mask.flatten().tolist(),
        raw_bbox_clipped=bool(x<0 or y<0 or x+w>width or y+h>height))


def filter_span(raw,start,end,label,surface,instances,image,vocab):
    if surface in POLICY['ambiguity_exclusions']:return 'ambiguous_surface',None
    normalized=[re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$','',t) for t in raw]
    reason=context_exclusion(normalized,start,end,label,raw_tokens=raw)
    if reason:return reason,None
    if not instances:return 'no_class_instance',None
    if len(instances)!=1:return 'multiple_class_instances',None
    if instances[0].get('iscrowd',0):return 'single_crowd_instance',None
    if end>POLICY['max_caption_length']-1:return 'truncated_span',None
    if any(vocab.get(t,3) in (0,1,2,3) for t in raw[start:end]):return 'unknown_token',None
    region=rectangle_mask(instances[0]['bbox'],image['width'],image['height'])
    if region is None:return 'invalid_bbox',None
    if region['patch_area_equivalents']<POLICY['minimum_patch_area']:return 'bbox_too_small',None
    if region['image_fraction']>POLICY['maximum_image_fraction']:return 'bbox_too_large',None
    return None,region


def cap_person(rows):
    other=[r for r in rows if r['category']!='person']
    people=[r for r in rows if r['category']=='person']
    people.sort(key=lambda r:hash_order('coco-route-person-cap-v1:',r['span_id']))
    count=min(len(people),len(other)//3)
    return other+people[:count],len(people)-count


def manual_sample(rows):
    byclass=defaultdict(list)
    for r in rows:byclass[r['category']].append(r)
    classes=sorted(byclass,key=lambda c:hash_order('coco-route-manual-class-v1:',c))
    chosen=[]
    for c in classes[:16]:
        chosen.append(min(byclass[c],key=lambda r:hash_order('coco-route-manual-record-v1:',r['span_id'])))
    remaining=sorted((r for r in rows if r['span_id'] not in {s['span_id'] for s in chosen}),
        key=lambda r:hash_order('coco-route-manual-record-v1:',r['span_id']))
    return chosen+remaining[:16-len(chosen)]


def write(path,value):
    raw=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    if path.exists():assert path.read_bytes()==raw,'Preserve existing sealed data; use a fresh version if policy changes.'
    else:path.write_bytes(raw)
    return sha(raw)


def main(version='v1'):
    global POLICY,context_exclusion,OUT
    policy_file='MAPPING_POLICY_V1.json';summary_file='FULL_DATA_PREPARATION.json'
    if version=='v1.1':
        from manifest_policy_v11 import POLICY as updated,context_exclusion as updated_context
        POLICY,context_exclusion=updated,updated_context
        OUT=HERE/'review/noun_box_train10k_v1_1'
        policy_file='MAPPING_POLICY_V11.json';summary_file='FULL_DATA_PREPARATION_V11.json'
    assert json.loads((HERE/policy_file).read_text(encoding='utf-8'))==POLICY
    assert sha(DATA.read_bytes())=='2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
    a=json.loads((HERE.parent/'vg_region_caption/review/loaded_A.json').read_text(encoding='utf-8'))['json']
    groups=a['groups.json']['train'];assert len(groups)==10000
    data=json.loads(DATA.read_text(encoding='utf-8'))
    annotations_path=HERE/'review/full_annotations/train10k_instances.json'
    provenance=json.loads(annotations_path.with_suffix('.provenance.json').read_text(encoding='utf-8'))
    assert sha(annotations_path.read_bytes())==provenance['filtered_file_sha256']
    annotations=json.loads(annotations_path.read_text(encoding='utf-8'))
    images={r['id']:r for r in annotations['images']};assert set(images)=={r['coco_id'] for r in groups}
    categories={r['id']:r['name'] for r in annotations['categories']}
    byclass=defaultdict(list)
    for r in annotations['annotations']:byclass[(r['image_id'],categories[r['category_id']])].append(r)
    aliases=runpy.run_path(str(ROOT/'captioning/object_consistency.py'))['ALIASES']
    counter=Counter()
    for r in data['images']:
        if r['split'] in ('train','restval'):
            for s in r['sentences'][:5]:counter.update(s['raw'].lower().split())
    vocab={w:i+4 for i,(w,_) in enumerate(counter.most_common(9996))}
    vocab.update({'<pad>':0,'<bos>':1,'<eos>':2,'<unk>':3})
    vocab_sha=sha(json.dumps(vocab,sort_keys=True).encode())
    assert vocab_sha=='427a33573c9302661282dd4553e87ccd82138f5903cea25823afc7eddb49bd3c','Mismatch against actual smoke tokenizer receipt.'
    rejected=Counter();rows=[];mask_bank={};candidate_count=0;captions=0
    for n,group in enumerate(groups,1):
        iid=group['coco_id'];r=data['images'][group['image_id']];im=images[iid]
        assert r['cocoid']==iid and r['imgid']==group['image_id'] and r['split'] in ('train','restval')
        assert r['filename']==im['file_name']==group['filename'] and len(r['sentences'])>=5
        for j,s in enumerate(r['sentences'][:5]):
            captions+=1;raw,found=spans(s['raw'],aliases)
            for start,end,label,surface in found:
                candidate_count+=1;instances=byclass[(iid,label)]
                reason,region=filter_span(raw,start,end,label,surface,instances,im,vocab)
                if reason:rejected[reason]+=1;continue
                mask_key=f'{iid}:{label}'
                if mask_key not in mask_bank:
                    mask_bank[mask_key]=dict(coco_id=iid,category=label,annotation_id=instances[0]['id'],
                        bbox_xywh=instances[0]['bbox'],image_size=[im['width'],im['height']],**region)
                identifier=f'{iid}:{j}:{start}:{end}:{label}'
                rows.append(dict(span_id=sha(identifier.encode())[:24],coco_id=iid,karpathy_imgid=group['image_id'],
                    filename=group['filename'],caption_index=j,caption=s['raw'],category=label,surface=surface,
                    span_words=[start,end],raw_span=raw[start:end],target_token_positions=list(range(start+1,end+1)),
                    decoder_logit_positions=list(range(start,end)),token_ids=[vocab[t] for t in raw[start:end]],mask_key=mask_key))
        if n%2000==0:print('Mapped locked TRAIN captions:',n,'/10000',flush=True)
    before_cap=len(rows);rows,dropped=cap_person(rows);rejected['person_cap']=dropped
    rows.sort(key=lambda r:(r['karpathy_imgid'],r['caption_index'],r['span_words'],r['category']))
    assert len({r['span_id'] for r in rows})==len(rows)
    assert sum(rejected.values())+len(rows)==candidate_count and captions==50000
    keep_masks={r['mask_key'] for r in rows};mask_bank={k:mask_bank[k] for k in sorted(keep_masks)}
    counts=Counter(r['category'] for r in rows);image_ids={r['coco_id'] for r in rows}
    assert counts['person']<=len(rows)*POLICY['person_maximum_fraction']
    class_gate=sum(v>=50 for v in counts.values())
    gates=dict(images=len(image_ids)>=POLICY['minimum_images'],spans=len(rows)>=POLICY['minimum_unique_spans'],
        classes=class_gate>=POLICY['minimum_classes_with_50_spans'])
    byimage=defaultdict(list)
    for r in rows:byimage[r['coco_id']].append(r['span_id'])
    schedule=[dict(coco_id=r['coco_id'],karpathy_imgid=r['image_id'],
        span_ids=sorted(byimage[r['coco_id']],key=lambda s:hash_order('coco-route-schedule-v1:',s))) for r in groups]
    assert sum(len(r['span_ids']) for r in schedule)==len(rows)
    if version=='v1.1':
        originals=json.loads((HERE/'review/noun_box_train10k_v1/manual16.json').read_text(encoding='utf-8'))
        retained={r['span_id'] for r in rows};sample=[]
        for original in originals:
            start,end=original['span_words'];raw=original['caption'].lower().split();iid=original['coco_id'];label=original['category']
            reason,_=filter_span(raw,start,end,label,original['surface'],byclass[(iid,label)],images[iid],vocab)
            if reason is None and original['span_id'] not in retained:reason='person_cap'
            sample.append(dict(original,v11_kept=original['span_id'] in retained,v11_rejection_reason=reason))
    else:sample=manual_sample(rows)
    assert len(sample)==16 and len({r['span_id'] for r in sample})==16
    OUT.mkdir(parents=True,exist_ok=True)
    payloads=dict(policy=POLICY,spans=rows,mask_bank=mask_bank,supervision_schedule=schedule)
    hashes={name+'.json':write(OUT/(name+'.json'),payload) for name,payload in payloads.items()}
    write(OUT/'manual16.json',sample if version=='v1.1' else [dict(**r,box=mask_bank[r['mask_key']]) for r in sample])
    result=dict(status='COVERAGE_GATE_PASS_MANUAL_REVIEW_PENDING' if all(gates.values()) else 'COVERAGE_GATE_FAIL_NO_PILOT',
        train_images=10000,captions=captions,lexical_candidate_spans=candidate_count,eligible_before_person_cap=before_cap,
        retained_images=len(image_ids),retained_unique_annotation_spans=len(rows),retained_mask_count=len(mask_bank),
        person_spans=counts['person'],person_fraction=counts['person']/len(rows) if rows else 0,
        retained_classes=len(counts),classes_with_at_least_50_spans=class_gate,class_span_counts=dict(sorted(counts.items())),
        rejected_by_first_reason=dict(rejected),gates=gates,policy=POLICY,
        source_provenance=provenance,vocabulary_sha256=vocab_sha,
        implementation_sha256={n:sha((HERE/n).read_bytes()) for n in ['prepare_manifest.py','manifest_policy.py','audit.py']+(['manifest_policy_v11.py'] if version=='v1.1' else [])},
        payload_sha256=hashes,unique_span_definition='COCO image + source caption index among first5 + word span + class',
        main_caption_images_preserved=len(schedule),manual16_distinct_classes=len({r['category'] for r in sample}),
        manual_review_completed=False,test_used=False,model_run=False,
        schedule_scope='Per-image supervision only; main minibatch/RNG order and numeric recipe will be locked before pilot.')
    write(OUT/'preparation_summary.json',result)
    write(HERE/summary_file,result)
    print(json.dumps({k:result[k] for k in ['status','retained_images','retained_unique_annotation_spans','person_fraction','classes_with_at_least_50_spans','gates']},indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--version',choices=['v1','v1.1'],default='v1')
    main(parser.parse_args().version)
