"""Audit existing caption/box joins and token coordinates, without torch or models."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import runpy
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCES = Path('C:/KhoaLuanTotNghiep/Code/output/vcoco_alignment_week1/sources')
AMBIGUOUS = {'train','orange','oranges','bear','bears','mouse','mice','remote','remotes',
    'sink','sinks','tie','ties','book','books','plant','plants','table','tables',
    'monitor','monitors','vehicle','vehicles','ball','balls','glove','gloves','racket','rackets'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spans(text, aliases):
    # Retain the actual tokenizer's whitespace positions; punctuation is only a matching view.
    raw=text.lower().split()
    normalized=[re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$', '', t) for t in raw]
    used=set(); found=[]; present=set(normalized)
    for alias,label in aliases:
        if alias[0] not in present:
            continue
        for start in range(len(raw)-len(alias)+1):
            end=start+len(alias)
            if used.intersection(range(start,end)) or tuple(normalized[start:end])!=alias:
                continue
            used.update(range(start,end))
            found.append((start,end,label,' '.join(alias)))
    return raw,found


def geometry(boxes,width,height,include_mask=False):
    axis=(np.arange(14)+.5)/14
    yy,xx=np.meshgrid(axis,axis,indexing='ij')
    centers=np.zeros((14,14),bool); fraction=np.zeros((14,14))
    # Exact area of the union of axis-aligned boxes within each patch; overlaps counted once.
    normalized=[]
    for x,y,w,h in boxes:
        assert all(np.isfinite([x,y,w,h])) and w>0 and h>0
        a,b=max(0.,x/width),max(0.,y/height)
        c,d=min(1.,(x+w)/width),min(1.,(y+h)/height)
        assert c>a and d>b
        normalized.append([a,b,c,d])
        centers|=(xx>=a)&(xx<c)&(yy>=b)&(yy<d)
    for row in range(14):
        for col in range(14):
            a,b,c,d=col/14,row/14,(col+1)/14,(row+1)/14
            clipped=[(max(a,x),max(b,y),min(c,z),min(d,t)) for x,y,z,t in normalized if min(c,z)>max(a,x) and min(d,t)>max(b,y)]
            cuts=sorted({v for x,y,z,t in clipped for v in (x,z)})
            area=0.
            for left,right in zip(cuts,cuts[1:]):
                intervals=sorted((y,t) for x,y,z,t in clipped if x<(left+right)/2<z)
                last=-1.
                for low,high in intervals:
                    area+=(right-left)*max(0.,high-max(last,low)); last=max(last,high)
            fraction[row,col]=area*196
    assert fraction.min()>=-1e-10 and fraction.max()<=1+1e-10
    result=dict(normalized_xyxy=normalized, patch_centers_in_union=int(centers.sum()),
        patch_area_equivalents=float(fraction.sum()), patches_with_positive_overlap=int((fraction>0).sum()),
        union_area_fraction=float(fraction.sum()/196))
    if include_mask: result['patch_area_fraction']=fraction.clip(0,1).tolist()
    return result


def main():
    aliases=runpy.run_path(str(REPO/'captioning/object_consistency.py'))['ALIASES']
    # Check punctuation alignment, longest multiword match, no duplicate overlap, and union area.
    raw,s=spans('A hot dog, next to a dog.',aliases)
    assert [(a,b,c) for a,b,c,_ in s]==[(1,3,'hot dog'),(6,7,'dog')]
    assert raw[2]=='dog,' and raw[6]=='dog.'
    g=geometry([[0,0,50,100],[25,0,50,100]],100,100)
    assert abs(g['union_area_fraction']-.75)<1e-10
    tiny=geometry([[0,0,1,1]],1000,1000)
    assert tiny['patch_centers_in_union']==0 and tiny['patches_with_positive_overlap']==1
    artifact=HERE.parent/'vg_region_caption/review/loaded_A.json'
    a=json.loads(artifact.read_text(encoding='utf-8'))['json']
    groups=a['groups.json']['train']; assert len(groups)==10000
    dataset_path=SOURCES/'dataset_coco.json'; boxes_path=SOURCES/'instances_vcoco_trainval2014.json'
    assert sha(dataset_path)=='2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
    data=json.loads(dataset_path.read_text(encoding='utf-8'))
    boxes=json.loads(boxes_path.read_text(encoding='utf-8'))
    provenance=json.loads(boxes_path.with_suffix('.provenance.json').read_text(encoding='utf-8'))
    assert sha(boxes_path)==provenance['output_sha256']
    images={x['id']:x for x in boxes['images']}; cats={x['id']:x['name'] for x in boxes['categories']}
    byclass=defaultdict(list); crowd=Counter()
    for b in boxes['annotations']:
        if b.get('iscrowd',0): crowd[b['image_id']]+=1; continue
        assert b['image_id'] in images
        byclass[(b['image_id'],cats[b['category_id']])].append(b)
    counter=Counter()
    for r in data['images']:
        if r['split'] in ('train','restval'):
            for sentence in r['sentences'][:5]: counter.update(sentence['raw'].lower().split())
    vocab={w:i+4 for i,(w,_) in enumerate(counter.most_common(9996))}
    vocab.update({'<pad>':0,'<bos>':1,'<eos>':2,'<unk>':3})
    counts=Counter(); candidates=[]; supported_images=set(); safe_images=set(); classcounts=Counter()
    geo_cache={}
    for r in groups:
        iid=r['coco_id']; record=data['images'][r['image_id']]
        assert record['imgid']==r['image_id'] and record['cocoid']==iid and record['filename']==r['filename']
        assert record['split'] in ('train','restval')
        counts['train_images']+=1
        if iid in images:
            counts['images_with_annotation_source']+=1
            assert images[iid]['file_name']==r['filename']
        for j,sentence in enumerate(record['sentences'][:5]):
            counts['captions']+=1; raw,found=spans(sentence['raw'],aliases)
            for start,end,label,surface in found:
                counts['lexical_candidate_spans']+=1
                if iid not in images: counts['spans_without_annotation_source']+=1;continue
                bs=byclass[(iid,label)]
                if not bs: counts['spans_without_non_crowd_class_box']+=1;continue
                counts['spans_with_non_crowd_class_boxes']+=1; supported_images.add(iid)
                if end>29: counts['matched_spans_outside_decoder_target_window']+=1;continue
                if any(t not in vocab for t in raw[start:end]): counts['matched_spans_with_unk']+=1;continue
                if surface in AMBIGUOUS: counts['matched_spans_excluded_lexical_ambiguity']+=1;continue
                counts['rule_filtered_candidate_spans']+=1;safe_images.add(iid);classcounts[label]+=1
                if len(bs)>1:counts['rule_filtered_multiple_instance_spans']+=1
                key=(iid,label)
                if key not in geo_cache:
                    im=images[iid]; geo_cache[key]=geometry([b['bbox'] for b in bs],im['width'],im['height'])
                geo=geo_cache[key]
                if geo['patch_centers_in_union']==0:counts['rule_filtered_zero_patch_center_spans']+=1
                if geo['patch_area_equivalents']<1:counts['rule_filtered_union_smaller_than_one_patch_area_spans']+=1
                candidates.append(dict(coco_id=iid,karpathy_imgid=r['image_id'],filename=r['filename'],caption_index=j,
                    caption=sentence['raw'],raw_whitespace_tokens=raw,span_words=[start,end],
                    target_token_positions=list(range(start+1,end+1)),decoder_logit_positions=list(range(start,end)),
                    raw_span=raw[start:end],token_ids=[vocab[t] for t in raw[start:end]],category=label,
                    annotation_ids=[b['id'] for b in bs],bbox_xywh=[b['bbox'] for b in bs],
                    image_size=[images[iid]['width'],images[iid]['height']],geometry=geo))
    counts['images_with_any_class_box_matched_span']=len(supported_images)
    counts['images_with_rule_filtered_candidate_span']=len(safe_images)
    counts['images_missing_annotation_source']=10000-counts['images_with_annotation_source']
    candidates.sort(key=lambda r:hashlib.sha256(f"coco-grounding-audit-v1:{r['coco_id']}:{r['caption_index']}:{r['span_words']}".encode()).hexdigest())
    sample=candidates[:12]; assert len(sample)==12
    result=dict(status='PARTIAL_BOX_SOURCE_ONLY_INTERFACE_UNVERIFIED_NO_PILOT',
        sources=dict(dataset_sha256=sha(dataset_path),box_sha256=sha(boxes_path),artifact_sha256=sha(artifact),
            tokenizer_sha256=sha(REPO/'captioning/tokenizer.py'),aliases_sha256=sha(REPO/'captioning/object_consistency.py')),
        box_scope=dict(images=len(images),annotations=len(boxes['annotations']),categories=len(cats),
            provenance=provenance,full_coco_annotations_available=False),counts=dict(counts),class_span_counts=dict(classcounts),
        lexical_policy=dict(ambiguity_exclusions=sorted(AMBIGUOUS),matching='longest alias first; whitespace-token-aligned punctuation normalization',
            noun_pos_verified=False,tokenizer='Static replay of lower().split(),Counter.most_common(9996),BOS offset,30-token truncation; torch tokenizer not executed',
            vocabulary_matches_checkpoint_verified=False),manual_sample_selection='First12 ascending SHA256(coco-grounding-audit-v1:coco_id:caption_index:span_words); from rule-filtered candidates before manual reading',
        manual_sample=sample,model_run=False,new_annotations_downloaded=False)
    HERE.mkdir(exist_ok=True)
    (HERE/'DATA_AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(counts,indent=2));print('Sample:')
    for r in sample: print(json.dumps({k:r[k] for k in ['coco_id','karpathy_imgid','caption','raw_span','category','span_words','decoder_logit_positions','bbox_xywh','image_size']},ensure_ascii=False))


if __name__=='__main__':
    main()
