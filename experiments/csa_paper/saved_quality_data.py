"""Saved text contracts only; no models or metric dependencies."""
import hashlib
import json
import math

def check_rows(rows,expected):
    if len(rows)!=len(expected) or len({x['coco_id'] for x in rows})!=len(rows) or {x['coco_id'] for x in rows}!=set(expected):
        raise ValueError('Missing/duplicate/foreign caption IDs')
    if any(not isinstance(x['caption'],str) for x in rows):raise ValueError('Non-text caption')

def common_refs(references,bundle):
    all_refs={x['coco_id']:x['references'] for x in references}
    ids=bundle['cohort_ids']
    if len(ids)!=len(set(ids)) or len(ids)!=4000:raise ValueError('Expected4000unique IDs')
    if not set(ids)<=set(all_refs):raise ValueError('Old4000 not contained in new5000')
    selected=[dict(coco_id=i,references=all_refs[i][:5]) for i in sorted(ids)]
    if any(len(x['references'])!=5 for x in selected):raise ValueError('Missing reference')
    digest=hashlib.sha256(json.dumps(selected,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    if digest!=bundle['reference_sha256']:raise ValueError('Old/new reference identities differ')
    order=bundle.get('legacy_order',sorted(ids))
    if len(order)!=len(ids) or set(order)!=set(ids):raise ValueError('Wrong legacy input order')
    by_id={x['coco_id']:x for x in selected}
    return [by_id[i] for i in order]

def chair_counts(chair,rows):
    sentences=chair['sentences'];ids={x['coco_id'] for x in rows};mapping={x['coco_id']:x['caption'] for x in rows}
    if len(sentences)!=len(ids) or {x['image_id'] for x in sentences}!=ids:raise ValueError('CHAIR ID coverage')
    if any(x['caption']!=mapping[x['image_id']] for x in sentences):raise ValueError('CHAIR captions differ')
    counts=dict(images=len(ids),hallucinated_captions=sum(bool(x['mscoco_hallucinated_words']) for x in sentences),
        hallucinated_mentions=sum(len(x['mscoco_hallucinated_words']) for x in sentences),
        object_mentions=sum(len(x['mscoco_generated_words']) for x in sentences),
        recalled_unique=sum(len(set(x['mscoco_generated_words'])&set(x['mscoco_gt_words'])) for x in sentences),
        reference_unique=sum(len(set(x['mscoco_gt_words'])) for x in sentences))
    metric=dict(CHAIRs=counts['hallucinated_captions']/len(ids),CHAIRi=counts['hallucinated_mentions']/max(1,counts['object_mentions']),
                Recall=counts['recalled_unique']/max(1,counts['reference_unique']))
    if any(not math.isfinite(v) or abs(v-chair['overall_metrics'][k])>1e-12 for k,v in metric.items()):raise ValueError('CHAIR count aggregation differs')
    return metric,counts

def align_cider_scores(references,values,rows):
    if len(references)!=len(values):raise ValueError('CIDEr score/reference length mismatch')
    indexed={r['coco_id']:float(v) for r,v in zip(references,values)}
    if len(indexed)!=len(references) or {r['coco_id'] for r in rows}!=set(indexed):raise ValueError('CIDEr score IDs differ')
    return [indexed[r['coco_id']] for r in rows]
