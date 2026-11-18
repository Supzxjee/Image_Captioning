"""FullTRAIN prompt transport/stream checks and incremental-vector assembly; no neural imports."""
from collections import Counter
import json
from pathlib import Path
import re
import numpy as np
from real_bank_data import read,save,sha
from prompt_adapter_data import tensor_batch,PROMPT_SHA
from audit_assets import DETECTION_SHA

BASE_RECEIPT_SHA='3c3416d1d6d21ba693e4b5c9833d89b0479e89a712196453340fb9ad79d21a42'
BASE_MANIFEST_SHA='24c1c75fda9be90f6e1409aedc8d2bbecb04cb78e5570443a3f813b402b66486'
MAPPING_FILES=('cohort.json','prompt_texts.json','prompt_links.jsonl','summary.json','review_examples.json')


def checked_payloads(root,receipt):
    root=Path(root)
    for n,h in receipt['payload_sha256'].items():
        if Path(n).name!=n or '/' in n or '\\' in n or not (root/n).is_file() or sha(root/n)!=h:raise ValueError('Missing/changed payload:'+n)


def checked_base(root):
    choices=[p for p in Path(root).rglob('cache_receipt.json') if sha(p)==BASE_RECEIPT_SHA]
    if len(choices)!=1:raise ValueError('Need one exact169 prompt cache; no caption checkpoint is loaded')
    root=choices[0].parent;receipt=read(choices[0]);checked_payloads(root,receipt)
    if receipt['status']!='PROMPT_TEXT_CACHE_COMPLETE' or sha(root/'cache_manifest.json')!=BASE_MANIFEST_SHA:raise ValueError('Wrong169 identity')
    spec=read(Path(__file__).with_name('RERANK_TEST_INPUTS.json'))['clip'];manifest=read(root/'cache_manifest.json')
    if manifest['model']!=spec['id'] or manifest['revision']!=spec['revision'] or manifest['files_sha256']!=spec['files_sha256']:raise ValueError('CLIP169 space/assets differ')
    texts=read(root/'prompt_texts.json');vectors=np.load(root/'prompt_embeddings.npy',allow_pickle=False)
    validate_vectors(texts,vectors)
    if len(texts)!=318:raise ValueError('Expected318basevectors')
    return root,texts,vectors,manifest


def validate_vectors(texts,vectors):
    if texts!=sorted(set(texts)) or not texts or any(not isinstance(t,str) or not t.strip() for t in texts):raise ValueError('Invalid text vocabulary')
    if vectors.shape!=(len(texts),512) or vectors.dtype!=np.float32 or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=2e-6,rtol=0):
        raise ValueError('Invalid vector shape/dtype/norm')


def checked_mapping(root,base_root=None,diagnostic_path=None):
    root=Path(root);r=read(root/'receipt.json')
    if r['status']!='PROMPT_ROI_MAPPING_AUDIT_COMPLETE' or set(r['payload_sha256'])!=set(MAPPING_FILES):raise ValueError('Incomplete full mapping receipt')
    if r['source_inputs_sha256']!={'prompts':PROMPT_SHA,'detections':DETECTION_SHA}:raise ValueError('Mapping inference sources changed')
    if r['roi_package_sha256']!=sha(Path(__file__).with_name('SAVE_FULL_ROI_V134_PACKAGE.json')):raise ValueError('ROI134 identity changed')
    checked_payloads(root,r)
    cohort=read(root/'cohort.json');texts=read(root/'prompt_texts.json');summary=read(root/'summary.json')
    if len(cohort['train'])!=113287 or len(cohort['val'])!=5000 or cohort['test'] or summary['images']!=118287 or summary['test_used']:
        raise ValueError('Expected full113287TRAIN+5000VALID/noTEST')
    old={}
    if base_root is not None:
        with (Path(base_root)/'prompt_links.jsonl').open(encoding='utf-8') as f:
            for line in f:
                row=json.loads(line);old[row['coco_id']]=row
        if len(old)!=15000:raise ValueError('Wrong169link coverage')
    with (root/'prompt_links.jsonl').open(encoding='utf-8') as f:
        validation=validate_records(map(json.loads,f),cohort,texts,summary,old,diagnostic_path)
    return texts,validation


def record_differences(new,old,path=''):
    """Only computed FP64 drift with identical finite FP32 values can be accepted."""
    if not isinstance(new,(dict,list)) and type(new)==type(old) and new==old:return []
    if isinstance(new,dict) and isinstance(old,dict):
        result=[]
        for k in sorted(set(new)|set(old)):
            p=path+'.'+k if path else k
            if k not in new or k not in old:result.append(dict(path=p,old=old.get(k),new=new.get(k),safe=False))
            else:result.extend(record_differences(new[k],old[k],p))
        return result
    if isinstance(new,list) and isinstance(old,list) and len(new)==len(old):
        return [d for i,(a,b) in enumerate(zip(new,old)) for d in record_differences(a,b,f'{path}[{i}]')]
    computed=bool(re.fullmatch(r'objects\[\d+\]\.(?:IoU|roi_bbox\[\d+\])|relations\[\d+\]\.geometry\[\d+\]',path))
    numeric=type(new) in (int,float) and type(old) in (int,float)
    delta=abs(new-old) if numeric else None
    safe=bool(computed and numeric and np.isfinite([new,old]).all() and delta<=1e-12 and
              np.isfinite(np.float32(new)) and np.isfinite(np.float32(old)) and np.float32(new).tobytes()==np.float32(old).tobytes())
    return [dict(path=path,old=old,new=new,safe=safe,abs_delta=delta)]


def validate_records(rows,cohort,texts,summary,old_records,diagnostic_path=None):
    expected={int(i):g for g in ('train','val') for i in cohort[g]}
    if len(expected)!=len(cohort['train'])+len(cohort['val']) or cohort['test']:raise ValueError('Split overlap/duplicates/TEST')
    if texts!=sorted(set(texts)) or not texts:raise ValueError('Invalid unique text list')
    seen=set();used=set();matched=set();counts={g:Counter() for g in ('train','val')};chunk=[]
    exact=safe_drift=unsafe=0;diff_fields=Counter();examples=[];maximum=0.;first_bad=None
    dummy=np.zeros((len(texts),512),np.float32)
    for row in rows:
        i=row['coco_id'];g=row['group']
        if set(row)!={'coco_id','filename','group','roi_count','source_mode','objects','relations','counts'} or type(i)!=int or i in seen or expected.get(i)!=g:
            raise ValueError('Foreign/duplicate ID or reference/supervision in inference payload')
        if not 1<=row['roi_count']<=150:raise ValueError('Invalid ROI count')
        seen.add(i);chunk.append(row)
        if i in old_records:
            differences=record_differences(row,old_records[i])
            if not differences:exact+=1
            else:
                accepted=all(d['safe'] for d in differences)
                safe_drift+=int(accepted);unsafe+=int(not accepted)
                for d in differences:
                    diff_fields[d['path']]+=1
                    if d.get('abs_delta') is not None and d['safe']:maximum=max(maximum,d['abs_delta'])
                    if not d['safe'] and first_bad is None:first_bad=dict(coco_id=i,**d)
                if len(examples)<20:examples.append(dict(coco_id=i,accepted=accepted,differences=differences[:12]))
            matched.add(i)
        for o in row['objects']:used.add(o['text'])
        for rel in row['relations']:used.add(rel['text'])
        counts[g].update(images=1,kept_objects=len(row['objects']),mapped_relations=len(row['relations']))
        if len(chunk)==128:tensor_batch(chunk,dummy,texts);chunk=[]
    if chunk:tensor_batch(chunk,dummy,texts)
    if seen!=set(expected) or used!=set(texts) or matched!=set(old_records):raise ValueError('Incomplete inference/text/base169coverage')
    diagnostic=dict(policy='derived_fp64_atol1e-12_same_finite_fp32_v1' if old_records else 'no_prior_cache_comparison',compared_records=len(matched),exact_records=exact,
        safe_derived_drift_records=safe_drift,unsafe_records=unsafe,max_accepted_abs_delta=maximum,
        difference_fields=dict(diff_fields),examples=examples,first_unsafe=first_bad,semantic_or_fp32_changes_allowed=False)
    if diagnostic_path is not None:save(Path(diagnostic_path),diagnostic)
    if unsafe:raise ValueError('Full mapping changed existing169instance/text/ROI or FP32 input: '+str(first_bad)+'; inspect comparison_diagnostic.json')
    for g,c in counts.items():
        if any(summary['counts'][g].get(k)!=v for k,v in c.items()):raise ValueError('Full mapping count differs')
    return dict(images=len(seen),groups={g:dict(c) for g,c in counts.items()},unique_texts=len(texts),
        old169_records_exactly_preserved=exact,old169_records_safe_derived_drift=safe_drift,
        old169_records_model_inputs_preserved=len(matched),comparison_policy=diagnostic['policy'],
        max_accepted_derived_abs_delta=maximum,test_used=False)


def assemble(texts,old_texts,old_vectors,new_texts,new_vectors):
    validate_vectors(old_texts,old_vectors)
    if not set(old_texts)<=set(texts) or new_texts!=sorted(set(texts)-set(old_texts)):raise ValueError('Wrong incremental vocabulary')
    if new_texts:validate_vectors(new_texts,new_vectors)
    elif new_vectors.shape!=(0,512):raise ValueError('Nonempty encoding for no new texts')
    source={t:old_vectors[i] for i,t in enumerate(old_texts)};source.update({t:new_vectors[i] for i,t in enumerate(new_texts)})
    result=np.stack([source[t] for t in texts]);validate_vectors(texts,result)
    index={t:i for i,t in enumerate(texts)}
    if any(not np.array_equal(result[index[t]],old_vectors[i]) for i,t in enumerate(old_texts)):raise ValueError('Reused169vector bytes changed')
    return result
