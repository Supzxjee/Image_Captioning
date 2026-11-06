"""USER-RUN CPU preparation only; no caption model or metric evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
from real_bank_data import read,save,sha,validate_arrays
from prompt_adapter_data import PROMPT_SHA,detector_regions,join_entry,inference_record,tensor_batch
from audit_assets import detection_index
from prompt_scratch_data import checked_cache


def map_records(metadata,arrays,prompts,detections):
    if len({r['coco_id'] for r in metadata})!=len(metadata):raise ValueError('Duplicate TEST IDs')
    result=[]
    for j,meta in enumerate(metadata):
        if meta['group']!='test' or meta['source_split']!='test':raise ValueError('Non-TEST image in preparation')
        name=meta['filename']
        if name not in prompts or name not in detections:raise ValueError('Missing original UPT/detector TEST entry:'+name)
        if int(arrays['ids'][j])!=meta['coco_id']:raise ValueError('ROI ID order differs')
        lo,hi=map(int,arrays['offsets'][j:j+2])
        regions=detector_regions(detections[name],meta['width'],meta['height'],arrays['boxes'][lo:hi],meta['region_kinds'])
        result.append(inference_record(meta,join_entry(prompts[name],regions)))
        if len(result)%100==0:print('TEST prompt/ROI',len(result),'/',len(metadata),flush=True)
    return result


def prepare(cfg):
    from prepare_test_regions import prepare as roi_prepare
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    spec=read(Path(__file__).with_name('RERANK_TEST_INPUTS.json'))['test_receipt']
    if sha(cfg['prompts'])!=PROMPT_SHA:raise ValueError('Changed UPT prompt JSON')
    prompts=read(cfg['prompts'])
    dataset=read(cfg['dataset_json']);test=[r for r in dataset['images'] if r['split']=='test']
    missing=[r['filename'] for r in test if r['filename'] not in prompts]
    save(out/'source_coverage.json',dict(test_images=len(test),missing_prompt_count=len(missing),missing_examples=missing[:20]))
    if len(test)!=5000 or missing:raise ValueError('Original prompt JSON lacks full TEST5000 coverage; send review, no substitution')
    roi_prepare(dict(dataset_json=cfg['dataset_json'],detections=cfg['detections'],visual_cache=cfg['visual_cache'],
        image_root=cfg['image_root'],output=str(out)))
    for n,h in spec['files'].items():
        if sha(out/n)!=h:raise ValueError('TEST ROI differs from historical fixed cache:'+n)
    metadata=read(out/'test_metadata.json');vocab=read(out/'vocabulary.json')
    with np.load(out/'test_regions.npz',allow_pickle=False) as f:arrays={k:f[k] for k in f.files}
    validate_arrays(arrays,metadata,vocab)
    rows=map_records(metadata,arrays,prompts,detection_index(read(cfg['detections'])))
    _,_,old,old_texts,inventory=checked_cache(cfg['prompt_root'])
    texts=sorted({o['text'] for r in rows for o in r['objects']+r['relations']});new=sorted(set(texts)-set(old_texts))
    lookup={t:old[i] for i,t in enumerate(old_texts)};encoded=None
    if new:
        from prompt_text_cache import encode_cpu
        vectors,encoded=encode_cpu(new,cfg['hf_cache'])
        expected=read(Path(__file__).with_name('RERANK_TEST_INPUTS.json'))['clip']
        if encoded['files_sha256']!=expected['files_sha256']:raise ValueError('Changed text CLIP space')
        lookup.update(zip(new,vectors))
    vectors=np.stack([lookup[t] for t in texts]).astype(np.float32) if texts else np.empty((0,512),np.float32)
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=2e-6,rtol=0):raise ValueError('Invalid text vectors')
    for offset in range(0,len(rows),32):tensor_batch(rows[offset:offset+32],vectors,texts)
    np.save(out/'prompt_embeddings.npy',vectors);save(out/'prompt_texts.json',texts)
    with (out/'prompt_links.jsonl').open('w',encoding='utf-8',newline='\n') as f:
        for r in rows:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
    summary=dict(status='PROMPT_TEST_PREPARATION_COMPLETE',test_images=5000,caption_generation=False,caption_training=False,caption_scoring=False,
        with_objects=sum(bool(r['objects']) for r in rows),with_relations=sum(bool(r['relations']) for r in rows),
        objects=sum(len(r['objects']) for r in rows),relations=sum(len(r['relations']) for r in rows),
        texts=len(texts),reused_texts=len(texts)-len(new),new_texts=new,source175_files=inventory['cache_files'],new_encoding=encoded)
    save(out/'summary.json',summary)
    names=('test_regions.npz','test_metadata.json','vocabulary.json','test_receipt.json','prompt_links.jsonl','prompt_texts.json','prompt_embeddings.npy','summary.json','source_coverage.json')
    save(out/'receipt.json',dict(status=summary['status'],payload_sha256={n:sha(out/n) for n in names},prompt_source_sha256=PROMPT_SHA))
    print('TEST PREPARATION COMPLETE; no captions generated/scored. Preserve entire folder.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args();prepare(read(a.config))
