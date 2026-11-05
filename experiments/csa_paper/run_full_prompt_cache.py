"""USER-RUN CPU fullTRAIN prompt cache; reuse169 vectors and encode only unseen strings."""
import argparse
from pathlib import Path
import shutil
import time
import numpy as np
from real_bank_data import read,save,sha
from full_prompt_cache_data import checked_base,checked_mapping,assemble,MAPPING_FILES,BASE_RECEIPT_SHA,BASE_MANIFEST_SHA
from prompt_text_cache import encode_cpu


def inspect(cfg):
    base,old_texts,old_vectors,old_manifest=checked_base(cfg['base_root'])
    texts,validation=checked_mapping(cfg['mapping_root'],base,Path(cfg['output'])/'comparison_diagnostic.json')
    new=sorted(set(texts)-set(old_texts))
    if not set(old_texts)<=set(texts):raise ValueError('FullTRAIN cache lost old169texts')
    plan=dict(status='FULL_PROMPT_CACHE_PLAN_VALIDATED',images=validation['images'],validation=validation,
        unique_texts=len(texts),reused_texts=len(old_texts),new_texts=new,encode_new_count=len(new),
        base169_receipt_sha256=BASE_RECEIPT_SHA,base169_manifest_sha256=BASE_MANIFEST_SHA,
        mapping_receipt_sha256=sha(Path(cfg['mapping_root'])/'receipt.json'),models_executed=False,test_used=False)
    save(Path(cfg['output'])/'cache_plan.json',plan)
    print('FullTRAIN prompt cache plan:',dict(images=validation['images'],reuse=len(old_texts),encode_new=len(new),unique_texts=len(texts)),flush=True)
    return base,old_texts,old_vectors,old_manifest,texts,validation,new,plan


def incremental_vectors(texts,old_texts,old_vectors,hf_cache):
    new=sorted(set(texts)-set(old_texts));meta=dict(model_executed=False,max_tokens=0,seconds=0.)
    if new:
        vectors,meta=encode_cpu(new,hf_cache);meta['model_executed']=True
    else:vectors=np.empty((0,512),np.float32)
    return assemble(texts,old_texts,old_vectors,new,vectors),meta


def cache(cfg):
    started=time.monotonic();out=Path(cfg['output'])
    if (out/'cache_receipt.json').exists():raise ValueError('Preserve completed cache; use a fresh output')
    base,old_texts,old_vectors,old_manifest,texts,validation,new,plan=inspect(cfg)
    result,encoded=incremental_vectors(texts,old_texts,old_vectors,cfg['hf_cache'])
    if encoded['model_executed'] and (encoded['spec']['id']!=old_manifest['model'] or encoded['spec']['revision']!=old_manifest['revision'] or encoded['files_sha256']!=old_manifest['files_sha256']):
        raise ValueError('Incremental CLIP space differs from169')
    mapping=Path(cfg['mapping_root'])
    np.save(out/'prompt_embeddings.npy',result)
    for n in MAPPING_FILES:shutil.copy2(mapping/n,out/n)
    shutil.copy2(mapping/'receipt.json',out/'mapping_receipt.json')
    save(out/'reuse_report.json',dict(base169_receipt_sha256=BASE_RECEIPT_SHA,base169_manifest_sha256=BASE_MANIFEST_SHA,
        reused_vectors=len(old_texts),new_vectors=len(new),total_vectors=len(texts),reused_vectors_exact=True,
        old169_records_exactly_preserved=validation['old169_records_exactly_preserved'],
        old169_records_model_inputs_preserved=validation.get('old169_records_model_inputs_preserved',validation['old169_records_exactly_preserved']),
        old169_records_safe_derived_drift=validation.get('old169_records_safe_derived_drift',0),
        comparison_policy=validation.get('comparison_policy','exact_v1'),
        model_executed_for_new_texts=encoded['model_executed'],existing_vectors_reencoded=False))
    manifest=dict(schema='csa_prompt_full_karpathy_v1',status='FULL_PROMPT_TEXT_CACHE_COMPLETE',
        model=old_manifest['model'],revision=old_manifest['revision'],files_sha256=old_manifest['files_sha256'],
        base169_receipt_sha256=BASE_RECEIPT_SHA,mapping_receipt_sha256=plan['mapping_receipt_sha256'],
        validation=validation,texts=len(texts),shape=list(result.shape),dtype='float32',device='cpu',frozen=True,
        max_tokens=max(old_manifest['max_tokens'],encoded['max_tokens']),truncation=False,
        new_encoding=encoded,training=False,test_used=False,seconds=time.monotonic()-started)
    save(out/'cache_manifest.json',manifest)
    names=list(MAPPING_FILES)+['prompt_embeddings.npy','mapping_receipt.json','reuse_report.json','cache_manifest.json']
    if (out/'comparison_diagnostic.json').is_file():names.append('comparison_diagnostic.json')
    save(out/'cache_receipt.json',dict(schema=manifest['schema'],status=manifest['status'],payload_sha256={n:sha(out/n) for n in names}))
    print('FULL PROMPT TEXT CACHE COMPLETE:118287images,',len(texts),'vectors. No caption training/TEST.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('inspect','cache'));p.add_argument('--config',required=True)
    a=p.parse_args();globals()[a.mode](read(a.config))
