"""USER-RUN CPU frozen CLIP cache. Validation helpers are data-only."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import time
import numpy as np
from real_bank_data import read,save,sha
from prompt_adapter_data import tensor_batch

RECEIPT_SHA='2edaef51011684a01d948bb54df6607b9b8a7dba6da3de713e6245adae91429a'
SOURCE_FILES=('receipt.json','summary.json','cohort.json','prompt_links.jsonl','prompt_texts.json','review_examples.json')


def source_pattern():
    return r'(^|/)('+ '|'.join(re.escape(n) for n in SOURCE_FILES)+r')$'


def checked_source(root):
    matches=[p for p in Path(root).rglob('receipt.json') if sha(p)==RECEIPT_SHA]
    if len(matches)!=1:raise ValueError('Need one exact mapping receipt from SaveVersion167, no latest substitute')
    root=matches[0].parent;r=read(matches[0])
    if r['status']!='PROMPT_ROI_MAPPING_AUDIT_COMPLETE':raise ValueError('Mapping incomplete')
    for n,h in r['payload_sha256'].items():
        if Path(n).name!=n or not (root/n).is_file() or sha(root/n)!=h:raise ValueError('Missing/changed167 payload:'+n)
    return root,r


def validate_links(root):
    cohort=read(root/'cohort.json');texts=read(root/'prompt_texts.json');summary=read(root/'summary.json')
    if len(texts)!=318 or texts!=sorted(set(texts)) or any(not isinstance(s,str) or not s.strip() for s in texts):raise ValueError('Wrong318text list')
    expected={i:g for g in ('train','val') for i in cohort[g]}
    if len(expected)!=15000 or len(cohort['train'])!=10000 or len(cohort['val'])!=5000 or cohort['test']:raise ValueError('Wrong mapping cohort')
    used=set();seen=set();counts={g:Counter() for g in ('train','val')};dummy=np.zeros((318,512),np.float32)
    with (root/'prompt_links.jsonl').open(encoding='utf-8') as f:
        for line in f:
            row=json.loads(line);i=row['coco_id'];g=row['group']
            if i in seen or expected.get(i)!=g or set(row)!={'coco_id','filename','group','roi_count','source_mode','objects','relations','counts'}:
                raise ValueError('Unexpected/leaked inference fields or duplicate/foreign ID')
            seen.add(i);tensor_batch([row],dummy,texts)
            for obj in row['objects']:used.add(obj['text'])
            for rel in row['relations']:used.add(rel['text'])
            counts[g]['images']+=1;counts[g]['kept_objects']+=len(row['objects']);counts[g]['mapped_relations']+=len(row['relations'])
    if seen!=set(expected) or used!=set(texts):raise ValueError('Incomplete link/text coverage')
    for g,c in counts.items():
        if any(summary['counts'][g][k]!=v for k,v in c.items()):raise ValueError('Link counts differ from167 summary')
    return texts,dict(images=len(seen),unique_texts=len(texts),groups={k:dict(v) for k,v in counts.items()},test_used=False)


def encode_cpu(texts,hf_cache):
    if not texts or len(set(texts))!=len(texts):raise ValueError('Need unique nonempty prompt texts')
    # Neural imports and execution occur only in the user-run cache phase.
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import CLIPModel,CLIPTokenizer
    if tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])<(2,6):raise RuntimeError('Torch>=2.6 required; do not replace Torch automatically')
    torch.set_num_threads(min(4,torch.get_num_threads()))
    spec=read(Path(__file__).with_name('RERANK_TEST_INPUTS.json'))['clip'];files={};started=time.monotonic()
    for name,want in spec['files_sha256'].items():
        print('Pinned CLIP asset:',name,flush=True)
        path=Path(hf_hub_download(spec['id'],name,revision=spec['revision'],cache_dir=hf_cache,token=False))
        files[name]=sha(path)
        if files[name]!=want:raise ValueError('Changed CLIP asset:'+name)
    tokenizer=CLIPTokenizer.from_pretrained(path.parent,local_files_only=True)
    model,info=CLIPModel.from_pretrained(path.parent,local_files_only=True,use_safetensors=False,weights_only=True,
        trust_remote_code=False,torch_dtype=torch.float32,attn_implementation='eager',output_loading_info=True)
    allowed={'text_model.embeddings.position_ids','vision_model.embeddings.position_ids'}
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs') or set(info.get('unexpected_keys',[]))-allowed:raise ValueError('CLIP loading keys differ')
    if model.config.projection_dim!=512 or model.config.text_config.max_position_embeddings!=77:raise ValueError('CLIP config differs')
    model.requires_grad_(False).eval().to('cpu');chunks=[];maximum=0
    for start in range(0,len(texts),32):
        tokens=tokenizer(texts[start:start+32],padding=True,truncation=False,return_tensors='pt')
        maximum=max(maximum,tokens.input_ids.shape[1])
        if maximum>77:raise ValueError('Overlong prompt; truncation forbidden')
        with torch.inference_mode():
            vectors=model.get_text_features(**tokens).float();norms=vectors.norm(dim=-1,keepdim=True)
            if not torch.isfinite(vectors).all() or not (norms>0).all():raise ValueError('Invalid CLIP vectors')
            chunks.append((vectors/norms).numpy())
        print('CLIP text CPU',min(start+32,len(texts)),'/',len(texts),flush=True)
    result=np.concatenate(chunks)
    if result.shape!=(len(texts),512) or result.dtype!=np.float32 or not np.allclose(np.linalg.norm(result,axis=1),1,rtol=0,atol=2e-6):raise ValueError('Embedding cache shape/norm differs')
    return result,dict(spec=spec,files_sha256=files,loading_info=info,max_tokens=maximum,seconds=time.monotonic()-started)


def cache(cfg):
    root,receipt=checked_source(cfg['source_root']);texts,validation=validate_links(root)
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    result,encoded=encode_cpu(texts,cfg['hf_cache']);spec=encoded['spec']
    np.save(out/'prompt_embeddings.npy',result)
    for n in ('prompt_texts.json','prompt_links.jsonl','cohort.json','summary.json'):shutil.copy2(root/n,out/n)
    manifest=dict(status='PROMPT_TEXT_CACHE_COMPLETE',source_version='ducanh2403/2-vi-cai-kia-qua-nhieu/167',
        mapping_receipt_sha256=RECEIPT_SHA,source_payload_sha256=receipt['payload_sha256'],validation=validation,
        model=spec['id'],revision=spec['revision'],files_sha256=encoded['files_sha256'],loading_info=encoded['loading_info'],
        texts=318,shape=[318,512],dtype='float32',device='cpu',frozen=True,training=False,test_used=False,
        max_tokens=encoded['max_tokens'],truncation=False,seconds=time.monotonic()-started)
    save(out/'cache_manifest.json',manifest)
    names=('prompt_embeddings.npy','prompt_texts.json','prompt_links.jsonl','cohort.json','summary.json','cache_manifest.json')
    save(out/'cache_receipt.json',dict(status='PROMPT_TEXT_CACHE_COMPLETE',payload_sha256={n:sha(out/n) for n in names}))
    print('PROMPT TEXT CACHE COMPLETE; no caption model trained. Preserve entire folder.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args();cache(read(args.config))
