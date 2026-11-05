"""Byte-preserving CPU transport for exact169 cache and exact161 weights; no models."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def pattern(names):
    return r'(^|/)('+'|'.join(re.escape(n) for n in names)+r')$'


def locate(root,name,expected):
    matches=[p for p in Path(root).rglob(name) if p.is_file() and sha(p)==expected]
    if len(matches)!=1:
        raise ValueError(f'Need one exact {name}; found {len(matches)} matching SHA, no version fallback')
    return matches[0]


def package_full_cache(download,destination,inventory):
    """Preserve exact175 fresh cache, without caption weights or neural imports."""
    destination=Path(destination);files=inventory['cache_files']
    if inventory['cache_version']!='ducanh2403/2-vi-cai-kia-qua-nhieu/175':raise ValueError('Expected exact175')
    if destination.exists():raise FileExistsError('Preserve existing destination')
    if any(Path(n).name!=n or '/' in n or '\\' in n for n in files):raise ValueError('Unsafe payload name')
    root=locate(download,'cache_receipt.json',files['cache_receipt.json']).parent
    for n,h in files.items():
        if not (root/n).is_file() or sha(root/n)!=h:raise ValueError('Missing/changed175 payload:'+n)
    read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    receipt=read('cache_receipt.json');manifest=read('cache_manifest.json');state=read('run_status.json')
    expected={'cohort.json','prompt_texts.json','prompt_links.jsonl','summary.json','review_examples.json','prompt_embeddings.npy','mapping_receipt.json','reuse_report.json','cache_manifest.json'}
    if receipt['status']!='FULL_PROMPT_TEXT_CACHE_COMPLETE' or set(receipt['payload_sha256'])!=expected or any(files.get(n)!=h for n,h in receipt['payload_sha256'].items()):raise ValueError('Full cache receipt contract differs')
    if state!=dict(complete=True,training=False,test_used=False,caption_weights_loaded=False):raise ValueError('Wrong cache run status')
    if manifest['status']!='FULL_PROMPT_TEXT_CACHE_COMPLETE' or manifest['cache_mode']!='fresh_all' or manifest['shape']!=[410,512] or manifest['training'] or manifest['test_used']:raise ValueError('Wrong full cache manifest')
    texts=read('prompt_texts.json');cohort=read('cohort.json');vectors=np.load(root/'prompt_embeddings.npy',allow_pickle=False)
    if len(texts)!=410 or texts!=sorted(set(texts)) or any(not isinstance(t,str) or not t.strip() for t in texts):raise ValueError('Wrong full text vocabulary')
    if vectors.shape!=(410,512) or vectors.dtype!=np.float32 or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=2e-6,rtol=0):raise ValueError('Embedding shape/dtype/norm differs')
    if len(cohort['train'])!=len(set(cohort['train'])) or len(cohort['train'])!=113287 or len(cohort['val'])!=len(set(cohort['val'])) or len(cohort['val'])!=5000 or cohort['test'] or set(cohort['train'])&set(cohort['val']):raise ValueError('Full cohort coverage differs')
    destination.mkdir(parents=True)
    for n,h in files.items():
        shutil.copy2(root/n,destination/n)
        if sha(destination/n)!=h:raise ValueError('Copied bytes changed:'+n)
    report=dict(status='FULL_PROMPT_V175_PRESERVED',cache_source=inventory['cache_version'],files_sha256=files,
        source_review_zip_sha256=inventory['source_review_zip_sha256'],original_bytes=True,embedding_shape=[410,512],
        models_executed=False,training=False,test_used=False)
    (destination/'transport_receipt.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def package(cache_download,weight_download,destination,inventory):
    destination=Path(destination)
    if destination.exists():raise FileExistsError('Preserve existing destination; choose a fresh one')
    files=inventory['cache_files']
    if any(Path(n).name!=n or '/' in n or '\\' in n for n in files):raise ValueError('Unsafe payload name')
    receipt_path=locate(cache_download,'cache_receipt.json',files['cache_receipt.json'])
    root=receipt_path.parent
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt['status']!='PROMPT_TEXT_CACHE_COMPLETE':raise ValueError('Incomplete cache')
    if any(n not in files or files[n]!=h for n,h in receipt['payload_sha256'].items()):raise ValueError('Cache receipt identities differ')
    if set(receipt['payload_sha256'])!={'prompt_embeddings.npy','prompt_texts.json','prompt_links.jsonl','cohort.json','summary.json','cache_manifest.json'}:
        raise ValueError('Cache receipt payload coverage differs')
    for n,h in files.items():
        if not (root/n).is_file() or sha(root/n)!=h:raise ValueError('Missing/changed169 payload:'+n)
    state=json.loads((root/'run_status.json').read_text(encoding='utf-8'))
    if not state['complete'] or state['training'] or state['test_used']:raise ValueError('Wrong cache run status')
    vectors=np.load(root/'prompt_embeddings.npy',allow_pickle=False)
    if vectors.shape!=(318,512) or vectors.dtype!=np.float32 or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=2e-6,rtol=0):
        raise ValueError('Embedding shape/dtype/norm differs')
    weight=locate(weight_download,'best.bin',inventory['checkpoint_sha256'])
    # Binary identity is checked without deserializing weights or importing Torch.
    destination.mkdir(parents=True)
    for n in files:shutil.copy2(root/n,destination/n)
    shutil.copy2(weight,destination/'csa_ce161_best.bin')
    expected={**files,'csa_ce161_best.bin':inventory['checkpoint_sha256']}
    for n,h in expected.items():
        if sha(destination/n)!=h:raise ValueError('Copied bytes changed:'+n)
    report=dict(status='PROMPT_PILOT_INPUTS_PRESERVED',cache_source=inventory['cache_version'],
        checkpoint_source=inventory['checkpoint_version'],files_sha256=expected,
        original_bytes=True,models_executed=False,training=False,test_used=False)
    (destination/'transport_receipt.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report
