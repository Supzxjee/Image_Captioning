"""Byte-preserving cache transport; no API/model imports."""
import hashlib
import json
from pathlib import Path
import shutil

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def package(source,destination,inventory):
    source,destination=Path(source),Path(destination)
    if inventory['kernel_version']!='ducanh2403/2-vi-cai-kia-qua-nhieu/130':raise ValueError('Expected exact Output130')
    if destination.exists():raise ValueError('Preserve existing destination; use a fresh path')
    for name,h in inventory['files'].items():
        if Path(name).name!=name or not (source/name).is_file() or sha(source/name)!=h:raise ValueError('Missing/changed output: '+name)
    read=lambda name:json.loads((source/name).read_text(encoding='utf-8'))
    receipt=read('cache_receipt.json');smoke=read('smoke_summary.json');manifest=read('cache_manifest.json')
    if receipt['status']!='SEMANTIC_BANK_CACHE_COMPLETE' or smoke['status']!='REAL_INPUT_SMOKE_COMPLETE':raise ValueError('Incomplete source')
    for name,h in receipt['payload_sha256'].items():
        if inventory['files'].get(name)!=h:raise ValueError('Cache receipt payload differs')
    if manifest['source_bank_sha256']!=receipt['inputs']['overlay_sha256']['bank_words_corrected.json']:raise ValueError('Source bank differs')
    destination.mkdir(parents=True)
    for name,h in inventory['files'].items():
        shutil.copyfile(source/name,destination/name)
        if sha(destination/name)!=h:raise ValueError('Copied bytes differ: '+name)
    report=dict(status='SEMANTIC_BANK_V130_TRANSPORT_COMPLETE',source_kernel_version=inventory['kernel_version'],
        source_review_zip_sha256=inventory['source_review_zip_sha256'],files_sha256=inventory['files'],
        original_bytes_preserved=True,model_executed=False,also_requires_dataset='ducanh2403/csa-real-v126-lemma-fixed')
    (destination/'transport_receipt.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report
