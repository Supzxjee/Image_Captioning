"""Exact139 checkpoint transport; byte checks only, never deserialize weights."""
import hashlib
import json
from pathlib import Path
import shutil

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def package(download,destination,inventory):
    download,destination=Path(download),Path(destination)
    if inventory['kernel_version']!='ducanh2403/2-vi-cai-kia-qua-nhieu/139':raise ValueError('Expected exact version139')
    if destination.exists():raise ValueError('Preserve existing destination; choose a fresh path')
    paths={}
    for name,h in inventory['files'].items():
        if Path(name).name!=name:raise ValueError('Unexpected payload path')
        matches=[p for p in download.rglob(name) if p.is_file() and sha(p)==h]
        if len(matches)!=1:raise ValueError(f'Output139: expected one verified {name}, got {len(matches)}; no latest fallback')
        paths[name]=matches[0]
    read=lambda n:json.loads(paths[n].read_text(encoding='utf-8'))
    r=read('result.json');best=read('best_receipt.json');last=read('checkpoint_receipt.json')
    if r['status']!='FULL_XE_10_COMPLETE' or (r['completed_epochs'],r['updates'],r['best_epoch'])!=(10,177110,9):raise ValueError('Wrong training endpoint')
    if (best['validated_epoch'],best['updates'],best['sha256'])!=(9,159399,inventory['files']['best.bin']):raise ValueError('Wrong best checkpoint')
    if (last['epoch'],last['updates'],last['sha256'])!=(11,177110,inventory['files']['checkpoint.bin']):raise ValueError('Wrong latest checkpoint')
    destination.mkdir(parents=True)
    for name,p in paths.items():
        shutil.copy2(p,destination/name)
        if sha(destination/name)!=inventory['files'][name]:raise ValueError('Copy bytes differ:'+name)
    receipt=dict(status='CSA_XE139_TRANSPORT_COMPLETE',source=inventory['kernel_version'],files_sha256=inventory['files'],
        best_validated_epoch=9,latest_trained_epoch=10,weights_deserialized=False,
        also_requires=['ducanh2403/csa-full-roi-karpathy-v134','ducanh2403/csa-semantic-banks-v130'])
    (destination/'transport_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt
