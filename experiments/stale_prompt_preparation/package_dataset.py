"""Preserve the audited runnable cache; no model or credential imports."""
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def package(source,destination,inventory):
    names={'aux_rank0.h5','aux_rank1.h5','benchmark_inputs.h5','receipt.json','protocol.json','manifest.json',
        'cache_rank0.json','cache_rank1.json','benchmark_rank0.json','benchmark_rank1.json',
        'four_scores_rank0.json','four_scores_rank1.json','four_condition_interpretation.json','panel.html','summary.json'}
    if inventory['schema']!='stale_prompt_2000_transport_v1' or set(inventory['files'])!=names:raise ValueError('Unexpected inventory.')
    source=Path(source).resolve();destination=Path(destination).resolve()
    if source==destination or source in destination.parents or destination in source.parents:raise ValueError('Use separate, non-nested source/destination.')
    for name,record in inventory['files'].items():
        if not (source/name).is_file() or sha(source/name)!=record['sha256']:raise ValueError('Missing/changed source: '+name)
        print('Verified source:',name,flush=True)
    receipt=json.loads((source/'receipt.json').read_text())
    if receipt['status']!='PREPARATION_COMPLETE' or receipt['images']!=2000 or receipt['full_pilot_run']:raise ValueError('Not the completed untrained cache.')
    panel_names=receipt['panel_sha256']
    for name,expected in panel_names.items():
        if Path(name).name!=name or not (source/'panel'/name).is_file() or sha(source/'panel'/name)!=expected:raise ValueError('Missing/changed panel: '+name)
    destination.mkdir(parents=True,exist_ok=True)
    allowed=names|{'transport_receipt.json'}|{'panel__'+n for n in panel_names}
    if any(p.is_symlink() or p.name not in allowed for p in destination.iterdir()):raise ValueError('Unrelated staging files preserved; use fresh destination.')
    needed=sum((source/n).stat().st_size for n in names if not (destination/n).exists())
    if shutil.disk_usage(destination).free<needed+256*1024**2:raise RuntimeError('Insufficient space for verified copy.')
    for name in sorted(names):
        target=destination/name
        if target.exists():
            if not target.is_file() or sha(target)!=inventory['files'][name]['sha256']:raise ValueError('Changed destination preserved: '+name)
            continue
        part=destination/(name+'.partial')
        shutil.copyfile(source/name,part)
        if sha(part)!=inventory['files'][name]['sha256']:raise ValueError('Copied bytes differ: '+name)
        part.replace(target);print('Copied:',name,flush=True)
    # Preserve panel pixels separately without modifying any receipted payload.
    for name,expected in panel_names.items():
        if Path(name).name!=name or not (source/'panel'/name).is_file() or sha(source/'panel'/name)!=expected:raise ValueError('Missing/changed panel: '+name)
        target=destination/('panel__'+name)
        if target.exists() and sha(target)!=expected:raise ValueError('Changed panel destination preserved.')
        if not target.exists():shutil.copyfile(source/'panel'/name,target)
    report=dict(status='DATASET_FOLDER_READY',schema=inventory['schema'],payload_files=len(names),
        files=inventory['files'],panel_files={('panel__'+n):s for n,s in panel_names.items()},
        source_review_zip_sha256=inventory['source_review_zip_sha256'],trained=False,
        note='15 root payloads unchanged, including3HDF5. Panel files flattened as panel__NAME; original panel.html uses panel/ paths. Recreate panel/ from these when viewing. Historical logs/source and legacy_subset.pt excluded; main pilot uses original legacy dataset.')
    (destination/'transport_receipt.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report
