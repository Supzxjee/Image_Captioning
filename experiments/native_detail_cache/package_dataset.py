"""Verify/copy the audited H/L cache for dataset upload; standard library only."""
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def validate_inventory(inventory):
    names={'cache_receipt.json','cache_spec.json','manifest.json'}
    names|={f'{group}_{i:03d}.{suffix}' for group,count in [('train',20),('val',10)] for i in range(count) for suffix in ('h5','json')}
    if inventory['schema']!='native_detail_HL_15k_transport_v1' or set(inventory['files'])!=names:raise ValueError('Unexpected pinned payload inventory.')


def locate(inventory,source=None):
    validate_inventory(inventory);expected=inventory['files']['cache_receipt.json']['sha256']
    if source is not None:
        candidates=[Path(source)/'cache_receipt.json']
    else:
        candidates=list(Path('/kaggle/input').rglob('cache_receipt.json'))
        local=Path('/kaggle/working/native_detail_HL_15k/cache_receipt.json')
        if local.exists():candidates.append(local)
    matches={p.parent.resolve() for p in candidates if p.is_file() and sha(p)==expected}
    if len(matches)!=1:raise ValueError(f'Found {len(matches)} matching completed cache folders. Attach the completed cache notebook Output, not the review ZIP; set SOURCE to the folder containing cache_spec.json if needed.')
    return matches.pop()


def package(source,destination,inventory):
    validate_inventory(inventory);source=Path(source).resolve();destination=Path(destination).resolve()
    if source==destination or source in destination.parents or destination in source.parents:raise ValueError('Source and destination must be separate non-nested folders.')
    destination.mkdir(parents=True,exist_ok=True);expected=inventory['files']
    allowed=set(expected)|{n+'.partial' for n in expected}|{'transport_receipt.json'}
    if any(p.is_symlink() or p.name not in allowed for p in destination.iterdir()):raise ValueError('Staging contains unrelated files; preserve it and choose a fresh DESTINATION.')
    needed=sum(r['bytes'] for name,r in expected.items() if not (destination/name).is_file())
    if shutil.disk_usage(destination).free<needed+256*1024**2:raise RuntimeError('Insufficient disk for the verified cache copy plus256MiB.')
    # Validate the entire source first; missing/changed shards stop before any upload is possible.
    for name,record in expected.items():
        p=source/name
        if not p.is_file() or p.stat().st_size!=record['bytes'] or sha(p)!=record['sha256']:raise ValueError('Source payload missing/changed: '+name)
        print('Source verified:',name,flush=True)
    for name,record in expected.items():
        target=destination/name
        if target.exists():
            if not target.is_file() or target.stat().st_size!=record['bytes'] or sha(target)!=record['sha256']:raise ValueError('Existing destination differs; preserving: '+name)
            continue
        temporary=destination/(name+'.partial')
        with (source/name).open('rb') as reader,temporary.open('wb') as writer:shutil.copyfileobj(reader,writer,8*1024**2)
        if sha(temporary)!=record['sha256']:raise ValueError('Copied bytes differ: '+name)
        temporary.replace(target);print('Copied and verified:',name,flush=True)
    if any(p.name not in set(expected)|{'transport_receipt.json'} for p in destination.iterdir()):raise ValueError('Uncommitted partial staging files remain; preserved. Choose a fresh DESTINATION.')
    report=dict(status='DATASET_FOLDER_READY',schema=inventory['schema'],files=expected,payload_files=63,
        source_review_zip_sha256=inventory['source_review_zip_sha256'],source_spec_sha256=inventory['source_spec_sha256'],
        bytes=sum(r['bytes'] for r in expected.values()),trained=False,caption_evaluated=False,
        note='Flat dataset preserves all30HDF5 files and33 verification/manifest files byte-for-byte. Historical logs/source remain in the original notebook Output and GitHub review.')
    (destination/'transport_receipt.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('VERIFIED DATASET FOLDER READY:',destination,flush=True)
    return report
