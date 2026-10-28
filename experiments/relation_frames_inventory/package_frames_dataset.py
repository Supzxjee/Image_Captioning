"""Verify/copy exact version105 cache bytes; standard library only, no API/model."""
import hashlib
import json
from pathlib import Path
import re
import shutil

NAMES={'receipt.json','inputs.h5','relation_frames_manifest.bin','images.json','detector_boxes.json'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def package(source,destination,inventory):
    expected=inventory['files']
    if inventory['schema']!='relation_frames_v2_transport_v1' or set(expected)!=NAMES:raise ValueError('Wrong inventory')
    if any(not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h) for h in expected.values()):raise ValueError('Invalid pinned hashes')
    source=Path(source).resolve();destination=Path(destination).resolve()
    if source==destination or source in destination.parents or destination in source.parents:raise ValueError('Use separate non-nested folders')
    for n,h in expected.items():
        p=source/n
        if p.is_symlink() or not p.is_file() or sha(p)!=h:raise ValueError('Missing/changed source: '+n)
        print('Verified source:',n,flush=True)
    receipt=json.loads((source/'receipt.json').read_text(encoding='utf-8'))
    if receipt['status']!='CACHE_COPY_COMPLETE_PENDING_LABEL_AUDIT' or receipt['images']!=612 or receipt['legacy_prompt_coverage']!=612:
        raise ValueError('Wrong completed cache')
    if any(receipt[k] for k in ('feature_extraction','caption_training','neural_forward')):raise ValueError('Unexpected model execution')
    if receipt['payload_sha256']!={n:h for n,h in expected.items() if n!='receipt.json'}:raise ValueError('Receipt manifest differs')
    destination.mkdir(parents=True,exist_ok=True)
    allowed=NAMES|{n+'.partial' for n in NAMES}|{'transport_receipt.json'}
    if any(p.is_symlink() or p.name not in allowed for p in destination.iterdir()):raise ValueError('Unrelated destination preserved')
    needed=sum((source/n).stat().st_size for n in expected if not (destination/n).exists())
    if shutil.disk_usage(destination).free<needed+256*1024**2:raise RuntimeError('Insufficient copy space')
    for n,h in expected.items():
        target=destination/n
        if target.exists():
            if not target.is_file() or sha(target)!=h:raise ValueError('Changed destination preserved: '+n)
            continue
        partial=destination/(n+'.partial');shutil.copyfile(source/n,partial)
        if sha(partial)!=h:raise ValueError('Copy differs: '+n)
        partial.replace(target)
    if any(p.name not in NAMES|{'transport_receipt.json'} for p in destination.iterdir()):raise ValueError('Partial files remain')
    report=dict(status='DATASET_FOLDER_READY',schema=inventory['schema'],files=expected,payload_files=5,
        bytes=sum((source/n).stat().st_size for n in expected),source_kernel_version=inventory['kernel_version'],
        source_review_zip_sha256=inventory['source_review_zip_sha256'],neural_forward=False,caption_training=False,
        note='Five original files unchanged at dataset root; full HDF5 hash verified during transport.')
    (destination/'transport_receipt.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report
