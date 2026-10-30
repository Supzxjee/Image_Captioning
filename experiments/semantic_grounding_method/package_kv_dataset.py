"""Verify/copy original CPU preparation bytes; no API, HDF5 reader, or model."""
import hashlib
import json
from pathlib import Path
import re
import shutil

NAMES = {'auxiliary_facts.json','cpu_preparation.json','detector_boxes.json',
         'inputs.h5','metadata.json','protocol.json','source_receipts.json','statistics.json'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def package(source, destination, inventory):
    expected = inventory['files']
    if inventory['schema'] != 'relation_kv_transport_v1' or set(expected) != NAMES:
        raise ValueError('Wrong inventory')
    if any(not re.fullmatch('[0-9a-f]{64}', h) for h in expected.values()):
        raise ValueError('Invalid pinned hashes')
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Use separate non-nested folders')
    for name, expected_sha in expected.items():
        path = source/name
        if path.is_symlink() or not path.is_file() or sha(path) != expected_sha:
            raise ValueError('Missing/changed source: '+name)
        print('Verified source:',name,flush=True)
    receipt = json.loads((source/'cpu_preparation.json').read_bytes())
    if receipt['status'] != 'READY_FOR_MODEL_PREFLIGHT' or receipt['rows'] != 5472:
        raise ValueError('Wrong completed preparation')
    if any(receipt[k] for k in ('neural_forward','feature_extraction','checkpoint_loaded')):
        raise ValueError('Unexpected model execution')
    if receipt['payload_sha256'] != {n:h for n,h in expected.items() if n != 'cpu_preparation.json'}:
        raise ValueError('Receipt payload manifest differs')
    destination.mkdir(parents=True,exist_ok=True)
    if any(p.is_symlink() or p.name not in NAMES|{'transport_receipt.json'} for p in destination.iterdir()):
        raise ValueError('Unrelated or partial destination preserved')
    needed = sum((source/n).stat().st_size for n in expected if not (destination/n).exists())
    if shutil.disk_usage(destination).free < needed+256*1024**2:
        raise RuntimeError('Insufficient copy space')
    for name, expected_sha in expected.items():
        target = destination/name
        if target.exists():
            if not target.is_file() or sha(target) != expected_sha:
                raise ValueError('Changed destination preserved: '+name)
            continue
        partial = destination/(name+'.partial')
        shutil.copyfile(source/name,partial)
        if sha(partial) != expected_sha:
            raise ValueError('Copy differs: '+name)
        partial.replace(target)
    report = dict(status='DATASET_FOLDER_READY',schema=inventory['schema'],files=expected,
                  source_kernel_version=inventory['kernel_version'],payload_files=len(NAMES),
                  source_review_zip_sha256=inventory['source_review_zip_sha256'],
                  full_payload_hashes_verified=True,model_executed=False,
                  note='Original eight files unchanged at dataset root. No HDF5/model execution.')
    (destination/'transport_receipt.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report
