"""Copy a verified hidden-pair cache to a small Kaggle Dataset-ready folder; stdlib only."""
import hashlib
import json
from pathlib import Path
import shutil

RECEIPT_SHA = '873f6da0e1b4c4f7d7408b67c593c07ce56e5b4ca3fa1db5f9ce5e04aa188920'
REQUIRED = {'cohort.json', 'features_rank0.h5', 'features_rank1.h5', 'pairs_rank0.jsonl', 'pairs_rank1.jsonl'}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            result.update(block)
    return result.hexdigest()


def locate(source=None):
    if source is not None:
        folder = Path(source)
        if not (folder / 'cache_receipt.json').is_file():
            raise FileNotFoundError(f'SOURCE must contain cache_receipt.json: {folder}')
        return folder
    local = Path('/kaggle/working/upt_hidden_pairs_11k')
    if (local / 'cache_receipt.json').is_file() and sha(local / 'cache_receipt.json') == RECEIPT_SHA:
        return local
    candidates = [p.parent for p in Path('/kaggle/input').rglob('cache_receipt.json') if sha(p) == RECEIPT_SHA]
    if len(candidates) != 1:
        raise RuntimeError(f'Found {len(candidates)} matching cache folders. Attach completed cache notebook Output, then set SOURCE to its exact folder. The review ZIP alone is insufficient.')
    return candidates[0]


def package(source, destination, receipt_sha=RECEIPT_SHA):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Source and destination must be separate, non-nested folders.')
    receipt_path = source / 'cache_receipt.json'
    if sha(receipt_path) != receipt_sha:
        raise ValueError('Receipt differs from the audited cache; do not silently repackage another run.')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt['status'] != 'CACHE_COMPLETE' or receipt['schema'] != 'upt_hidden_pair_caption_cache_v1':
        raise ValueError('Incomplete or unknown cache schema.')
    if set(receipt['files']) != REQUIRED:
        raise ValueError('Cache payload inventory differs.')
    expected = dict(receipt['files'], **{'cache_receipt.json': receipt_sha})
    for name, expected_sha in expected.items():
        path = source / name
        if not path.is_file() or sha(path) != expected_sha:
            raise ValueError('Missing or changed source file: ' + name)
        print('Source verified:', name, flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    if any(p.name not in set(expected) | {'transport_receipt.json'} | {n + '.partial' for n in expected}
           for p in destination.iterdir()):
        raise ValueError('Destination contains unrelated files; choose a fresh folder.')
    needed = sum((source / name).stat().st_size for name in expected if not (destination / name).is_file())
    if shutil.disk_usage(destination).free < needed + 256 * 1024**2:
        raise RuntimeError('Not enough free disk space for cache copy plus256MiB.')
    for name, expected_sha in expected.items():
        final = destination / name
        if final.exists():
            if not final.is_file() or sha(final) != expected_sha:
                raise ValueError('Existing destination differs; preserving it: ' + name)
            print('Already verified:', name, flush=True)
            continue
        temporary = destination / (name + '.partial')
        with (source / name).open('rb') as reader, temporary.open('wb') as writer:
            shutil.copyfileobj(reader, writer, 8 * 1024**2)
        if sha(temporary) != expected_sha:
            raise ValueError('Copied bytes differ: ' + name)
        temporary.replace(final)
        print('Copied and verified:', name, flush=True)
    report = dict(status='DATASET_FOLDER_READY', source=str(source), payload_files=6,
                  files=expected, bytes=sum((destination / n).stat().st_size for n in expected),
                  model_or_training_executed=False,
                  next='Save notebook Output, then create a private Kaggle Dataset from this output folder. No API token required.')
    (destination / 'transport_receipt.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('DATASET FOLDER READY:', destination, flush=True)
    print('Payload MiB:', round(report['bytes'] / 1024**2, 2), flush=True)
    return report
