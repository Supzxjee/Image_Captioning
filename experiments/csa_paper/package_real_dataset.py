"""Byte-preserving CPU transport only; no API, NumPy archive or model loading."""
import hashlib
import json
from pathlib import Path
import re
import shutil

NAMES = {'vocabulary.json', 'bank_words.json', 'pilot_metadata.json', 'regions.npz',
         'receipt.json', 'summary.json', 'source_receipt.json', 'software_freeze.txt'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def package(source, destination, inventory):
    expected = inventory['files']
    if inventory['kernel_version'] != 'ducanh2403/2-vi-cai-kia-qua-nhieu/126' or set(expected) != NAMES:
        raise ValueError('Transport requires exact version126/eight original files')
    if any(not re.fullmatch('[0-9a-f]{64}', h) for h in expected.values()):
        raise ValueError('Invalid pinned digest')
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Separate non-nested folders required')
    for name, digest in expected.items():
        if (source/name).is_symlink() or not (source/name).is_file() or sha(source/name) != digest:
            raise ValueError('Missing/changed source payload: '+name)
    receipt = json.loads((source/'receipt.json').read_bytes())
    summary = json.loads((source/'summary.json').read_bytes())
    if receipt['status'] != 'CPU_PREPARATION_COMPLETE' or summary['status'] != 'CPU_PREPARATION_COMPLETE':
        raise ValueError('Source is not completed CPU preparation')
    if receipt['schema'] != 'csa_paper_words_roi_v1' or receipt['semantic_embeddings_created'] or receipt['training_ready']:
        raise ValueError('Unexpected source schema or training state')
    if summary['regions']['images'] != 320 or summary['regions']['train_images'] != 256 or summary['regions']['val_images'] != 64:
        raise ValueError('Wrong pilot cohort')
    if receipt['payload_sha256'] != {n: expected[n] for n in ('vocabulary.json', 'bank_words.json', 'pilot_metadata.json', 'regions.npz')}:
        raise ValueError('Receipt manifest differs')
    destination.mkdir(parents=True, exist_ok=True)
    if any(p.is_symlink() or p.name not in NAMES | {'transport_receipt.json'} for p in destination.iterdir()):
        raise ValueError('Preserve unrelated/partial destination; choose a fresh folder')
    needed = sum((source/n).stat().st_size for n in expected if not (destination/n).exists())
    if shutil.disk_usage(destination).free < needed + (128 << 20):
        raise RuntimeError('Insufficient copy space')
    for name, digest in expected.items():
        target = destination/name
        if target.exists():
            if not target.is_file() or sha(target) != digest:
                raise ValueError('Changed destination preserved: '+name)
            continue
        partial = destination/(name+'.partial')
        shutil.copyfile(source/name, partial)
        if sha(partial) != digest:
            raise ValueError('Copy digest differs: '+name)
        partial.replace(target)
    report = dict(status='DATASET_FOLDER_READY', source_kernel_version=inventory['kernel_version'],
        source_review_zip_sha256=inventory['source_review_zip_sha256'], files=expected,
        full_original_file_hashes_verified=True, original_bytes_unchanged=True,
        model_executed=False, semantic_embeddings_created=False, semantic_word_lists_cleared=False,
        lexical_warning='rid18575 vs ride881: suspected lemma problem, CPU token trace required before CLIP text encoding')
    (destination/'transport_receipt.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    return report
