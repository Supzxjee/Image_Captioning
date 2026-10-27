"""Recover only an absent gzip metadata file, with byte-exact pinned provenance."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def restore_metadata_view(source,pinned_metadata,metadata_sha,workdir=None):
    source=Path(source).resolve();pinned_metadata=Path(pinned_metadata).resolve()
    receipt=json.loads((source/'receipt.json').read_text())
    expected={'inputs.h5','initial_head.bin','gain_metadata.json.gz','object_targets.json'}
    if receipt['status']!='PREPARATION_COMPLETE' or set(receipt['payload_sha256'])!=expected:
        raise ValueError('Wrong preparation receipt; no metadata recovery allowed')
    if receipt['metadata_sha256']!=metadata_sha or receipt['payload_sha256']['gain_metadata.json.gz']!=metadata_sha:
        raise ValueError('Metadata identity differs; do not replace it')
    if file_sha(pinned_metadata)!=metadata_sha:raise ValueError('Pinned source metadata bytes differ')
    metadata=source/'gain_metadata.json.gz'
    if metadata.is_file():
        if file_sha(metadata)!=metadata_sha:raise ValueError('Existing metadata is corrupt; preserve it and inspect')
        return source
    for n in expected-{'gain_metadata.json.gz'}:
        if not (source/n).is_file():raise ValueError('Only missing metadata can be recovered; also missing: '+n)
    parent=Path(workdir) if workdir is not None else Path(tempfile.gettempdir())
    parent.mkdir(parents=True,exist_ok=True)
    view=Path(tempfile.mkdtemp(prefix='grounded-gain-restored-',dir=parent))
    # Read-only input remains untouched. Never copy the multi-GB HDF5 payload.
    for n in sorted(expected|{'receipt.json'}):
        origin=pinned_metadata if n=='gain_metadata.json.gz' else source/n
        if n=='inputs.h5':
            try:os.link(origin,view/n)
            except OSError:os.symlink(origin,view/n)
        else:shutil.copyfile(origin,view/n)
    if file_sha(view/'gain_metadata.json.gz')!=metadata_sha:raise ValueError('Restored metadata differs')
    record=dict(source_dataset=str(source),pinned_metadata=str(pinned_metadata),metadata_sha256=metadata_sha,
        restored_file='gain_metadata.json.gz',hdf5_copied=False,receipt_unchanged=True)
    (view/'metadata_restoration.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print('Restored exact metadata in local view:',view,'; original dataset untouched; HDF5 linked, not copied.',flush=True)
    return view
