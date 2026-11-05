"""Preserve exact173 mapping payloads after download; no API or neural execution."""
from pathlib import Path
import shutil
from real_bank_data import read,save,sha
from full_prompt_cache_data import MAPPING_FILES,checked_payloads


def preserve(download,destination,expected_source):
    download=Path(download);destination=Path(destination)
    if destination.exists():raise FileExistsError('Preserve existing recovered mapping')
    sources=[p for p in download.rglob('source_receipt.json') if read(p)==expected_source]
    if len(sources)!=1:raise ValueError('Expected one source receipt matching original173 source e294c5c/27pins')
    candidates=[p for p in download.rglob('receipt.json') if read(p).get('status')=='PROMPT_ROI_MAPPING_AUDIT_COMPLETE']
    if len(candidates)!=1:raise ValueError('Expected one completed mapping173 receipt')
    root=candidates[0].parent;r=read(candidates[0])
    if set(r['payload_sha256'])!=set(MAPPING_FILES):raise ValueError('Wrong mapping payload contract')
    checked_payloads(root,r)
    destination.mkdir(parents=True)
    for n in MAPPING_FILES+('receipt.json',):
        shutil.copy2(root/n,destination/n)
        if sha(destination/n)!=sha(root/n):raise ValueError('Recovered copy changed:'+n)
    save(destination.parent/'mapping_source_receipt.json',expected_source)
    save(destination.parent/'recovery_receipt.json',dict(source='ducanh2403/2-vi-cai-kia-qua-nhieu/173',
        mapping_receipt_sha256=sha(destination/'receipt.json'),payload_sha256=r['payload_sha256'],mapping_rerun=False,original_bytes=True))
    return destination
