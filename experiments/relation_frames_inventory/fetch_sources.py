"""Fetch only the three public metadata payloads pinned in source_receipt.json."""
import hashlib
import json
from pathlib import Path
import urllib.request

root=Path(__file__).resolve().parent/'sources'
receipt=json.loads((root/'source_receipt.json').read_text(encoding='utf-8'))
for name, entry in receipt['files'].items():
    target=root/(name+'.jsonl')
    raw=target.read_bytes() if target.is_file() else urllib.request.urlopen(entry['url'],timeout=60).read()
    if hashlib.sha256(raw).hexdigest()!=entry['sha256'] or len(raw)!=entry['bytes']:
        raise ValueError('Pinned source differs: '+name)
    if not target.is_file():
        target.write_bytes(raw)
    print(name, len(raw), 'verified')
