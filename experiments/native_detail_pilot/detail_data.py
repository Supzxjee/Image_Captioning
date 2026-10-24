"""Pure locked H/L schedule and consuming-cache verification."""
import hashlib
import json
from pathlib import Path

CACHE_RECEIPT_SHA='0f2e0f15d82b2c2c3ce09b2132cc204723dbd094485700a2d0ea4b74b1d43f5e'
A_SHA='630a4b6d4bcef84515bbd44a52b6474811377bda53caa9722c5806795fa7f901'
BASE_SHA='d2796e30b4e7454798fac5c513be714dca69918b423910233aee95b161e0e81e'
DATA_SHA='2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
LEGACY_SHA='d7910e9bf87e3aad86d806c77ec336b34793e3bef68a3f662fdcb2e9456ca38d'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');tmp.replace(p)


def schedule(rows):
    if len(rows)!=10000 or len({r['coco_id'] for r in rows})!=10000:raise ValueError('Expected locked train10k.')
    order=sorted(range(10000),key=lambda i:hashlib.sha256(('native-detail-HL-schedule-v1:'+str(rows[i]['coco_id'])).encode()).hexdigest())
    first=set(order[:5000])
    return [[int((i in first) != bool(epoch%2)) for i in range(10000)] for epoch in range(4)] #1=H,0=L


def verify_cache(root,inventory):
    root=Path(root)
    if digest(root/'cache_receipt.json')!=CACHE_RECEIPT_SHA:raise ValueError('Cache is not the audited completed15k run.')
    for name,record in inventory['files'].items():
        path=root/name
        if not path.is_file() or path.stat().st_size!=record['bytes'] or digest(path)!=record['sha256']:raise ValueError('Missing/corrupt cached payload: '+name)
        print('HL payload verified:',name,flush=True)
    manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'));groups={s:[r for part in manifest for r in part['rows'] if r['group']==s] for s in ('train','val')}
    if len(groups['val'])!=5000 or {r['coco_id'] for r in groups['train']}&{r['coco_id'] for r in groups['val']}:raise ValueError('Cache split mismatch.')
    schedule(groups['train'])
    return manifest,groups
