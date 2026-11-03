"""Explicit Output139→15epoch identity migration; no model imports."""
import hashlib
import json
from pathlib import Path
from real_bank_data import HERE,read,sha,save
from full_xe_data import validate_state
from run_best9_decode import saved_text


def resolve_resume_root(root,destination,package=None,receipt_root=None):
    root=Path(root);destination=Path(destination)
    valid=[]
    for p in sorted(root.rglob('checkpoint_receipt.json')):
        if not p.is_file():continue
        try:
            r=read(p);b=read(p.parent/'best_receipt.json')
            if sha(p.parent/'checkpoint.bin')==r['sha256'] and sha(p.parent/'best.bin')==b['sha256']:
                valid.append((p.parent,r['sha256'],b['sha256']))
        except (OSError,ValueError,KeyError,TypeError):continue
    if valid:
        if len({(r[1],r[2]) for r in valid})!=1:raise ValueError('Multiple different verified checkpoints; narrow RESUME_ROOT')
        return valid[0][0]
    package=package or read(HERE/'SAVE_XE139_PACKAGE.json')['files']
    receipt_root=Path(receipt_root) if receipt_root is not None else HERE/'full_xe10_review_text'
    found={}
    for name in ('checkpoint.bin','best.bin'):
        files=[p for p in sorted(root.rglob(name)) if p.is_file()]
        digests={p:sha(p) for p in files}
        if name=='checkpoint.bin' and any(h!=package[name] for h in digests.values()):
            raise ValueError('Unverified non139 latest checkpoint present; preserve progress and narrow RESUME_ROOT')
        candidates=[p for p,h in digests.items() if h==package[name]]
        if not candidates:raise ValueError('Missing exact Output139 '+name+'; cannot recover metadata from another checkpoint')
        found[name]=candidates[0]
    receipts={n:read(receipt_root/n) for n in ('checkpoint_receipt.json','best_receipt.json')}
    if receipts['checkpoint_receipt.json']['sha256']!=package['checkpoint.bin'] or receipts['best_receipt.json']['sha256']!=package['best.bin']:
        raise ValueError('Pinned Output139 receipt digest differs')
    destination.mkdir(parents=True,exist_ok=True)
    for name,p in found.items():
        link=destination/name
        if not link.exists():link.symlink_to(p.resolve())
        if sha(link)!=package[name]:raise ValueError('Recovery link bytes differ')
    for n,r in receipts.items():save(destination/n,r)
    save(destination/'receipt_recovery.json',dict(kind='exact139-receipts-for-verified-binaries',original_root=str(root),
        binaries={n:dict(path=str(p),sha256=package[n]) for n,p in found.items()},weights_deserialized=False))
    return destination


def check_resume_identity(old,current,digest,state):
    validate_state(state,total_epochs=current['epochs'])
    if old==current:return 'identical-current-run'
    raw,_=saved_text();pinned=json.loads(raw['input_identity.json'])
    package=read(HERE/'SAVE_XE139_PACKAGE.json')['files']
    if hashlib.sha256(raw['input_identity.json']).hexdigest()!=package['input_identity.json']:raise ValueError('Changed legacy identity receipt')
    if current['epochs']!=15 or digest!=package['checkpoint.bin'] or old!=pinned:
        raise ValueError('Resume identity differs; only exact latest139 can migrate to15 epochs')
    if (state['epoch'],state['batch_cursor'],state['phase'],state['updates'])!=(11,0,'train',177110):raise ValueError('Not completed10epoch checkpoint')
    excluded={'source','schema','epochs'}
    if {k:v for k,v in old.items() if k not in excluded}!={k:v for k,v in current.items() if k not in excluded}:
        raise ValueError('Resume operational inputs/software differ')
    allowed={'run_full_xe.py','full_xe_data.py'}
    for n,h in old['source'].items():
        if n not in allowed and current['source'].get(n)!=h:raise ValueError('Unapproved inherited source change: '+n)
    return 'exact139-latest10-to15'
