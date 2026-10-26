"""Pinned inputs and schedules; no ML imports."""
import hashlib
import json
from pathlib import Path
import numpy as np

RECEIPT_SHA='867ecb73d880e791c35065863504cee9fd44a3bad63bac326a3c87e5106be51e'
PROTOCOL_SHA='9b7ea3a035a36fa0ab99477ed3291f026ac43d190dcba7567b66f22d60a1bf4a'
MANIFEST_SHA='f33e9682fa53301efaba458c839d3ce627ebb65858f10f083bd5e4aaed17e6d3'
CHAIR_SHA='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3'
POLICY=dict(version='stale-prompt-CT-v1',updates=1250,global_batch=8,aux_every=5,aux_images=2000,
    seed=42,lr=5e-6,weight_decay=.01,betas=[.9,.999],eps=1e-8,clip_grad_norm=1.,
    dropout='eval with gradients',max_arm_seconds=7200,staging_max_seconds=7200,
    timer_scope='Separate2h staging and2h training/endgreedy; setup,hash validation,initial A prediction and CPU scoring outside these timers; total notebook may exceed4h. Not a total2h cap or ETA.',
    checkpoint='end only',selection='No interim dev/selection, no active-case filtering',
    objective='C:mainCE+.1KL(A||student)+.1neutralCE+.1visual; T:C+.1stale',
    gate='T-A CIDEr>=.01, pairedCI95lower(T-A)>0 and lower(T-C)>0; CHAIRs/i not worse than A; Recall>=A-.005',
    novelty='Unestablished; caption efficacy and semantic alignment are separate claims',
    independent_Argon_review='Not received at packaging; user authorized bounded C/T')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def locate_cache(root):
    root=Path(root)
    candidates=list(root.rglob('receipt.json')) if root.is_dir() else []
    matches=[p.parent for p in candidates if sha(p)==RECEIPT_SHA]
    if len(matches)!=1:raise ValueError(f'Expected exactly one audited preparation receipt; found {len(matches)} under {root}. Attach full cache, not review ZIP; narrow root if duplicate copies.')
    return matches[0]


def checked_payloads(root):
    root=Path(root)
    if sha(root/'receipt.json')!=RECEIPT_SHA:raise ValueError('Wrong receipt.')
    receipt=json.loads((root/'receipt.json').read_text(encoding='utf-8'))
    if receipt['status']!='PREPARATION_COMPLETE' or receipt['images']!=2000 or receipt['full_pilot_run']:raise ValueError('Incomplete/changed preparation.')
    for name,expected in receipt['payload_sha256'].items():
        if Path(name).name!=name or not (root/name).is_file() or sha(root/name)!=expected:raise ValueError('Missing/changed cache payload: '+name)
        print('Verified cache:',name,flush=True)
    if sha(root/'protocol.json')!=PROTOCOL_SHA or sha(root/'manifest.json')!=MANIFEST_SHA:raise ValueError('Protocol/manifest differs from audited run.')
    return receipt


def schedule():
    main=np.random.default_rng(42).permutation(10000).reshape(1250,8)
    # The preparation fixed rank split is kept. All1000 rows/rank are consumed once.
    auxiliary=np.random.default_rng(42).permutation(1000).reshape(250,4)
    return main,auxiliary


def validate_auxiliary(f,rows):
    if not f.attrs.get('complete') or f['coco_ids'][:].tolist()!=[r['coco_id'] for r in rows]:raise ValueError('Auxiliary completion/identity/order differs.')
    shapes=dict(visual_original=(197,768),visual_occluded=(197,768),prompt_original=(20,512),prompt_neutral=(20,512),mask_original=(20,),mask_neutral=(20,))
    for name,shape in shapes.items():
        a=f[name]
        if a.shape!=(len(rows),*shape):raise ValueError('Auxiliary shape differs: '+name)
        if name.startswith('mask'):
            if a.dtype.kind not in 'bu':raise ValueError('Auxiliary mask dtype differs.')
        elif a.dtype!=np.dtype('float32'):raise ValueError('Auxiliary feature dtype differs.')
        for start in range(0,len(rows),32):
            values=a[start:start+32]
            if not np.isfinite(values).all() or (name.startswith('mask') and not np.isin(values,[0,1]).all()):raise ValueError('Invalid auxiliary values: '+name)
