"""Sealed metadata and protocol guards; no torch or model execution."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

A_SHA='630a4b6d4bcef84515bbd44a52b6474811377bda53caa9722c5806795fa7f901'
BASE_SHA='d2796e30b4e7454798fac5c513be714dca69918b423910233aee95b161e0e81e'
DATA_SHA='2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
LEGACY_SHA='d7910e9bf87e3aad86d806c77ec336b34793e3bef68a3f662fdcb2e9456ca38d'
DETECTION_SHA='498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c'
NOUN_SHA='bde873ba77124333488b263a052419dc80e412c1e089178cdf423d2836e1b146'
CHAIR_SHA='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3'
POLICY=dict(version='grounded-gain-v0.3',seed=42,images=2000,epochs=2,updates=500,
    global_batch=8,lr=1e-4,weight_decay=.01,betas=[.9,.999],eps=1e-8,clip=1.,
    kl=.1,aux=.1,rho=.8,kappa=2.,margin=.05,headroom=.1,rank_scale=196.,
    quadrature=64,verification_quadrature=128,atol=1e-5,rtol=1e-4,
    prepare_seconds=1800,preflight_seconds=600,arm_seconds=2700)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def save(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n',encoding='utf-8')


def order(text):return hashlib.sha256(text.encode()).hexdigest()


def group_error(group, special):
    """Whole-group decisions; never discard a difficult alternative alone."""
    prefix=group['prefix'];alts=group['alternatives']
    if not prefix or prefix[0]!=special['bos'] or any(t in (special['pad'],special['eos']) for t in prefix):return 'invalid_prefix'
    if not alts:return 'empty_group'
    forbidden=set(special.values())
    for a in alts:
        t=a['tokens'];m=np.asarray(a['mask'],dtype=float)
        if not t or any(v in forbidden or not 0<=v<10000 for v in t) or len(prefix)+len(t)>30:return 'invalid_target'
        if m.shape!=(196,) or not np.isfinite(m).all() or (m<0).any() or (m>1).any() or not m.any():return 'invalid_mask'
    if len({a['annotation_id'] for a in alts})!=1:return 'multiple_instances'
    if any(a['mask']!=alts[0]['mask'] for a in alts):return 'multiple_masks'
    seqs=[tuple(a['tokens']) for a in alts]
    for i,a in enumerate(seqs):
        for b in seqs[i+1:]:
            if a==b:return 'duplicate_event'
            if a[:len(b)]==b or b[:len(a)]==a:return 'overlapping_events'
    return None


def build_metadata(source, locked, vocab, baseline):
    source=Path(source)
    if sha(source/'manifest_receipt.json')!=NOUN_SHA:raise ValueError('Wrong sealed noun receipt')
    receipt=json.loads((source/'manifest_receipt.json').read_text())
    for n,h in receipt['payload_sha256'].items():
        if Path(n).name!=n or sha(source/n)!=h:raise ValueError('Noun payload mismatch: '+n)
    spans=json.loads((source/'spans.json').read_text());masks=json.loads((source/'mask_bank.json').read_text())
    train={r['coco_id']:r for r in locked['train']};val=locked['val']
    ids=sorted({s['coco_id'] for s in spans}&train.keys(),key=lambda i:(order(f'grounded-gain-v1:{i}'),i))[:2000]
    if len(ids)!=2000 or set(ids)&{r['coco_id'] for r in val}:raise ValueError('Wrong/disjoint cohort')
    special={k:vocab[f'<{k}>'] for k in ('bos','pad','eos','unk')}
    candidates=defaultdict(dict);selected=set(ids);counts=Counter()
    for s in spans:
        if s['coco_id'] not in selected:continue
        words=s['caption'].lower().split();tokens=[special['bos']]+[vocab.get(w,special['unk']) for w in words]
        pos=s['target_token_positions'];first=min(pos);target=[tokens[i] for i in pos]
        if target!=s['token_ids'] or pos!=list(range(first,first+len(pos))) or s['decoder_logit_positions']!=[p-1 for p in pos]:raise ValueError('Sealed token positions differ')
        row=train[s['coco_id']]
        if s['filename']!=row['filename'] or s['karpathy_imgid']!=row['image_id']:raise ValueError('Image identity differs')
        key=(s['coco_id'],tuple(tokens[:first]));m=masks[s['mask_key']]
        a=dict(tokens=target,annotation_id=m['annotation_id'],mask=m['patch_area_fraction'],category=s['category'])
        # Preserve mask identity while collapsing only genuinely identical events.
        ident=(tuple(target),m['annotation_id'],tuple(a['mask']))
        candidates[key][ident]=a
    byimage=defaultdict(list)
    for (i,prefix),alternatives in candidates.items():
        alts=sorted(alternatives.values(),key=lambda a:(a['tokens'],a['annotation_id']))
        g=dict(prefix=list(prefix),alternatives=alts)
        error=group_error(g,special)
        if error:counts[error]+=1;continue
        mask=alts[0]['mask']
        g['mask']=mask;g['annotation_id']=alts[0]['annotation_id'];g['categories']=sorted({a['category'] for a in alts})
        for a in alts:del a['mask']
        byimage[str(i)].append(g)
    for i,groups in byimage.items():
        groups.sort(key=lambda g:order(f"grounded-gain-group-v1:{i}:"+','.join(map(str,g['prefix']))))
    if {r['image_id'] for r in baseline}!={r['image_id'] for r in val}:raise ValueError('Baseline coverage differs')
    return dict(policy=POLICY,groups=dict(train=[train[i] for i in ids],val=val),word2idx=vocab,
        auxiliary=dict(byimage),baseline=baseline,source_receipt_sha256=NOUN_SHA,
        coverage=dict(selected_images=2000,eligible_groups=sum(map(len,byimage.values())),
            images_with_groups=len(byimage),exclusions=dict(counts),
            selected_groups_two_epochs=sum(min(2,len(g)) for g in byimage.values()),
            scope='Weak head/name joins; not human gold; no neural execution'))


def load_metadata(path):
    import gzip
    return json.loads(gzip.decompress(Path(path).read_bytes()).decode('utf-8'))


def schedule(meta,epoch):
    ids=np.random.default_rng(42+epoch).permutation(2000).reshape(250,8)
    return ids


def checked_cache(root,meta_sha):
    root=Path(root);r=json.loads((root/'receipt.json').read_text())
    if r['status']!='PREPARATION_COMPLETE' or r['metadata_sha256']!=meta_sha or r['policy']!=POLICY:raise ValueError('Wrong/incomplete preparation')
    for n,h in r['payload_sha256'].items():
        if Path(n).name!=n or sha(root/n)!=h:raise ValueError('Missing/changed preparation payload: '+n)
    return r
