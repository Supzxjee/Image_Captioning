"""Data-only contracts for the user-run CSA beam diagnostic."""
import gzip
import json
import math
from pathlib import Path
import numpy as np
from real_bank_data import read, save, sha, validate_arrays

CHECKPOINT_SHA='58ce5dc0e30ac26add9bac8e24a2e9e03b91532bfdeece588997395bf9cfc16b'
MODEL_SHA='139edbb1b73caf3dadaf5c960b25542e282f88baf2ce9bdf141de922ceacbc09'


def baseline():
    return json.loads(gzip.decompress(Path(__file__).with_name('CANDIDATE_BASELINE161.json.gz').read_bytes()))


def content(ids):
    if not ids or ids[0]!=1:raise ValueError('Missing BOS')
    values=ids[1:]
    if 2 in values:values=values[:values.index(2)]
    if any(type(x)!=int or not 3<=x<10000 for x in values):raise ValueError('Invalid beam token')
    return values


def check_record(row, expected, vocab):
    candidates=row['candidates']
    if row['coco_id']!=expected['coco_id'] or len(candidates)!=5:raise ValueError('Candidate ID/width differs')
    scores=[x['logprob'] for x in candidates]
    if any(not math.isfinite(s) for s in scores) or scores!=sorted(scores,reverse=True):raise ValueError('Invalid beam scores')
    for c in candidates:
        ids=c['ids'];tokens=content(ids)
        if not 2<=len(ids)<=21 or (2 not in ids and len(ids)!=21):raise ValueError('Beam length differs')
        if 2 in ids and any(ids[ids.index(2)+1:]):raise ValueError('Non-PAD after EOS')
        if c['caption']!=' '.join(vocab[t] for t in tokens):raise ValueError('Caption/token mismatch')
    first=candidates[0]
    if first['caption']!=expected['caption'] or content(first['ids'])!=content(expected['ids']):
        raise ValueError('Rank0 differs from saved161: '+str(row['coco_id']))
    error=abs(first['logprob']-expected['score'])
    if error>2e-5:raise ValueError('Rank0 logprob differs: '+str(row['coco_id']))
    return error


def prepare(cfg):
    out=Path(cfg['output']);root=Path(cfg['full_root']);here=Path(__file__).parent
    package=read(here/'SAVE_FULL_ROI_V134_PACKAGE.json')['files']
    roots=[root] if (root/'receipt.json').is_file() else [p.parent for p in root.rglob('receipt.json') if sha(p)==package['receipt.json']]
    if len(roots)!=1:raise ValueError('Expected unique fullROI134 root')
    root=roots[0]
    for n in ('receipt.json','manifest.json','vocabulary.json'):
        if sha(root/n)!=package[n]:raise ValueError('Changed ROI input: '+n)
    manifest=read(root/'manifest.json');vocab=read(root/'vocabulary.json');saved=baseline()
    metadata=[];parts=[];offsets=[0];features=[]
    entries=[e for e in manifest['shards'] if e['group']=='val']
    if len(entries)!=5:raise ValueError('Expected five VALID shards')
    for e in entries:
        for n,h in e['files'].items():
            if package.get(n)!=h or sha(root/n)!=h:raise ValueError('Changed VALID shard: '+n)
        rows=read(root/next(n for n in e['files'] if n.endswith('.json')))['images']
        with np.load(root/next(n for n in e['files'] if n.endswith('.npz')),allow_pickle=False) as f:a={k:f[k] for k in f.files}
        validate_arrays(a,rows,vocab)
        metadata.extend(rows);parts.extend(a['ids'].tolist());features.append(a['features'])
        offsets.extend((a['offsets'][1:]+offsets[-1]).tolist())
        print('VALID-only stage',len(metadata),'/5000',flush=True)
    if len(metadata)!=5000 or len(set(parts))!=5000 or parts!=saved['order'] or vocab!=saved['vocabulary']:
        raise ValueError('VALID order/vocabulary differs from161')
    refs=[dict(coco_id=r['coco_id'],references=r['captions']) for r in metadata]
    if refs!=saved['references']:raise ValueError('Changed full VALID reference lists')
    stage=Path(cfg['stage']);stage.mkdir()
    arrays=dict(features=np.concatenate(features),offsets=np.array(offsets,np.int64),ids=np.array(parts,np.int64))
    for k,a in arrays.items():np.save(stage/(k+'.npy'),a)
    save(out/'validation_references.json',refs);save(out/'vocabulary.json',vocab)
    save(out/'prepared.json',dict(status='VALID_CANDIDATES_PREPARED',images=5000,test_used=False,
        checkpoint_sha256=CHECKPOINT_SHA,files_sha256={k+'.npy':sha(stage/(k+'.npy')) for k in arrays}))


def image_batch(data,indices):
    sizes=np.diff(data['offsets'])[indices]
    x=np.zeros((len(indices),int(sizes.max()),768),np.float32);valid=np.zeros(x.shape[:2],bool)
    for j,i in enumerate(indices):
        lo,hi=data['offsets'][i:i+2];x[j,:hi-lo]=data['features'][lo:hi];valid[j,:hi-lo]=True
    return x,valid


def choose_oracle(rows, per_rank):
    values=np.asarray(per_rank,dtype=float)
    if values.shape!=(5,len(rows)) or not np.isfinite(values).all():raise ValueError('Wrong candidate CIDEr matrix')
    chosen=values.argmax(axis=0)
    selected=[dict(coco_id=r['coco_id'],caption=r['candidates'][int(k)]['caption']) for r,k in zip(rows,chosen)]
    return selected,chosen.tolist(),float(values.max(axis=0).mean())
