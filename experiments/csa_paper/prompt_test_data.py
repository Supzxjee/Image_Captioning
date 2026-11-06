"""Frozen TEST input and foreign-text controls; NumPy only."""
import hashlib
import json
from pathlib import Path
import numpy as np
from real_bank_data import read,sha
from prompt_adapter_data import tensor_batch

HERE=Path(__file__).resolve().parent


def source_checkpoint(root,arm):
    spec=read(HERE/'PROMPT_SCST_RESULT.json')['arms'][arm];root=Path(root)
    candidates=[root] if root.is_file() else sorted(root.rglob('checkpoint.bin'))
    matches=[p for p in candidates if sha(p)==spec['checkpoint_sha256']]
    if len(matches)!=1:raise ValueError('Need exact '+arm+' checkpoint.bin from version'+str(spec['save_version']))
    return matches[0]


def cache_root(mount):
    inv=read(HERE/'PROMPT_TEST_PREPARATION_RESULT.json')['transport_inventory'];matches=[p for p in Path(mount).rglob('receipt.json') if sha(p)==inv['cache_files']['receipt.json']]
    if len(matches)!=1:raise ValueError('Need unique exact181 TEST receipt')
    root=matches[0].parent
    for n,h in inv['cache_files'].items():
        if Path(n).name!=n or not (root/n).is_file() or sha(root/n)!=h:raise ValueError('Changed/missing TEST181 payload:'+n)
    return root,inv


def load_inputs(root):
    root=Path(root)
    with np.load(root/'test_regions.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
    rows=[json.loads(line) for line in (root/'prompt_links.jsonl').read_text(encoding='utf-8').splitlines()]
    texts=read(root/'prompt_texts.json');vectors=np.load(root/'prompt_embeddings.npy',allow_pickle=False)
    if len(rows)!=5000 or [r['coco_id'] for r in rows]!=a['ids'].tolist() or any(r['group']!='test' for r in rows):raise ValueError('Wrong TEST order/group')
    if vectors.shape!=(258,512) or vectors.dtype!=np.float32 or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=2e-6,rtol=0):raise ValueError('Wrong181 vectors')
    return a,rows,texts,vectors,read(root/'vocabulary.json')


def donor_plan(rows):
    plan={};records=[]
    for kind in ('objects','relations'):
        slots=[(r['coco_id'],j,o['text']) for r in rows for j,o in enumerate(r[kind])]
        slots.sort(key=lambda x:hashlib.sha256(f'42:{kind}:{x[0]}:{x[1]}'.encode()).hexdigest())
        for i,(coco,j,text) in enumerate(slots):
            k=(i+1)%len(slots)
            while slots[k][0]==coco:
                k=(k+1)%len(slots)
                if k==i:raise ValueError('Need foreign-image text donors for '+kind)
            other,slot,donor=slots[k];plan[kind,coco,j]=donor
            records.append(dict(kind=kind,coco_id=coco,slot=j,donor_coco_id=other,donor_slot=slot,text=donor,unchanged_text=donor==text))
    return plan,records


def prompt_batch(rows,vectors,texts,mode,donors):
    p=tensor_batch(rows,vectors,texts)
    if mode=='normal':return p
    if mode not in ('zero','shuffle'):raise ValueError('Unknown intervention')
    lookup={s:i for i,s in enumerate(texts)}
    for kind,key in (('objects','object_text'),('relations','relation_text')):
        if mode=='zero':p[key].fill(0);continue
        for i,row in enumerate(rows):
            for j,_ in enumerate(row[kind]):p[key][i,j]=vectors[lookup[donors[kind,row['coco_id'],j]]]
    return p


def image_batch(arrays,indices):
    counts=arrays['offsets'][np.asarray(indices)+1]-arrays['offsets'][indices]
    x=np.zeros((len(indices),int(counts.max()),768),np.float32);valid=np.zeros(x.shape[:2],bool)
    for j,i in enumerate(indices):
        lo,hi=arrays['offsets'][i:i+2];x[j,:hi-lo]=arrays['features'][lo:hi];valid[j,:hi-lo]=True
    return x,valid
