"""Pure NumPy pilot batching and cached-bank contracts, no neural imports."""
from pathlib import Path
import numpy as np
from real_bank_data import HERE,read,sha,batch

def events(images,group):
    return [(i,j) for i,m in enumerate(images) if m['group']==group for j in range(len(m['sequences']))]

def epoch_batches(items,epoch,size=32):
    order=np.random.default_rng(42+epoch).permutation(len(items))
    return [[items[int(i)] for i in order[start:start+size]] for start in range(0,len(order),size)]

def learning_rate(step):
    if not 1<=step<=600:raise ValueError('Pilot step outside1..600')
    return 8e-6+(1e-4-8e-6)*min((step-1)/119,1)

def event_batch(arrays,images,items):
    x,valid,_=batch(arrays,images,[i for i,j in items])
    seq=np.array([images[i]['sequences'][j] for i,j in items],np.int64)
    seq=seq[:,:int((seq!=0).sum(1).max())]
    return x,valid,seq

def target_tokens(images,items):
    return sum(sum(t!=0 for t in images[i]['sequences'][j][1:]) for i,j in items)

def load_banks(root,bank,inputs):
    root=Path(root)
    if not root.is_dir():raise FileNotFoundError('Add Input csa-semantic-banks-v130 or correct BANK_ROOT; directory not mounted: '+str(root))
    if not (root/'cache_receipt.json').is_file():
        found=list(root.rglob('cache_receipt.json'))
        if len(found)!=1:raise ValueError('Expected one semantic cache root')
        root=found[0].parent
    package=read(HERE/'SAVE_BANK_V130_PACKAGE.json')
    for name,h in package['files'].items():
        if not (root/name).is_file() or sha(root/name)!=h:raise ValueError('Missing/changed Output130 payload: '+name)
    receipt=read(root/'cache_receipt.json')
    if receipt['inputs']!=inputs or receipt['status']!='SEMANTIC_BANK_CACHE_COMPLETE':raise ValueError('Bank source identity differs')
    vectors=[]
    with np.load(root/'semantic_banks.npz',allow_pickle=False) as f:
        if set(f.files)!={'noun_vectors','verb_vectors','noun_words','verb_words'}:raise ValueError('Unexpected bank members')
        for kind in ('noun','verb'):
            words=[x['word'] for x in bank[kind]];v=f[kind+'_vectors']
            if f[kind+'_words'].tolist()!=words or v.shape!=(len(words),512) or v.dtype!=np.float32 or not np.isfinite(v).all():
                raise ValueError('Bank shape/order/precision differs')
            if not np.allclose(np.linalg.norm(v,axis=1),1,atol=2e-6):raise ValueError('Bank not normalized')
            vectors.append(v.copy())
    return vectors,package
