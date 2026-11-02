"""NumPy-only grouped sampler and within-rank image deduplication."""
import numpy as np
from full_cache_data import collate

def grouped_order(data,train_events,seed=42):
    images=np.asarray(data['caption_image'][:train_events])
    if len(images)!=train_events or (np.diff(images)<0).any() or not (data['group'][images]==0).all():raise ValueError('Expected sorted TRAIN caption/image mapping')
    boundaries=np.r_[0,np.flatnonzero(np.diff(images))+1,len(images)]
    order=np.random.default_rng(seed).permutation(len(boundaries)-1)
    return np.concatenate([np.arange(boundaries[i],boundaries[i+1],dtype=np.int64) for i in order])

def rank_events(indices,rank):
    if rank not in (0,1) or not 16<len(indices)<=32:raise ValueError('Expected two ranks and17..32globalcaptionevents')
    return np.asarray(indices[rank*16:(rank+1)*16],np.int64)

def shared_collate(data,indices):
    indices=np.asarray(indices,np.int64);images=data['caption_image'][indices]
    _,first,inverse=np.unique(images,return_index=True,return_inverse=True)
    features,valid,_=collate(data,indices[first])
    seq=np.asarray(data['sequences'][indices],np.int64);seq=seq[:,:int((seq!=0).sum(1).max())]
    return features,valid,seq,inverse.astype(np.int64)

def ddp_loss_multiplier(world,global_tokens):
    if world!=2 or global_tokens<=0:raise ValueError('Expected world2/nonemptyglobaltargets')
    return world/global_tokens
