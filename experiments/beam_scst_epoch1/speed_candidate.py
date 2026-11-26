"""Same-shape cached input batching and deferred finite-logit synchronization."""
import numpy as np


def batch_rows(data, rows, device):
    import torch
    rows=np.asarray(rows,dtype=np.int64)
    if rows.ndim!=1 or len(rows)==0 or rows.min()<0 or rows.max()>=data.stops[-1]:
        raise ValueError('Invalid cached image indices')
    parts=np.searchsorted(data.stops,rows,side='right')
    patches=[];prompts=[];ids=[];positions=[]
    for part in np.unique(parts):
        loc=np.flatnonzero(parts==part)
        offsets=rows[loc]-(data.stops[part-1] if part else 0)
        patches.append(np.array(data.patches[part][offsets],dtype=np.float32,copy=True))
        ids.extend(map(int,data.ids[part][offsets]))
        prompts.append(data.context[part].images(offsets,device))
        positions.extend(loc.tolist())
    if len(patches)==1:
        return torch.from_numpy(patches[0]).to(device),prompts[0],ids
    order=np.argsort(positions)
    x=torch.from_numpy(np.concatenate(patches,axis=0)[order]).to(device)
    index=torch.from_numpy(order).to(device)
    prompt={key:torch.cat([p[key] for p in prompts]).index_select(0,index) for key in prompts[0]}
    return x,prompt,[ids[i] for i in order]


def train_beams_fast(model, memory, width=5, maximum=20):
    import torch
    if width<2 or maximum<1:raise ValueError('Need at least two beams and one action')
    b,n,d=memory.shape
    vocab,device=model.head.out_features,memory.device
    if width>vocab-2:raise ValueError('Beam width exceeds valid vocabulary')
    seq=torch.ones((b,width,1),dtype=torch.long,device=device)
    scores=torch.full((b,width),-torch.inf,device=device);scores[:,0]=0
    done=torch.zeros((b,width),dtype=torch.bool,device=device)
    history=memory.new_zeros((b,width,0))
    expanded=memory[:,None].expand(-1,width,-1,-1).reshape(b*width,n,d)
    forbidden=torch.arange(vocab,device=device)<2
    finite=torch.ones((),dtype=torch.bool,device=device)
    finished=None
    for _ in range(maximum):
        logits=model.decode(expanded,seq.reshape(b*width,-1))[:,-1].float()
        finite.logical_and_(torch.isfinite(logits).all())
        logp=logits.log_softmax(-1).view(b,width,vocab)
        logp=logp.masked_fill(forbidden,-torch.inf)
        if finished is None:
            finished=torch.full_like(logp,-torch.inf);finished[:,:,0]=0
        logp=torch.where(done[:,:,None],finished,logp)
        scores,flat=(scores[:,:,None]+logp.detach()).flatten(1).topk(width,dim=1)
        parent,token=flat//vocab,flat%vocab
        chosen=logp.gather(1,parent[:,:,None].expand(-1,-1,vocab)).gather(2,token[:,:,None]).squeeze(-1)
        history=torch.cat((history.gather(1,parent[:,:,None].expand(-1,-1,history.shape[-1])),chosen[:,:,None]),-1)
        seq=torch.cat((seq.gather(1,parent[:,:,None].expand(-1,-1,seq.shape[-1])),token[:,:,None]),-1)
        done=done.gather(1,parent)|token.eq(2)
        if done.all():break
    # Check every visited logit, including masked/unselected values, before backward.
    if not finite:raise FloatingPointError('Nonfinite beam logits')
    if history.shape[-1]<maximum:
        padding=maximum-history.shape[-1]
        history=torch.cat((history,history.new_zeros((b,width,padding))),-1)
        seq=torch.cat((seq,seq.new_zeros((b,width,padding))),-1)
    if not torch.isfinite(history).all():raise FloatingPointError('Nonfinite selected beam log probabilities')
    return seq,history
