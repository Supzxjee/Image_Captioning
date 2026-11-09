"""User-run batched beam search; no model import at module load."""
def beam_search(model,memory,width=5,maximum=20):
    import torch
    if model.training or not 1<=width<=model.head.out_features-2:raise ValueError('Beam requires eval model/valid width')
    b,n,d=memory.shape;vocab=model.head.out_features;device=memory.device
    seq=torch.ones((b,width,1),dtype=torch.long,device=device);score=torch.full((b,width),-torch.inf,device=device);score[:,0]=0
    done=torch.zeros((b,width),dtype=torch.bool,device=device);expanded=memory[:,None].expand(-1,width,-1,-1).reshape(b*width,n,d)
    for _ in range(maximum):
        logits=model.decode(expanded,seq.reshape(b*width,-1))[:,-1].float();logp=logits.log_softmax(-1).view(b,width,vocab)
        logp[:,:,:2]=-torch.inf;logp=logp.masked_fill(done[:,:,None],-torch.inf);logp[:,:,0]=torch.where(done,torch.zeros_like(score),logp[:,:,0])
        score,flat=(score[:,:,None]+logp).flatten(1).topk(width,dim=1)
        parent=flat//vocab;token=flat%vocab
        seq=torch.cat((seq.gather(1,parent[:,:,None].expand(-1,-1,seq.shape[-1])),token[:,:,None]),-1)
        done=done.gather(1,parent)|(token==2)
        if done.all():break
    return seq,score

def check_paths(model,memory,seq,scores):
    import torch
    from torch.nn import functional as F
    b,k,length=seq.shape;vocab=model.head.out_features
    if not torch.isfinite(scores).all() or (scores[:,:-1]<scores[:,1:]).any() or not (seq[:,:,0]==1).all():raise ValueError('Invalid beam paths/scores')
    target=seq[:,:,1:];before_eos=(target==2).cumsum(-1)-(target==2).long();mask=before_eos==0
    if ((target<2)&mask).any() or ((target!=0)&~mask).any():raise ValueError('Invalid BOS/PAD/EOS path')
    mem=memory[:,None].expand(-1,k,-1,-1).reshape(b*k,*memory.shape[1:]);prefix=seq[:,:,:-1].reshape(b*k,-1)
    logits=model.decode(mem,prefix).float();logp=F.log_softmax(logits,-1);logp[:,:,:2]=-torch.inf
    # Exclude post-EOS PAD without multiplying negative infinity by zero.
    per=logp.gather(-1,target.reshape(b*k,-1,1)).squeeze(-1).masked_fill(~mask.reshape(b*k,-1),0).sum(-1).view(b,k)
    torch.testing.assert_close(scores,per,rtol=1e-4,atol=2e-4)

def text_from_ids(ids,words):
    out=[]
    for word in ids[1:]:
        word=int(word)
        if word==2:break
        if word>1:out.append(words[word])
    return ' '.join(out)
