"""Frozen-feature caption reranking with a bounded learned residual.

Patch attention is caption-conditioned spatial evidence, not proof of correct
instance binding. References and CIDEr targets are never inputs to forward().
"""
import torch
from torch import nn
import torch.nn.functional as F

MODES=('scores','global','regional')


class CaptionRanker(nn.Module):
    def __init__(self,mode,hidden=128,scale=.2):
        super().__init__()
        if mode not in MODES:raise ValueError('Unknown ranker mode.')
        self.mode,self.scale=mode,scale
        if mode!='scores':
            self.image=nn.Linear(512,hidden);self.text=nn.Linear(512,hidden)
        if mode=='regional':
            self.patch=nn.Linear(768,hidden);self.word=nn.Linear(512,hidden)
            self.attention=nn.MultiheadAttention(hidden,4,batch_first=True,dropout=0.)
        width=6+(3*hidden if mode!='scores' else 0)+(hidden if mode=='regional' else 0)
        self.head=nn.Sequential(nn.Linear(width,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def represent(self,batch):
        f=batch['scores'].float();pieces=[f]
        if self.mode!='scores':
            image=self.image(batch['image_global'].float())[:,None,:].expand(-1,f.size(1),-1)
            text=self.text(batch['text_global'].float())
            pieces.extend([image,text,image*text])
        if self.mode=='regional':
            b,k=f.shape[:2]
            query=self.word(batch['text_tokens'].float()).flatten(0,1)
            patches=self.patch(batch['patches'].float())
            key=patches[:,None,:,:].expand(-1,k,-1,-1).flatten(0,1)
            attended,_=self.attention(query,key,key,need_weights=False)
            mask=batch['text_mask'].flatten(0,1).unsqueeze(-1).to(attended.dtype)
            pooled=(attended*mask).sum(1)/mask.sum(1).clamp_min(1)
            pieces.append(pooled.reshape(b,k,-1))
        return torch.cat(pieces,-1)

    def forward(self,batch):
        residual=self.scale*torch.tanh(self.head(self.represent(batch)).squeeze(-1))
        base=batch['fixed_scores'].double()
        score=base+residual.double()
        return score.masked_fill(~batch['candidate_mask'].bool(),float('-inf')),residual


def ranking_loss(scores,targets,mask,residual):
    """Weighted pair loss; a CIDEr preference is not a verified semantic negative."""
    delta=targets[:,:,None]-targets[:,None,:]
    valid=mask[:,:,None]&mask[:,None,:]&(delta>1e-3)
    difference=scores[:,:,None]-scores[:,None,:]
    # Index before softplus: padded -inf must never enter differentiable math.
    if valid.any():
        weights=delta[valid].clamp(max=2.)
        loss=(F.softplus(-difference[valid]/.1)*weights).sum()/weights.sum()
    else:loss=residual.sum()*0
    return loss+.01*residual[mask].square().mean()


def select_rank(scores,mask):
    if not mask.bool().any(-1).all():raise ValueError('Image without candidates.')
    return scores.masked_fill(~mask.bool(),float('-inf')).argmax(-1)
