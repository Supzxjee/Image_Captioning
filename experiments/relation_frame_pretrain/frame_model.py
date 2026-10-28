"""User-run Torch model. Never imported by local metadata/static checks."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class FrameModel(nn.Module):
    def __init__(self,embedding):
        super().__init__()
        if embedding.shape!=(10000,512):raise ValueError('Wrong frozen A embedding')
        self.embedding=nn.Embedding.from_pretrained(embedding.detach().clone(),freeze=True,padding_idx=0)
        self.entity=nn.Linear(768,256)
        self.query=nn.Linear(512,256)
        self.key=nn.Linear(770,256)
        self.value=nn.Linear(770,256)
        self.classes=nn.Embedding(80,32)
        self.frame=nn.Sequential(nn.Linear(840,512),nn.ReLU(),nn.Linear(512,512),nn.LayerNorm(512))
        self.initial=nn.Linear(512,256)
        self.gru=nn.GRU(512,256,batch_first=True)
        self.output=nn.Linear(256,10000)
        self.truth=nn.Linear(512,16)

    def encode(self,b):
        v=b['visual'];a=b['areas']
        pooled=torch.bmm(a[:,:2]/a[:,:2].sum(-1,keepdim=True),v)
        e=self.entity(pooled);q=self.query(e.flatten(1))
        # Separate ordered role masks; union rectangle supplies the intervening context.
        patch=torch.cat([v,a[:,:2].transpose(1,2)],dim=-1)
        scores=(self.key(patch)*q[:,None,:]).sum(-1)/math.sqrt(256)
        scores=scores+torch.log(a[:,2].clamp_min(1e-30))
        scores=scores.masked_fill(a[:,2]<=0,-torch.inf)
        u=torch.bmm(scores.softmax(-1).unsqueeze(1),self.value(patch)).squeeze(1)
        c=self.classes(b['classes']).flatten(1)
        return self.frame(torch.cat([e.flatten(1),u,b['geometry'],c],dim=-1))

    def lexical_loss(self,r,ids,targets,weights):
        h=torch.tanh(self.initial(r)).unsqueeze(0)
        out,_=self.gru(self.embedding(ids),h)
        logits=self.output(out)
        ce=F.cross_entropy(logits.transpose(1,2),targets,reduction='none')
        return (ce*weights).sum(-1)

    def objective(self,b,arm):
        r=self.encode(b);plus=r[::2];minus=r[1::2]
        pos=self.lexical_loss(plus,b['ids'],b['targets'],b['weights']).mean()
        observed=self.truth(r).gather(1,b['relation'].repeat_interleave(2)[:,None]).squeeze(1)
        dp,dm=observed[::2],observed[1::2]
        truth=.5*(F.softplus(-dp).mean()+F.softplus(dm).mean())
        pair=F.softplus(-(dp-dm)).mean()
        loss=pos if arm=='RelPos' else pos+.5*truth+.5*pair
        return loss,dict(pos=pos.detach(),truth=truth.detach(),pair=pair.detach())

    def ranking_gap(self,b):
        r=self.encode(b);args=[b[k].repeat_interleave(2,dim=0) for k in ('ids','targets','weights')]
        scores=-self.lexical_loss(r,*args)
        return scores[::2]-scores[1::2]
