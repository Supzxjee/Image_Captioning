"""Same visual evidence/decoder in both arms; only the decoder cross loss differs."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize


class KVUpdate(nn.Module):
    def __init__(self,dim=512,rank=8):
        super().__init__();self.a=nn.Parameter(torch.empty(2,rank,dim));self.b=nn.Parameter(torch.zeros(2,dim,rank))
        nn.init.normal_(self.a,std=.02)
    def forward(self,weight):
        update=torch.bmm(self.b,self.a)
        return weight+torch.cat((torch.zeros_like(update[0]),update[0],update[1]),0)


class Evidence(nn.Module):
    def __init__(self,projection):
        super().__init__();self.key=copy.deepcopy(projection);self.value=copy.deepcopy(projection)
        self.key.requires_grad_(True);self.value.requires_grad_(True)
        self.position=nn.Linear(4,512,bias=False);self.geometry=nn.Linear(8,512,bias=False)
        self.query=nn.Parameter(torch.randn(6,512)*.02);self.role=nn.Embedding(3,512);self.norm=nn.LayerNorm(512)
        axis=(torch.arange(14)+.5)/14;y,x=torch.meshgrid(axis,axis,indexing='ij')
        self.register_buffer('xy',torch.stack((x.flatten(),y.flatten()),-1));self.register_buffer('roles',torch.tensor([0,1,2,2,2,2]))
    def forward(self,patches,boxes,geo):
        n=len(patches);xy=self.xy[None].expand(n,-1,-1)
        union=torch.cat((boxes.min(1).values[:,:2],boxes.max(1).values[:,2:]),-1)
        all_boxes=torch.cat((boxes,union[:,None]),1)
        inside=((xy[:,None]>=all_boxes[:,:,None,:2]).all(-1)&(xy[:,None]<=all_boxes[:,:,None,2:]).all(-1))
        allowed=inside[:,self.roles].clone();empty=~allowed.any(-1)
        centers=(all_boxes[:,:,:2]+all_boxes[:,:,2:])/2
        nearest=((xy[:,None]-centers[:,self.roles,None])**2).sum(-1).argmin(-1)
        rows,slots=torch.where(empty);allowed[rows,slots,nearest[rows,slots]]=True
        spatial=torch.cat((xy,inside[:,:2].transpose(1,2).float()),-1)
        keys=self.key(patches)+self.position(spatial)
        q=self.query[None]+self.geometry(geo)[:,None]
        attention=(torch.einsum('nqd,npd->nqp',q,keys)/math.sqrt(512)).masked_fill(~allowed,-torch.inf).softmax(-1)
        tokens=self.norm(torch.bmm(attention,F.relu(self.value(patches))))+self.role(self.roles)[None]
        return tokens,attention


class PhraseModel(nn.Module):
    def __init__(self,decoder,projection):
        super().__init__();self.decoder=decoder.requires_grad_(False);self.evidence=Evidence(projection)
        for layer in self.decoder.transformer_decoder.layers:
            parametrize.register_parametrization(layer.multihead_attn,'in_proj_weight',KVUpdate())
    def memory(self,patches,global_memory,boxes,geometry,off=False):
        z,attention=self.evidence(patches,boxes,geometry)
        pad=torch.zeros((len(z),global_memory.shape[1]+6),dtype=torch.bool,device=z.device)
        if off:z=torch.zeros_like(z);pad[:,-6:]=True
        return torch.cat((global_memory,z),1),pad,attention
    def forward(self,patches,global_memory,boxes,geometry,targets,positive_indices,cross_weight):
        memory,padding,_=self.memory(patches,global_memory,boxes,geometry)
        scores,nll=self.scores(memory,padding,targets)
        scores=scores.reshape(-1,2,2);nll=nll.reshape(-1,2,2)
        positive=scores.gather(-1,positive_indices[...,None]).squeeze(-1)
        negative=scores.gather(-1,(1-positive_indices)[...,None]).squeeze(-1)
        ce=nll.gather(-1,positive_indices[...,None]).mean()
        cross=F.softplus(negative-positive).mean()
        return ce+cross_weight*cross,ce.detach(),cross.detach()
    def scores(self,memory,padding,targets):
        memory=memory.repeat_interleave(2,0);padding=padding.repeat_interleave(2,0)
        logits=self.decoder(targets[:,:-1],memory,tgt_key_padding_mask=targets[:,:-1].eq(0),memory_key_padding_mask=padding).float()
        logp=logits.log_softmax(-1).gather(-1,targets[:,1:,None]).squeeze(-1)
        keep=targets[:,1:].ne(0);words=keep&targets[:,1:].ne(2)
        assert bool(words.any(-1).all()) and bool(torch.isfinite(logp).all())
        scores=(logp*words).sum(-1)/words.sum(-1)
        nll=-(logp*keep).sum(-1)/keep.sum(-1)
        return scores,nll
    @torch.no_grad()
    def greedy(self,memory,padding,max_length=20):
        seq=torch.ones((len(memory),1),dtype=torch.long,device=memory.device);done=torch.zeros(len(seq),dtype=torch.bool,device=seq.device)
        for _ in range(max_length-1):
            logits=self.decoder(seq,memory,tgt_key_padding_mask=seq.eq(0),memory_key_padding_mask=padding)[:,-1]
            nxt=logits.argmax(-1);nxt=torch.where(done,torch.zeros_like(nxt),nxt)
            seq=torch.cat((seq,nxt[:,None]),1);done|=nxt.eq(2)
            if bool(done.all()):break
        return seq
