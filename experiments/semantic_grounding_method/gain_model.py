"""User-run additive branch and analytic probability-gain attribution."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from gain_data import POLICY


class GainHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.q=nn.Linear(512,128,bias=False);self.k=nn.Linear(512,128,bias=False)
        self.v=nn.Linear(512,128,bias=False);self.o=nn.Linear(128,512,bias=False)
        for p in (self.q,self.k,self.v):nn.init.xavier_uniform_(p.weight)
        nn.init.zeros_(self.o.weight)

    def parts(self,h,z):
        h=F.layer_norm(h,(512,));z=F.layer_norm(z,(512,))
        a=torch.softmax(self.q(h)@self.k(z).transpose(-1,-2)/math.sqrt(128),-1)
        u=self.o(F.normalize(self.v(z),dim=-1,eps=1e-6))
        return a,u

    def forward(self,h,z,vocab):
        a,u=self.parts(h,z)
        return F.linear(a@u,vocab)


class Captioner(nn.Module):
    def __init__(self,base,head):
        super().__init__();self.base=base.requires_grad_(False).eval();self.head=head

    @torch.no_grad()
    def memory(self,batch):
        memory,padding=self.base.build_memory(batch['visual'],batch['prompt'],batch['mask'])
        z=self.base.encoder.relu(self.base.encoder.vis_projection(batch['visual'][:,1:]))
        return memory,padding,z

    @torch.no_grad()
    def states(self,ids,memory,padding,tgt_padding=None):
        d=self.base.decoder
        h=d.transformer_decoder(tgt=d.pos_encoder(d.embedding(ids)*math.sqrt(d.d_model)),memory=memory,
            tgt_mask=d.causal_mask(ids.size(1),ids.device),tgt_key_padding_mask=tgt_padding,memory_key_padding_mask=padding)
        return h.detach(),d.fc_out(h).detach()

    def logits(self,ids,memory,padding,z):
        h,base=self.states(ids,memory,padding)
        return base+self.head(h,z,self.base.decoder.fc_out.weight),base

    @torch.no_grad()
    def generate(self,memory,padding,z,tok,sample=False,baseline=False):
        ids=torch.full((len(memory),1),tok.bos_idx,device=memory.device,dtype=torch.long)
        live=torch.ones(len(memory),device=memory.device,dtype=torch.bool);masks=[]
        for _ in range(29):
            h,base=self.states(ids,memory,padding)
            logits=base[:,-1] if baseline else base[:,-1]+self.head(h[:,-1:],z,self.base.decoder.fc_out.weight)[:,0]
            if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite rollout')
            token=torch.multinomial(logits.softmax(-1),1).flatten() if sample else logits.argmax(-1)
            token=torch.where(live,token,tok.pad_idx);masks.append(live.clone())
            ids=torch.cat([ids,token[:,None]],1);live=live&token.ne(tok.eos_idx)
            if not live.any():break
        return ids,torch.stack(masks,1)

    def auxiliary(self,group,memory,padding,z,nodes=64):
        device=memory.device;traces=[]
        for alt in group['alternatives']:
            target=torch.tensor(alt['tokens'],device=device,dtype=torch.long)
            ids=torch.tensor(group['prefix']+alt['tokens'][:-1],device=device)[None]
            h,base=self.states(ids,memory,padding)
            first=len(group['prefix'])-1;h=h[:,first:];base=base[0,first:]
            a,u=self.head.parts(h,z)
            traces.append((target,base,a[0],u[0],F.linear(a@u,self.base.decoder.fc_out.weight)[0]))
        return attributed(traces,self.base.decoder.fc_out.weight,torch.tensor(group['mask'],device=device),nodes)


def attributed(traces,vocab,mask,nodes=64):
    """Integrate all alternative paths jointly; never allocate patch x vocabulary."""
    n,w=np.polynomial.legendre.leggauss(nodes)
    weights=torch.as_tensor(w/2,device=mask.device,dtype=torch.float32)
    positions=torch.as_tensor((n+1)/2,device=mask.device,dtype=torch.float32)

    def at(alpha):
        alpha=torch.as_tensor(alpha,device=mask.device,dtype=torch.float32).reshape(-1)
        logevents=[];effects=[]
        for target,base,a,u,correction in traces:
            logits=base[None]+alpha[:,None,None]*correction[None];logp=logits.log_softmax(-1);p=logp.exp()
            logevents.append(logp.gather(2,target[None,:,None].expand(len(alpha),-1,1)).sum((1,2)))
            direction=vocab[target]-p@vocab
            effects.append((a[None]*(direction@u.T)).sum(1))
        logs=torch.stack(logevents,1);mass=logs.logsumexp(1)
        if not bool(torch.isfinite(mass).all()) or bool((mass.detach()>1e-6).any()):raise ValueError('Invalid disjoint event mass')
        contributions=logs.softmax(1)[:,:,None]*torch.stack(effects,1)
        return mass,contributions

    f0,_=at(0.);f1,end=at(1.);f0=f0[0];f1=f1[0];end=end[0];delta=f1-f0
    _,effect=at(positions);components=(weights[:,None,None]*effect).sum(0)
    grouped=components.sum(0);cg=(grouped*mask).sum();cu=delta-cg
    margin=torch.minimum(delta.new_tensor(.05),.1*(-f0.detach()).clamp_min(0))
    gain=F.relu(margin-delta);cancel=F.relu(cg.abs()+cu.abs()-2*F.relu(delta))
    direction=F.relu(.8*F.relu(delta)-cg)
    # Sum signed alternatives BEFORE the nonlinear ranking penalty.
    endpoint=end.sum(0);d=F.relu(mask[:,None]-mask[None,:]);den=d.sum()
    rank=(d*F.relu(196*(endpoint[None,:]-endpoint[:,None]))).sum()/den if float(den)>0 else delta*0
    residual=(components.double().sum()-delta.double()).abs()
    return dict(gain=gain,cancel=cancel,direction=direction,rank=rank,delta=delta,cg=cg,cu=cu,
        components=components,endpoint=endpoint,residual=residual,rank_defined=(den>0).to(delta.dtype))


def check_completeness(result):
    tolerance=POLICY['atol']+POLICY['rtol']*result['delta'].detach().abs()
    if not bool(result['residual']<=tolerance):raise AssertionError('Quadrature completeness failed; preserve logs, do not relax tolerance')
