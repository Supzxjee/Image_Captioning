"""USER-RUN neural helpers: current-policy sampling and matching rescoring."""
import torch
from torch import nn
from torch.nn import functional as F


def action_logits(logits):
    logits=logits.float().clone()
    logits[...,0:2]=-torch.inf
    return logits


@torch.no_grad()
def rollout(base,vo,vr,valid,sample,maximum=20):
    if base.training:raise ValueError('SCST rollout requires dropout disabled')
    seq=torch.ones((len(vo),1),device=vo.device,dtype=torch.long)
    active=torch.ones(len(vo),device=vo.device,dtype=torch.bool);masks=[];logps=[]
    for _ in range(maximum):
        logits=base.decode(seq,vo,vr,valid)[:,-1].float()
        if not torch.isfinite(logits).all():raise ValueError('Nonfinite rollout logits')
        logp=F.log_softmax(action_logits(logits),-1)
        token=torch.multinomial(logp.exp(),1).squeeze(1) if sample else logp.argmax(-1)
        logps.append(logp.gather(1,token[:,None]).squeeze(1).masked_fill(~active,0))
        masks.append(active.clone());token=token.masked_fill(~active,0)
        seq=torch.cat((seq,token[:,None]),1);active=active & token.ne(2)
        if not active.any():break
    return seq,torch.stack(masks,1),torch.stack(logps,1)


def selected_logps(logits,actions,mask):
    if logits.shape[:2]!=actions.shape or actions.shape!=mask.shape:raise ValueError('Action/mask shapes differ')
    return F.log_softmax(action_logits(logits),-1).gather(-1,actions[...,None]).squeeze(-1).masked_fill(~mask,0)


def policy_loss(logits,actions,mask,advantage):
    return -(selected_logps(logits,actions,mask).sum(1)*advantage.detach()).mean()


def captions(seq,vocab):
    rows=[]
    for ids in seq.detach().cpu().tolist():
        content=[]
        for token in ids[1:]:
            if token==2:break
            if token>1:content.append(vocab[token])
        rows.append(' '.join(content))
    return rows


class PolicyReader(nn.Module):
    def __init__(self,base):super().__init__();self.base=base
    def forward(self,x,valid,sample_prefix,ce_prefix):
        vo,vr=self.base.encode(x,valid)
        return self.base.decode(sample_prefix,vo,vr,valid),self.base.decode(ce_prefix,vo,vr,valid)
