"""USER-RUN neural module; preserves the existing two-memory CSA decoder."""
import torch
from torch import nn


class RegionPromptAdapter(nn.Module):
    def __init__(self, dim=512, heads=8, relation=False):
        super().__init__()
        self.relation=relation;self.heads=heads
        self.project=nn.Sequential(nn.Linear(dim*(3 if relation else 2)+(8 if relation else 4),dim),nn.GELU(),nn.Linear(dim,dim))
        self.attention=nn.MultiheadAttention(dim,heads,dropout=0,batch_first=True)
        self.output=nn.Linear(dim,dim,bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, memory, object_memory, region_valid, p, text_mode='normal'):
        if text_mode not in ('normal','zero','shuffle'):raise ValueError('Unknown prompt intervention')
        valid=p['relation_valid' if self.relation else 'object_valid']
        text=p['relation_text' if self.relation else 'object_text']
        if valid.dtype!=torch.bool or valid.shape!=text.shape[:2] or text.shape[-1]!=memory.shape[-1]:
            raise ValueError('Prompt mask/text shape differs')
        if not torch.isfinite(text).all():raise ValueError('Nonfinite text embedding')
        if text_mode=='zero':text=torch.zeros_like(text)
        elif text_mode=='shuffle':text=text.roll(1,0)
        b,n,d=memory.shape;k=valid.shape[1]
        if k<1 or valid.shape[0]!=b:raise ValueError('Expected fixed nonempty padded prompt slots')
        rows=torch.arange(b,device=memory.device)[:,None]
        indices=p['relation_pair' if self.relation else 'object_roi']
        if indices.dtype!=torch.long:raise ValueError('ROI indices must be int64')
        expected=valid.shape+(2,) if self.relation else valid.shape
        if indices.shape!=expected or ((indices<0)|(indices>=n)).any():raise ValueError('Prompt ROI out of range')
        if self.relation:
            s,o=indices.unbind(-1)
            if ((s==o)&valid).any():raise ValueError('Collapsed relation endpoints')
            if (valid & ~(region_valid[rows,s]&region_valid[rows,o])).any():raise ValueError('Pair refers to padded ROI')
            spatial=p['relation_geometry']
            parts=[text,object_memory[rows,s],object_memory[rows,o],spatial]
            linked=(torch.arange(n,device=memory.device)[None,:,None]==s[:,None,:]) | (torch.arange(n,device=memory.device)[None,:,None]==o[:,None,:])
        else:
            if (valid & ~region_valid[rows,indices]).any():raise ValueError('Object refers to padded ROI')
            spatial=p['object_box'];parts=[text,object_memory[rows,indices],spatial]
            linked=torch.arange(n,device=memory.device)[None,:,None]==indices[:,None,:]
        if spatial.shape!=valid.shape+((8,) if self.relation else (4,)) or not torch.isfinite(spatial).all():
            raise ValueError('Invalid spatial tensor')
        allowed=linked & valid[:,None,:] & region_valid[:,:,None]
        active=allowed.any(-1)
        # Empty rows need one finite softmax position, then an exactly zero residual.
        safe=allowed.clone();safe[:,:,0]|=~active
        slots=self.project(torch.cat(parts,-1)).masked_fill(~valid[...,None],0)
        mask=(~safe)[:,None].expand(-1,self.heads,-1,-1).reshape(b*self.heads,n,k)
        delta=self.attention(memory,slots,slots,attn_mask=mask,need_weights=False)[0]
        delta=delta.masked_fill(~active[...,None],0)
        return self.output(delta).masked_fill(~region_valid[...,None],0)


class PromptCSA(nn.Module):
    def __init__(self, base):
        super().__init__();self.base=base
        self.object_prompt=RegionPromptAdapter(base.c.dim,base.c.heads)
        self.relation_prompt=RegionPromptAdapter(base.c.dim,base.c.heads,relation=True)

    def encode(self, regions, valid, prompts=None, text_mode='normal', enabled=True):
        vo,vr=self.base.encode(regions,valid)
        if prompts is None or not enabled:return vo,vr
        do=self.object_prompt(vo,vo,valid,prompts,text_mode)
        dr=self.relation_prompt(vr,vo,valid,prompts,text_mode)
        return vo+do,vr+dr

    def decode(self, ids, vo, vr, valid):return self.base.decode(ids,vo,vr,valid)

    def forward(self, regions, valid, ids, prompts=None):
        return self.decode(ids,*self.encode(regions,valid,prompts),valid)
