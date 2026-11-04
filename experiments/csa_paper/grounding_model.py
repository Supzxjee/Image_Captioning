"""USER-RUN ONLY: collect actual final object-attention probabilities for loss."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class GroundingAttention(nn.MultiheadAttention):
    collect = False
    log_attention = None

    def forward(self,query,key,value,key_padding_mask=None,need_weights=False,**kwargs):
        if not self.collect:
            self.log_attention=None
            return super().forward(query,key,value,key_padding_mask=key_padding_mask,need_weights=need_weights,**kwargs)
        if not self.batch_first or not self._qkv_same_embed_dim or self.bias_k is not None or self.bias_v is not None or self.add_zero_attn:
            raise ValueError('Unsupported object MHA contract')
        if kwargs.get('attn_mask') is not None or kwargs.get('is_causal',False):raise ValueError('Object attention is noncausal')
        b,t,d=query.shape;n=key.shape[1];h=self.num_heads;head=d//h
        wq,wk,wv=self.in_proj_weight.chunk(3,dim=0)
        bq,bk,bv=self.in_proj_bias.chunk(3) if self.in_proj_bias is not None else (None,None,None)
        q=F.linear(query,wq,bq).reshape(b,t,h,head).transpose(1,2)
        k=F.linear(key,wk,bk).reshape(b,n,h,head).transpose(1,2)
        v=F.linear(value,wv,bv).reshape(b,n,h,head).transpose(1,2)
        logits=(q*(head**-.5))@k.transpose(-1,-2)
        if key_padding_mask is not None:
            if key_padding_mask.dtype!=torch.bool or key_padding_mask.all(1).any():raise ValueError('Invalid region mask')
            logits=logits.masked_fill(key_padding_mask[:,None,None,:],-torch.inf)
        self.log_attention=logits.float().log_softmax(-1)
        weights=F.dropout(self.log_attention.exp(),p=self.dropout,training=self.training)
        context=(weights.to(v.dtype)@v).transpose(1,2).reshape(b,t,d)
        return self.out_proj(context), None


def install(model):
    old=model.decoder[-1].object_attn
    new=GroundingAttention(old.embed_dim,old.num_heads,dropout=old.dropout,bias=old.in_proj_bias is not None,batch_first=True)
    new.load_state_dict(old.state_dict(),strict=True)
    new.to(device=old.in_proj_weight.device,dtype=old.in_proj_weight.dtype)
    new.train(old.training);model.decoder[-1].object_attn=new
    return new


def grounding_sum(log_attention,positive):
    """Return differentiable summed NLL and eligible count; avoid all-inf rows."""
    if positive.dtype!=torch.bool or positive.shape!=(log_attention.shape[0],log_attention.shape[2],log_attention.shape[3]):
        raise ValueError('Positive mask must be B,T,N')
    eligible=positive.any(-1);count=int(eligible.sum())
    if not count:
        # Masked padding has -inf logp: multiplying its sum by zero would yield NaN.
        return log_attention.masked_fill(~torch.isfinite(log_attention),0).sum()*0, 0
    logp=log_attention.permute(0,2,1,3)[eligible]
    mask=positive[eligible][:,None,:].expand_as(logp)
    mass=logp.masked_fill(~mask,-torch.inf).flatten(1).logsumexp(-1)-math.log(log_attention.shape[1])
    if not torch.isfinite(mass).all():raise ValueError('Invalid/empty positive attention mass')
    return -mass.sum(),count
