"""No semantic-bank alignment: three visual Transformer layers, unchanged decoder."""
import torch
from torch import nn
from model import CSACaptioner,FFN,zero_pad


class VisualLayer(nn.Module):
    def __init__(self,c):
        super().__init__();self.self_attn=nn.MultiheadAttention(c.dim,c.heads,c.dropout,batch_first=True)
        self.norm=nn.LayerNorm(c.dim);self.drop=nn.Dropout(c.dropout);self.ff=FFN(c)

    def forward(self,x,valid):
        context=self.self_attn(x,x,x,key_padding_mask=~valid,need_weights=False)[0]
        return zero_pad(self.ff(zero_pad(self.norm(x+self.drop(context)),valid)),valid)


class PlainCaptioner(CSACaptioner):
    def __init__(self,c):
        # Preserve common layers' seed42 initialization from the original CSA constructor.
        bank=torch.zeros(1,c.dim)
        super().__init__(c,bank,bank)
        for n in ('noun_agg','verb_agg','object1','object2','relation','nouns','verbs'):delattr(self,n)
        self.visual_layers=nn.ModuleList([VisualLayer(c) for _ in range(3)])
        self.visual_layers.apply(self._init)
        for layer in self.visual_layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight);nn.init.zeros_(layer.self_attn.in_proj_bias)

    def encode(self,regions,valid):
        if valid.dtype!=torch.bool or regions.ndim!=3 or valid.shape!=regions.shape[:2] or not valid.any(dim=1).all():
            raise ValueError('Every image needs a valid ROI')
        if not torch.isfinite(regions).all():raise ValueError('Nonfinite ROI')
        v=zero_pad(self.project(regions),valid)
        for layer in self.visual_layers:v=layer(v,valid)
        return v,v
