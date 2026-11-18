"""User-run neural baseline: one patch memory, standard causal decoder."""
import math
import torch
from torch import nn

class PatchCaptioner(nn.Module):
    def __init__(self,vocab,dim=512,heads=8,layers=4,hidden=2048,dropout=.1,maximum=64):
        super().__init__();self.dim=dim
        self.project=nn.Sequential(nn.Linear(768,dim),nn.LayerNorm(dim))
        self.embedding=nn.Embedding(vocab,dim,padding_idx=0)
        p=torch.arange(maximum,dtype=torch.float32)[:,None];scale=torch.exp(torch.arange(0,dim,2)*(-math.log(10000)/dim))
        pos=torch.zeros(maximum,dim);pos[:,0::2]=torch.sin(p*scale);pos[:,1::2]=torch.cos(p*scale)
        self.register_buffer('positions',pos,persistent=True);self.dropout=nn.Dropout(dropout)
        layer=nn.TransformerDecoderLayer(dim,heads,hidden,dropout,batch_first=True,norm_first=False)
        self.decoder=nn.TransformerDecoder(layer,layers,norm=nn.LayerNorm(dim));self.head=nn.Linear(dim,vocab)
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:nn.init.zeros_(module.bias)
            elif isinstance(module,nn.Embedding):nn.init.normal_(module.weight,std=.02);module.weight.data[0].zero_()
            elif isinstance(module,nn.MultiheadAttention):nn.init.xavier_uniform_(module.in_proj_weight);nn.init.zeros_(module.in_proj_bias)

    def encode(self,patches):
        if patches.ndim!=3 or patches.shape[1:]!=(196,768):raise ValueError('Wrong patch shape')
        return self.project(patches)

    def decode(self,memory,prefix):
        if prefix.ndim!=2 or len(prefix)!=len(memory):raise ValueError('Wrong memory/prefix shape')
        if prefix.shape[1]>len(self.positions) or (prefix==0).all(1).any():raise ValueError('Empty or too long prefix')
        text=self.dropout(self.embedding(prefix)*math.sqrt(self.dim)+self.positions[:prefix.shape[1]])
        causal=torch.ones(prefix.shape[1],prefix.shape[1],dtype=torch.bool,device=prefix.device).triu(1)
        output=self.decoder(text,memory,tgt_mask=causal,tgt_key_padding_mask=prefix==0)
        return self.head(output)

    def forward(self,patches,prefix):return self.decode(self.encode(patches),prefix)

    @torch.no_grad()
    def greedy(self,patches,maximum=20):
        ids=torch.ones(len(patches),1,dtype=torch.long,device=patches.device);done=torch.zeros(len(patches),dtype=torch.bool,device=patches.device)
        for _ in range(maximum):
            word=self(patches,ids)[:,-1].argmax(-1);word=word.masked_fill(done,0)
            ids=torch.cat((ids,word[:,None]),1);done|=word==2
            if done.all():break
        return ids
