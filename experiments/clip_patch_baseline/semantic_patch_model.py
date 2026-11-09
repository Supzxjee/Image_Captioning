"""User-run bank/prompt patch memory; unchanged baseline causal decoder."""
import torch
from torch import nn
from patch_model import PatchCaptioner
from bank_layers import Config,SemanticAggregation,Alignment


class BankMemory(nn.Module):
    def __init__(self):
        super().__init__();self.aggregate=SemanticAggregation(Config(vocab=10000,dropout=0))
        self.align=Alignment();self.attention=nn.MultiheadAttention(512,8,dropout=0,batch_first=True)
        nn.init.zeros_(self.attention.out_proj.weight);nn.init.zeros_(self.attention.out_proj.bias)
    def forward(self,visual,bank):
        prototypes,_=self.aggregate(bank)
        selected,_=self.align(visual,prototypes,torch.ones(visual.shape[:2],dtype=torch.bool,device=visual.device))
        return visual+self.attention(visual,selected,selected,need_weights=False)[0]


class PromptMemory(nn.Module):
    def __init__(self,input_dim):
        super().__init__();self.slot=nn.Sequential(nn.Linear(input_dim,512),nn.ReLU(),nn.Linear(512,512))
        self.attention=nn.MultiheadAttention(512,8,dropout=0,batch_first=True)
        nn.init.zeros_(self.attention.out_proj.weight);nn.init.zeros_(self.attention.out_proj.bias)
    def forward(self,visual,slot_inputs,support,valid):
        slots=self.slot(slot_inputs)
        allowed=support.transpose(1,2).gt(0)&valid[:,None,:]
        active=allowed.any(-1)
        # Give inactive queries one finite key, then zero their entire residual.
        safe=allowed.clone();safe[:,:,0]|=~active
        mask=(~safe)[:,None].expand(-1,8,-1,-1).reshape(-1,visual.shape[1],slots.shape[1])
        delta=self.attention(visual,slots,slots,attn_mask=mask,need_weights=False)[0]
        return visual+delta.masked_fill(~active[...,None],0)


class SemanticPatchCaptioner(nn.Module):
    def __init__(self,vocab,nouns,verbs):
        super().__init__()
        # Construct the baseline FIRST; never reinitialize its modules recursively.
        self.base=PatchCaptioner(vocab)
        self.register_buffer('noun_bank',nouns);self.register_buffer('verb_bank',verbs)
        self.object_bank=BankMemory();self.relation_bank=BankMemory()
        self.object_prompt=PromptMemory(512+512+4)
        self.relation_prompt=PromptMemory(512+512+512+8)
        self.gate=nn.Linear(1024,512);nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)
    @property
    def project(self):return self.base.project
    @property
    def head(self):return self.base.head
    def memories(self,patches,p):
        visual=self.base.encode(patches);mo=self.object_bank(visual,self.noun_bank);mr=self.relation_bank(mo,self.verb_bank)
        obj=torch.cat((p['object_text'],torch.bmm(p['object_weights'],mo),p['object_box']),-1)
        rel=torch.cat((p['relation_text'],torch.bmm(p['subject_weights'],mo),torch.bmm(p['target_weights'],mo),p['relation_geometry']),-1)
        vo=self.object_prompt(mo,obj,p['object_weights'],p['object_valid'])
        vr=self.relation_prompt(mr,rel,p['subject_weights']+p['target_weights'],p['relation_valid'])
        return vo,vr
    def encode(self,patches,p):
        vo,vr=self.memories(patches,p);gate=self.gate(torch.cat((vo,vr),-1)).sigmoid()
        return gate*vo+(1-gate)*vr
    def decode(self,memory,prefix):return self.base.decode(memory,prefix)
    def forward(self,patches,prefix,p):return self.decode(self.encode(patches,p),prefix)
