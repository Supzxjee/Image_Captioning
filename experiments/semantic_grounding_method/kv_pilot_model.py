"""User-run neural module. This file must never be imported by local static checks."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from kv_pilot_data import PAIRS,PARTNER,PREDICATES,INVERSE


class RelationReader(nn.Module):
    def __init__(self,embedding,fields,arm):
        super().__init__()
        self.arm = arm
        self.fields = {r['text']:i for i,r in enumerate(fields['fields'])}
        table = torch.stack([embedding.weight[r['ids']].detach().mean(0) for r in fields['fields']])
        self.register_buffer('text',table)
        self.register_buffer('partner',torch.tensor(PARTNER))
        self.register_buffer('left',torch.tensor([i for i,j in PAIRS]))
        self.register_buffer('right',torch.tensor([j for i,j in PAIRS]))
        self.register_buffer('pred_fields',torch.tensor([self.fields[p] for p in PREDICATES]))
        self.visual = nn.Linear(768,128)
        self.tp = nn.Linear(512,128)
        self.ts = nn.Linear(512,128)
        self.tobj = nn.Linear(512,128)
        self.wg = nn.Linear(8,128,bias=False)
        self.f = nn.Sequential(nn.Linear(648,256),nn.GELU(),nn.Linear(256,128))
        self.text_mlp = nn.Sequential(nn.Linear(384,128),nn.GELU(),nn.Linear(128,128))
        self.q = nn.Linear(512,64,bias=False)
        self.k = nn.Linear(128,64,bias=False)
        self.v = nn.Linear(128,128,bias=False)
        self.o = nn.Linear(128,512,bias=False)
        self.bias = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.o.weight)

    @torch.no_grad()
    def features(self,batch):
        raw = batch['visual'][:,1:]
        roi = batch['roi']/batch['roi'].sum(-1,keepdim=True).clamp_min(1e-12)
        union = batch['union']/batch['union'].sum(-1,keepdim=True).clamp_min(1e-12)
        return dict(roi=torch.bmm(roi,raw),union=torch.bmm(union,raw),geometry=batch['geometry'],valid=batch['pair_valid'])

    def hidden(self,features):
        roi = self.visual(features['roi'])
        a,b = roi[:,self.left],roi[:,self.right]
        union = self.visual(features['union'])
        geom = features['geometry']
        predicate = self.tp(self.text[self.pred_fields])
        batch = len(roi)
        def repeated(x):return x[:,:,None].expand(-1,-1,3,-1)
        pp = predicate[None,None].expand(batch,30,-1,-1)
        interaction = self.wg(geom)[:,:,None]*pp
        source = torch.cat([repeated(a),repeated(b),repeated(union),repeated(geom),pp,interaction],-1)
        return self.f(source).reshape(batch,90,128)

    def bank(self,features):
        h = self.hidden(features)
        value_h = (h+h[:,self.partner])*.5 if self.arm=='E' else h
        valid = features['valid'][:,:,None].expand(-1,-1,3).reshape(len(h),90)
        keys,values = self.k(h),self.v(value_h)
        keys = torch.cat([keys,keys.new_zeros((len(h),1,64))],1)
        values = torch.cat([values,values.new_zeros((len(h),1,128))],1)
        valid = torch.cat([valid,torch.ones((len(h),1),device=h.device,dtype=torch.bool)],1)
        active_h = torch.cat([h[valid[:,:90]],h.new_zeros((1,128))],0).detach()
        active_v = values[valid].detach()
        self.last_bank_stats = dict(h_norm_mean=active_h.norm(dim=-1).mean(),h_std=active_h.std(unbiased=False),
            value_norm_mean=active_v.norm(dim=-1).mean(),value_std=active_v.std(unbiased=False))
        return keys,values,valid

    def context(self,hidden,bank):
        keys,values,valid = bank
        scores = self.q(hidden)@keys.transpose(-1,-2)/math.sqrt(64)
        attention = scores.masked_fill(~valid[:,None],-torch.inf).softmax(-1)
        return attention@values

    def query(self,triples):
        ix = torch.tensor([[self.fields[s],self.fields[p],self.fields[o]] for s,p,o in triples],device=self.text.device)
        e = self.text[ix]
        return F.normalize(self.text_mlp(torch.cat([self.ts(e[:,0]),self.tp(e[:,1]),self.tobj(e[:,2])],-1)),dim=-1,eps=1e-6)

    def fact_scores(self,features,facts,effective=False):
        h = self.hidden(features)
        slots = torch.tensor([PAIRS.index((r['subject_slot'],r['object_slot']))*3+PREDICATES.index(r['predicate'])
                              for r in facts],device=h.device)
        ii = torch.arange(len(h),device=h.device)
        views = torch.stack([h[ii,slots],h[ii,self.partner[slots]]],1)
        if effective and self.arm=='E':
            views = views.mean(1,keepdim=True).expand(-1,2,-1)
        q1 = self.query([(r['subject'],r['predicate'],r['object']) for r in facts])
        q2 = self.query([(r['object'],INVERSE[r['predicate']],r['subject']) for r in facts])
        score = 10*(F.normalize(views,dim=-1,eps=1e-6)*torch.stack([q1,q2],1)).sum(-1)+self.bias
        return score


class Captioner(nn.Module):
    def __init__(self,base,fields,arm):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        self.teacher = copy.deepcopy(base.decoder).requires_grad_(False)
        for layer in base.decoder.transformer_decoder.layers[1:]:layer.requires_grad_(True)
        self.reader = RelationReader(base.decoder.embedding,fields,arm)
        self.eval()  # Deterministic eval-mode policy, as in the native C0 recipe.

    @torch.no_grad()
    def inputs(self,batch):
        memory,padding = self.base.build_memory(batch['visual'],batch['prompt'],batch['mask'])
        return dict(memory=memory,padding=padding,features=self.reader.features(batch))

    def logits(self,ids,x,teacher=False,reader_off=False):
        if teacher:
            return self.teacher(ids,x['memory'],memory_key_padding_mask=x['padding'])
        d = self.base.decoder
        hidden = d.transformer_decoder(tgt=d.pos_encoder(d.embedding(ids)*math.sqrt(d.d_model)),
            memory=x['memory'],tgt_mask=d.causal_mask(ids.size(1),ids.device),memory_key_padding_mask=x['padding'])
        if not reader_off:
            bank = x['bank'] if 'bank' in x else self.reader.bank(x['features'])
            context = self.reader.context(hidden,bank)
            contribution = self.reader.o(context)
            self.last_read_stats = dict(context_norm_mean=context.detach().norm(dim=-1).mean(),
                contribution_norm_mean=contribution.detach().norm(dim=-1).mean())
            hidden = hidden+contribution
        return d.fc_out(hidden)

    @torch.no_grad()
    def generate(self,x,tok,sample=False,teacher=False,reader_off=False,trace=False):
        if not teacher and not reader_off:
            x = dict(x,bank=self.reader.bank(x['features']))
        ids = torch.full((len(x['memory']),1),tok.bos_idx,device=x['memory'].device,dtype=torch.long)
        live = torch.ones(len(ids),device=ids.device,dtype=torch.bool)
        masks,chosen = [],[]
        for _ in range(29):
            logits = self.logits(ids,x,teacher,reader_off)[:,-1].float()
            if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite policy logits')
            next_token = torch.multinomial(logits.softmax(-1),1).flatten() if sample else logits.argmax(-1)
            next_token = torch.where(live,next_token,tok.pad_idx)
            masks.append(live.clone())
            if trace:chosen.append(logits.log_softmax(-1).gather(1,next_token[:,None]).flatten())
            ids = torch.cat([ids,next_token[:,None]],1)
            live = live & next_token.ne(tok.eos_idx)
            if not live.any():break
        result = (ids,torch.stack(masks,1))
        return result+(torch.stack(chosen,1),) if trace else result
