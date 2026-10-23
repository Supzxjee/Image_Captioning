"""Frozen captioner plus a 108752-parameter visual noun-evidence head."""
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from pilot_model import PilotModel


def logmeanexp(values,weights=None):
    maximum=values.max(-2,keepdim=True).values
    centered=(values-maximum).exp()
    pooled=centered.mean(-2) if weights is None else (centered*weights[...,None]).sum(-2)/weights.sum(-1,keepdim=True)
    return maximum.squeeze(-2)+pooled.log()


def global_mean(local_sum,count):
    total=local_sum.detach().new_tensor(float(count));dist.all_reduce(total)
    return local_sum*(dist.get_world_size()/total.clamp_min(1)),total


class EvidenceHead(nn.Module):
    def __init__(self,prior):
        super().__init__()
        p=torch.as_tensor(prior,dtype=torch.float32)
        if p.shape!=(80,) or not bool(((p>0)&(p<1)).all()):raise ValueError('Invalid TRAIN prior.')
        self.net=nn.Sequential(nn.LayerNorm(768,elementwise_affine=False),nn.Linear(768,128),nn.GELU(),nn.Linear(128,80))
        self.register_buffer('prior_logit',p.log()-torch.log1p(-p))
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():self.net[-1].bias.copy_(self.prior_logit)
        assert sum(x.numel() for x in self.parameters())==108752

    def forward(self,visual):
        z=self.net(visual[:,1:])
        if z.shape[1:]!=(196,80):raise ValueError('Expected CLS+196 CLIP patches.')
        g=logmeanexp(z)
        return z,g,(g-self.prior_logit).clamp(-2,2)


class DecoderProxy:
    """Plain callable avoids registering frozen decoder weights a second time."""
    def __init__(self,owner):self.owner=owner
    def __call__(self,tgt,memory,tgt_key_padding_mask=None,memory_key_padding_mask=None):
        owner=self.owner
        with torch.no_grad():
            logits=owner.base.decoder(tgt,memory,tgt_key_padding_mask=tgt_key_padding_mask,memory_key_padding_mask=memory_key_padding_mask)
        if owner.condition=='off':return logits
        correction=owner.current_d
        repeats=len(tgt)//len(correction)
        if len(tgt)!=len(correction)*repeats:raise ValueError('Incorrect image/reference expansion.')
        correction=correction.repeat_interleave(repeats,0)[:,owner.candidate_classes]
        allowed=torch.ones((*tgt.shape,len(owner.candidate_ids)),dtype=torch.bool,device=tgt.device)
        positions=torch.arange(tgt.shape[1],device=tgt.device)[None,:]
        for prefix,columns in owner.guards:
            match=torch.ones_like(tgt,dtype=torch.bool)
            for offset,valid_ids in enumerate(prefix):
                match&=torch.isin(tgt.roll(offset,1),valid_ids)&(positions>=offset)
            allowed[:,:,columns]&=~match[:,:,None]
        delta=.5*correction[:,None,:]*allowed
        return logits.index_add(-1,owner.candidate_ids,delta)


class EvidenceModel(PilotModel):
    def __init__(self,base,prior,lexical,arm):
        nn.Module.__init__(self)
        if arm not in ('G','S'):raise ValueError(arm)
        self.base=base.eval().requires_grad_(False);self.arm=arm;self.head=EvidenceHead(prior)
        self.register_buffer('candidate_ids',torch.tensor(lexical['candidate_ids'],dtype=torch.long))
        self.register_buffer('candidate_classes',torch.tensor(lexical['candidate_classes'],dtype=torch.long))
        self.guards=[]
        # Guards are tiny nonpersistent index tables, materialized on the active device below.
        self.guard_data=lexical['guards'];self.condition='on';self.current_d=None;self.detail={}
        self.proxy=DecoderProxy(self)

    @property
    def decoder(self):return self.proxy

    def memory(self,batch,intervention='on'):
        if intervention not in ('on','off','train_mean','shuffle'):raise ValueError(intervention)
        self.condition=intervention
        if not self.guards:
            self.guards=[([torch.tensor(ids,device=self.candidate_ids.device) for ids in r['prefix_ids_reversed']],
                torch.tensor(r['columns'],device=self.candidate_ids.device)) for r in self.guard_data]
        with torch.no_grad():memory,padding=self.base.build_memory(batch['visual'],batch['legacy'],batch['legacy_mask'])
        self.current_z,self.current_g,self.current_d=self.head(batch['visual'])
        if intervention in ('train_mean','shuffle'):self.current_d=batch['override_d']
        return memory,padding

    def forward(self,batch,sample,active,advantage,scst_weight,pad_idx):
        from captioning.scst import policy_loss
        memory,padding=self.memory(batch)
        targets=batch['captions'].flatten(0,1);mask=batch['caption_mask'].flatten(0,1)
        logits=self.teacher(memory,padding,targets[:,:-1],mask[:,:-1])
        ce=F.cross_entropy(logits.flatten(0,1),targets[:,1:].reshape(-1),ignore_index=pad_idx)
        rl=ce.new_zeros(())
        if scst_weight:
            sample_logits=self.teacher(memory,padding,sample[:,:-1])
            rl=policy_loss(sample_logits,sample[:,1:],active,advantage)
        observed=batch['observed'];y=batch['presence']
        presence_sum=(F.binary_cross_entropy_with_logits(self.current_g,y,reduction='none')*observed).sum()
        presence,nlabels=global_mean(presence_sum,int(observed.sum().item()))
        images,slots=batch['region_valid'].nonzero(as_tuple=True)
        if len(images):
            z=self.current_z[images]
            pooled=logmeanexp(z,batch['region_masks'][images,slots]) if self.arm=='S' else logmeanexp(z)
            positive=F.softplus(-pooled.gather(-1,batch['region_class'][images,slots,None]).squeeze(-1))
            negative=(y[images]==0)&(observed[images]>0)
            loss=positive+(F.softplus(pooled)*negative).sum(-1)/negative.sum(-1).clamp_min(1)
            region_sum=loss.sum()
        else:region_sum=self.current_z.sum()*0
        region,nregions=global_mean(region_sum,len(images))
        stats=torch.stack([presence_sum.detach(),region_sum.detach()]);dist.all_reduce(stats)
        self.detail=dict(presence_loss_global=float(stats[0]/nlabels.clamp_min(1)),region_loss_global=float(stats[1]/nregions.clamp_min(1)),
            local_regions=len(images),global_regions=int(nregions.item()),global_observed_labels=int(nlabels.item()),
            local_image_ids=batch['image_id'].cpu().tolist(),correction_abs_mean=float(self.current_d.detach().abs().mean()),
            correction_saturation_fraction=float((self.current_d.detach().abs()>=2).float().mean()))
        return ce+scst_weight*rl+.1*presence+.1*region,ce.detach(),rl.detach()
