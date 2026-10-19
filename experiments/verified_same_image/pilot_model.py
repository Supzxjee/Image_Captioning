"""Same evidence architecture as E; replace cross-image pseudo-pairs with verified pairs."""
import torch
from torch.nn import functional as F
from evidence_core import EvidenceMemory, PilotModel as OriginalPilotModel


def pair_loss(scores, positive_index, valid):
    positive=scores.gather(1,positive_index[:,None]).squeeze(1)
    negative=scores.gather(1,(1-positive_index)[:,None]).squeeze(1)
    margin=positive-negative
    if not bool(valid.any()): return scores.sum()*0,dict(pairs=0,accuracy=None,margin=None)
    return F.relu(.2-margin[valid]).mean(),dict(pairs=int(valid.sum()),
        accuracy=float((margin[valid]>0).float().mean().detach()),margin=float(margin[valid].mean().detach()))


class PilotModel(OriginalPilotModel):
    def __init__(self,base,arm):
        if arm not in ('A','F'): raise ValueError(arm)
        super().__init__(base,'A' if arm=='A' else 'E')
        self.arm=arm

    def forward(self,batch,sample,active,advantage,scst_weight,pad_idx):
        from captioning.scst import policy_loss
        memory,padding=self.memory(batch)
        targets=batch['captions'].flatten(0,1); masks=batch['caption_mask'].flatten(0,1)
        logits=self.teacher(memory,padding,targets[:,:-1],masks[:,:-1])
        ce=F.cross_entropy(logits.flatten(0,1),targets[:,1:].reshape(-1),ignore_index=pad_idx)
        rl=ce.new_zeros(())
        if scst_weight: rl=policy_loss(self.teacher(memory,padding,sample[:,:-1]),sample[:,1:],active,advantage)
        alignment=ce.new_zeros(());info=dict(pairs=0,accuracy=None,margin=None)
        if self.extra is not None:
            n=len(batch['aux_visual'])
            patches=batch['aux_visual'][:,None,1:].expand(-1,2,-1,-1).reshape(n*2,196,768)
            text=batch['aux_text'].reshape(n*2,512)
            boxes=batch['aux_bounds'].reshape(n*2,3,4)
            roles=torch.full((n*2,),2,dtype=torch.long,device=text.device)
            _,scores,_=self.extra.attend(patches,text,roles,boxes)
            alignment,info=pair_loss(scores.reshape(n,2),batch['aux_positive'],batch['aux_valid'])
        self.last_alignment=dict(loss=float(alignment.detach()),**info)
        return ce+scst_weight*rl+.1*alignment,ce.detach(),rl.detach()
