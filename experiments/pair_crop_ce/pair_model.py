"""Same evidence encoder in both arms; patch coordinates are in original image space."""
import math
import torch
from torch.nn import functional as F
from phrase_model import Evidence,PhraseModel


class MappedEvidence(Evidence):
    def forward(self,packed,boxes,geo):
        assert packed.shape[1:]==(196,770)
        patches,xy=packed[:,:,:768],packed[:,:,768:]
        union=torch.cat((boxes.min(1).values[:,:2],boxes.max(1).values[:,2:]),-1)
        all_boxes=torch.cat((boxes,union[:,None]),1)
        inside=((xy[:,None]>=all_boxes[:,:,None,:2]).all(-1)&(xy[:,None]<=all_boxes[:,:,None,2:]).all(-1))
        allowed=inside[:,self.roles].clone();empty=~allowed.any(-1)
        centers=(all_boxes[:,:,:2]+all_boxes[:,:,2:])/2
        nearest=((xy[:,None]-centers[:,self.roles,None])**2).sum(-1).argmin(-1)
        rows,slots=torch.where(empty);allowed[rows,slots,nearest[rows,slots]]=True
        spatial=torch.cat((xy,inside[:,:2].transpose(1,2).float()),-1)
        keys=self.key(patches)+self.position(spatial)
        q=self.query[None]+self.geometry(geo)[:,None]
        attention=(torch.einsum('nqd,npd->nqp',q,keys)/math.sqrt(512)).masked_fill(~allowed,-torch.inf).softmax(-1)
        tokens=self.norm(torch.bmm(attention,F.relu(self.value(patches))))+self.role(self.roles)[None]
        return tokens,attention


class PairModel(PhraseModel):
    def __init__(self,decoder,projection):
        super().__init__(decoder,projection)
        mapped=MappedEvidence(projection)
        mapped.load_state_dict(self.evidence.state_dict(),strict=True)
        self.evidence=mapped
