"""Text-conditioned verification of an ordered actor, predicate and target."""
import torch
from torch import nn


class TripletVerifier(nn.Module):
    def __init__(self,dim=512):
        super().__init__()
        self.pair=nn.Sequential(nn.Linear(4*dim+10,128),nn.GELU(),nn.LayerNorm(128))
        self.text=nn.Sequential(nn.Linear(dim,128),nn.GELU(),nn.LayerNorm(128))
        self.joint=nn.Sequential(nn.Linear(384,64),nn.GELU(),nn.Linear(64,1))

    def forward(self,views,geometry,text):
        a,b=views[:,0].detach(),views[:,1].detach()
        pair=self.pair(torch.cat([a,b,a*b,a-b,geometry],dim=-1))
        target=self.text(text.detach())
        pair=pair[:,None,:].expand_as(target)
        return self.joint(torch.cat([pair,target,pair*target],dim=-1)).squeeze(-1)
