"""Ordered region evidence from one shared union-crop patch grid.

Regions are geometric supports, not segmentation or ground-truth contact.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

REGIONS=('subject','object','intersection','bridge','outside','union_cls')


def boxes_in_crop(subject, target, crop, width, height):
    x,y,x2,y2=crop
    if x2<=x or y2<=y:raise ValueError('Empty crop.')
    result=[]
    for box in (subject,target):
        if len(box)!=4 or not all(math.isfinite(v) for v in box) or not 0<=box[0]<box[2]<=1 or not 0<=box[1]<box[3]<=1:
            raise ValueError('Invalid normalized box.')
        result.append([(box[0]*width-x)/(x2-x),(box[1]*height-y)/(y2-y),
                       (box[2]*width-x)/(x2-x),(box[3]*height-y)/(y2-y)])
    return torch.tensor(result,dtype=torch.float32).clamp(0,1)


def region_weights(boxes, grid=14):
    if boxes.shape!=(2,4) or grid<1 or not torch.isfinite(boxes).all():raise ValueError('Invalid ROI/grid.')
    if not ((boxes[:,:2]>=0).all() and (boxes[:,2:]<=1).all() and (boxes[:,2:]>boxes[:,:2]).all()):
        raise ValueError('Empty/out-of-bounds ROI.')
    boxes=boxes.to(dtype=torch.float64)
    yy,xx=torch.meshgrid(torch.arange(grid,device=boxes.device,dtype=torch.float64)/grid,
                        torch.arange(grid,device=boxes.device,dtype=torch.float64)/grid,indexing='ij')
    cells=torch.stack([xx.flatten(),yy.flatten(),xx.flatten()+1/grid,yy.flatten()+1/grid],dim=-1)
    def intersection(a,b):return torch.cat([torch.maximum(a[:2],b[:2]),torch.minimum(a[2:],b[2:])])
    def coverage(box):
        lo=torch.maximum(cells[:,:2],box[:2]);hi=torch.minimum(cells[:,2:],box[2:])
        return (hi-lo).clamp_min(0).prod(-1)*grid**2
    a,b=boxes;overlap=intersection(a,b)
    ac=(a[:2]+a[2:])/2;bc=(b[:2]+b[2:])/2
    bridge=torch.cat([(torch.minimum(ac,bc)-.5/grid).clamp(0,1),
                      (torch.maximum(ac,bc)+.5/grid).clamp(0,1)])
    wa,wb,wi=coverage(a),coverage(b),coverage(overlap)
    # Exact rectangular inclusion/exclusion, including sub-patch small boxes.
    wbri=(coverage(bridge)-coverage(intersection(bridge,a))-coverage(intersection(bridge,b))
          +coverage(intersection(bridge,overlap))).clamp(0,1)
    outside=(1-wa-wb+wi).clamp(0,1)
    weights=torch.stack([wa,wb,wi,wbri,outside]).clamp(0,1)
    weights[weights<1e-12]=0.  # Remove floating-point slivers in exactly empty regions.
    return weights.float()


def pool_regions(patches, cls, boxes):
    grid=math.isqrt(patches.shape[0])
    if grid**2!=patches.shape[0] or patches.ndim!=2 or cls.shape!=(patches.shape[1],):
        raise ValueError('Expected square patch grid and matching CLS.')
    weights=region_weights(boxes.to(patches.device),grid)
    mass=weights.sum(-1);valid=mass>1e-8
    pooled=weights@patches/mass.clamp_min(1e-8)[:,None]
    pooled[~valid]=0
    features=F.normalize(torch.cat([pooled,cls[None]],dim=0),dim=-1)
    return features,torch.cat([valid,valid.new_ones(1)]),torch.cat([mass/patches.shape[0],mass.new_ones(1)])


class InteractionRegionVerifier(nn.Module):
    """Candidate-text conditioned head; matched ablation repeats union CLS."""
    def __init__(self,vision_dim=768,text_dim=512):
        super().__init__()
        self.pair=nn.Sequential(nn.Linear(6*vision_dim+22,128),nn.GELU(),nn.LayerNorm(128))
        self.text=nn.Sequential(nn.Linear(text_dim,128),nn.GELU(),nn.LayerNorm(128))
        self.joint=nn.Sequential(nn.Linear(384,64),nn.GELU(),nn.Linear(64,1))

    def forward(self,regions,valid,mass,geometry,text,union_only=False):
        evidence=regions.detach()
        if union_only:evidence=evidence[:,-1:, :].expand_as(evidence)
        evidence=evidence*valid[:,:,None]
        pair=self.pair(torch.cat([evidence.flatten(1),valid.float(),mass,geometry],dim=-1))
        target=self.text(text.detach());pair=pair[:,None].expand_as(target)
        return self.joint(torch.cat([pair,target,pair*target],dim=-1)).squeeze(-1)
