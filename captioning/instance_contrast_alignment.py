"""Frozen text targets; learn ordered pair appearance without heuristic positives."""
import torch
from torch import nn
from torch.nn import functional as F


class InstanceContrastAlignment(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.pair_projection = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(),
                                             nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, views, text):
        # views: A,B,union in this order. Text has one candidate per relation.
        pair = F.normalize(self.pair_projection(views.flatten(1)), dim=-1)
        return torch.einsum('bd,bkd->bk', pair, F.normalize(text.detach(), dim=-1))


class ResidualInstanceAlignment(nn.Module):
    """Keep frozen union/text cosine and learn only a bounded additive correction."""
    def __init__(self, dim=512, scale=.1):
        super().__init__()
        if not 0 < scale <= 1:
            raise ValueError('Residual scale must be in (0,1].')
        self.scale = scale
        self.correction = nn.Sequential(nn.Linear(dim*3, dim), nn.GELU(),
                                         nn.LayerNorm(dim), nn.Linear(dim, dim))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, views, text):
        fixed = F.normalize(text.detach(), dim=-1)
        base = torch.einsum('bd,bkd->bk', views[:, 2].detach(), fixed)
        raw = torch.einsum('bd,bkd->bk', self.correction(views.detach().flatten(1)), fixed)
        delta = self.scale * torch.tanh(raw)
        return base + delta, base, delta


class PairOnlyResidualAlignment(ResidualInstanceAlignment):
    """Ablation: union remains a frozen prior, correction sees only A and B."""
    def __init__(self, dim=512, scale=.1):
        super().__init__(dim=dim, scale=scale)
        self.correction = nn.Sequential(nn.Linear(dim*2, dim), nn.GELU(),
                                         nn.LayerNorm(dim), nn.Linear(dim, dim))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)

    def forward(self, views, text):
        fixed = F.normalize(text.detach(), dim=-1)
        base = torch.einsum('bd,bkd->bk', views[:, 2].detach(), fixed)
        raw = torch.einsum('bd,bkd->bk', self.correction(views[:, :2].detach().flatten(1)), fixed)
        delta = self.scale * torch.tanh(raw)
        return base + delta, base, delta


def supervised_alignment_loss(scores, labels, edges, temperature=.1, margin=.2,
                              contrast_weight=.2, edge_kinds=None):
    known = labels >= 0
    if not known.any():
        raise ValueError('Batch has no known supervision.')
    classification = F.binary_cross_entropy_with_logits(
        scores[known] / temperature, labels[known].to(scores.dtype))
    ranking = scores.sum() * 0
    if edges.numel():
        i, r, j, c = edges.T
        if not ((labels[i, r] == 1).all() and (labels[j, c] == 0).all()):
            raise ValueError('Contrast edges must connect confirmed positive to confirmed zero.')
        penalties = F.relu(margin - scores[i, r] + scores[j, c])
        if edge_kinds is None:
            ranking = penalties.mean()
        else:
            if edge_kinds.shape != penalties.shape or not ((edge_kinds == 0) | (edge_kinds == 1)).all():
                raise ValueError('Expected one predicate(0)/actor(1) kind per edge.')
            # Each present type has equal weight; missing types are not fake zero losses.
            ranking = torch.stack([penalties[edge_kinds == kind].mean()
                                   for kind in (0, 1) if (edge_kinds == kind).any()]).mean()
    return classification + contrast_weight * ranking, classification, ranking
