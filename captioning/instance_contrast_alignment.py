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


def supervised_alignment_loss(scores, labels, edges, temperature=.1, margin=.2,
                              contrast_weight=.2):
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
        ranking = F.relu(margin - scores[i, r] + scores[j, c]).mean()
    return classification + contrast_weight * ranking, classification, ranking
