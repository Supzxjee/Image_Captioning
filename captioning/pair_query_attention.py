"""Small candidate-conditioned reader of frozen CLIP patch tokens.

Attention is an internal weighting, never a contact or causal explanation label.
"""
import torch
from torch import nn
from torch.nn import functional as F


def patch_evidence(tokens, support, projection):
    """Label-free fixed compression; all arms receive the identical patch bank."""
    if tokens.ndim != 3 or tokens.shape[1:] != (197, 768) or support.shape != (len(tokens), 5, 196):
        raise ValueError('Expected 14x14 CLIP patches and five geometric supports.')
    if projection.shape[0] != 768 or not all(torch.isfinite(v).all() for v in (tokens, support, projection)):
        raise ValueError('Nonfinite or incompatible patch evidence.')
    content = F.normalize(F.normalize(tokens[:, 1:].float(), dim=-1) @ projection, dim=-1)
    yy, xx = torch.meshgrid(torch.arange(14), torch.arange(14), indexing='ij')
    xy = torch.stack([(xx.flatten()+.5)/14, (yy.flatten()+.5)/14], -1).to(content)
    evidence = torch.cat([content, support.transpose(1, 2), xy[None].expand(len(tokens), -1, -1)], -1)
    weights = support[:, :2].float()
    pooled = weights @ content / weights.sum(-1, keepdim=True).clamp_min(1e-8)
    return evidence, pooled


class PairQueryAttention(nn.Module):
    def __init__(self, patch_dim=71, text_dim=512, hidden=32):
        super().__init__()
        self.pair = nn.Sequential(nn.Linear(2*64+10, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.text = nn.Sequential(nn.Linear(text_dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.query = nn.Linear(2*hidden, hidden)
        self.key = nn.Linear(patch_dim, hidden)
        self.value = nn.Linear(patch_dim, hidden)
        self.score = nn.Sequential(nn.Linear(3*hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.hidden = hidden

    def forward(self, patches, endpoints, geometry, text, mode='attention', return_weights=False):
        if mode not in ('attention', 'mean_pool', 'query_blind'): raise ValueError('Unknown control.')
        if endpoints.shape != (len(patches), 2, 64) or geometry.shape != (len(patches), 10):
            raise ValueError('Invalid ordered endpoint/box evidence.')
        pair = self.pair(torch.cat([endpoints.detach().flatten(1), geometry.detach()], -1))
        words = self.text(text.detach())
        pair = pair[:, None].expand(-1, text.shape[1], -1)
        query = self.query(torch.cat([pair, words], -1))
        keys, values = self.key(patches.detach()), self.value(patches.detach())
        if mode == 'query_blind':
            # Query networks remain trainable, but the reader sees no supplied pair/text.
            blank_pair = self.pair(torch.zeros_like(torch.cat([endpoints.flatten(1), geometry], -1)))
            blank_words = self.text(torch.zeros_like(text[:, :1]))
            reader = self.query(torch.cat([blank_pair[:, None], blank_words], -1))
            logits = torch.einsum('brd,bpd->brp', reader, keys)/self.hidden**.5
            weights = logits.softmax(-1).expand(-1, text.shape[1], -1)
        elif mode == 'mean_pool':
            weights = patches.new_full((len(patches), text.shape[1], patches.shape[1]), 1/patches.shape[1])
        else:
            logits = torch.einsum('brd,bpd->brp', query, keys)/self.hidden**.5
            weights = logits.softmax(-1)
        attended = weights @ values
        scores = self.score(torch.cat([query, attended, query*attended], -1)).squeeze(-1)
        return (scores, weights) if return_weights else scores
