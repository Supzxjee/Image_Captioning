"""CSA equations/conventions from IMPLEMENTATION_PROTOCOL; self-written, no C0."""
import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    vocab: int
    dim: int = 512
    heads: int = 8
    hidden: int = 2048
    prototypes: int = 150
    layers: int = 3
    dropout: float = .2
    raw_dim: int = 768


def zero_pad(x, valid):
    return x.masked_fill(~valid[..., None], 0)


class FFN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(c.dim, c.hidden), nn.ReLU(), nn.Dropout(c.dropout),
                                 nn.Linear(c.hidden, c.dim), nn.Dropout(c.dropout))
        self.norm = nn.LayerNorm(c.dim, eps=1e-5)

    def forward(self, x):
        return self.norm(x + self.net(x))


class SemanticAggregation(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.wa = nn.Linear(c.dim, c.prototypes, bias=False)
        self.mlp = nn.Sequential(nn.Linear(c.prototypes, c.prototypes), nn.ReLU(),
                                 nn.Linear(c.prototypes, c.prototypes))
        self.wb = nn.Linear(c.dim, c.dim, bias=False)

    def forward(self, bank):
        weights = self.mlp(self.wa(bank)).float().softmax(dim=0)
        return weights.T.to(bank.dtype) @ self.wb(bank), weights


class Alignment(nn.Module):
    def forward(self, visual, prototypes, valid):
        if prototypes.ndim == 2:
            prototypes = prototypes.unsqueeze(0).expand(visual.shape[0], -1, -1)
        cosine = torch.bmm(F.normalize(prototypes.float(), dim=-1, eps=1e-6),
                           F.normalize(visual.float(), dim=-1, eps=1e-6).transpose(1, 2))
        weights = (cosine / .1).masked_fill(~valid[:, None, :], -torch.inf).softmax(dim=-1)
        relevance = (weights * cosine).sum(dim=-1)
        selected = relevance.softmax(dim=-1)[..., None] * prototypes
        return selected.to(visual.dtype), weights


class Fusion(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.align = Alignment()
        self.self_attn = nn.MultiheadAttention(c.dim, c.heads, c.dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(c.dim, c.heads, c.dropout, batch_first=True)
        self.norm_s, self.norm_c = nn.LayerNorm(c.dim), nn.LayerNorm(c.dim)
        self.ff_s, self.ff_c = FFN(c), FFN(c)
        self.drop = nn.Dropout(c.dropout)
        self.gate = nn.Linear(2 * c.dim, 1, bias=False)

    def forward(self, visual, prototypes, valid):
        selected, _ = self.align(visual, prototypes, valid)
        context = self.self_attn(visual, visual, visual, key_padding_mask=~valid, need_weights=False)[0]
        own = zero_pad(self.norm_s(visual + self.drop(context)), valid)
        own = zero_pad(self.ff_s(own), valid)
        context = self.cross_attn(own, selected, selected, need_weights=False)[0]
        cross = zero_pad(self.norm_c(own + self.drop(context)), valid)
        cross = zero_pad(self.ff_c(cross), valid)
        gate = self.gate(torch.cat((own, cross), dim=-1)).sigmoid()
        return zero_pad(gate * own + (1 - gate) * cross, valid), selected


class DecoderLayer(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(c.dim, c.heads, c.dropout, batch_first=True)
        self.object_attn = nn.MultiheadAttention(c.dim, c.heads, c.dropout, batch_first=True)
        self.relation_attn = nn.MultiheadAttention(c.dim, c.heads, c.dropout, batch_first=True)
        self.norm = nn.LayerNorm(c.dim)
        self.drop = nn.Dropout(c.dropout)
        self.gate_o, self.gate_r = nn.Linear(2 * c.dim, c.dim), nn.Linear(2 * c.dim, c.dim)
        self.ffn = FFN(c)

    def forward(self, x, previous_word, vo, vr, text_valid, region_valid):
        causal = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
        h = self.self_attn(x, x, x, attn_mask=causal, key_padding_mask=~text_valid, need_weights=False)[0]
        h = zero_pad(self.norm(x + self.drop(h)), text_valid)
        co = self.object_attn(h, vo, vo, key_padding_mask=~region_valid, need_weights=False)[0]
        cr = self.relation_attn(h, vr, vr, key_padding_mask=~region_valid, need_weights=False)[0]
        go = self.gate_o(torch.cat((co, previous_word), dim=-1)).sigmoid()
        gr = self.gate_r(torch.cat((cr, previous_word), dim=-1)).sigmoid()
        return zero_pad(self.ffn(go * co + gr * cr), text_valid)


class CSACaptioner(nn.Module):
    def __init__(self, c, nouns, verbs):
        super().__init__()
        if c.dim % 2 or c.dim % c.heads or c.vocab < 4 or c.layers < 1 or c.prototypes < 1:
            raise ValueError('Invalid model dimensions')
        for bank in (nouns, verbs):
            if bank.ndim != 2 or bank.shape[1] != c.dim or not len(bank) or not torch.isfinite(bank).all():
                raise ValueError('Empty, nonfinite or incompatible semantic bank')
        self.c = c
        self.register_buffer('nouns', nouns.detach().float().clone())
        self.register_buffer('verbs', verbs.detach().float().clone())
        self.project = nn.Linear(c.raw_dim, c.dim)
        self.noun_agg, self.verb_agg = SemanticAggregation(c), SemanticAggregation(c)
        self.object1, self.object2, self.relation = Fusion(c), Fusion(c), Fusion(c)
        self.embedding = nn.Embedding(c.vocab, c.dim, padding_idx=0)
        self.decoder = nn.ModuleList([DecoderLayer(c) for _ in range(c.layers)])
        self.head = nn.Linear(c.dim, c.vocab)
        self.apply(self._init)
        for module in self.modules():
            if isinstance(module, nn.MultiheadAttention):
                nn.init.xavier_uniform_(module.in_proj_weight)
                nn.init.zeros_(module.in_proj_bias)
        nn.init.normal_(self.embedding.weight, std=c.dim ** -.5)
        with torch.no_grad():
            self.embedding.weight[0].zero_()

    @staticmethod
    def _init(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def encode(self, regions, valid):
        if valid.dtype != torch.bool or regions.ndim != 3 or valid.shape != regions.shape[:2] or not valid.any(dim=1).all():
            raise ValueError('Every image requires at least one valid region')
        if not torch.isfinite(regions).all():
            raise ValueError('Nonfinite visual feature')
        v = zero_pad(self.project(regions), valid)
        nouns, _ = self.noun_agg(self.nouns)
        verbs, _ = self.verb_agg(self.verbs)
        v, selected = self.object1(v, nouns, valid)
        vo, _ = self.object2(v, selected, valid)
        vr, _ = self.relation(vo, verbs, valid)
        return vo, vr

    def decode(self, ids, vo, vr, valid):
        if ids.ndim != 2 or ids.shape[1] == 0 or not (ids[:, 0] == 1).all():
            raise ValueError('Prefix must start with BOS, with nonempty time dimension')
        raw = self.embedding(ids)
        position = torch.arange(ids.shape[1], device=ids.device, dtype=torch.float32)[:, None]
        frequency = torch.exp(torch.arange(0, self.c.dim, 2, device=ids.device).float() * (-math.log(10000) / self.c.dim))
        pe = torch.empty(ids.shape[1], self.c.dim, device=ids.device)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * frequency), torch.cos(position * frequency)
        text_valid = ids != 0
        x = zero_pad(raw * math.sqrt(self.c.dim) + pe.to(raw.dtype), text_valid)
        for layer in self.decoder:
            x = layer(x, raw, vo, vr, text_valid, valid)
        return self.head(x)

    def forward(self, regions, valid, ids):
        return self.decode(ids, *self.encode(regions, valid), valid)


def xe_loss(logits, target):
    mask = target != 0
    if not mask.any():
        raise ValueError('XE batch has no target tokens')
    nll = F.cross_entropy(logits.flatten(0, 1), target.flatten(), reduction='none', ignore_index=0).view_as(target)
    return (nll * mask).sum() / mask.sum()


@torch.no_grad()
def beam_search(model, vo, vr, valid, beams=5, maximum=20):
    """Per-image beams; no hidden length normalization, BOS/PAD cannot be emitted."""
    if model.training or maximum < 1 or not 1 <= beams <= model.c.vocab - 2:
        raise ValueError('Beam search requires eval mode and valid width/length')
    results = []
    for i in range(len(vo)):
        seq = torch.ones(1, 1, dtype=torch.long, device=vo.device)
        scores = torch.zeros(1, device=vo.device)
        finished = torch.zeros(1, device=vo.device, dtype=torch.bool)
        for _ in range(maximum):
            count = len(seq)
            logits = model.decode(seq, vo[i:i+1].expand(count, -1, -1),
                                  vr[i:i+1].expand(count, -1, -1), valid[i:i+1].expand(count, -1))[:, -1]
            logp = logits.float().log_softmax(-1)
            logp[:, :2] = -torch.inf
            logp[finished] = -torch.inf
            logp[finished, 0] = 0
            best, flat = (scores[:, None] + logp).flatten().topk(beams)
            parent, token = flat // model.c.vocab, flat % model.c.vocab
            seq = torch.cat((seq[parent], token[:, None]), dim=1)
            finished = finished[parent] | (token == 2)
            scores = best
            if finished.all():
                break
        results.append((seq, scores))
    return results


def beam_mean_loss(model, regions, valid, sequences, rewards, maximum=20):
    """Gradient rescore; rewards [B,K] are externally computed CIDEr-D and detached."""
    b, k, length = sequences.shape
    if rewards.shape != (b, k) or length < 2 or length > maximum + 1:
        raise ValueError('Invalid beam/reward shape')
    flat = sequences.reshape(b * k, length)
    logits = model(regions.repeat_interleave(k, 0), valid.repeat_interleave(k, 0), flat[:, :-1])
    target = flat[:, 1:]
    # Include first EOS, exclude every later position (even malformed non-PAD tokens).
    eos_before = (target == 2).cumsum(dim=1) - (target == 2).long()
    mask = (target != 0) & (eos_before == 0)
    logp = logits.float().log_softmax(-1).gather(-1, target[..., None]).squeeze(-1)
    score = (logp * mask).sum(-1).reshape(b, k) / maximum
    advantage = rewards.detach() - rewards.detach().mean(dim=1, keepdim=True)
    return -(advantage * score).mean()
