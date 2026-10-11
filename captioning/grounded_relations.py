"""Instance-pair relation grounding for CLIP patch features."""
from pathlib import PurePosixPath

import torch
import torch.nn as nn
import torch.nn.functional as F


def align_grounded_relation_cache(cache, image_paths):
    """Validate a fixed-size grounded-relation cache and align it by filename."""
    if not isinstance(cache, dict):
        raise ValueError('Grounded relation cache must be a mapping.')
    indexed = {}
    required = {
        'tokens', 'negative_tokens', 'subject_boxes', 'object_boxes',
        'confidences', 'mask',
    }
    for key, entry in cache.items():
        filename = PurePosixPath(str(key).replace('\\', '/')).name
        if filename in indexed:
            raise ValueError(f'Ambiguous grounded relation filename: {filename}')
        if not isinstance(entry, dict) or not required.issubset(entry):
            raise ValueError(f'Invalid grounded relation entry for {filename}.')
        relation_count = entry['tokens'].size(0)
        if entry['tokens'].shape != (relation_count, 512):
            raise ValueError(f'{filename}: expected relation tokens (R, 512).')
        if entry['negative_tokens'].shape != entry['tokens'].shape:
            raise ValueError(f'{filename}: negative relation tokens do not match positives.')
        if entry['subject_boxes'].shape != (relation_count, 4):
            raise ValueError(f'{filename}: invalid subject boxes.')
        if entry['object_boxes'].shape != (relation_count, 4):
            raise ValueError(f'{filename}: invalid object boxes.')
        if entry['confidences'].shape != (relation_count,):
            raise ValueError(f'{filename}: invalid relation confidences.')
        if entry['mask'].shape != (relation_count,):
            raise ValueError(f'{filename}: invalid relation mask.')
        indexed[filename] = entry
    return {
        path: indexed[PurePosixPath(str(path).replace('\\', '/')).name]
        for path in image_paths
        if PurePosixPath(str(path).replace('\\', '/')).name in indexed
    }


def _pool_patch_boxes(patch_features, boxes, valid):
    """Pool 14x14 CLIP patches inside normalized boxes, with nearest fallback."""
    batch, patch_count, dim = patch_features.shape
    if patch_count != 196:
        raise ValueError(f'Expected 196 CLIP patches, got {patch_count}.')
    centers = (torch.arange(14, device=patch_features.device, dtype=boxes.dtype) + 0.5) / 14
    yy, xx = torch.meshgrid(centers, centers, indexing='ij')
    patch_x = xx.reshape(1, 1, 196)
    patch_y = yy.reshape(1, 1, 196)
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    inside = ((patch_x >= x1.unsqueeze(-1)) & (patch_x <= x2.unsqueeze(-1)) &
              (patch_y >= y1.unsqueeze(-1)) & (patch_y <= y2.unsqueeze(-1)) &
              valid.unsqueeze(-1))
    counts = inside.sum(dim=-1)
    pooled = torch.einsum('brp,bpd->brd', inside.to(patch_features.dtype), patch_features)
    pooled = pooled / counts.clamp_min(1).unsqueeze(-1).to(pooled.dtype)
    center_x = ((x1 + x2) / 2).unsqueeze(-1)
    center_y = ((y1 + y2) / 2).unsqueeze(-1)
    distances = (patch_x - center_x).square() + (patch_y - center_y).square()
    nearest_index = distances.argmin(dim=-1)
    nearest = patch_features.gather(
        1, nearest_index.unsqueeze(-1).expand(-1, -1, dim))
    return torch.where((counts > 0).unsqueeze(-1), pooled, nearest)


def relative_geometry(subject_boxes, object_boxes):
    """Return direction, scale, overlap and containment evidence for each pair."""
    sx1, sy1, sx2, sy2 = subject_boxes.unbind(dim=-1)
    ox1, oy1, ox2, oy2 = object_boxes.unbind(dim=-1)
    sw, sh = (sx2 - sx1).clamp_min(1e-6), (sy2 - sy1).clamp_min(1e-6)
    ow, oh = (ox2 - ox1).clamp_min(1e-6), (oy2 - oy1).clamp_min(1e-6)
    scx, scy = (sx1 + sx2) / 2, (sy1 + sy2) / 2
    ocx, ocy = (ox1 + ox2) / 2, (oy1 + oy2) / 2
    ix = (torch.minimum(sx2, ox2) - torch.maximum(sx1, ox1)).clamp_min(0)
    iy = (torch.minimum(sy2, oy2) - torch.maximum(sy1, oy1)).clamp_min(0)
    intersection = ix * iy
    subject_area, object_area = sw * sh, ow * oh
    union = (subject_area + object_area - intersection).clamp_min(1e-6)
    return torch.stack([
        (ocx - scx) / sw, (ocy - scy) / sh,
        torch.log(ow / sw), torch.log(oh / sh),
        intersection / union,
        intersection / subject_area.clamp_min(1e-6),
        intersection / object_area.clamp_min(1e-6),
        torch.sqrt((ocx - scx).square() + (ocy - scy).square()),
    ], dim=-1)


class PairGroundedRelationAlignment(nn.Module):
    """Ground relation text in its subject/object patches before query fusion."""
    def __init__(self, embed_dim, num_heads, dropout=0.1, relation_dim=512):
        super().__init__()
        self.relation_projection = nn.Sequential(
            nn.Linear(relation_dim, embed_dim), nn.LayerNorm(embed_dim))
        self.pair_projection = nn.Sequential(
            nn.Linear(embed_dim * 3 + 8, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim))
        self.evidence_gate = nn.Linear(embed_dim * 2 + 1, 1)
        self.relation_norm = nn.LayerNorm(embed_dim)
        self.query_to_relation = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.query_gate = nn.Linear(embed_dim * 2, embed_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.evidence_gate.weight)
        nn.init.constant_(self.evidence_gate.bias, -1.0)
        nn.init.zeros_(self.query_gate.weight)
        nn.init.constant_(self.query_gate.bias, -2.0)

    def forward(self, visual_queries, projected_visual_tokens, relation_tokens,
                negative_relation_tokens, subject_boxes, object_boxes,
                relation_confidences, relation_mask):
        valid = relation_mask.to(torch.bool)
        has_relations = valid.any(dim=1)
        if projected_visual_tokens.shape[1] != 197:
            raise ValueError('Pair grounding requires raw 197-token CLIP visual features.')
        patches = projected_visual_tokens[:, 1:]
        subject = _pool_patch_boxes(patches, subject_boxes, valid)
        obj = _pool_patch_boxes(patches, object_boxes, valid)
        union_boxes = torch.stack([
            torch.minimum(subject_boxes[..., 0], object_boxes[..., 0]),
            torch.minimum(subject_boxes[..., 1], object_boxes[..., 1]),
            torch.maximum(subject_boxes[..., 2], object_boxes[..., 2]),
            torch.maximum(subject_boxes[..., 3], object_boxes[..., 3]),
        ], dim=-1)
        union = _pool_patch_boxes(patches, union_boxes, valid)
        geometry = relative_geometry(subject_boxes, object_boxes).to(patches.dtype)
        pair_features = self.pair_projection(torch.cat([subject, obj, union, geometry], dim=-1))
        positive = self.relation_projection(relation_tokens)
        negative = self.relation_projection(negative_relation_tokens)
        confidence = relation_confidences.to(positive.dtype).unsqueeze(-1)
        evidence = torch.sigmoid(self.evidence_gate(
            torch.cat([pair_features, positive, confidence], dim=-1)))
        grounded = self.relation_norm(positive + evidence * pair_features)

        safe_valid = valid.clone()
        if (~has_relations).any():
            safe_valid[~has_relations, 0] = True
            grounded = grounded.clone()
            grounded[~has_relations, 0] = 0
        attended, _ = self.query_to_relation(
            visual_queries, grounded, grounded,
            key_padding_mask=~safe_valid, need_weights=False)
        gate = torch.sigmoid(self.query_gate(torch.cat([visual_queries, attended], dim=-1)))
        aligned = visual_queries + self.dropout(gate * attended)
        aligned = torch.where(has_relations[:, None, None], aligned, visual_queries)
        diagnostics = {
            'pair_features': pair_features,
            'positive_relation_features': positive,
            'negative_relation_features': negative,
            'mask': valid,
            'confidences': relation_confidences,
            'evidence_gate': evidence.squeeze(-1),
        }
        return aligned, diagnostics


def counterfactual_relation_loss(diagnostics, margin=0.2):
    """Rank the observed predicate above a hard counterfactual for the same pair."""
    valid = diagnostics['mask']
    if not valid.any():
        return diagnostics['pair_features'].sum() * 0
    pair = F.normalize(diagnostics['pair_features'][valid], dim=-1)
    positive = F.normalize(diagnostics['positive_relation_features'][valid], dim=-1)
    negative = F.normalize(diagnostics['negative_relation_features'][valid], dim=-1)
    positive_score = (pair * positive).sum(dim=-1)
    negative_score = (pair * negative).sum(dim=-1)
    weights = diagnostics['confidences'][valid].to(positive_score.dtype).clamp_min(0.05)
    losses = F.relu(margin - positive_score + negative_score)
    return (losses * weights).sum() / weights.sum()
