"""Instance-pair relation grounding for CLIP patch features."""
from pathlib import PurePosixPath

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTION_PREDICATES = frozenset({'holding', 'wearing', 'carrying', 'riding'})


def action_relation_mask(entry):
    """Mark heuristic action labels eligible for appearance-only alignment."""
    mask = torch.zeros_like(entry['mask'], dtype=torch.bool)
    for index, relation in enumerate(entry.get('relations', [])):
        if index < mask.numel() and relation.get('relation') in ACTION_PREDICATES:
            mask[index] = bool(entry['mask'][index])
    return mask


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
    def __init__(self, embed_dim, num_heads, dropout=0.1, relation_dim=512,
                 grounding_version='v1', verified_scale=0.1):
        super().__init__()
        if grounding_version not in {'v1', 'appearance_v2', 'verified_v3'}:
            raise ValueError('Unknown grounded relation version.')
        self.grounding_version = grounding_version
        self.verified_scale = verified_scale
        # This scorer never sees relative geometry or trainable text projections.
        self.appearance_projection = (nn.Sequential(
            nn.Linear(embed_dim * 3, relation_dim), nn.GELU(), nn.LayerNorm(relation_dim)
        ) if grounding_version in {'appearance_v2', 'verified_v3'} else None)
        self.verified_visual_projection = (nn.Sequential(
            nn.Linear(relation_dim, embed_dim), nn.LayerNorm(embed_dim)
        ) if grounding_version == 'verified_v3' else None)
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
        if grounding_version == 'verified_v3':
            for component in (self.appearance_projection, self.pair_projection,
                              self.evidence_gate, self.query_gate):
                component.requires_grad_(False)

    def _verified_forward(self, visual_queries, subject, obj, union, relation_tokens,
                          negative_tokens, confidences, valid):
        appearance = self.appearance_projection(torch.cat([subject, obj, union], dim=-1))
        pair = F.normalize(appearance, dim=-1)
        positive = F.normalize(relation_tokens.detach(), dim=-1)
        negative = F.normalize(negative_tokens.detach(), dim=-1)
        positive_score = (pair * positive).sum(-1)
        gap = positive_score - (pair * negative).sum(-1)
        required_margin = (0.5 * (positive - negative).norm(dim=-1)).clamp_max(0.2)
        accepted = valid & (confidences > 0) & (positive_score > 0) & (gap >= required_margin)
        # A verification weight, not a calibrated probability or learned query gate.
        weights = (confidences.clamp(0, 1) * torch.sigmoid(positive_score / 0.1) *
                   torch.sigmoid((gap - required_margin) / 0.1)) * accepted
        grounded = self.relation_norm(
            self.relation_projection(relation_tokens) + self.verified_visual_projection(appearance))
        has_accepted = accepted.any(dim=1)
        safe = accepted.clone()
        safe[~has_accepted, 0] = True
        values = grounded * weights.unsqueeze(-1)
        attended, _ = self.query_to_relation(
            visual_queries, grounded, values, key_padding_mask=~safe, need_weights=False)
        row_weight = weights.sum(-1) / accepted.sum(-1).clamp_min(1)
        aligned = visual_queries + self.dropout(
            self.verified_scale * row_weight[:, None, None] * attended)
        aligned = torch.where(has_accepted[:, None, None], aligned, visual_queries)
        return aligned, {
            'appearance_features': appearance,
            'fixed_positive_features': relation_tokens.detach(),
            'fixed_negative_features': negative_tokens.detach(),
            'mask': valid, 'confidences': confidences,
            'evidence_gate': weights, 'accepted_mask': accepted,
            'verification_weights': weights,
        }

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
        if self.grounding_version == 'verified_v3':
            return self._verified_forward(visual_queries, subject, obj, union, relation_tokens,
                                          negative_relation_tokens, relation_confidences, valid)
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
        if self.appearance_projection is not None:
            diagnostics['appearance_features'] = self.appearance_projection(
                torch.cat([subject, obj, union], dim=-1))
            diagnostics['fixed_positive_features'] = relation_tokens.detach()
            diagnostics['fixed_negative_features'] = negative_relation_tokens.detach()
            if patches.size(0) > 1:
                # Use a different local-batch image but retain original boxes.
                # Loss filters cases with similar annotated relation phrases.
                other_patches = patches.roll(1, dims=0)
                other_subject = _pool_patch_boxes(other_patches, subject_boxes, valid)
                other_object = _pool_patch_boxes(other_patches, object_boxes, valid)
                other_union = _pool_patch_boxes(other_patches, union_boxes, valid)
                diagnostics['wrong_image_features'] = self.appearance_projection(
                    torch.cat([other_subject, other_object, other_union], dim=-1))
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


def appearance_relation_loss(diagnostics, action_mask, margin=0.2,
                             wrong_image_weight=0.5, duplicate_threshold=0.9):
    """Action-only predicate ranking plus a conservative cross-image ranking term.

    Text targets are fixed cached CLIP EOS vectors. Cross-image negatives are weak:
    missing annotations do not prove the relation is absent. Exclude other images
    containing near-identical positive phrases to reduce false negatives.
    """
    features = diagnostics['appearance_features']
    valid = diagnostics['mask'].bool() & action_mask.bool()
    zero = features.sum() * 0
    if not valid.any():
        return zero, {'predicate_loss': zero, 'wrong_image_loss': zero,
                      'actions': 0, 'eligible_wrong_images': 0}
    positive = F.normalize(diagnostics['fixed_positive_features'].detach(), dim=-1)
    negative = F.normalize(diagnostics['fixed_negative_features'].detach(), dim=-1)
    appearance = F.normalize(features, dim=-1)
    positive_score = (appearance * positive).sum(-1)
    negative_score = (appearance * negative).sum(-1)
    weights = diagnostics['confidences'].to(features.dtype).clamp_min(0.05)
    # Similar CLIP phrases have a small maximum attainable cosine-score gap.
    # Cap the requested gap at half that bound rather than impose an impossible loss.
    predicate_margin = torch.minimum(
        torch.full_like(positive_score, margin), 0.5 * (positive - negative).norm(dim=-1))
    predicate = (F.relu(predicate_margin - positive_score + negative_score)[valid] *
                 weights[valid]).sum() / weights[valid].sum()
    wrong_loss, eligible_count = zero, 0
    if 'wrong_image_features' in diagnostics and wrong_image_weight > 0:
        other_text = positive.roll(1, dims=0)
        other_valid = diagnostics['mask'].bool().roll(1, dims=0)
        similarities = torch.einsum('brd,bsd->brs', positive, other_text)
        duplicates = ((similarities >= duplicate_threshold) & other_valid[:, None, :]).any(-1)
        eligible = valid & ~duplicates
        eligible_count = int(eligible.sum().item())
        if eligible.any():
            wrong_features = F.normalize(diagnostics['wrong_image_features'], dim=-1)
            wrong_score = (wrong_features * positive).sum(-1)
            wrong_loss = (F.relu(margin - positive_score + wrong_score)[eligible] *
                          weights[eligible]).sum() / weights[eligible].sum()
    return predicate + wrong_image_weight * wrong_loss, {
        'predicate_loss': predicate, 'wrong_image_loss': wrong_loss,
        'actions': int(valid.sum().item()), 'eligible_wrong_images': eligible_count,
    }
