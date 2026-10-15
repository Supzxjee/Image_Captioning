"""Instance-bound verification, selective prompts and causal decoder evidence routing.

Unknown training labels are not a third negative class. The third inference
decision is abstention, determined by calibration support and localization.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class EvidenceRelationVerifier(nn.Module):
    def __init__(self, dim=512, hidden=64):
        super().__init__()
        self.dim, self.hidden = dim, hidden
        self.pair = nn.Sequential(nn.Linear(5 * dim + 10, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.text = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.correction = nn.Linear(3 * hidden, 1)
        nn.init.zeros_(self.correction.weight)
        nn.init.zeros_(self.correction.bias)
        self.offset = nn.Parameter(torch.tensor(-3.0))

    def forward(self, views, geometry, text):
        if views.ndim != 3 or views.shape[1:] != (3, self.dim):
            raise ValueError('Expected ordered subject, object, union features.')
        if geometry.shape != (len(views), 10) or text.ndim != 3 or text.shape[0] != len(views) or text.shape[2] != self.dim:
            raise ValueError('Feature dimensions differ.')
        if not all(torch.isfinite(x).all() for x in (views, geometry, text)):
            raise ValueError('Nonfinite evidence.')
        a, b, union = F.normalize(views.detach(), dim=-1).unbind(1)
        target = F.normalize(text.detach(), dim=-1)
        pair = self.pair(torch.cat([a, b, union, a * b, a - b, geometry], -1))
        word = self.text(target)
        pair = pair[:, None].expand_as(word)
        logits = 10 * torch.einsum('bd,bkd->bk', union, target) + self.offset
        logits = logits + self.correction(torch.cat([pair, word, pair * word], -1)).squeeze(-1)
        # These features remain attached during verifier training; inference consumers
        # freeze the verifier before adapting the caption decoder.
        return logits, F.layer_norm(pair + word, (self.hidden,))


def masked_verification_loss(logits, labels):
    if labels.shape != logits.shape or not ((labels == -1) | (labels == 0) | (labels == 1)).all():
        raise ValueError('Labels must be -1 unknown, 0 contradicted, 1 supported.')
    known = labels >= 0
    if not known.any():
        raise ValueError('No known verification labels.')
    return F.binary_cross_entropy_with_logits(logits[known], labels[known].to(logits.dtype))


def fit_temperature(logits, labels):
    """Fit one bounded scalar on calibration images only; never on evaluation."""
    known = labels >= 0
    if not known.any() or set(labels[known].tolist()) != {0, 1}:
        raise ValueError('Temperature calibration needs both known outcomes.')
    logits, labels = logits[known].detach(), labels[known].float()
    candidates = torch.logspace(-1, 1, 81, device=logits.device)
    losses = torch.stack([F.binary_cross_entropy_with_logits(logits / t, labels) for t in candidates])
    return float(candidates[losses.argmin()])


def fit_selective_policy(probabilities, labels, precision=.9, min_known=5):
    """Choose observed high-precision tails on calibration, NOT a guarantee.

    No supported tail defaults to >1; no contradicted tail defaults to <0.
    Unknowns never count toward precision or minimum calibration support.
    """
    if not 0 < precision <= 1 or min_known < 1 or probabilities.shape != labels.shape:
        raise ValueError('Invalid calibration policy settings.')
    if probabilities.ndim != 2 or not torch.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError('Invalid probabilities.')
    if not ((labels == -1) | (labels == 0) | (labels == 1)).all():
        raise ValueError('Invalid labels.')
    positive, negative, support = [], [], []
    for c in range(labels.shape[1]):
        known = labels[:, c] >= 0
        p, y = probabilities[known, c], labels[known, c]
        hi, lo, hi_n, lo_n = 1.01, -.01, 0, 0
        if len(p) and (y == 1).any() and (y == 0).any():
            for threshold in sorted(set(p.tolist())):
                take = p >= threshold
                if int(take.sum()) >= min_known and int((y[take] == 1).sum()) / int(take.sum()) >= precision:
                    hi, hi_n = threshold, int(take.sum())
                    break
            for threshold in sorted(set(p.tolist()), reverse=True):
                take = p <= threshold
                if threshold < hi and int(take.sum()) >= min_known and int((y[take] == 0).sum()) / int(take.sum()) >= precision:
                    lo, lo_n = threshold, int(take.sum())
                    break
        positive.append(hi); negative.append(lo)
        support.append({'known': len(p), 'positive_tail_known': hi_n, 'negative_tail_known': lo_n})
    return {'positive': positive, 'negative': negative, 'calibration_support': support,
            'requested_known_precision': precision, 'minimum_known': min_known,
            'warning': 'Empirical calibration precision only; no statistical or out-of-distribution guarantee.'}


def selective_decisions(probabilities, valid, policy):
    if probabilities.shape != valid.shape or probabilities.ndim != 2:
        raise ValueError('Expected one localization flag per pair/predicate.')
    if not torch.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError('Invalid verification probabilities.')
    hi = probabilities.new_tensor(policy['positive'])
    lo = probabilities.new_tensor(policy['negative'])
    if hi.shape != (probabilities.shape[1],) or lo.shape != hi.shape or not torch.isfinite(hi).all() or not torch.isfinite(lo).all() or not (lo < hi).all():
        raise ValueError('Invalid selective thresholds.')
    result = torch.full_like(probabilities, -1, dtype=torch.long)
    result[valid.bool() & (probabilities <= lo)] = 0
    result[valid.bool() & (probabilities >= hi)] = 1
    return result


def verified_prompt_memory(rows, features, probabilities, decisions, policy, max_relations=3):
    """Serialize and pack exactly the same selected instance-bound claims.

    Labels/annotation IDs are deliberately not read by selection. Existing box
    hashes define identities. The cap priority is threshold margin, not a
    claim that probabilities are perfectly calibrated across predicates.
    """
    from .selected_relation_prompts import build_selected_prompts, entity_id
    from .supervised_relations import RELATIONS
    if probabilities.shape != decisions.shape or features.shape[:2] != probabilities.shape or len(rows) != len(features):
        raise ValueError('Prompt evidence coverage differs.')
    if probabilities.shape[1] != len(RELATIONS):
        raise ValueError('Unexpected predicate order.')
    scores = probabilities.detach().clone()
    scores[decisions != 1] = -2.
    prompts = build_selected_prompts([{**row, 'scores': scores[i].tolist()} for i, row in enumerate(rows)],
                                     policy['positive'], max_relations=max_relations)
    lookup = {}
    for i, row in enumerate(rows):
        s = entity_id(row['filename'], row['subject'], row['subject_box'])
        o = entity_id(row['filename'], row['object'], row['object_box'])
        for c, predicate in enumerate(RELATIONS):
            key = row['filename'], s, predicate, o
            if key in lookup:
                raise ValueError('Duplicate instance/predicate evidence.')
            lookup[key] = i, c
    memory = {}
    for filename, entry in prompts.items():
        indices = [lookup[(filename, r['subject_instance'], r['relation'], r['object_instance'])]
                   for r in entry['relations']]
        entry['verification'] = 'selective_evidence_v1; empirical calibration, not guaranteed correctness'
        for claim in entry['relations']:
            claim['verification_probability'] = claim['score']
            claim['verification_status'] = 'supported_prediction'
        f = torch.stack([features[i, c] for i, c in indices]) if indices else features.new_empty((0, features.shape[-1]))
        p = torch.stack([probabilities[i, c] for i, c in indices]) if indices else probabilities.new_empty((0,))
        memory[filename] = {'features': f.detach().unsqueeze(0), 'probabilities': p.detach().unsqueeze(0),
                            'accepted': torch.ones((1, len(indices)), dtype=torch.bool, device=features.device)}
    return prompts, memory


class StepwiseEvidenceRouter(nn.Module):
    """Zero-initialized residual: ordinary learned gate vs reliability-bound gate."""
    def __init__(self, model_dim=512, evidence_dim=64, use_reliability=True):
        super().__init__()
        self.use_reliability = use_reliability
        self.key = nn.Linear(evidence_dim, model_dim)
        self.value = nn.Linear(evidence_dim, model_dim)
        self.gate = nn.Linear(2 * model_dim, 1)
        self.output = nn.Linear(model_dim, model_dim, bias=False)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.gate.bias, -2.)

    def forward(self, hidden, evidence, probabilities, accepted):
        if hidden.ndim != 3 or evidence.ndim != 3 or evidence.shape[:2] != probabilities.shape or accepted.shape != probabilities.shape or hidden.shape[0] != evidence.shape[0]:
            raise ValueError('Evidence batch/slot dimensions differ.')
        if not torch.isfinite(hidden).all() or not torch.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
            raise ValueError('Invalid routing input.')
        # Verification is frozen: caption CE must not increase its own confidence.
        p = probabilities.detach()
        valid = accepted.bool() & (p > 0)
        if not torch.isfinite(evidence[valid]).all():
            raise ValueError('Nonfinite accepted evidence.')
        if evidence.shape[1] == 0:
            return hidden, hidden.new_zeros((*hidden.shape[:2], 1))
        safe = evidence.detach().masked_fill(~valid.unsqueeze(-1), 0)
        logits = torch.einsum('btd,brd->btr', hidden, self.key(safe)) / math.sqrt(hidden.shape[-1])
        # Finite sentinel plus explicit zero/renormalization handles all-masked rows.
        weights = logits.masked_fill(~valid[:, None], torch.finfo(logits.dtype).min).softmax(-1)
        weights = weights * valid[:, None]
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        context = weights @ self.value(safe)
        gate = self.gate(torch.cat([hidden, context], -1)).sigmoid()
        reliability = (weights * p[:, None]).sum(-1, keepdim=True)
        bound = reliability if self.use_reliability else valid.any(-1)[:, None, None].to(hidden.dtype)
        gate = gate * bound
        return hidden + self.output(context) * gate, gate


class VerifiedEvidenceCaptionDecoder(nn.Module):
    """Opt-in adapter around the current CaptionDecoder, with explicit evidence.

    Save this adapter separately. Old decoder checkpoints load into base unchanged.
    Merely training a verifier does NOT train these routing parameters.
    """
    def __init__(self, base, evidence_dim=64, use_reliability=True):
        super().__init__()
        self.base = base
        self.router = StepwiseEvidenceRouter(base.d_model, evidence_dim, use_reliability)

    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None,
                evidence=None, return_gate=False):
        if evidence is None:
            logits = self.base(tgt, memory, tgt_key_padding_mask, memory_key_padding_mask)
            return (logits, None) if return_gate else logits
        hidden = self.base.pos_encoder(self.base.embedding(tgt) * math.sqrt(self.base.d_model))
        hidden = self.base.transformer_decoder(tgt=hidden, memory=memory,
            tgt_mask=self.base.causal_mask(tgt.size(1), tgt.device),
            tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        hidden, gate = self.router(hidden, evidence['features'], evidence['probabilities'], evidence['accepted'])
        logits = self.base.fc_out(hidden)
        return (logits, gate) if return_gate else logits
