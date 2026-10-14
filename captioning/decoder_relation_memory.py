"""Explicit relation slots for caption cross-attention; no invented text IDs."""
import torch


def append_relation_memory(memory, padding_mask, diagnostics, norm):
    """Append text + instance-pair features, independently of the visual-query gate.

    Confidence is a detector-derived weight, not a calibrated relation probability.
    Unsupported/zero-confidence slots are both zeroed and attention-masked.
    """
    confidence = diagnostics['confidences'].to(memory.dtype).clamp(0, 1)
    valid = diagnostics['mask'].bool() & (confidence > 0)
    pair = diagnostics['pair_features']
    text = diagnostics['positive_relation_features']
    tokens = confidence.unsqueeze(-1) * norm(text + confidence.unsqueeze(-1) * pair)
    tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0)
    return torch.cat([memory, tokens], dim=1), torch.cat([padding_mask, ~valid], dim=1)
