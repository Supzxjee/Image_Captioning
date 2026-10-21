"""Same caption path as A; auxiliary shared Q-Former/decoder, no new parameters."""
import torch
from torch.nn import functional as F
from hidden_model import HiddenModel


def masked_queries(encoder, visual, keep):
    """Exclude CLS and mask cross-attention keys at EVERY Q-Former layer."""
    if visual.shape[1:] != (197, 768) or keep.shape != visual[:, 1:, 0].shape:
        raise ValueError('Expected cached197 tokens and196 patch mask')
    if bool((keep.sum(1) < 4).any()):
        raise ValueError('Insufficient patch evidence; no nearest fallback')
    qformer = encoder.qformer
    if qformer is None or qformer.prompt_conditioned:
        raise ValueError('Expected unconditioned baseline Q-Former')
    tokens = encoder.dropout(encoder.relu(encoder.vis_projection(visual[:, 1:])))
    queries = qformer.query_tokens.expand(visual.size(0), -1, -1)
    # Same equations/weights as VisualQueryLayer; only retrieval adds a key mask.
    for layer in qformer.layers:
        update, _ = layer.self_attn(queries, queries, queries, need_weights=False)
        queries = layer.self_norm(queries + layer.dropout(update))
        update, _ = layer.cross_attn(queries, tokens, tokens,
                                     key_padding_mask=~keep, need_weights=False)
        queries = layer.cross_norm(queries + layer.dropout(update))
        queries = layer.ffn_norm(queries + layer.dropout(layer.ffn(queries)))
    return queries


class RegionCaptionModel(HiddenModel):
    def __init__(self, base, arm):
        super().__init__(base, 'A')
        if arm not in ('G', 'T'):
            raise ValueError(arm)
        self.arm, self.last_aux = arm, {}

    def phrase_loss(self, batch, pad_idx):
        keep = batch['aux_patch_mask']
        if self.arm == 'T':
            keep = torch.ones_like(keep)
        memory = masked_queries(self.base.encoder, batch['aux_visual'], keep)
        targets = batch['aux_caption']
        logits = self.decoder(targets[:, :-1], memory,
            tgt_key_padding_mask=targets[:, :-1].eq(pad_idx),
            memory_key_padding_mask=torch.zeros(memory.shape[:2], dtype=torch.bool, device=memory.device))
        # Mean over nonpadding target tokens, including EOS, just like caption CE.
        loss = F.cross_entropy(logits.flatten(0, 1), targets[:, 1:].reshape(-1), ignore_index=pad_idx)
        self.last_aux = dict(phrase_ce_loss=float(loss.detach()), phrase_weight=.1,
            aux_samples=targets.size(0), mean_aux_patches=float(keep.sum(1).float().mean()),
            mean_original_region_patches=float(batch['aux_patch_mask'].sum(1).float().mean()))
        return loss

    def training_diagnostics(self):
        return self.last_aux

    def forward(self, batch, sample, active, advantage, scst_weight, pad_idx):
        loss, ce, rl = super().forward(batch, sample, active, advantage, scst_weight, pad_idx)
        return loss + .1 * self.phrase_loss(batch, pad_idx), ce, rl
