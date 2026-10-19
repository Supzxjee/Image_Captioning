"""One optional region-memory branch. Gates learn caption utility, not verified truth."""
import torch
from torch import nn
from torch.nn import functional as F

from pilot_data import INPUT_DIM


class RegionMemory(nn.Module):
    def __init__(self, arm, dim=512):
        super().__init__()
        if arm not in 'BCD' or len(arm) != 1:
            raise ValueError('Expected B, C or D.')
        self.arm = arm
        self.visual = nn.Sequential(nn.Linear(INPUT_DIM, dim), nn.GELU(), nn.LayerNorm(dim))
        self.role = nn.Embedding(3, dim)
        if arm in 'CD':
            self.text = nn.Sequential(nn.Linear(512, dim), nn.LayerNorm(dim))
        if arm == 'D':
            self.gates = nn.ModuleList([nn.Sequential(nn.Linear(dim * 3, 128), nn.GELU(), nn.Linear(128, 1))
                                        for _ in range(3)])
            for gate in self.gates:
                nn.init.zeros_(gate[-1].weight)
                nn.init.zeros_(gate[-1].bias)
        self.norm = nn.LayerNorm(dim)

    def forward(self, regions, text, roles, valid):
        visual = self.visual(regions) + self.role(roles)
        gates = torch.ones_like(valid, dtype=visual.dtype)
        if self.arm == 'B':
            tokens = self.norm(visual)
        elif self.arm == 'C':
            tokens = self.norm(visual + self.text(text))
        else:
            semantic = self.text(text)
            joint = torch.cat([visual, semantic, visual * semantic], dim=-1)
            scores = torch.cat([gate(joint) for gate in self.gates], dim=-1)
            gates = torch.sigmoid(scores.gather(-1, roles.unsqueeze(-1)).squeeze(-1))
            # Keep the gate AFTER LayerNorm; normalizing g*v would remove scalar gating.
            tokens = self.norm(visual) * gates.unsqueeze(-1)
        return tokens.masked_fill(~valid.unsqueeze(-1), 0), gates


class PilotModel(nn.Module):
    def __init__(self, base, arm):
        super().__init__()
        self.base = base
        self.arm = arm
        self.extra = RegionMemory(arm) if arm != 'A' else None
        self.last_gates = None

    @property
    def decoder(self):
        return self.base.decoder

    def memory(self, batch, intervention='on'):
        memory, padding = self.base.build_memory(batch['visual'], batch['legacy'], batch['legacy_mask'])
        if self.extra is None:
            return memory, padding
        text, regions = batch['text'], batch['regions']
        if intervention == 'text_swap':
            text = batch['swap_text']
        elif intervention == 'region_swap':
            regions = batch['swap_regions']
        elif intervention not in ('on', 'off'):
            raise ValueError(intervention)
        tokens, gates = self.extra(regions, text, batch['roles'], batch['valid'])
        self.last_gates = gates.detach()
        padding_extra = ~batch['valid']
        if intervention == 'off':
            tokens = torch.zeros_like(tokens)
            padding_extra = torch.ones_like(padding_extra)
        return torch.cat([memory, tokens], dim=1), torch.cat([padding, padding_extra], dim=1)

    def teacher(self, memory, padding, targets, masks=None):
        b, slots, dim = memory.shape
        repeats = targets.size(0) // b
        if targets.size(0) != b * repeats:
            raise ValueError('Caption and image batch sizes differ.')
        expanded = memory[:, None].expand(b, repeats, slots, dim).reshape(b * repeats, slots, dim)
        expanded_pad = padding[:, None].expand(b, repeats, slots).reshape(b * repeats, slots)
        return self.decoder(targets, expanded, tgt_key_padding_mask=masks, memory_key_padding_mask=expanded_pad)

    def forward(self, batch, sample, active, advantage, scst_weight, pad_idx):
        from captioning.scst import policy_loss
        memory, padding = self.memory(batch)
        targets = batch['captions'].flatten(0, 1)
        masks = batch['caption_mask'].flatten(0, 1)
        logits = self.teacher(memory, padding, targets[:, :-1], masks[:, :-1])
        ce = F.cross_entropy(logits.flatten(0, 1), targets[:, 1:].reshape(-1), ignore_index=pad_idx)
        rl = ce.new_zeros(())
        if scst_weight:
            sample_logits = self.teacher(memory, padding, sample[:, :-1])
            rl = policy_loss(sample_logits, sample[:, 1:], active, advantage)
        loss = ce + scst_weight * rl
        return loss, ce.detach(), rl.detach()
