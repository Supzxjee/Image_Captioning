"""UPT pair-vector projection into the unchanged caption memory. No verb text/gate."""
import torch
from torch import nn
from pilot_model import PilotModel


class HiddenModel(PilotModel):
    def __init__(self, base, arm):
        nn.Module.__init__(self)
        if arm not in ('A', 'H'):
            raise ValueError(arm)
        self.base, self.arm = base, arm
        self.extra = nn.Sequential(nn.Linear(1024, 512), nn.LayerNorm(512)) if arm == 'H' else None

    def memory(self, batch, intervention='on'):
        memory, padding = self.base.build_memory(batch['visual'], batch['legacy'], batch['legacy_mask'])
        if intervention not in ('on', 'off', 'pair_swap'):
            raise ValueError(intervention)
        if self.extra is None or intervention == 'off':
            return memory, padding
        features, valid = ((batch['swap_hidden'], batch['swap_valid']) if intervention == 'pair_swap'
                           else (batch['hidden'], batch['valid']))
        if not bool(valid.any()):
            return memory, padding
        tokens = self.extra(features).masked_fill(~valid[..., None], 0)
        return torch.cat((memory, tokens), 1), torch.cat((padding, ~valid), 1)
