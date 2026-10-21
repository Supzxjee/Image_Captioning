"""Keep baseline memory, append Vo/Vr from its 32 Q-Former queries."""
import torch
from torch import nn
from pilot_model import PilotModel


class CSAModel(PilotModel):
    def __init__(self, base, core):
        nn.Module.__init__(self)
        self.base, self.extra = base, core

    def memory(self, batch, intervention='on'):
        if intervention not in ('on', 'off'):
            raise ValueError(intervention)
        memory, padding = self.base.build_memory(batch['visual'], batch['legacy'], batch['legacy_mask'])
        if intervention == 'off':
            return memory, padding
        if memory.shape[1] != batch['legacy'].shape[1] + 32 or bool(padding[:, -32:].any()):
            raise ValueError('Expected legacy followed by 32 valid baseline Q-Former queries.')
        objects, relations, _author_mask = self.extra(memory[:, -32:])
        valid = torch.zeros(memory.shape[0], 64, device=memory.device, dtype=torch.bool)
        return torch.cat((memory, objects, relations), 1), torch.cat((padding, valid), 1)
