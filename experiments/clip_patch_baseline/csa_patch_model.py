"""Paper-structure CSA encoder/parallel decoder; CLIP-patch + UPT extensions disclosed."""
import torch
from torch import nn
from bank_layers import CSACaptioner, Config
from semantic_patch_model import PromptMemory


class CSAPatchCaptioner(nn.Module):
    def __init__(self, vocab, nouns, verbs):
        super().__init__()
        self.core = CSACaptioner(Config(vocab=vocab), nouns, verbs)
        self.object_prompt = PromptMemory(512 + 512 + 4)
        self.relation_prompt = PromptMemory(512 + 512 + 512 + 8)

    @property
    def head(self):
        return self.core.head

    def streams(self, patches, p):
        if patches.ndim != 3 or patches.shape[1:] != (196, 768):
            raise ValueError('Expected196CLIPpatches, not ROI indices/features')
        valid = torch.ones(patches.shape[:2], dtype=torch.bool, device=patches.device)
        vo, vr = self.core.encode(patches, valid)
        obj = torch.cat((p['object_text'], torch.bmm(p['object_weights'], vo), p['object_box']), -1)
        rel = torch.cat((p['relation_text'], torch.bmm(p['subject_weights'], vo),
                         torch.bmm(p['target_weights'], vo), p['relation_geometry']), -1)
        return (self.object_prompt(vo, obj, p['object_weights'], p['object_valid']),
                self.relation_prompt(vr, rel, p['subject_weights'] + p['target_weights'], p['relation_valid']))

    def encode(self, patches, p):
        # Packaging for existing beam expansion only; decode restores both streams.
        return torch.cat(self.streams(patches, p), dim=1)

    def decode(self, memory, prefix):
        if memory.ndim != 3 or memory.shape[1:] != (392, 512):
            raise ValueError('Expected packaged pair of196x512memories')
        vo, vr = memory.split(196, dim=1)
        valid = torch.ones(vo.shape[:2], dtype=torch.bool, device=vo.device)
        return self.core.decode(prefix, vo, vr, valid)

    def forward(self, patches, prefix, p):
        return self.decode(self.encode(patches, p), prefix)
