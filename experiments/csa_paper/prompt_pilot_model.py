"""USER-RUN neural glue; no change to CSA banks, decoder or SCST action policy."""
import torch
from torch import nn


class PromptPolicyReader(nn.Module):
    def __init__(self,model,enabled):
        super().__init__();self.model=model;self.enabled=enabled

    def forward(self,x,valid,prompts,sampled_prefix,ce_prefix):
        vo,vr=self.model.encode(x,valid,prompts,enabled=self.enabled)
        return self.model.decode(sampled_prefix,vo,vr,valid),self.model.decode(ce_prefix,vo,vr,valid)


def configure(base,enabled):
    from prompt_adapter_model import PromptCSA
    model=PromptCSA(base).eval()
    if not enabled:
        model.object_prompt.requires_grad_(False);model.relation_prompt.requires_grad_(False)
    return model


def to_device(batch,device):
    arrays,prompts,ids=batch
    return [torch.from_numpy(a).to(device) for a in arrays],{k:torch.from_numpy(v).to(device) for k,v in prompts.items()},ids


def reseed(rank):
    import random
    import numpy as np
    random.seed(1234+rank);np.random.seed(1234+rank);torch.manual_seed(1234+rank);torch.cuda.manual_seed(1234+rank)
