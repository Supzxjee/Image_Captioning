"""USER-RUN full XE reader, with shared image memories and explicit prompt inputs."""
import hashlib
import torch
from torch import nn
from prompt_adapter_model import PromptCSA


class ScratchCaptioner(nn.Module):
    def __init__(self,base,enabled):
        super().__init__();self.model=PromptCSA(base);self.enabled=enabled
        if not enabled:
            self.model.object_prompt.requires_grad_(False);self.model.relation_prompt.requires_grad_(False)

    @property
    def c(self):return self.model.base.c
    @property
    def nouns(self):return self.model.base.nouns
    @property
    def verbs(self):return self.model.base.verbs

    def encode(self,x,valid,prompts):return self.model.encode(x,valid,prompts,enabled=self.enabled)
    def decode(self,ids,vo,vr,valid):return self.model.decode(ids,vo,vr,valid)

    def forward(self,x,valid,ids,prompts,inverse=None):
        vo,vr=self.encode(x,valid,prompts)
        if inverse is not None:
            vo=vo.index_select(0,inverse);vr=vr.index_select(0,inverse);valid=valid.index_select(0,inverse)
        return self.decode(ids,vo,vr,valid)

    def common_digest(self):
        h=hashlib.sha256()
        for name,value in sorted(self.model.base.state_dict().items()):
            h.update(name.encode());h.update(str(tuple(value.shape)).encode());h.update(str(value.dtype).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()


def to_device(values,device):
    x,v,seq,p,inverse=values
    return [torch.from_numpy(a).to(device) for a in (x,v,seq)]+[{k:torch.from_numpy(a).to(device) for k,a in p.items()},None if inverse is None else torch.from_numpy(inverse).to(device)]


def beam_preflight(model,x,valid,prompts):
    from model import beam_search
    from full_xe_decode import check_beam_rows
    model.eval()
    with torch.inference_mode():
        rows=beam_search(model,*model.encode(x,valid,prompts),valid,beams=5,maximum=20)
        check_beam_rows([(seq.cpu().tolist(),scores.cpu().tolist()) for seq,scores in rows],len(x),model.c.vocab)
    return dict(images=len(x),beams=5,maximum=20,passed=True,decoder='original per-image beam',prompt_enabled=model.enabled)
