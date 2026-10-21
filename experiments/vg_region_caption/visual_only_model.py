"""GV: same regional phrase likelihood as G, without auxiliary decoder gradients."""
import torch
from torch.func import functional_call
from torch.nn import functional as F
from region_model import RegionCaptionModel,masked_queries


class VisualOnlyModel(RegionCaptionModel):
    def __init__(self,base):
        super().__init__(base,'G')
        self.arm='GV'

    def phrase_loss(self,batch,pad_idx):
        keep=batch['aux_patch_mask']
        memory=masked_queries(self.base.encoder,batch['aux_visual'],keep)
        targets=batch['aux_caption']
        # Read CURRENT decoder values every forward; only their parameter gradients stop.
        params={name:value.detach() for name,value in self.decoder.named_parameters()}
        buffers=dict(self.decoder.named_buffers())
        logits=functional_call(self.decoder,(params,buffers),(targets[:,:-1],memory),
            dict(tgt_key_padding_mask=targets[:,:-1].eq(pad_idx),
                 memory_key_padding_mask=torch.zeros(memory.shape[:2],dtype=torch.bool,device=memory.device)),
            strict=True)
        loss=F.cross_entropy(logits.flatten(0,1),targets[:,1:].reshape(-1),ignore_index=pad_idx)
        self.last_aux=dict(phrase_ce_loss=float(loss.detach()),phrase_weight=.1,
            aux_samples=targets.size(0),mean_aux_patches=float(keep.sum(1).float().mean()),
            mean_original_region_patches=float(keep.sum(1).float().mean()),aux_decoder_gradient_enabled=False)
        return loss
