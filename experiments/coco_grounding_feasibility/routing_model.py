"""Training-only weak noun/patch loss; unchanged baseline inference and parameters."""
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from pilot_model import PilotModel
from route_probe import RouteProbe, route


def global_span_mean(local_sum, count):
    total = local_sum.detach().new_tensor(float(count))
    dist.all_reduce(total)
    # DDP averages gradients across ranks, so compensate once here.
    return local_sum * (dist.get_world_size() / total.clamp_min(1)), total


def auxiliary(parts, batch, arm):
    image, slot = batch['span_valid'].nonzero(as_tuple=True)
    positions = batch['span_positions'][image, slot]
    valid = batch['span_token_valid'][image, slot]
    caption = image * 5 + batch['span_caption'][image, slot]
    paths = parts['paths'][caption[:, None], positions]
    spatial = paths.sum(-1)
    bbox = (paths * batch['span_masks'][image, slot, None, :]).sum(-1)
    mass = spatial if arm == 'B' else bbox
    lengths = valid.sum(-1).clamp_min(1)
    per_span = (-(mass + 1e-8).log() * valid).sum(-1) / lengths
    local_sum = per_span.sum() + parts['paths'].sum() * 0
    loss, count = global_span_mean(local_sum, len(image))
    def span_sum(values):
        return ((values * valid).sum(-1) / lengths).sum().detach()
    stats = torch.stack([local_sum.detach(),
        span_sum(parts['decoder_q_mass'][caption[:, None], positions]),
        span_sum(spatial), span_sum(parts['cls_path_mass'][caption[:, None], positions]),
        span_sum(bbox), batch['span_person'][image, slot].sum().float()])
    dist.all_reduce(stats)
    means = (stats[:5] / count.clamp_min(1)).cpu().tolist()
    details = dict(zip(['aux_global_mean', 'decoder_q_mass', 'spatial_mass', 'cls_mass', 'bbox_mass'], means))
    details.update(local_spans=len(image), global_spans=int(count.item()),
                   global_person_spans=int(stats[5].item()),
                   local_image_ids=batch['image_id'].detach().cpu().tolist())
    return loss, details


class RoutingModel(PilotModel):
    def __init__(self, base, arm):
        nn.Module.__init__(self)
        if arm not in ('B', 'R'):
            raise ValueError(arm)
        self.base, self.arm, self.extra = base, arm, None
        self.details = {}

    def forward(self, batch, sample, active, advantage, scst_weight, pad_idx):
        from captioning.scst import policy_loss
        with RouteProbe(self.base) as probe:
            memory, padding = self.memory(batch)
            targets = batch['captions'].flatten(0, 1)
            masks = batch['caption_mask'].flatten(0, 1)
            logits = self.teacher(memory, padding, targets[:, :-1], masks[:, :-1])
            ce = F.cross_entropy(logits.flatten(0, 1), targets[:, 1:].reshape(-1), ignore_index=pad_idx)
            # Caption expansion is image-major: five references per image, never five image repeats.
            cw = probe.weights['qformer'].repeat_interleave(5, dim=0)
            dw = probe.weights['decoder']
            parts = route(dw, cw, batch['legacy'].shape[1], dw.new_ones((len(dw), 196)))
            aux, self.details = auxiliary(parts, batch, self.arm)
        # Remove observer before sampled teacher; only reference CE positions supervise routing.
        rl = ce.new_zeros(())
        if scst_weight:
            sample_logits = self.teacher(memory, padding, sample[:, :-1])
            rl = policy_loss(sample_logits, sample[:, 1:], active, advantage)
        return ce + scst_weight * rl + .05 * aux, ce.detach(), rl.detach()

    def training_diagnostics(self):
        return self.details
