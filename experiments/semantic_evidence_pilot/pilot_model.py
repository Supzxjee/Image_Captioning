"""Text queries image values; no text residual in visual evidence or matching score."""
import torch
from torch import nn
from torch.nn import functional as F


class EvidenceMemory(nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.query = nn.Linear(512, dim)
        self.role = nn.Embedding(3, dim)
        self.key = nn.Linear(768, dim)
        self.value = nn.Linear(768, dim)
        self.position = nn.Linear(4, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        axis = (torch.arange(14, dtype=torch.float32) + .5) / 14
        y, x = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('xy', torch.stack((x.flatten(), y.flatten()), -1))

    def attend(self, patches, text, roles, bounds):
        # bounds: subject, object, allowed box; values contain image features only.
        xy = self.xy[None].expand(len(text), -1, -1)
        inside = (xy[:, None] >= bounds[:, :, None, :2]).all(-1) & (xy[:, None] <= bounds[:, :, None, 2:]).all(-1)
        allowed = inside[:, 2].clone()
        empty = ~allowed.any(-1)
        center = (bounds[:, 2, :2] + bounds[:, 2, 2:]) / 2
        nearest = ((xy-center[:, None])**2).sum(-1).argmin(-1)
        allowed[torch.arange(len(text), device=text.device)[empty], nearest[empty]] = True
        spatial = torch.cat((xy, inside[:, :2].transpose(1, 2).float()), -1)
        q = self.query(text) + self.role(roles)
        keys = self.key(patches) + self.position(spatial)
        weights = ((keys*q[:, None]).sum(-1) / q.shape[-1]**.5).masked_fill(~allowed, -torch.inf).softmax(-1)
        evidence = self.norm((weights[:, :, None]*self.value(patches)).sum(1))
        score = (F.normalize(q, dim=-1)*F.normalize(evidence, dim=-1)).sum(-1)
        return evidence, score, weights

    def forward(self, patches, text, roles, valid, bounds):
        shape = valid.shape
        tokens, scores, attention = self.attend(patches.flatten(0,1), text.flatten(0,1), roles.flatten(), bounds.flatten(0,1))
        return tokens.reshape(*shape, -1).masked_fill(~valid[..., None], 0), scores.reshape(shape), attention.reshape(*shape,196)

    def alignment(self, patches, text, roles, valid, bounds, phrase_ids, image_ids, positive_scores):
        """Local-rank pseudo-pair hinge loss; uncertain negatives are a stated limit."""
        n, slots = valid.shape
        words, role, ok = phrase_ids.flatten(), roles.flatten(), valid.flatten()
        owner = torch.arange(n, device=text.device).repeat_interleave(slots)
        # Exclude every exact phrase known anywhere in the target image, not just its positive slot.
        known = ((words[None, :, None] == phrase_ids[:, None, :]) & valid[:, None, :]).any(-1)
        eligible = ok[:, None] & ok[None, :] & (role[:, None] == role[None, :])
        eligible &= image_ids[owner, None] != image_ids[owner][None, :]
        eligible &= ~known[owner] & ~known[owner].T
        has = eligible.any(-1)
        donor = eligible.long().argmax(-1)
        index = torch.nonzero(has, as_tuple=False).flatten()
        if not len(index):
            return positive_scores.sum()*0, dict(pairs=0, accuracy=None, margin=None)
        donor = donor[index]
        p, t, r, b = patches.flatten(0,1), text.flatten(0,1), roles.flatten(), bounds.flatten(0,1)
        _, wrong_text, _ = self.attend(p[index], t[donor], r[index], b[index])
        _, wrong_region, _ = self.attend(p[donor], t[index], r[index], b[donor])
        positive = positive_scores.flatten()[index]
        losses = (F.relu(.2-positive+wrong_text)+F.relu(.2-positive+wrong_region)) / 2
        margin = torch.minimum(positive-wrong_text, positive-wrong_region)
        return losses.mean(), dict(pairs=len(index), accuracy=float((margin>0).float().mean().detach()),
                                   margin=float(margin.mean().detach()))


class PilotModel(nn.Module):
    def __init__(self, base, arm):
        super().__init__()
        if arm not in ('A','E'): raise ValueError(arm)
        self.base, self.arm = base, arm
        self.extra = EvidenceMemory() if arm=='E' else None
        self.last_gates = None
        self.last_alignment = dict(loss=0., pairs=0, accuracy=None, margin=None)

    @property
    def decoder(self): return self.base.decoder

    def memory(self, batch, intervention='on'):
        memory, padding = self.base.build_memory(batch['visual'], batch['legacy'], batch['legacy_mask'])
        if self.extra is None: return memory, padding
        shape = batch['valid'].shape
        patches = batch['visual'][:, None, 1:].expand(-1, shape[1], -1, -1)
        text, bounds = batch['text'], batch['bounds']
        if intervention=='text_swap': text=batch['swap_text']
        elif intervention=='region_swap':
            joint = batch['swap_valid']
            patches = torch.where(joint[...,None,None], batch['swap_visual'][:,None,1:].expand_as(patches), patches)
            bounds = torch.where(joint[...,None,None], batch['swap_bounds'], bounds)
        elif intervention not in ('on','off'): raise ValueError(intervention)
        tokens, scores, attention = self.extra(patches, text, batch['roles'], batch['valid'], bounds)
        self.last_gates = torch.ones_like(scores)
        self.last_attention = attention.detach()
        self._evidence_inputs = (patches, text, batch['roles'], batch['valid'], bounds, scores)
        extra_padding = ~batch['valid']
        if intervention=='off':
            tokens=torch.zeros_like(tokens);extra_padding=torch.ones_like(extra_padding)
        return torch.cat((memory,tokens),1), torch.cat((padding,extra_padding),1)

    def teacher(self, memory, padding, targets, masks=None):
        b, slots, dim = memory.shape
        repeats = targets.size(0)//b
        if repeats*b != targets.size(0): raise ValueError('Caption/image batch mismatch')
        m=memory[:,None].expand(b,repeats,slots,dim).reshape(-1,slots,dim)
        p=padding[:,None].expand(b,repeats,slots).reshape(-1,slots)
        return self.decoder(targets,m,tgt_key_padding_mask=masks,memory_key_padding_mask=p)

    def forward(self, batch, sample, active, advantage, scst_weight, pad_idx):
        from captioning.scst import policy_loss
        memory,padding=self.memory(batch)
        targets=batch['captions'].flatten(0,1); masks=batch['caption_mask'].flatten(0,1)
        logits=self.teacher(memory,padding,targets[:,:-1],masks[:,:-1])
        ce=F.cross_entropy(logits.flatten(0,1),targets[:,1:].reshape(-1),ignore_index=pad_idx)
        rl=ce.new_zeros(())
        if scst_weight:
            rl=policy_loss(self.teacher(memory,padding,sample[:,:-1]),sample[:,1:],active,advantage)
        alignment=ce.new_zeros(())
        info=dict(pairs=0,accuracy=None,margin=None)
        if self.extra is not None:
            p,t,r,v,b,s=self._evidence_inputs
            alignment,info=self.extra.alignment(p,t,r,v,b,batch['phrase_ids'],batch['image_id'],s)
        self.last_alignment=dict(loss=float(alignment.detach()),**info)
        return ce+scst_weight*rl+.1*alignment, ce.detach(), rl.detach()
