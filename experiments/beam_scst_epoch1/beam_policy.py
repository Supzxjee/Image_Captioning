"""USER-RUN differentiable beam histories. Importing this module needs no Torch."""


def train_beams(model, memory, width=5, maximum=20):
    import torch
    if width < 2 or maximum < 1:
        raise ValueError('Need at least two beams and one action')
    b, n, d = memory.shape
    vocab, device = model.head.out_features, memory.device
    if width > vocab-2:
        raise ValueError('Beam width exceeds valid vocabulary')
    seq = torch.ones((b, width, 1), dtype=torch.long, device=device)
    scores = torch.full((b, width), -torch.inf, device=device)
    scores[:, 0] = 0
    done = torch.zeros((b, width), dtype=torch.bool, device=device)
    history = memory.new_zeros((b, width, 0))
    expanded = memory[:, None].expand(-1, width, -1, -1).reshape(b*width, n, d)
    for _ in range(maximum):
        logits = model.decode(expanded, seq.reshape(b*width, -1))[:, -1].float()
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite beam logits')
        logp = logits.log_softmax(-1).view(b, width, vocab)
        forbidden = torch.arange(vocab, device=device) < 2
        logp = logp.masked_fill(forbidden, -torch.inf)
        finished = torch.full_like(logp, -torch.inf)
        finished[:, :, 0] = 0
        logp = torch.where(done[:, :, None], finished, logp)
        # Selection is discrete; selected path log probabilities retain their graph.
        scores, flat = (scores[:, :, None]+logp.detach()).flatten(1).topk(width, dim=1)
        parent, token = flat//vocab, flat % vocab
        chosen = logp.gather(1, parent[:, :, None].expand(-1, -1, vocab)).gather(2, token[:, :, None]).squeeze(-1)
        history = torch.cat((history.gather(1, parent[:, :, None].expand(-1, -1, history.shape[-1])), chosen[:, :, None]), -1)
        seq = torch.cat((seq.gather(1, parent[:, :, None].expand(-1, -1, seq.shape[-1])), token[:, :, None]), -1)
        done = done.gather(1, parent) | token.eq(2)
        if done.all():
            break
    if history.shape[-1] < maximum:
        padding = maximum-history.shape[-1]
        history = torch.cat((history, history.new_zeros((b, width, padding))), -1)
        seq = torch.cat((seq, seq.new_zeros((b, width, padding))), -1)
    if not torch.isfinite(history).all():
        raise FloatingPointError('Nonfinite selected beam log probabilities')
    return seq, history


def image_losses(logps, rewards):
    if logps.ndim != 3 or rewards.shape != logps.shape[:2]:
        raise ValueError('Expected image x beam x action and image x beam reward')
    advantage = rewards.detach()-rewards.detach().mean(-1, keepdim=True)
    return -(logps.mean(-1)*advantage).mean(-1)


def rank_microbatch(start, rank, micro, real_images=113287):
    """Two ranks x four images x four microbatches; dummy rows have weight zero."""
    if rank not in (0, 1) or micro not in range(4) or not 0 <= start < real_images:
        raise ValueError('Invalid epoch cursor/rank/microbatch')
    offsets = [start+micro*8+rank*4+i for i in range(4)]
    return [i if i < real_images else 0 for i in offsets], [i < real_images for i in offsets]


def mechanism_guard(device):
    """USER-RUN only: beam parent history, EOS and weighted-tail gradients."""
    import torch
    from patch_beam import beam_search
    class Toy(torch.nn.Module):
        def __init__(self, eos_first=False):
            super().__init__()
            self.head = torch.nn.Linear(3, 9)
            self.eos_first = eos_first
        def decode(self, memory, prefix):
            x = memory.mean(1)[:, None, :].expand(-1, prefix.shape[1], -1)
            # Deterministic prefix dependence tests parent-history gathering.
            x = x+prefix.float()[:, :, None]*.03
            logits = self.head(x)
            if self.eos_first:
                offset = torch.zeros_like(logits)
                offset[:, :, 2] = 50
                logits = logits+offset
            return logits
    for eos_first in (False, True):
        toy = Toy(eos_first).to(device).eval()
        memory = torch.randn(3, 4, 3, device=device)
        seq, logps = train_beams(toy, memory, 5, 6)
        with torch.no_grad():
            expected, score = beam_search(toy, memory, 5, 6)
        padded = torch.cat((expected, expected.new_zeros((3, 5, seq.shape[-1]-expected.shape[-1]))), -1)
        torch.testing.assert_close(seq, padded, rtol=0, atol=0)
        torch.testing.assert_close(logps.sum(-1), score, rtol=1e-5, atol=1e-5)
        after_eos = seq[:, :, 1:].eq(2).cumsum(-1)-seq[:, :, 1:].eq(2).long() > 0
        if torch.count_nonzero(logps.masked_select(after_eos)):
            raise ValueError('Post-EOS actions affect loss')
        reward = torch.arange(5, device=device).float().expand(3, -1)
        loss = image_losses(logps, reward).mean()
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in toy.parameters()):
            raise ValueError('Beam history lost its gradient')
    # Compare the actual weighted image loss gradients to an explicit formula.
    for count in (7, 32):
        lp = torch.randn(32, 5, 20, device=device, requires_grad=True)
        reward = torch.randn(32, 5, device=device)
        reference = image_losses(lp[:count], reward[:count]).mean()
        g = torch.autograd.grad(reference, lp, retain_graph=True)[0]
        distributed = lp.sum()*0
        for rank in (0, 1):
            for micro in range(4):
                rows, keep = rank_microbatch(0, rank, micro, count)
                weights = torch.tensor(keep, device=device, dtype=torch.float32)
                distributed = distributed+(image_losses(lp[rows], reward[rows])*weights).sum()/count
        actual = torch.autograd.grad(distributed, lp)[0]
        torch.testing.assert_close(actual, g, rtol=1e-6, atol=1e-7)
    return dict(beam_path_and_score=True, selected_history_autograd=True, EOS_mask=True,
                image_mean_reward_baseline=True, tail7_and_global32_weighted_gradient=True)
