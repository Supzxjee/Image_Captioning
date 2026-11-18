"""User-run KL regularizer and tensor preflight. Do not import locally."""
import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F


def reference_kl(logits, reference_logits, mask):
    """Sequence-sum KL(current||XE), image mean; PAD/BOS excluded, EOS included."""
    if logits.shape != reference_logits.shape or logits.ndim != 3 or logits.shape[-1] < 3:
        raise ValueError('KL logit shapes differ')
    if mask.shape != logits.shape[:2] or mask.dtype != torch.bool:
        raise ValueError('Wrong KL action mask')
    current = logits[..., 2:].float()
    reference = reference_logits.detach()[..., 2:].float()
    if not torch.isfinite(current).all() or not torch.isfinite(reference).all():
        raise ValueError('Nonfinite supported KL logits')
    logp = F.log_softmax(current, dim=-1)
    logq = F.log_softmax(reference, dim=-1)
    tokens = (logp.exp() * (logp-logq)).sum(-1)
    return tokens.masked_fill(~mask, 0).sum(1).mean()


def smoke(output):
    from scst_pilot_model import policy_loss
    if not torch.cuda.is_available():
        raise RuntimeError('KL neural preflight needs CUDA')
    torch.manual_seed(20261119)
    device = torch.device('cuda', 0)
    logits = torch.randn(2, 3, 7, device=device, requires_grad=True)
    reference = logits.detach().clone().requires_grad_(True)
    mask = torch.tensor([[True, True, False], [True, False, False]], device=device)
    actions = torch.tensor([[2, 4, 0], [3, 0, 0]], device=device)
    advantage = torch.tensor([.3, -.2], device=device)
    kl = reference_kl(logits, reference, mask)
    if abs(float(kl)) > 1e-6:
        raise ValueError('Initial equal-policy KL is not zero')
    rl = policy_loss(logits, actions, mask, advantage)
    baseline_grad, = torch.autograd.grad(rl, logits, retain_graph=True)
    beta0_grad, = torch.autograd.grad(rl+0*kl, logits, retain_graph=True)
    torch.testing.assert_close(beta0_grad, baseline_grad, rtol=1e-6, atol=1e-7)
    initial_grad, = torch.autograd.grad(kl, logits)
    torch.testing.assert_close(initial_grad, torch.zeros_like(initial_grad), rtol=0, atol=1e-6)
    changed = logits.detach().clone()
    changed[..., 3] += 1
    changed.requires_grad_(True)
    shifted = reference_kl(changed, reference, mask)
    shifted.backward()
    if not torch.isfinite(shifted) or float(shifted) <= 0 or not torch.isfinite(changed.grad).all():
        raise ValueError('Shifted KL/gradient invalid')
    if reference.grad is not None or not torch.any(changed.grad):
        raise ValueError('KL must update current policy only')
    if changed.grad[~mask].abs().max() != 0 or changed.grad[..., :2].abs().max() != 0:
        raise ValueError('Padding/PAD/BOS received KL gradient')
    ignored = changed.detach().clone()
    ignored[..., :2] = float('-inf')
    torch.testing.assert_close(reference_kl(ignored, reference, mask), shifted.detach())
    if float(reference_kl(ignored, reference, torch.zeros_like(mask))) != 0:
        raise ValueError('Empty action mask should give zero KL')
    report = dict(equal_policy_zero=True, beta0_gradient_parity=True,
                  teacher_detached=True, shifted_policy_positive=True,
                  padding_and_special_logits_ignored=True, finite_gradient=True,
                  torch_version=str(torch.__version__), GPU=torch.cuda.get_device_name(0),
                  captioner_executed=False)
    Path(output).write_text(json.dumps(report, indent=2))
    print('KL tensor/CUDA preflight PASS', report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    smoke(parser.parse_args().output)
