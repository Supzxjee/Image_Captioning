"""USER-RUN ONLY: tensor/gradient tests and synthetic production-size smoke."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import traceback

import torch
from torch.nn import functional as F

from model import CSACaptioner, Config, beam_mean_loss, beam_search, xe_loss


def close(a, b):
    torch.testing.assert_close(a, b, atol=2e-5, rtol=1e-4)


def check(device):
    torch.manual_seed(42)
    c = Config(vocab=13, dim=16, heads=4, hidden=32, prototypes=5, layers=2, dropout=0, raw_dim=8)
    model = CSACaptioner(c, F.normalize(torch.randn(9, 16), dim=-1),
                         F.normalize(torch.randn(7, 16), dim=-1)).to(device).eval()
    regions = torch.randn(2, 3, 8, device=device)
    valid = torch.tensor([[True, True, False], [True, True, True]], device=device)
    prefix = torch.tensor([[1, 4, 5, 6], [1, 7, 8, 9]], device=device)
    with torch.no_grad():
        base = model(regions, valid, prefix)
        _, weights = model.noun_agg(model.nouns)
        close(weights.sum(0), torch.ones(c.prototypes, device=device))
        p, _ = model.noun_agg(model.nouns)
        selected, spatial = model.object1.align(model.project(regions), p, valid)
        close(spatial.sum(-1), torch.ones_like(spatial.sum(-1)))
        assert (spatial[0, :, 2] == 0).all()
        assert selected.shape == (2, c.prototypes, c.dim)
        extra = torch.randn(2, 2, 8, device=device)
        padded = model(torch.cat((regions, extra), 1),
                       torch.cat((valid, torch.zeros(2, 2, dtype=torch.bool, device=device)), 1), prefix)
        close(base, padded)
        order = torch.tensor([2, 0, 1], device=device)
        close(base, model(regions[:, order], valid[:, order], prefix))
        changed = prefix.clone(); changed[:, 2:] = 10
        close(base[:, :2], model(regions, valid, changed)[:, :2])
        for t in range(1, prefix.shape[1] + 1):
            close(base[:, t-1], model(regions, valid, prefix[:, :t])[:, -1])
        # Batched/single-image execution may select different floating-point kernels.
        for i in range(2):
            close(base[i:i+1], model(regions[i:i+1], valid[i:i+1], prefix[i:i+1]))
        vo, vr = model.encode(regions, valid)
        beams = beam_search(model, vo, vr, valid, beams=1, maximum=4)
        for i, (seq, _) in enumerate(beams):
            greedy = torch.ones(1, 1, dtype=torch.long, device=device)
            for _ in range(4):
                logits = model.decode(greedy, vo[i:i+1], vr[i:i+1], valid[i:i+1])[:, -1].clone()
                logits[:, :2] = -torch.inf
                token = logits.argmax(-1, keepdim=True)
                greedy = torch.cat((greedy, token), -1)
                if token.item() == 2:
                    break
            assert torch.equal(seq, greedy)
    target = torch.tensor([[4, 5, 2, 0], [7, 8, 9, 2]], device=device)
    loss = xe_loss(model(regions, valid, prefix), target)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    for layer in model.decoder:
        assert layer.gate_o.weight.grad.abs().sum() > 0
        assert layer.gate_r.weight.grad.abs().sum() > 0
    assert not model.nouns.requires_grad and not model.verbs.requires_grad
    # EOS contributes to XE, padding does not.
    logits = torch.zeros(1, 3, 13, device=device, requires_grad=True)
    xe_loss(logits, torch.tensor([[4, 2, 0]], device=device)).backward()
    assert logits.grad[0, 1].abs().sum() > 0 and logits.grad[0, 2].abs().sum() == 0
    model.zero_grad(set_to_none=True)
    seq = torch.tensor([[[1, 4, 2, 0], [1, 5, 6, 2]], [[1, 7, 2, 0], [1, 8, 9, 2]]], device=device)
    rewards = torch.tensor([[1., 0.], [.1, .9]], device=device)
    rl = beam_mean_loss(model, regions, valid, seq, rewards)
    assert torch.isfinite(rl)
    rl.backward()
    assert model.head.weight.grad is not None and model.head.weight.grad.abs().sum() > 0
    # Swapping both beams and rewards leaves the objective invariant.
    with torch.no_grad():
        close(rl, beam_mean_loss(model, regions, valid, seq.flip(1), rewards.flip(1)))
        close(beam_mean_loss(model, regions, valid, seq, torch.ones_like(rewards)), torch.zeros((), device=device))
    try:
        model.encode(regions, torch.zeros_like(valid))
    except ValueError:
        pass
    else:
        raise AssertionError('All-masked vision was accepted')
    eos_contract(device)
    return dict(config=asdict(c), checks='aggregation/masks/padding/permutation/causality/prefix/batch/greedy/gradients/XE/EOS/RL')


def eos_contract(device):
    class Fixture:
        training = False
        c = Config(vocab=5)

        def decode(self, ids, vo, vr, valid):
            logits = torch.full((len(ids), ids.shape[1], 5), -10., device=ids.device)
            logits[..., 2], logits[..., 4] = 2., 1.
            return logits
    v = torch.zeros(1, 1, 1, device=device)
    seq, scores = beam_search(Fixture(), v, v, torch.ones(1, 1, dtype=torch.bool, device=device), beams=2, maximum=3)[0]
    assert seq[0, :2].tolist() == [1, 2]
    assert (seq[0, 2:] == 0).all()
    logp = Fixture().decode(torch.ones(1, 1, dtype=torch.long, device=device), v, v, None).log_softmax(-1)[0, 0, 2]
    close(scores[0], logp)


def production_smoke(device):
    c = Config(vocab=10000)
    model = CSACaptioner(c, F.normalize(torch.randn(17, c.dim), dim=-1),
                         F.normalize(torch.randn(11, c.dim), dim=-1)).to(device)
    regions = torch.randn(1, 3, c.raw_dim, device=device)
    valid = torch.ones(1, 3, dtype=torch.bool, device=device)
    sequence = torch.tensor([[1, 4, 5, 6, 2]], device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-6, betas=(.9, .99), eps=1e-8)
    losses = []
    for step in range(2):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss = xe_loss(model(regions, valid, sequence[:, :-1]), sequence[:, 1:])
        assert torch.isfinite(loss)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step(); losses.append(float(loss.detach()))
        print(f'Synthetic XE update {step+1}/2: {losses[-1]:.6f}', flush=True)
    model.eval()
    with torch.no_grad():
        vo, vr = model.encode(regions, valid)
        beams, scores = beam_search(model, vo, vr, valid, beams=5, maximum=20)[0]
    optimizer.zero_grad(set_to_none=True)
    for group in optimizer.param_groups:
        group['lr'] = 4e-6
    # Fixed synthetic rewards only validate the beam-mean backward contract.
    rewards = torch.tensor([[0., .25, .5, .75, 1.]], device=device)
    loss = beam_mean_loss(model, regions, valid, beams[None], rewards)
    assert torch.isfinite(loss)
    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(config=asdict(c), trainable_parameters=sum(p.numel() for p in model.parameters()),
                xe_losses=losses, rl_synthetic_loss=float(loss.detach()), beam_ids=beams.cpu().tolist(),
                beam_scores=scores.cpu().tolist(), quality_metrics_computed=False,
                checkpoint_saved=False, dual_gpu_checked=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', required=True)
    args = parser.parse_args(); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    report = dict(status='INCOMPLETE', synthetic_only=True, model_executed_locally_by_agent=False,
                  torch_version=torch.__version__, cuda_available=torch.cuda.is_available())
    try:
        if not torch.cuda.is_available():
            raise RuntimeError('Select a GPU for the user-run smoke')
        device = torch.device('cuda:0')
        report['gpu'] = torch.cuda.get_device_name(device)
        print('Tiny tensor contracts', flush=True)
        report['tiny'] = check(device)
        print('Production dimensions, synthetic inputs/rewards', flush=True)
        report['production'] = production_smoke(device)
        torch.cuda.synchronize()
        report['peak_gpu_bytes'] = torch.cuda.max_memory_allocated()
        report['status'] = 'CORE_SMOKE_COMPLETE'
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['seconds'] = time.monotonic() - start
        (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
