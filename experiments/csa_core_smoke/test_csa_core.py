"""CPU parity of unchanged author equations + buffer engineering + memory interface."""
import gc
import json
from pathlib import Path
import sys
import torch
from torch import nn
from csa_core import CSACore, load_banks, reference_class
from csa_model import CSAModel


def main(directory):
    torch.set_num_threads(2)
    banks = load_banks(directory)
    source = Path(directory) / 'csa_net.py'
    results = []
    for n in (32, 150):
        torch.manual_seed(42)
        ref = reference_class(source, banks)().eval()
        adapter = CSACore(source, banks).eval()
        incompatible = adapter.core.load_state_dict(ref.state_dict(), strict=False)
        assert set(incompatible.missing_keys) == {'object_weight','relation_weight'}
        assert not incompatible.unexpected_keys
        assert sum(p.numel() for p in adapter.parameters()) == 23161644
        x = (torch.randn(2, n, 512) * .1).requires_grad_()
        y = x.detach().clone().requires_grad_()
        expected, actual = ref({'reg_feat': x}), adapter(y)
        errors = []
        for a, b in zip(expected[:2], actual[:2]):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
            errors.append(float((a-b).abs().max().detach()))
        assert torch.equal(expected[2], actual[2])
        weights = [torch.randn_like(t) for t in expected[:2]]
        sum((t*w).mean() for t,w in zip(expected[:2],weights)).backward()
        sum((t*w).mean() for t,w in zip(actual[:2],weights)).backward()
        torch.testing.assert_close(x.grad, y.grad, rtol=1e-4, atol=1e-6)
        compared = 0
        for name, p in ref.named_parameters():
            q = dict(adapter.core.named_parameters())[name]
            assert p.grad is not None and q.grad is not None, name
            assert torch.isfinite(q.grad).all(), name
            torch.testing.assert_close(p.grad, q.grad, rtol=1e-4, atol=1e-6)
            compared += 1
        assert y.grad.abs().sum() > 0
        for key in ('object_weight', 'relation_weight'):
            b = dict(adapter.core.named_buffers())[key]
            assert b.grad is None and not b.requires_grad and key in adapter.core.state_dict()
        results.append(dict(tokens=n, output_max_abs=errors, parameter_gradients_compared=compared))
        print('PARITY PASS', results[-1], flush=True)
        del ref, adapter, expected, actual, x, y
        gc.collect()

    class Base(nn.Module):
        def build_memory(self, visual, legacy, mask):
            return torch.cat((legacy, visual), 1), torch.cat((mask == 0, torch.zeros(visual.shape[:2], dtype=torch.bool)), 1)
    core = CSACore(source, banks).eval()
    model = CSAModel(Base(), core).eval()
    batch = dict(visual=torch.randn(2,32,512,requires_grad=True), legacy=torch.randn(2,3,512),
                 legacy_mask=torch.tensor([[1,1,0],[1,0,0]]))
    expected = model.base.build_memory(batch['visual'],batch['legacy'],batch['legacy_mask'])
    off = model.memory(batch,'off')
    for a,b in zip(off,expected):
        assert torch.equal(a,b)
    memory, mask = model.memory(batch)
    assert memory.shape == (2,99,512) and mask.shape == (2,99)
    assert torch.equal(memory[:,:35],expected[0]) and torch.equal(mask[:,:35],expected[1])
    assert not mask[:,35:].any()
    (memory[:,35:]*torch.randn_like(memory[:,35:])).mean().backward()
    assert batch['visual'].grad.abs().sum() > 0
    assert sum(float(p.grad.abs().sum()) for p in core.parameters() if p.grad is not None) > 0
    report = dict(status='CPU_PARITY_AND_INTERFACE_PASS', tolerance=dict(rtol=1e-4,atol=1e-6),
                  cases=results, baseline_bypass_exact=True, no_caption_quality_evaluation=True)
    (Path(directory)/'parity.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
