"""Synthetic CPU tests run by the user on Kaggle, no pretrained model downloads."""
import unittest
import torch
from torch import nn
from hidden_model import HiddenModel


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__(); self.embedding = nn.Embedding(9, 512); self.out = nn.Linear(512, 9)
    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        keep = (~memory_key_padding_mask).unsqueeze(-1)
        pooled = (memory * keep).sum(1) / keep.sum(1).clamp_min(1)
        return self.out(self.embedding(tgt) + pooled[:, None])


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__(); self.projection = nn.Linear(768, 512); self.decoder = FakeDecoder()
    def build_memory(self, visual, legacy, mask):
        memory = torch.cat((legacy, self.projection(visual)), 1)
        padding = torch.cat((mask.eq(0), torch.zeros(visual.size(0), visual.size(1), dtype=torch.bool, device=visual.device)), 1)
        return memory, padding


def batch(device='cpu', no_pairs=False):
    return dict(visual=torch.randn(2, 3, 768, device=device), legacy=torch.randn(2, 2, 512, device=device),
        legacy_mask=torch.ones(2, 2, dtype=torch.long, device=device), hidden=torch.randn(2, 8, 1024, device=device),
        valid=torch.full((2, 8), not no_pairs, dtype=torch.bool, device=device),
        swap_hidden=torch.randn(2, 8, 1024, device=device), swap_valid=torch.ones(2, 8, dtype=torch.bool, device=device),
        captions=torch.tensor([[[1, 4, 5, 2, 0]] * 5] * 2, device=device),
        caption_mask=torch.tensor([[[False, False, False, False, True]] * 5] * 2, device=device))


class Checks(unittest.TestCase):
    def test_initial_base_same_and_off_or_empty_exactly_preserves_memory(self):
        models = []
        for arm in ('A', 'H'):
            torch.manual_seed(42); models.append(HiddenModel(FakeBase(), arm).eval())
        for key, value in models[0].base.state_dict().items():
            torch.testing.assert_close(value, models[1].base.state_dict()[key], rtol=0, atol=0)
        data = batch(no_pairs=True); model = models[1]
        original = model.base.build_memory(data['visual'], data['legacy'], data['legacy_mask'])
        for condition in ('off', 'on'):
            actual = model.memory(data, condition)
            for a, b in zip(actual, original):
                torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_invalid_slots_are_masked_and_do_not_affect_decoder(self):
        torch.manual_seed(42); model = HiddenModel(FakeBase(), 'H').eval(); data = batch()
        data['valid'][:, 4:] = False
        memory, padding = model.memory(data)
        self.assertTrue(bool(padding[:, -4:].all()))
        expected = model.decoder(data['captions'][:, 0, :-1], memory, memory_key_padding_mask=padding)
        data['hidden'][:, 4:] = 100000
        m, p = model.memory(data)
        actual = model.decoder(data['captions'][:, 0, :-1], m, memory_key_padding_mask=p)
        torch.testing.assert_close(actual, expected, atol=5e-6, rtol=2e-5)

    def test_ce_scst_finite_gradients_and_empty_branch_has_no_grad(self):
        for arm in ('A', 'H'):
            for empty in (False, True):
                torch.manual_seed(42); model = HiddenModel(FakeBase(), arm).eval(); data = batch(no_pairs=empty)
                loss, _, _ = model(data, data['captions'][:, 0, :4], torch.ones(2, 3, dtype=torch.bool), torch.tensor([1., -.2]), .05, 0)
                loss.backward()
                self.assertTrue(bool(torch.isfinite(loss)))
                self.assertTrue(all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None))
                if model.extra is not None:
                    grads = [p.grad for p in model.extra.parameters() if p.grad is not None]
                    if empty:
                        self.assertTrue(all(not bool(g.any()) for g in grads))
                    else:
                        self.assertTrue(any(bool(g.any()) for g in grads))

    def test_actual_attention_decoder_empty_branch_reproduces_base(self):
        from captioning.models import CaptionDecoder
        torch.manual_seed(42); model = HiddenModel(FakeBase(), 'H').eval()
        model.base.decoder = CaptionDecoder(9, num_layers=1).eval()
        data = batch(no_pairs=True); mem, pad = model.base.build_memory(data['visual'], data['legacy'], data['legacy_mask'])
        tgt = data['captions'][:, 0, :-1]
        expected = model.decoder(tgt, mem, memory_key_padding_mask=pad)
        m, p = model.memory(data)
        torch.testing.assert_close(model.decoder(tgt, m, memory_key_padding_mask=p), expected, atol=5e-6, rtol=2e-5)
        data['valid'][1] = True
        m, p = model.memory(data)
        mixed = model.decoder(tgt, m, memory_key_padding_mask=p)
        torch.testing.assert_close(mixed[0], expected[0], atol=5e-6, rtol=2e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
