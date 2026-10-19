"""Run on Kaggle before expensive staging/training; no pretrained model downloads."""
import unittest
import torch
from torch import nn
from pilot_model import PilotModel, RegionMemory
from pilot_data import INPUT_DIM


class FakeDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(9, 512)
        self.out = nn.Linear(512, 9)

    def forward(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        keep = (~memory_key_padding_mask).unsqueeze(-1)
        value = (memory * keep).sum(1) / keep.sum(1).clamp_min(1)
        return self.out(self.embedding(tgt) + value[:, None])


class FakeBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(768, 512)
        self.decoder = FakeDecoder()

    def build_memory(self, visual, legacy, mask):
        memory = torch.cat([legacy, self.projection(visual[:, :2])], 1)
        padding = torch.cat([mask.eq(0), torch.zeros(visual.size(0), 2, dtype=torch.bool, device=visual.device)], 1)
        return memory, padding


def batch(device='cpu'):
    return dict(visual=torch.randn(2, 3, 768, device=device), legacy=torch.randn(2, 2, 512, device=device),
                legacy_mask=torch.ones(2, 2, dtype=torch.long, device=device),
                regions=torch.randn(2, 9, INPUT_DIM, device=device), text=torch.randn(2, 9, 512, device=device),
                roles=torch.tensor([[0,1,2]*3]*2, device=device), valid=torch.ones(2,9,dtype=torch.bool,device=device),
                captions=torch.tensor([[[1,4,5,2,0]]*5]*2, device=device),
                caption_mask=torch.tensor([[[False,False,False,False,True]]*5]*2, device=device))


class ModelChecks(unittest.TestCase):
    def test_gate_survives_normalization_and_invalid_slots_are_zero(self):
        torch.manual_seed(42); data = batch(); branch = RegionMemory('D').eval()
        low, gate = branch(data['regions'],data['text'],data['roles'],data['valid'])
        torch.testing.assert_close(gate, torch.full_like(gate, .5))
        with torch.no_grad():
            for g in branch.gates: g[-1].bias.fill_(4.)
        high, _ = branch(data['regions'],data['text'],data['roles'],data['valid'])
        self.assertGreater(float(high.norm()), float(low.norm())*1.9)
        invalid, _ = branch(data['regions'],data['text'],data['roles'],torch.zeros_like(data['valid']))
        self.assertEqual(int(torch.count_nonzero(invalid)), 0)

    def test_off_and_empty_evidence_preserve_original_decoder(self):
        torch.manual_seed(42); data = batch(); base = FakeBase().eval(); model = PilotModel(base,'D').eval()
        mem, pad = base.build_memory(data['visual'],data['legacy'],data['legacy_mask'])
        tgt = data['captions'][:,0,:-1]
        expected = base.decoder(tgt,mem,memory_key_padding_mask=pad)
        for intervention in ('off','on'):
            if intervention == 'on': data['valid'].zero_()
            m,p = model.memory(data,intervention)
            torch.testing.assert_close(model.decoder(tgt,m,memory_key_padding_mask=p),expected)

    def test_ce_and_scst_have_finite_gradients_for_all_arms(self):
        for arm in 'ABCD':
            torch.manual_seed(42); data = batch(); model = PilotModel(FakeBase(),arm).eval()
            sample = data['captions'][:,0,:4]
            loss,ce,rl = model(data,sample,torch.ones(2,3,dtype=torch.bool),torch.tensor([1.,-.3]),.05,0)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            if arm != 'A': self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.extra.parameters()))

    def test_actual_attention_decoder_ignores_masked_auxiliary_tokens(self):
        from captioning.models import CaptionDecoder
        torch.manual_seed(42); data = batch(); base = FakeBase().eval()
        base.decoder = CaptionDecoder(9,num_layers=1).eval()
        model = PilotModel(base,'D').eval()
        with torch.no_grad():
            memory,padding = base.build_memory(data['visual'],data['legacy'],data['legacy_mask'])
            tgt = data['captions'][:,0,:-1]
            expected = base.decoder(tgt,memory,memory_key_padding_mask=padding)
            m,p = model.memory(data,'off')
            torch.testing.assert_close(model.decoder(tgt,m,memory_key_padding_mask=p),expected,atol=2e-6,rtol=2e-5)


if __name__ == '__main__': unittest.main(verbosity=2)
