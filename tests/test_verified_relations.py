import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from captioning.grounded_relations import PairGroundedRelationAlignment
from captioning.checkpoints import load_checkpoint
from captioning.config import Config


class FixedAppearance(torch.nn.Module):
    def forward(self, values):
        output = values.new_zeros(*values.shape[:-1], 512)
        output[..., 0] = 1
        return output


class VerifiedRelationTests(unittest.TestCase):
    def fixture(self, scale=.1):
        torch.manual_seed(3)
        module = PairGroundedRelationAlignment(8, 2, dropout=0., grounding_version='verified_v3',
                                               verified_scale=scale).eval()
        module.appearance_projection = FixedAppearance()
        queries, visual = torch.randn(2, 3, 8), torch.randn(2, 197, 8)
        positive = torch.zeros(2, 2, 512)
        positive[..., 0] = 1
        boxes = torch.tensor([[[0., 0., .4, .4], [.5, .5, 1., 1.]]] * 2)
        inputs = [queries, visual, positive, -positive, boxes, boxes,
                  torch.ones(2, 2), torch.ones(2, 2, dtype=torch.bool)]
        return module, inputs

    def test_verified_evidence_changes_memory_without_learned_query_gate(self):
        module, inputs = self.fixture()
        output, diagnostics = module(*inputs)
        self.assertTrue(diagnostics['accepted_mask'].all())
        self.assertFalse(torch.equal(output, inputs[0]))
        self.assertFalse(module.query_gate.weight.requires_grad)
        output.sum().backward()
        self.assertGreater(module.verified_visual_projection[0].weight.grad.abs().sum().item(), 0)

    def test_counterfactual_relation_rejected_and_memory_exactly_preserved(self):
        module, inputs = self.fixture()
        inputs[2], inputs[3] = inputs[3], inputs[2]
        output, diagnostics = module(*inputs)
        self.assertFalse(diagnostics['accepted_mask'].any())
        self.assertTrue(torch.equal(output, inputs[0]))

    def test_zero_confidence_or_no_relations_cannot_inject_attention_bias(self):
        module, inputs = self.fixture()
        inputs[6].zero_()
        output, diagnostics = module(*inputs)
        self.assertFalse(diagnostics['accepted_mask'].any())
        self.assertTrue(torch.equal(output, inputs[0]))
        inputs[6].fill_(1)
        inputs[7].zero_()
        output, _ = module(*inputs)
        self.assertTrue(torch.equal(output, inputs[0]))

    def test_zero_scale_bypasses_verified_fusion(self):
        module, inputs = self.fixture(scale=0)
        output, diagnostics = module(*inputs)
        self.assertTrue(diagnostics['accepted_mask'].all())
        self.assertTrue(torch.equal(output, inputs[0]))

    def test_verifier_frozen_and_config_rejects_joint_loss(self):
        module = PairGroundedRelationAlignment(8, 2, grounding_version='verified_v3')
        self.assertFalse(any(p.requires_grad for p in module.appearance_projection.parameters()))
        with self.assertRaises(ValueError):
            Config(visual_adapter='qformer', grounded_relation_alignment=True,
                   grounded_relation_cache_path='cache.pt', grounded_relation_version='verified_v3',
                   relation_alignment_weight=.1)

    def test_warm_start_v2_only_initializes_new_fusion_projection(self):
        def encoder(version):
            result = torch.nn.Module()
            result.feature_extractor = None
            result.grounded_relation_alignment = True
            result.grounded_relation_version = version
            result.relation_alignment = PairGroundedRelationAlignment(8, 2, grounding_version=version)
            return result
        old = encoder('appearance_v2')
        new = encoder('verified_v3')
        decoder = torch.nn.Linear(8, 4)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'v2.pth'
            torch.save({'grounded_relation_alignment': True, 'grounded_relation_version': 'appearance_v2',
                        'encoder_state_dict': old.state_dict(), 'decoder_state_dict': decoder.state_dict()}, path)
            load_checkpoint(SimpleNamespace(encoder=new, decoder=decoder), path,
                            SimpleNamespace(word2idx={}), warm_start=True)
            self.assertTrue(torch.equal(new.relation_alignment.appearance_projection[0].weight,
                                        old.relation_alignment.appearance_projection[0].weight))


if __name__ == '__main__':
    unittest.main()
