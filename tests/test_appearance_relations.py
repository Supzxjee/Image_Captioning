"""Tests for the shortcut controls motivated by the E3 V1 diagnostic."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
import torch

from captioning.grounded_relations import (
    PairGroundedRelationAlignment, action_relation_mask, appearance_relation_loss,
)
from captioning.config import Config
from captioning.checkpoints import load_checkpoint


class AppearanceRelationTests(unittest.TestCase):
    def diagnostics(self):
        positive = torch.tensor([[[1., 0.]], [[0., 1.]]], requires_grad=True)
        negative = -positive.detach()
        return {
            'appearance_features': positive.detach().clone().requires_grad_(),
            'fixed_positive_features': positive,
            'fixed_negative_features': negative,
            'wrong_image_features': positive.detach().flip(0).clone().requires_grad_(),
            'mask': torch.ones(2, 1, dtype=torch.bool),
            'confidences': torch.ones(2, 1),
        }

    def test_appearance_without_image_sensitivity_cannot_satisfy_wrong_image_margin(self):
        d = self.diagnostics()
        d['wrong_image_features'] = d['appearance_features']
        loss, parts = appearance_relation_loss(d, d['mask'])
        self.assertEqual(parts['eligible_wrong_images'], 2)
        self.assertAlmostEqual(parts['wrong_image_loss'].item(), 0.2, places=6)
        self.assertGreater(loss.item(), 0)

    def test_correct_distinct_evidence_satisfies_both_margins(self):
        d = self.diagnostics()
        loss, parts = appearance_relation_loss(d, d['mask'])
        self.assertEqual(parts['eligible_wrong_images'], 2)
        self.assertEqual(loss.item(), 0)

    def test_duplicate_phrase_is_not_used_as_wrong_image_negative(self):
        d = self.diagnostics()
        d['fixed_positive_features'] = torch.ones(2, 1, 2)
        _, parts = appearance_relation_loss(d, d['mask'])
        self.assertEqual(parts['eligible_wrong_images'], 0)

    def test_text_targets_receive_no_gradients(self):
        d = self.diagnostics()
        d['appearance_features'] = torch.tensor([[[-1., 1.]], [[1., -1.]]], requires_grad=True)
        loss, _ = appearance_relation_loss(d, d['mask'])
        loss.backward()
        self.assertIsNone(d['fixed_positive_features'].grad)
        self.assertGreater(d['appearance_features'].grad.abs().sum().item(), 0)

    def test_spatial_relations_are_excluded_from_appearance_loss(self):
        entry = {'mask': torch.tensor([1, 1, 1]), 'relations': [
            {'relation': 'left_of'}, {'relation': 'holding'}, {'relation': 'above'}]}
        self.assertEqual(action_relation_mask(entry).tolist(), [False, True, False])
        d = self.diagnostics()
        loss, parts = appearance_relation_loss(d, torch.zeros_like(d['mask']))
        loss.backward()
        self.assertEqual(parts['actions'], 0)
        self.assertEqual(loss.item(), 0)

    def test_v2_retains_no_relation_bypass_and_has_visual_only_scorer(self):
        torch.manual_seed(3)
        module = PairGroundedRelationAlignment(8, 2, dropout=0., grounding_version='appearance_v2').eval()
        queries, visual = torch.randn(2, 3, 8), torch.randn(2, 197, 8)
        texts = torch.randn(2, 2, 512)
        boxes = torch.tensor([[[0., 0., .5, .5], [.5, .5, 1., 1.]]] * 2)
        output, d = module(queries, visual, texts, -texts, boxes, boxes.flip(1),
                           torch.ones(2, 2), torch.tensor([[1, 1], [0, 0]]))
        self.assertTrue(torch.equal(output[1], queries[1]))
        self.assertEqual(module.appearance_projection[0].in_features, 3 * 8)
        self.assertEqual(tuple(d['appearance_features'].shape), (2, 2, 512))
        self.assertIn('wrong_image_features', d)

    def test_v2_requires_grounded_architecture(self):
        with self.assertRaises(ValueError):
            Config(grounded_relation_version='appearance_v2')

    def test_checkpoint_loader_refuses_silent_v1_to_v2_architecture_switch(self):
        encoder = torch.nn.Linear(2, 2)
        encoder.grounded_relation_alignment = True
        encoder.grounded_relation_version = 'appearance_v2'
        model = SimpleNamespace(encoder=encoder, decoder=torch.nn.Linear(2, 2))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'v1.pth'
            torch.save({'grounded_relation_alignment': True}, path)
            with self.assertRaisesRegex(RuntimeError, 'Grounded relation versions differ'):
                load_checkpoint(model, path, SimpleNamespace(word2idx={}), warm_start=True)

    def test_v2_scorer_backpropagates_into_visual_evidence(self):
        torch.manual_seed(10)
        module = PairGroundedRelationAlignment(8, 2, dropout=0., grounding_version='appearance_v2')
        queries, visual = torch.randn(2, 3, 8), torch.randn(2, 197, 8, requires_grad=True)
        texts = torch.randn(2, 1, 512)
        boxes = torch.tensor([[[.1, .1, .4, .4]], [[.2, .3, .8, .9]]])
        _, d = module(queries, visual, texts, -texts, boxes, boxes,
                      torch.ones(2, 1), torch.ones(2, 1, dtype=torch.bool))
        loss, _ = appearance_relation_loss(d, d['mask'])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(visual.grad.abs().sum().item(), 0)
        self.assertGreater(module.appearance_projection[0].weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
