import unittest

try:
    import torch
    from captioning.grounded_relations import (
        PairGroundedRelationAlignment,
        counterfactual_relation_loss,
        relative_geometry,
    )
except (ImportError, OSError):
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is unavailable in this environment.')
class GroundedRelationTests(unittest.TestCase):
    def test_pair_grounding_preserves_images_without_relations(self):
        torch.manual_seed(5)
        module = PairGroundedRelationAlignment(8, 2, dropout=0.0).eval()
        queries = torch.randn(2, 3, 8)
        visual = torch.randn(2, 197, 8)
        relation = torch.randn(2, 2, 512)
        boxes = torch.tensor([
            [[0.0, 0.0, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]],
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        ])
        mask = torch.tensor([[1, 1], [0, 0]])
        output, diagnostics = module(
            queries, visual, relation, relation.flip(1), boxes, boxes.flip(1),
            torch.ones(2, 2), mask)
        self.assertEqual(tuple(output.shape), (2, 3, 8))
        self.assertTrue(torch.equal(output[1], queries[1]))
        self.assertEqual(tuple(diagnostics['pair_features'].shape), (2, 2, 8))

    def test_counterfactual_margin_rewards_correct_relation(self):
        diagnostics = {
            'pair_features': torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
            'positive_relation_features': torch.tensor([[[1.0, 0.0], [0.0, 0.0]]]),
            'negative_relation_features': torch.tensor([[[-1.0, 0.0], [0.0, 0.0]]]),
            'mask': torch.tensor([[1, 0]], dtype=torch.bool),
            'confidences': torch.tensor([[0.9, 0.0]]),
        }
        self.assertEqual(counterfactual_relation_loss(diagnostics, margin=0.2).item(), 0.0)
        diagnostics['positive_relation_features'], diagnostics['negative_relation_features'] = (
            diagnostics['negative_relation_features'], diagnostics['positive_relation_features'])
        self.assertGreater(counterfactual_relation_loss(diagnostics, margin=0.2).item(), 2.0)

    def test_geometry_keeps_direction_and_overlap(self):
        subject = torch.tensor([[[0.0, 0.0, 0.5, 1.0]]])
        obj = torch.tensor([[[0.5, 0.0, 1.0, 1.0]]])
        geometry = relative_geometry(subject, obj)
        self.assertGreater(geometry[0, 0, 0].item(), 0)
        self.assertEqual(geometry[0, 0, 4].item(), 0.0)


if __name__ == '__main__':
    unittest.main()
