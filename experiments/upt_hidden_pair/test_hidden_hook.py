"""Synthetic upstream-interface checks; no UPT checkpoint or images are loaded."""
import unittest
import torch
from torch import nn
from extract_hidden import infer_hidden


class Head(nn.Module):
    def __init__(self):
        super().__init__(); self.box_pair_predictor = nn.Linear(1024, 117)
    def forward(self, features, shapes, props):
        labels = props[0]['labels']; n_human = int(labels.eq(0).sum())
        pairs = [(s, o) for s in range(n_human) for o in range(len(labels)) if s != o]
        h = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        o = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        hidden = torch.arange(len(pairs), dtype=torch.float32)[:, None].expand(-1, 1024)
        logits = self.box_pair_predictor(hidden)
        return logits, [torch.zeros(2, len(pairs), 117)], [h], [o], [labels[o]], []


class Native(nn.Module):
    def __init__(self, labels):
        super().__init__(); self.interaction_head = Head(); self.labels = torch.tensor(labels)
    def forward(self, images):
        n = len(self.labels)
        props = dict(labels=self.labels, scores=torch.full((n,), .8), boxes=torch.tensor([[0., 0., 10., 10.]] * n))
        logits, _, hs, os, _, _ = self.interaction_head(None, None, [props])
        if not len(hs[0]):
            return []
        return [dict(boxes=props['boxes'], pairing=torch.stack((hs[0], os[0])), scores=torch.zeros(len(hs[0])),
                     labels=torch.zeros(len(hs[0]), dtype=torch.long), objects=self.labels[os[0]], size=torch.tensor([20, 20]))]


class Checks(unittest.TestCase):
    def test_features_are_preclassifier_rows_not_postprocessed_verb_rows(self):
        model = Native([0, 0, 3]).eval(); prediction, hidden = infer_hidden(model, torch.zeros(3, 20, 20))
        self.assertEqual(hidden.shape, (4, 1024))
        self.assertEqual(hidden[:, 0].tolist(), [0., 1., 2., 3.])
        self.assertEqual(list(zip(prediction['subject_indices'].tolist(), prediction['object_indices'].tolist())), [(0, 1), (0, 2), (1, 0), (1, 2)])
        self.assertEqual(len(model.interaction_head.box_pair_predictor._forward_hooks), 0)

    def test_no_human_empty_output_preserves_empty_features_and_cleans_hook(self):
        model = Native([3]).eval(); prediction, hidden = infer_hidden(model, torch.zeros(3, 20, 20))
        self.assertEqual(hidden.shape, (0, 1024))
        self.assertEqual(prediction['empty_reason'], 'no_human_object_pairs')
        self.assertEqual(len(model.interaction_head.box_pair_predictor._forward_hooks), 0)

    def test_wrong_human_order_is_rejected(self):
        with self.assertRaises(ValueError):
            infer_hidden(Native([3, 0]).eval(), torch.zeros(3, 20, 20))


if __name__ == '__main__':
    unittest.main(verbosity=2)
