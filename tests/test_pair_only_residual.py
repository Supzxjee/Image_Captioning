import unittest
import torch
from captioning.instance_contrast_alignment import PairOnlyResidualAlignment


class PairOnlyResidualTests(unittest.TestCase):
    def test_union_cannot_change_learned_correction(self):
        torch.manual_seed(42)
        model = PairOnlyResidualAlignment(dim=8)
        torch.nn.init.normal_(model.correction[-1].weight)
        model.eval()
        views = torch.nn.functional.normalize(torch.randn(3, 3, 8), dim=-1)
        text = torch.nn.functional.normalize(torch.randn(3, 6, 8), dim=-1)
        changed = views.clone()
        changed[:, 2] = torch.roll(changed[:, 2], 1, 0)
        with torch.no_grad():
            first, base1, delta1 = model(views, text)
            second, base2, delta2 = model(changed, text)
        self.assertTrue(torch.equal(delta1, delta2))
        self.assertFalse(torch.equal(base1, base2))
        self.assertTrue(torch.allclose(first-second, base1-base2, atol=1e-6))

    def test_initial_score_is_exact_clip(self):
        model = PairOnlyResidualAlignment(dim=8)
        views = torch.nn.functional.normalize(torch.randn(3, 3, 8), dim=-1)
        text = torch.nn.functional.normalize(torch.randn(3, 6, 8), dim=-1)
        score, base, delta = model(views, text)
        self.assertTrue(torch.equal(score, base))
        self.assertEqual(float(delta.detach().abs().max()), 0)


if __name__ == '__main__':
    unittest.main()
