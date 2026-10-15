import unittest
import torch
from captioning.pair_query_attention import PairQueryAttention, patch_evidence
from captioning.instance_contrast_alignment import supervised_alignment_loss
from run_pair_query_attention import partition


class PairQueryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.model = PairQueryAttention().eval()
        self.patches = torch.randn(2, 196, 71)
        self.endpoints = torch.randn(2, 2, 64)
        self.geometry = torch.randn(2, 10)
        self.text = torch.randn(2, 6, 512)

    def test_reader_depends_on_candidate_and_controls_do_not(self):
        args = (self.patches, self.endpoints, self.geometry, self.text)
        for mode in ('attention', 'mean_pool', 'query_blind'):
            scores, weights = self.model(*args, mode=mode, return_weights=True)
            changed = self.model(self.patches, self.endpoints.flip(1), self.geometry, self.text.roll(1,1), mode=mode, return_weights=True)
            self.assertFalse(torch.equal(scores, changed[0]))
            self.assertEqual(torch.equal(weights, changed[1]), mode != 'attention')
            self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(2,6), atol=1e-6))

    def test_spatial_evidence_stays_paired_when_permuted(self):
        order = torch.randperm(196)
        original = self.model(self.patches, self.endpoints, self.geometry, self.text)
        permuted = self.model(self.patches[:,order], self.endpoints, self.geometry, self.text)
        self.assertTrue(torch.allclose(original, permuted, atol=1e-6))
        self.assertFalse(torch.equal(original, self.model(torch.zeros_like(self.patches), self.endpoints, self.geometry, self.text)))

    def test_unknown_logits_have_no_loss_gradient(self):
        logits = torch.randn(2, 6, requires_grad=True)
        labels = torch.full((2,6), -1); labels[0,0]=1; labels[1,0]=0
        loss, _, _ = supervised_alignment_loss(logits, labels, torch.tensor([[0,0,1,0]]), temperature=1)
        loss.backward()
        self.assertEqual(float(logits.grad[labels<0].abs().sum()), 0)

    def test_no_backbone_gradients_and_geometry_is_not_a_label(self):
        patches = self.patches.clone().requires_grad_(True)
        self.model(patches, self.endpoints, self.geometry, self.text).sum().backward()
        self.assertIsNone(patches.grad)
        self.assertGreater(float(self.model.key.weight.grad.abs().sum()), 0)

    def test_fixed_projection_keeps_empty_endpoint_finite(self):
        tokens=torch.randn(2,197,768); support=torch.zeros(2,5,196)
        support[:,1]=1
        patches, endpoints=patch_evidence(tokens,support,torch.randn(768,64))
        self.assertEqual(patches.shape,(2,196,71))
        self.assertTrue(torch.equal(endpoints[:,0],torch.zeros(2,64)))
        self.assertTrue(torch.isfinite(patches).all())

    def test_whole_review_inventory_excluded_not_only_completed_cases(self):
        rows=[{'image_id':i,'filename':str(i)} for i in range(12)]
        rows.insert(2,dict(rows[1]))
        split=partition(rows,len(rows)-2,{'1','3','5'})
        self.assertFalse(set(split['fit_images']) & set(split['calibration_images']))
        self.assertTrue({1,3,5}.isdisjoint(split['fit_images']+split['calibration_images']))
        self.assertEqual(split['excluded_manifest_images_present'],[1,3,5])
        self.assertEqual(split,partition(rows,len(rows)-2,{'1','3','5'}))


if __name__ == '__main__': unittest.main()
