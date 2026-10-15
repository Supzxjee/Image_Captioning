import importlib
import types
import unittest
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch
import torch
from captioning.evidence_routing import (EvidenceRelationVerifier, masked_verification_loss,
    fit_selective_policy, selective_decisions, StepwiseEvidenceRouter,
    VerifiedEvidenceCaptionDecoder, verified_prompt_memory)


class EvidenceRoutingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_unknown_labels_have_zero_gradient_and_actions_are_not_exclusive(self):
        logits = torch.zeros(1, 3, requires_grad=True)
        masked_verification_loss(logits, torch.tensor([[1, 1, -1]])).backward()
        self.assertLess(float(logits.grad[0, 0]), 0)
        self.assertLess(float(logits.grad[0, 1]), 0)
        self.assertEqual(float(logits.grad[0, 2]), 0)

    def test_zip_and_extracted_root_use_same_bytes_without_recursive_search(self):
        from captioning.feature_archive import open_feature_archive
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'root'; root.mkdir()
            (root / 'manifest.json').write_bytes(b'{"source":1}')
            path = Path(tmp) / 'source.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('manifest.json', (root / 'manifest.json').read_bytes())
            with open_feature_archive(path) as archive, open_feature_archive(root) as directory:
                self.assertEqual(archive.read('manifest.json'), directory.read('manifest.json'))
                self.assertEqual(directory.namelist(), ['manifest.json'])
                with self.assertRaises(ValueError):
                    directory.read('../source.zip')

    def test_unknowns_do_not_create_calibration_support(self):
        p = torch.tensor([[.9], [.8], [.1], [.05], [.99]])
        labels = torch.tensor([[1], [1], [0], [0], [-1]])
        a = fit_selective_policy(p, labels, precision=1, min_known=2)
        p[-1] = 0
        self.assertEqual(a, fit_selective_policy(p, labels, precision=1, min_known=2))
        self.assertEqual(a['calibration_support'][0]['positive_tail_known'], 2)
        no_support = fit_selective_policy(p, labels, min_known=5)
        decision = selective_decisions(torch.tensor([[1.]]), torch.ones(1, 1, dtype=torch.bool), no_support)
        self.assertEqual(int(decision[0, 0]), -1)

    def test_localization_failure_abstains_even_with_high_score(self):
        policy = {'positive': [.8], 'negative': [.2]}
        result = selective_decisions(torch.tensor([[.99], [.01], [.5]]), torch.zeros(3, 1).bool(), policy)
        self.assertTrue((result == -1).all())

    def test_exact_ninety_percent_is_not_lost_to_float32_rounding(self):
        p = torch.linspace(.5, 1, 10).reshape(-1, 1)
        labels = torch.ones(10, 1).long(); labels[0] = 0
        policy = fit_selective_policy(p, labels, precision=.9, min_known=5)
        self.assertEqual(policy['positive'], [.5])
        self.assertEqual(policy['calibration_support'][0]['positive_tail_known'], 10)

    def test_verifier_keeps_order_and_union_and_freezes_extractor(self):
        model = EvidenceRelationVerifier(dim=8, hidden=4)
        views = torch.randn(2, 3, 8, requires_grad=True)
        text = torch.randn(2, 6, 8, requires_grad=True)
        geometry = torch.randn(2, 10)
        # A trained correction must be able to distinguish same-category targets.
        torch.nn.init.normal_(model.correction.weight)
        logits, evidence = model(views, geometry, text)
        self.assertEqual(evidence.shape, (2, 6, 4))
        swapped = views.detach().clone(); swapped[:, 0] = views[:, 1]; swapped[:, 1] = views[:, 0]
        self.assertFalse(torch.allclose(logits, model(swapped, geometry, text)[0]))
        changed = views.detach().clone(); changed[:, 2] = torch.randn(2, 8)
        self.assertFalse(torch.allclose(logits, model(changed, geometry, text)[0]))
        logits.sum().backward()
        self.assertIsNone(views.grad); self.assertIsNone(text.grad)

    def test_router_exact_fallback_zero_initialization_and_bound(self):
        router = StepwiseEvidenceRouter(8, 4)
        h, e = torch.randn(2, 3, 8), torch.randn(2, 2, 4)
        p = torch.tensor([[.8, .6], [.9, .7]], requires_grad=True)
        mask = torch.tensor([[1, 1], [0, 0]]).bool()
        out, gate = router(h, e, p, mask)
        self.assertTrue(torch.equal(out, h))
        torch.nn.init.eye_(router.output.weight)
        out, gate = router(h, e, p, mask)
        self.assertTrue(torch.equal(out[1], h[1]))
        self.assertFalse(torch.allclose(out[0], h[0]))
        self.assertTrue((gate <= .8).all())
        out.sum().backward()
        self.assertIsNone(p.grad)
        self.assertGreater(float(router.gate.weight.grad.abs().sum()), 0)
        # Rejected bad values cannot poison an otherwise valid memory.
        e[1] = float('nan')
        self.assertTrue(torch.isfinite(router(h, e, p, mask)[0]).all())
        empty, gate = router(h, e[:, :0], p[:, :0], mask[:, :0])
        self.assertTrue(torch.equal(empty, h))

    def test_decoder_is_causal_and_baseline_compatible(self):
        with patch.dict('sys.modules', {'transformers': types.SimpleNamespace(CLIPModel=object)}):
            models = importlib.import_module('captioning.models')
        base = models.CaptionDecoder(13, d_model=8, nhead=2, num_layers=1).eval()
        decoder = VerifiedEvidenceCaptionDecoder(base, evidence_dim=4).eval()
        tgt, memory = torch.tensor([[1, 2, 3]]), torch.randn(1, 5, 8)
        evidence = {'features': torch.randn(1, 2, 4), 'probabilities': torch.tensor([[.9, .8]]),
                    'accepted': torch.ones(1, 2).bool()}
        self.assertTrue(torch.equal(base(tgt, memory), decoder(tgt, memory)))
        torch.testing.assert_close(base(tgt, memory), decoder(tgt, memory, evidence=evidence), rtol=0, atol=0)
        torch.nn.init.eye_(decoder.router.output.weight)
        original = decoder(tgt, memory, evidence=evidence)
        changed = tgt.clone(); changed[0, 2] = 7
        torch.testing.assert_close(original[:, :2], decoder(changed, memory, evidence=evidence)[:, :2])
        self.assertFalse(torch.allclose(original, base(tgt, memory)))
        evidence['accepted'].zero_()
        torch.testing.assert_close(base(tgt, memory), decoder(tgt, memory, evidence=evidence), rtol=0, atol=0)

    def test_prompt_and_memory_keep_same_instance_and_ignore_annotation_labels(self):
        rows = [{'filename': 'a.jpg', 'subject': 'person', 'object': 'bicycle',
                 'subject_box': [0, 0, .3, 1], 'object_box': [.3, 0, .6, 1], 'labels': [-1] * 6},
                {'filename': 'a.jpg', 'subject': 'person', 'object': 'bicycle',
                 'subject_box': [0, 0, .3, 1], 'object_box': [.6, 0, 1, 1], 'labels': [1] * 6}]
        features = torch.randn(2, 6, 4); p = torch.full((2, 6), .1); p[0, 1] = .9
        policy = {'positive': [.8] * 6, 'negative': [.2] * 6}
        decisions = selective_decisions(p, torch.ones_like(p).bool(), policy)
        prompts, memory = verified_prompt_memory(rows, features, p, decisions, policy)
        self.assertEqual(len(prompts['a.jpg']['relations']), 1)
        torch.testing.assert_close(memory['a.jpg']['features'][0, 0], features[0, 1])
        rows[0]['labels'], rows[1]['labels'] = [0] * 6, [-1] * 6
        other, _ = verified_prompt_memory(rows, features, p, decisions, policy)
        self.assertEqual(prompts, other)


if __name__ == '__main__':
    unittest.main()
