import copy
import unittest
import torch
from captioning.factorized_evidence import (actor_examples, ActionSupportHead, target_features,
    ConditionalTargetHead, target_objective, reviewed_labels, image_folds, combine_decisions, relative_target_prior)
from tests.test_reviewed_targets import example


class FactorizedEvidenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42); torch.set_num_threads(2)

    def test_action_input_ignores_target_and_union_and_does_not_make_unknown_zero(self):
        rows, _ = example(); views = torch.randn(2, 3, 8)
        features, labels, keys, lookup = actor_examples(rows, views)
        altered = views.clone(); altered[:, 1:] = 100
        other, other_labels, _, _ = actor_examples(rows, altered)
        self.assertTrue(torch.equal(features, other))
        self.assertTrue(torch.equal(labels, other_labels))
        self.assertEqual(labels.tolist(), [[-1, 1, -1, -1, -1, -1]])
        self.assertEqual(lookup.tolist(), [0, 0])
        action = ActionSupportHead(dim=8)
        self.assertTrue(torch.equal(action(features)[lookup][0], action(features)[lookup][1]))
        bad = copy.deepcopy(rows); bad[1]['labels'][1] = 0
        with self.assertRaises(ValueError): actor_examples(bad, views)

    def test_target_starts_at_clip_prior_and_remains_bounded(self):
        views = torch.randn(2, 3, 8); words = torch.randn(2, 6, 8)
        geometry = torch.tensor([[0., 0., .3, 1., .4, 0., 1., 1., .5, 0.]] * 2)
        features = target_features(views, geometry, words)
        self.assertEqual(features.shape, (2, 6, 20))
        head = ConditionalTargetHead(); scores, prior, correction = head(features)
        self.assertEqual(sum(x.numel() for x in head.parameters()), 126)
        self.assertTrue(torch.equal(scores, prior))
        head.weight.data.fill_(100)
        self.assertTrue((head(features)[2].abs() <= .5).all())

    def test_target_unknowns_have_no_gradient_and_source_labels_stay_unchanged(self):
        rows, cases = example(); original = copy.deepcopy(rows)
        scores = torch.zeros(2, 6, requires_grad=True)
        loss, _, _ = target_objective(scores, rows, cases)
        loss.backward()
        self.assertLess(float(scores.grad[0, 1]), 0)
        self.assertGreater(float(scores.grad[1, 1]), 0)
        self.assertEqual(float(scores.grad[:, [0, 2, 3, 4, 5]].abs().sum()), 0)
        self.assertEqual(rows, original)

    def test_two_supported_targets_are_not_treated_as_competing_negatives(self):
        rows, cases = example()
        rows.append({**rows[0], 'object_id': 23, 'object_box': [.2, .1, .8, .9]})
        cases.append({**cases[0], 'case_id': 'second', 'positive_row_index': 2})
        labels = reviewed_labels(rows, cases)
        self.assertEqual(labels[:, 1].tolist(), [1, 0, 1])
        scores = torch.zeros(3, 6, requires_grad=True)
        target_objective(scores, rows, cases)[0].backward()
        self.assertLess(float(scores.grad[0, 1]), 0)
        self.assertLess(float(scores.grad[2, 1]), 0)
        self.assertGreater(float(scores.grad[1, 1]), 0)

    def test_nested_folds_hold_out_every_image_exactly_once(self):
        rows, cases = [], []
        for i in range(12):
            new_rows, new_cases = example()
            for r in new_rows: r['image_id'] = i; r['filename'] = f'{i}.jpg'
            cases.append({**new_cases[0], 'case_id': str(i), 'positive_row_index': len(rows), 'alternate_row_index': len(rows) + 1})
            rows.extend(new_rows)
        folds = image_folds(rows, cases)
        seen = []
        for f in folds:
            a, b, c = (set(f[k]) for k in ('fit_images', 'calibration_images', 'evaluation_images'))
            self.assertFalse(a & b or a & c or b & c)
            self.assertEqual(a | b | c, set(range(12)))
            seen.extend(c)
        self.assertEqual(sorted(seen), list(range(12)))
        self.assertEqual(folds, image_folds(rows, cases))

    def test_support_requires_both_factors_and_unknown_never_becomes_support(self):
        action = torch.tensor([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        target = torch.tensor([[-1, 0, 1]] * 3)
        self.assertEqual(combine_decisions(action, target).tolist(), [[-1, 0, -1], [0, 0, 0], [-1, 0, 1]])

    def test_relative_prior_ignores_common_context_shift_and_preserves_ranking(self):
        rows, _ = example(); rows[0]['subject'] = rows[1]['subject'] = 'person'
        prior = torch.randn(2, 6)
        relative = relative_target_prior(prior, rows)
        torch.testing.assert_close(relative, relative_target_prior(prior + torch.arange(6) * 10, rows), atol=3e-6, rtol=1e-5)
        torch.testing.assert_close(relative[0] - relative[1], prior[0] - prior[1])
        rows[0]['subject_id'] = 999; rows[0]['labels'] = [0] * 6
        self.assertTrue(torch.equal(relative, relative_target_prior(prior, rows)))

    def test_relative_reference_deduplicates_targets_and_allows_two_positives(self):
        rows, _ = example(); rows[0]['subject'] = rows[1]['subject'] = 'person'
        rows.append({**rows[1], 'object_id': 23, 'object_box': [.2, .1, .7, .9]})
        prior = torch.tensor([[3.] * 6, [3.] * 6, [0.] * 6])
        relative = relative_target_prior(prior, rows)
        self.assertTrue((relative[:2] > 0).all())
        duplicated_rows = rows + [copy.deepcopy(rows[1])]
        doubled = torch.cat([prior, prior[1:2]])
        torch.testing.assert_close(relative, relative_target_prior(doubled, duplicated_rows)[:3])
        self.assertFalse(relative.requires_grad)


if __name__ == '__main__': unittest.main()
