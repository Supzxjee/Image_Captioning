import copy
import unittest
import torch
from captioning.supervised_relations import (RELATION_SPECS, build_pairs, decode_action,
                                             contrast_edges, diagnostic_subset)
from captioning.instance_contrast_alignment import (InstanceContrastAlignment,
                                                    ResidualInstanceAlignment, supervised_alignment_loss)


class SupervisedInstanceTests(unittest.TestCase):
    def fixture(self):
        actions = [{'action_name': name, 'role_name': ['agent', role],
                    'ann_id': [1, 2], 'image_id': [100, 100], 'label': [0, -1],
                    'role_object_id': [1, 2, 0, 0]} for name, role, _ in RELATION_SPECS]
        actions[0].update(label=[1, 0], role_object_id=[1, 2, 10, 0])
        annotations = [{'id': aid, 'image_id': 100, 'category_id': 1 if aid < 10 else 2,
                        'bbox': [aid, 10, 20, 30], 'iscrowd': 0} for aid in (1, 2, 10, 11)]
        instances = {'images': [{'id': 100, 'width': 100, 'height': 100}],
                     'annotations': annotations,
                     'categories': [{'id': 1, 'name': 'person'}, {'id': 2, 'name': 'tie'}]}
        caption = [{'cocoid': 100, 'filename': 'test.jpg', 'split': 'train'}]
        return actions, instances, caption

    def test_role_column_order_and_agent_validation(self):
        action = self.fixture()[0][0]
        self.assertEqual(decode_action(action)[1]['roles']['obj'], 10)
        action['role_object_id'][0] = 99
        with self.assertRaises(ValueError):
            decode_action(action)

    def test_unannotated_target_is_unknown_and_confirmed_absence_is_zero(self):
        rows, _ = build_pairs(*self.fixture())
        indexed = {(r['subject_id'], r['object_id']): r for r in rows}
        self.assertEqual(indexed[(1, 10)]['labels'][0], 1)
        self.assertEqual(indexed[(1, 11)]['labels'][0], -1)
        self.assertEqual(indexed[(2, 10)]['labels'][0], 0)
        edges = contrast_edges(rows)
        self.assertTrue(any(e[-1] == 'actor' for e in edges))
        self.assertTrue(any(e[-1] == 'predicate' for e in edges))
        for i, r, j, c, _ in edges:
            self.assertEqual(rows[i]['labels'][r], 1)
            self.assertEqual(rows[j]['labels'][c], 0)

    def test_missing_positive_role_is_not_negative(self):
        actions, instances, caption = self.fixture()
        actions[1]['label'][0] = 1
        rows, _ = build_pairs(actions, instances, caption)
        self.assertTrue(all(r['labels'][1] == -1 for r in rows if r['subject_id'] == 1))

    def test_caption_heldout_excluded(self):
        actions, instances, caption = self.fixture()
        for split in ('val', 'test'):
            caption[0]['split'] = split
            rows, stats = build_pairs(actions, instances, caption)
            self.assertEqual(rows, [])
            self.assertEqual(stats['excluded_caption_heldout_subjects'], 2)

    def test_subset_deterministic_unique(self):
        rows, _ = build_pairs(*self.fixture())
        self.assertEqual(diagnostic_subset(rows, 4), diagnostic_subset(rows, 4))
        selected = diagnostic_subset(rows, 4)
        self.assertEqual(len({(r['subject_id'], r['object_id']) for r in selected}), len(selected))

    def test_subset_fill_does_not_favor_additional_positives(self):
        rows = [{'image_id': i, 'subject_id': i*2, 'object_id': i*2+1,
                 'labels': [1 if i < 20 else 0, 0]} for i in range(40)]
        selected = diagnostic_subset(rows, 12)
        self.assertEqual(sum(1 in r['labels'] for r in selected), 6)

    def test_unknown_scores_have_no_loss_or_gradient(self):
        scores = torch.tensor([[.2, -.3, .8]], requires_grad=True)
        labels = torch.tensor([[1, 0, -1]])
        edges = torch.empty((0, 4), dtype=torch.long)
        loss, _, _ = supervised_alignment_loss(scores, labels, edges)
        modified = scores.detach().clone()
        modified[0, 2] = -100
        other, _, _ = supervised_alignment_loss(modified, labels, edges)
        self.assertAlmostEqual(loss.item(), other.item())
        loss.backward()
        self.assertEqual(scores.grad[0, 2].item(), 0)

    def test_unknown_contrast_is_rejected(self):
        with self.assertRaises(ValueError):
            supervised_alignment_loss(torch.zeros(1, 2), torch.tensor([[1, -1]]),
                                      torch.tensor([[0, 0, 0, 1]]))

    def test_pair_head_backprop_fixed_text(self):
        model = InstanceContrastAlignment(dim=8)
        views = torch.randn(2, 3, 8)
        text = torch.randn(2, 2, 8, requires_grad=True)
        scores = model(views, text)
        loss, _, _ = supervised_alignment_loss(scores, torch.tensor([[1, 0], [0, 1]]),
                                               torch.tensor([[0, 0, 0, 1]]))
        loss.backward()
        self.assertIsNone(text.grad)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))

    def test_ap_ties_and_threshold_abstention(self):
        from run_instance_alignment_diagnostic import binary_average_precision, fit_thresholds
        self.assertAlmostEqual(binary_average_precision([1, 0, 0], [.5, .5, .5]), 1/3)
        self.assertEqual(binary_average_precision([1, 0], [.9, .1]), 1)
        self.assertIsNone(binary_average_precision([0, 0], [.9, .1]))
        self.assertEqual(fit_thresholds(torch.zeros(4, 6), torch.zeros(4, 6)), [1.01]*6)

    def test_type_balanced_ranking_not_dominated_by_duplicated_predicates(self):
        scores = torch.tensor([[.6, .5], [-.4, 0.]])
        labels = torch.tensor([[1, 0], [0, -1]])
        edges = torch.tensor([[0, 0, 0, 1], [0, 0, 1, 0]])
        _, _, ranking = supervised_alignment_loss(scores, labels, edges, edge_kinds=torch.tensor([0, 1]))
        duplicated = torch.cat([edges[:1].repeat(20, 1), edges[1:]])
        kinds = torch.tensor([0]*20 + [1])
        _, _, other = supervised_alignment_loss(scores, labels, duplicated, edge_kinds=kinds)
        self.assertAlmostEqual(ranking.item(), other.item())
        _, _, single = supervised_alignment_loss(scores, labels, edges[:1], edge_kinds=torch.tensor([0]))
        self.assertAlmostEqual(single.item(), 2 * ranking.item())

    def test_edge_metrics_keeps_image_support_and_ties(self):
        from run_instance_alignment_diagnostic import edge_metrics
        rows, _ = build_pairs(*self.fixture())
        result = edge_metrics(torch.zeros(len(rows), 6), rows)
        self.assertEqual(result['actor']['images'], 1)
        self.assertEqual(result['actor']['ties'], result['actor']['edges'])
        self.assertEqual(result['actor']['wins'], 0)

    def test_residual_starts_exactly_at_clip_and_is_bounded(self):
        model = ResidualInstanceAlignment(dim=8, scale=.1)
        views = torch.randn(3, 3, 8)
        text = torch.randn(3, 6, 8)
        scores, base, delta = model(views, text)
        self.assertTrue(torch.equal(scores, base))
        self.assertEqual(delta.abs().sum().item(), 0)
        scores.sum().backward()
        self.assertGreater(model.correction[-1].weight.grad.abs().sum().item(), 0)
        with torch.no_grad():
            model.correction[-1].weight.fill_(100)
        scores, base, delta = model(views, text)
        self.assertLessEqual(delta.abs().max().item(), .100001)

    def test_shared_center_changes_bce_but_not_contrast_ranking(self):
        scores = torch.tensor([[.3, .2], [.1, .4]])
        labels = torch.tensor([[1, 0], [0, 1]])
        edges = torch.tensor([[0, 0, 1, 0], [1, 1, 1, 0]])
        _, raw_bce, raw_rank = supervised_alignment_loss(scores, labels, edges)
        _, centered_bce, centered_rank = supervised_alignment_loss(scores-.4, labels, edges)
        self.assertAlmostEqual(raw_rank.item(), centered_rank.item())
        self.assertNotAlmostEqual(raw_bce.item(), centered_bce.item())


if __name__ == '__main__':
    unittest.main()
