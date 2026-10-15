import copy
import unittest
import torch
from captioning.reviewed_targets import validate_target_cases, target_metrics, target_changes


def example():
    a = {'image_id': 1, 'filename': 'a.jpg', 'subject_id': 11, 'object_id': 21,
         'subject_box': [0, 0, .3, 1], 'object_box': [.3, 0, .6, 1],
         'object': 'bicycle', 'labels': [-1, 1, -1, -1, -1, -1]}
    b = {**a, 'object_id': 22, 'object_box': [.6, 0, 1, 1], 'labels': [-1] * 6}
    return [a, b], [{'case_id': 'one', 'positive_row_index': 0, 'alternate_row_index': 1, 'relation_index': 1}]


class ReviewedTargetTests(unittest.TestCase):
    def test_all_abstained_is_not_a_success_even_if_ranking_is_correct(self):
        rows, cases = example(); scores = torch.zeros(2, 6); scores[0, 1] = 1
        decisions = torch.full((2, 6), -1)
        result, _ = target_metrics(scores, decisions, rows, cases)
        self.assertEqual(result['same_class']['ranking_wins'], 1)
        self.assertEqual(result['all']['both_abstained'], 1)
        self.assertEqual(result['all']['positive_supported_alternate_not_supported'], 0)
        self.assertIsNone(result['unique_claims_before_cap']['precision_on_reviewed'])

    def test_wrong_target_supported_counted_even_when_positive_ranks_higher(self):
        rows, cases = example(); scores = torch.zeros(2, 6); scores[0, 1] = 2; scores[1, 1] = 1
        decisions = torch.full((2, 6), -1); decisions[:, 1] = 1
        result, _ = target_metrics(scores, decisions, rows, cases)
        self.assertEqual(result['all']['ranking_wins'], 1)
        self.assertEqual(result['all']['both_supported'], 1)
        self.assertEqual(result['unique_claims_before_cap']['precision_on_reviewed'], .5)
        mask = decisions == 1; mask[1, 1] = False
        result, _ = target_metrics(scores, decisions, rows, cases, mask)
        self.assertEqual(result['unique_claims_after_cap']['precision_on_reviewed'], 1)

    def test_subject_or_target_identity_change_and_duplicate_rejected(self):
        rows, cases = example(); bad = copy.deepcopy(rows); bad[1]['subject_id'] = 12
        with self.assertRaises(ValueError): validate_target_cases(bad, cases)
        bad = copy.deepcopy(rows); bad[1]['object_box'] = bad[0]['object_box']
        with self.assertRaises(ValueError): validate_target_cases(bad, cases)
        with self.assertRaises(ValueError): validate_target_cases(rows, cases + cases)
        bad = copy.deepcopy(rows); bad[1]['labels'][1] = 0
        with self.assertRaises(ValueError): validate_target_cases(bad, cases)

    def test_case_duplicates_do_not_inflate_unique_positive_precision_denominator(self):
        rows, cases = example(); rows.append({**rows[1], 'object_id': 23, 'object_box': [.4, .1, .7, .9]})
        cases.append({**cases[0], 'case_id': 'two', 'alternate_row_index': 2})
        scores = torch.zeros(3, 6); scores[0, 1] = 1
        decisions = torch.full((3, 6), -1); decisions[0, 1] = 1
        result, _ = target_metrics(scores, decisions, rows, cases)
        self.assertEqual(result['all']['cases'], 2)
        self.assertEqual(result['unique_claims_before_cap']['unique_known_claims'], 3)
        self.assertEqual(result['unique_claims_before_cap']['unique_known_positive_claims'], 1)
        self.assertEqual(result['unique_claims_before_cap']['true_selected'], 1)

    def test_regressions_count_images_and_ties_are_not_wins(self):
        old = [{'case_id': str(i), 'image_id': 1, 'predicate': str(i), 'same_class': True, 'gap': 1.} for i in range(2)]
        new = [{**r, 'gap': 0.} for r in old]
        counts, changes = target_changes(old, new)
        self.assertEqual(counts['regressed'], {'cases': 2, 'images': 1})
        self.assertEqual(counts['fixed']['cases'], 0)
        self.assertEqual(changes[0]['gap_change'], -1)
        with self.assertRaises(ValueError): target_changes(old, new[:1])


if __name__ == '__main__': unittest.main()
