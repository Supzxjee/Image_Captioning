"""Synthetic metadata tests; no source labels, neural models or HDF5 required."""
import unittest

from kv_data import (INVERSE, POLICY, bank, candidates, canonical, fact_key, image_id,
                     match_record, merge_facts, multipositive_groups, partition, readiness,
                     template)


class DataChecks(unittest.TestCase):
    def record(self, **changes):
        row = dict(image_id=1, subject='cat', object='car', predicate='inside',
                   label=1, source_row=0, statement='the cat is inside the car',
                   subject_slot=0, object_slot=1, subject_gt_id=11, object_gt_id=22)
        return dict(row, **changes)

    def test_template_inverse_and_canonical_bytes(self):
        self.assertTrue(all(INVERSE[INVERSE[p]] == p for p in INVERSE))
        self.assertEqual(template('car', 'contains', 'cat'), 'the car contains the cat')
        self.assertEqual(template('cat', 'inside', 'car'), 'the cat is inside the car')
        self.assertEqual(canonical({'b': 2, 'a': 1}), b'{"a":1,"b":2}')
        with self.assertRaises(ValueError):
            template('cat', 'left of', 'car')

    def test_source_filter_does_not_reinterpret_frame_or_absence(self):
        row = dict(image='000000000001.jpg', caption='The cat is inside the car.',
                   relation='inside', label=1, vote_true_validator_id='[1,2,3]',
                   vote_false_validator_id='[]', reference_frame=None)
        kept, _ = candidates([row], {1: 'train'}, set(), {'cat', 'car'})
        self.assertEqual(len(kept), 1)
        self.assertIsNone(kept[0]['reference_frame'])
        self.assertFalse(candidates([row], {1: 'val'}, set(), {'cat', 'car'})[0])
        self.assertFalse(candidates([row], {1: 'train'}, {1}, {'cat', 'car'})[0])
        self.assertFalse(candidates([dict(row, vote_false_validator_id='[4]')],
                                    {1: 'train'}, set(), {'cat', 'car'})[0])
        self.assertFalse(candidates([dict(row, relation='left of')],
                                    {1: 'train'}, set(), {'cat', 'car'})[0])
        with self.assertRaises(ValueError):
            image_id(dict(row, image='unknown.jpg'))

    def test_bank_is_label_independent_and_ties_stable(self):
        objects = [dict(conf=.8, bbox=[0, 0, 4, 4], label='wrong'),
                   dict(conf=.8, bbox=[6, 6, 10, 10], label='also_wrong')]
        result = bank(objects, 10, 10)
        self.assertEqual([r['original_index'] for r in result], [0, 1])
        self.assertNotIn('label', result[0])
        self.assertEqual(result, bank([dict(o, label='different') for o in objects], 10, 10))
        with self.assertRaises(ValueError):
            bank([dict(conf=float('nan'), bbox=[0, 0, 1, 1])], 10, 10)

    def test_gt_does_not_substitute_a_missing_predicted_pair(self):
        row = self.record()
        image = dict(width=10, height=10)
        annotations = dict(cat=[dict(id=11, bbox=[0, 0, 4, 4], iscrowd=0)],
                           car=[dict(id=22, bbox=[6, 6, 4, 4], iscrowd=0)])
        proposals = bank([dict(conf=.5, bbox=[0, 0, 4, 4]),
                          dict(conf=.7, bbox=[6, 6, 10, 10])], 10, 10)
        result, why = match_record(row, image, annotations, proposals)
        self.assertIsNone(why)
        self.assertEqual((result['subject_slot'], result['object_slot']), (1, 0))
        self.assertIsNone(match_record(row, image, annotations, proposals[:1])[0])
        self.assertIsNone(match_record(row, image, dict(annotations, cat=[]), proposals)[0])

    def test_equivalent_views_count_one_fact_and_opposite_stays_distinct(self):
        row = self.record()
        inverse = self.record(subject='car', object='cat', predicate='contains',
                              subject_slot=1, object_slot=0, source_row=1,
                              subject_gt_id=22, object_gt_id=11)
        self.assertEqual(fact_key(row), fact_key(inverse))
        opposite = self.record(predicate='contains', source_row=2)
        self.assertNotEqual(fact_key(row), fact_key(opposite))
        facts, quarantine = merge_facts([row, inverse, opposite])
        self.assertEqual(len(facts), 2)
        self.assertFalse(quarantine)
        self.assertEqual(sorted(r['source_rows'] for r in facts), [[0, 1], [2]])

    def test_truth_conflicts_and_ambiguous_entities_are_unknown(self):
        self.assertFalse(merge_facts([self.record(), self.record(label=0, source_row=1)])[0])
        facts, quarantine = merge_facts([self.record(), self.record(subject_gt_id=99, source_row=1)])
        self.assertFalse(facts)
        self.assertEqual(quarantine[0]['reason'], 'ambiguous_gt_entities_for_proposal_fact')

    def test_partition_has_no_head_held_caption_training_images(self):
        facts = [dict(self.record(image_id=i), fact_id=str(i)) for i in range(1, 101)]
        parts = partition(facts, {1, 2, 3})
        self.assertTrue(parts['held'])
        self.assertTrue({1, 2, 3}.issubset({r['image_id'] for r in parts['fit']}))
        self.assertFalse({r['image_id'] for r in parts['fit']} &
                         {r['image_id'] for r in parts['held']})

    def test_mp_requires_observed_contradiction_and_readiness_fails_closed(self):
        positive = dict(fact_id='p', lexical_group=['cat', 'inside', 'car'],
                        label=1, image_id=1, family='containment')
        self.assertEqual(multipositive_groups([positive]), [])
        groups = multipositive_groups([positive, dict(positive, fact_id='n', label=0, image_id=2)])
        self.assertEqual(groups[0]['negative'], ['n'])
        result = readiness({'fit': [positive], 'held': []})
        self.assertFalse(result['passes_operational_gates'])
        self.assertTrue(result['not_statistical_power_or_semantic_success'])
        self.assertEqual(POLICY['maximum_objects'], 6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
