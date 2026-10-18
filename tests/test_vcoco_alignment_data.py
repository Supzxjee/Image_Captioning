import copy
import random
import unittest

from audit_vcoco_alignment_data import audit, caption_index, make_partition, summarize


def fixture():
    images = [{'cocoid': iid, 'filename': f'COCO_val2014_{iid:012d}.jpg', 'split': split}
              for split, start, count in [('train', 1, 10030), ('val', 10031, 5000), ('test', 15031, 2)]
              for iid in range(start, start + count)]

    def actions(ids):
        result = []
        for name, role in [('hold', 'obj'), ('ride', 'instr')]:
            n = len(ids)
            result.append({'action_name': name, 'ann_id': [100000 + i for i in ids],
                           'image_id': ids, 'label': [1, 0, -1] + [1] * (n - 3),
                           'role_name': ['agent', role],
                           'role_object_id': [100000 + i for i in ids] + [200000 + i for i in ids]})
        return result

    return {'images': images}, {'train': actions(list(range(1, 16)) + [10031]),
                                'val': actions(list(range(16, 21)) + [15031])}


class VCOCOAlignmentDataTests(unittest.TestCase):
    def test_official_heldout_excluded_and_groups_disjoint(self):
        dataset, actions = fixture()
        summary, groups = audit(dataset, actions)
        self.assertEqual(summary['source_heldout_images_excluded'], 2)
        self.assertEqual(len(groups['caption_pilot_train10000']), 10000)
        self.assertEqual(len(groups['caption_tune1000']), 1000)
        self.assertEqual(len(groups['caption_validation_remainder4000']), 4000)
        self.assertTrue(set(groups['relation_fit']) <= set(groups['caption_pilot_train10000']))
        self.assertFalse(set(groups['caption_pilot_train10000']) &
                         (set(groups['relation_calibration']) | set(groups['relation_dev'])))
        self.assertNotIn(10031, groups['relation_fit'] + groups['relation_calibration'])
        self.assertNotIn(15031, groups['relation_dev'])

    def test_unknown_missing_role_and_multiple_positive_predicates_remain_distinct(self):
        dataset, actions = fixture()
        actions['train'][0]['role_object_id'][len(actions['train'][0]['ann_id'])] = 0
        summary, _, _ = summarize(actions['train'], caption_index(dataset))
        hold = summary['relations']['holding']
        self.assertEqual(hold['actor_labels'], {'1': 13, '0': 1, '-1': 1})
        self.assertEqual(hold['positive_action_missing_role'], 1)
        self.assertEqual(hold['positive_role_links'], 12)
        self.assertEqual(summary['relations']['riding']['positive_role_links'], 13)

    def test_validation_matches_existing_scst_shuffle(self):
        dataset, actions = fixture()
        _, groups = audit(dataset, actions)
        expected = [r['cocoid'] for r in dataset['images'] if r['split'] == 'val']
        random.Random(20261017).shuffle(expected)
        self.assertEqual(groups['caption_tune1000'], sorted(expected[:1000]))

    def test_deterministic_groups_and_no_input_mutation(self):
        dataset, actions = fixture()
        original = copy.deepcopy(actions)
        self.assertEqual(audit(dataset, actions), audit(dataset, actions))
        self.assertEqual(original, actions)

    def test_inconsistent_or_duplicate_caption_identity_rejected(self):
        dataset, _ = fixture()
        dataset['images'][0]['cocoid'] = 99
        with self.assertRaises(ValueError):
            caption_index(dataset)
        dataset, _ = fixture()
        dataset['images'].append(copy.deepcopy(dataset['images'][0]))
        with self.assertRaises(ValueError):
            caption_index(dataset)

    def test_source_overlap_and_missing_image_rejected(self):
        dataset, actions = fixture()
        actions['val'] = copy.deepcopy(actions['train'])
        with self.assertRaises(ValueError):
            audit(dataset, actions)
        dataset, actions = fixture()
        dataset['images'] = dataset['images'][1:]
        with self.assertRaises(ValueError):
            audit(dataset, actions)

    def test_small_pools_fail_instead_of_silently_changing_protocol(self):
        dataset, _ = fixture()
        with self.assertRaises(ValueError):
            make_partition(caption_index(dataset), {1, 2}, {3})


if __name__ == '__main__':
    unittest.main()
