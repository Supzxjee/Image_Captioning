import copy
import unittest
from captioning.selected_relation_prompts import build_selected_prompts


class SelectedPromptTests(unittest.TestCase):
    def row(self):
        return {'filename': 'image.jpg', 'subject': 'person', 'object': 'bicycle',
                'subject_box': [0, 0, .5, 1], 'object_box': [.3, .3, 1, 1],
                'scores': [.1, .8, .1, .1, .1, .1], 'labels': [0, 1, -1, 0, 0, 0],
                'subject_id': 123, 'object_id': 456}

    def test_labels_and_gt_ids_do_not_affect_selection(self):
        row = self.row()
        changed = {**row, 'labels': [1]*6, 'subject_id': 999, 'object_id': 888}
        self.assertEqual(build_selected_prompts([row], [.5]*6), build_selected_prompts([changed], [.5]*6))

    def test_fallback_and_exact_threshold(self):
        row = self.row()
        result = build_selected_prompts([row], [.9]*6)['image.jpg']
        self.assertEqual(result['mode'], 'object_only')
        self.assertEqual(result['relations'], [])
        self.assertEqual(len(build_selected_prompts([row], [.8]*6)['image.jpg']['relations']), 1)

    def test_duplicate_claims_dedup_and_multi_relations_survive(self):
        row = self.row()
        row['scores'][0] = .9
        entry = build_selected_prompts([row, copy.deepcopy(row)], [.5]*6)['image.jpg']
        self.assertEqual(len(entry['relations']), 2)
        self.assertEqual(len(entry['objects']), 2)
        capped = build_selected_prompts([row], [.5]*6, max_relations=1)['image.jpg']
        self.assertEqual(capped['capped_claims'], 1)

    def test_distinct_instances_same_class_not_merged(self):
        row = self.row()
        second = {**row, 'subject_box': [.6, 0, 1, .5]}
        entry = build_selected_prompts([row, second], [.5]*6)['image.jpg']
        self.assertEqual(len(entry['objects']), 3)
        self.assertEqual(len(entry['relations']), 2)

    def test_inventory_retains_objects_without_person_pairs(self):
        inventory = {'cat.jpg': [{'label': 'cat', 'bbox': [0, 0, 1, 1], 'confidence': .9}],
                     'empty.jpg': []}
        result = build_selected_prompts([], [.5]*6, object_inventory=inventory)
        self.assertEqual(result['cat.jpg']['mode'], 'object_only')
        self.assertIn('cat', result['cat.jpg']['prompt'])
        self.assertEqual(result['empty.jpg']['mode'], 'no_objects')
        self.assertEqual(result['empty.jpg']['prompt'], '')

    def test_all_person_object_candidates_have_no_gt_labels(self):
        from captioning.detector_candidates import person_object_pairs
        image = {'filename': 'x.jpg', 'image_id': 1, 'filepath': 'train2014', 'caption_split': 'train'}
        objects = [{'label': 'person', 'bbox': [0, 0, .3, 1], 'confidence': .9},
                   {'label': 'person', 'bbox': [.7, 0, 1, 1], 'confidence': .8},
                   {'label': 'bicycle', 'bbox': [.1, .3, .9, 1], 'confidence': .7}]
        pairs = person_object_pairs(image, objects)
        self.assertEqual(len(pairs), 4)
        self.assertTrue(all('labels' not in row and row['subject_id'] != row['object_id'] for row in pairs))


if __name__ == '__main__':
    unittest.main()
