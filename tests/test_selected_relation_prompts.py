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


if __name__ == '__main__':
    unittest.main()
