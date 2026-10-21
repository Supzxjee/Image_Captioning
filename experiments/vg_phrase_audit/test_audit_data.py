import io
import json
import unittest
from audit_data import iter_json_array, allow_image, match_pair, mapped_label, contains


class Checks(unittest.TestCase):
    def test_stream_json_boundaries_and_truncation(self):
        rows = [{'id': 1, 'text': 'Đức Anh'}, {'id': 2, 'data': [1, 2]}]
        text = json.dumps(rows, ensure_ascii=False, indent=2)
        for chunk in [1, 2, 7, 31, 1000]:
            self.assertEqual(list(iter_json_array(io.StringIO(text), chunk)), rows)
        for text in ['[{"id":1}', '[1 2]', '[1,]', '[1]garbage']:
            with self.assertRaises(ValueError):
                list(iter_json_array(io.StringIO(text), 3))

    def test_identity_split_not_source_filepath_controls_admission(self):
        coco = {i: dict(split=s) for i, s in enumerate(['train', 'restval', 'val', 'test'])}
        cache = {i: dict(split='train') for i in range(4)}
        self.assertTrue(allow_image(0, coco, cache, {2, 3}, set()))
        self.assertTrue(allow_image(1, coco, cache, {2, 3}, set()))
        for cid in [2, 3, None, 999]:
            self.assertFalse(allow_image(cid, coco, cache, {2, 3}, set()))
        self.assertFalse(allow_image(0, coco, cache, {2, 3}, {0}))

    def test_instance_geometry_and_class_both_required(self):
        s, o = [0, 0, .4, 1], [.5, .1, 1, 1]
        record = dict(selected_labels=[[0, 1], [0, 1]],
                      boxes_original_normalized=[[[.6, 0, 1, 1], o], [s, o]])
        self.assertEqual([x[0] for x in match_pair(record, 0, 1, s, o)], [1])
        self.assertEqual(match_pair(record, 0, 2, s, o), [])
        record['boxes_original_normalized'][0] = [s, o]
        self.assertEqual(len(match_pair(record, 0, 1, s, o)), 2)

    def test_class_and_lexical_filters_are_conservative(self):
        self.assertEqual(mapped_label(dict(names=['man'])), 0)
        self.assertIsNone(mapped_label(dict(names=['man', 'horse'])))
        self.assertIsNone(mapped_label(dict(names=['vehicle'])))
        self.assertTrue(contains('a man riding a horse', 'ride', stem=True))
        self.assertFalse(contains('a man beside a horse', 'ride', stem=True))


if __name__ == '__main__':
    unittest.main(verbosity=2)
