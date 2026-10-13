import unittest
from captioning.detector_pair_matching import normalized_detection_box, match_instances, matched_pairs


class DetectorPairMatchingTests(unittest.TestCase):
    def test_pixel_and_normalized_conversion_match(self):
        expected = [.1, .2, .5, .8]
        self.assertEqual(normalized_detection_box([10, 20, 50, 80], 100, 100), expected)
        self.assertEqual(normalized_detection_box(expected, 100, 100, 'normalized_xyxy'), expected)

    def test_wrong_class_cannot_match_even_with_perfect_box(self):
        gt = {1: {'label': 'person', 'box': [0, 0, 1, 1]}}
        det = [{'label': 'dog', 'box': [0, 0, 1, 1], 'confidence': .99}]
        self.assertEqual(match_instances(gt, det), {})

    def test_one_detection_cannot_match_two_gt_instances(self):
        gt = {1: {'label': 'person', 'box': [0, 0, 1, 1]},
              2: {'label': 'person', 'box': [0, 0, .9, 1]}}
        det = [{'label': 'person', 'box': [0, 0, 1, 1], 'confidence': .9}]
        self.assertEqual(set(match_instances(gt, det)), {1})

    def test_matched_box_replaces_crop_but_preserves_labels_and_gt(self):
        row = {'subject_id': 1, 'object_id': 2, 'subject_box': [0, 0, .5, 1],
               'object_box': [.5, 0, 1, 1], 'labels': [1, -1, 0]}
        matches = {1: {'box': [.01, 0, .49, 1], 'iou': .9, 'confidence': .8},
                   2: {'box': [.51, 0, 1, 1], 'iou': .9, 'confidence': .8}}
        result, missed = matched_pairs([row], matches)
        self.assertEqual(result[0]['labels'], row['labels'])
        self.assertEqual(result[0]['gt_subject_box'], row['subject_box'])
        self.assertEqual(result[0]['subject_box'], matches[1]['box'])
        self.assertEqual(sum(missed.values()), 0)
        result, missed = matched_pairs([row], {1: matches[1]})
        self.assertEqual(result, [])
        self.assertEqual(missed['object'], 1)


if __name__ == '__main__':
    unittest.main()
