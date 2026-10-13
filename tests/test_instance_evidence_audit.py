import unittest
from captioning.instance_evidence_audit import ambiguous_regions, subject_swap_indices


class EvidenceAuditTests(unittest.TestCase):
    def test_cross_class_conflict_without_merging_other_regions(self):
        items = [{'label': 'sandwich', 'bbox': [0, 0, .2, .2]},
                 {'label': 'hot dog', 'bbox': [0, 0, .2, .2]},
                 {'label': 'sandwich', 'bbox': [.5, .5, .7, .7]},
                 {'label': 'person', 'bbox': [0, 0, 1, 1]}]
        result = ambiguous_regions({'a.jpg': items})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['iou'], 1)
        self.assertNotEqual(result[0]['a'], result[0]['b'])

    def test_subject_swaps_never_cross_images_or_objects(self):
        rows = [dict(filename='a', subject_id='s1', object_id='o'),
                dict(filename='a', subject_id='s2', object_id='o'),
                dict(filename='b', subject_id='s3', object_id='o'),
                dict(filename='a', subject_id='s4', object_id='other')]
        self.assertEqual(subject_swap_indices(rows), [(0, 1), (1, 0)])


if __name__ == '__main__':
    unittest.main()
