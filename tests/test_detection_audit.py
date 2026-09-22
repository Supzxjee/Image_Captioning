import unittest

from captioning.detection_audit import audit_detections


class DetectionAuditTests(unittest.TestCase):
    def test_reports_coverage_bbox_confidence_and_duplicates(self):
        records = {
            'a.jpg': {'filename': 'a.jpg', 'split': 'train'},
            'b.jpg': {'filename': 'b.jpg', 'split': 'val'},
            'c.jpg': {'filename': 'c.jpg', 'split': 'test'},
        }
        source = {
            '/old/a.jpg': {'objects': [
                {'label': 'Person', 'bbox': [0, 0, 50, 50], 'conf': .9},
                {'label': 'person', 'bbox': [1, 1, 50, 50], 'conf': .8},
                {'label': 'dog', 'bbox': [-1, 1, 3, 3], 'conf': .2},
            ]},
            'b.jpg': {'objects': [{'label': 'cat', 'bbox': [5, 5, 4, 8], 'conf': 1.2}]},
            'extra.jpg': {'objects': []},
        }
        report, rows, labels = audit_detections(
            source, records, image_size=lambda *_: (100, 100))
        self.assertEqual(report['scope']['missing_source_images'], 1)
        self.assertEqual(report['scope']['extra_source_images'], 1)
        self.assertEqual(report['detections']['total'], 4)
        self.assertEqual(report['confidence']['below_0.25'], 1)
        self.assertEqual(report['issues']['out_of_bounds_bbox'], 1)
        self.assertEqual(report['issues']['nonpositive_bbox'], 1)
        self.assertEqual(report['issues']['same_label_duplicate_pairs'], 1)
        self.assertEqual(labels['person'], 2)
        self.assertEqual(rows[2]['missing_source'], 1)

    def test_invalid_entry_is_flagged_instead_of_crashing(self):
        report, _, _ = audit_detections(
            {'a.jpg': {'prompt': 'person'}},
            {'a.jpg': {'filename': 'a.jpg', 'split': 'train'}})
        self.assertEqual(report['issues']['invalid_entry'], 1)


if __name__ == '__main__':
    unittest.main()
