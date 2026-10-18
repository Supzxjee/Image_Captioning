import unittest

from audit_vcoco_detector_coverage import candidate_inventory, measure
from tests.test_vcoco_grounded_caption_data import fixture
from prepare_vcoco_grounded_caption_data import grounded_images


def detection(label, box, conf=.9):
    return {'label': label, 'bbox': box, 'conf': conf}


class DetectorCoverageTests(unittest.TestCase):
    def data(self):
        actions, instances, captions = fixture()
        for index, row in enumerate(instances['annotations']):
            row['bbox'] = [20 + 80 * index, 20, 30, 50]
        records, _ = grounded_images(actions, instances, captions, {100})
        objects = [detection('person' if r['category_id'] == 1 else 'horse',
                             [r['bbox'][0], r['bbox'][1], r['bbox'][0] + r['bbox'][2], r['bbox'][1] + r['bbox'][3]])
                   for r in instances['annotations']]
        return records, instances, {records[0]['filename']: {'objects': objects}}

    def test_all_links_multiple_predicates_and_shared_candidates(self):
        records, instances, indexed = self.data()
        stats, reports = measure(records, indexed, instances, 10)
        self.assertEqual(stats['localized_positive_links'], 3)
        self.assertEqual(stats['link_localization_recall'], 1)
        self.assertEqual(stats['candidate_pairs'], 12)  # 3 people x 4 other detections
        self.assertTrue(all(r['status'] == 'both_matched' for r in reports[0]['links']))

    def test_missing_record_vs_empty_detection_remain_distinct(self):
        records, instances, indexed = self.data()
        missing, _ = measure(records, {}, instances, 10)
        empty, _ = measure(records, {records[0]['filename']: {'objects': []}}, instances, 10)
        self.assertEqual(missing['missing_cache_images'], 1)
        self.assertEqual(empty['missing_cache_images'], 0)
        self.assertEqual(empty['empty_retained_detection_images'], 1)
        self.assertEqual(missing['positive_links'], empty['positive_links'])
        self.assertEqual(empty['link_localization_recall'], 0)

    def test_missing_object_is_counted_for_each_predicate(self):
        records, instances, indexed = self.data()
        indexed[records[0]['filename']]['objects'] = indexed[records[0]['filename']]['objects'][:3]
        stats, reports = measure(records, indexed, instances, 10)
        self.assertEqual(stats['localized_positive_links'], 0)
        self.assertTrue(all(r['status'] == 'object_missing' for r in reports[0]['links']))

    def test_wrong_class_cannot_match(self):
        records, instances, indexed = self.data()
        for row in indexed[records[0]['filename']]['objects'][3:]:
            row['label'] = 'bicycle'
        stats, _ = measure(records, indexed, instances, 10)
        self.assertEqual(stats['localized_positive_links'], 0)

    def test_all_gt_compete_not_only_relation_endpoints(self):
        records, instances, indexed = self.data()
        # New unlinked horse exactly matches the only horse detection; linked horse
        # overlaps but must not steal this detection merely because it is a role.
        instances['annotations'].append({'id': 20, 'image_id': 100, 'category_id': 2,
                                         'bbox': [261, 20, 30, 50], 'iscrowd': 0})
        indexed[records[0]['filename']]['objects'] = indexed[records[0]['filename']]['objects'][:3] + [detection('horse', [261, 20, 291, 70])]
        stats, reports = measure(records, indexed, instances, 10)
        self.assertEqual(stats['localized_positive_links'], 0)
        self.assertEqual(reports[0]['ground_truth_instances_for_matching'], 6)

    def test_low_confidence_nms_and_cap_applied_without_role_labels(self):
        entry = {'objects': [detection('person', [0, 0, 20, 40]),
                             detection('person', [0, 0, 20, 40], .8),
                             detection('horse', [30, 0, 60, 40], .7),
                             detection('cup', [65, 0, 75, 20], .4)]}
        items, pairs, _ = candidate_inventory(entry, 100, 100, 10)
        self.assertEqual(len(items), 2)
        self.assertEqual(pairs, 1)
        self.assertEqual(candidate_inventory(entry, 100, 100, 1)[1], 0)

    def test_no_positive_link_does_not_imply_false_candidates(self):
        records, instances, indexed = self.data()
        records[0]['relations'] = []
        stats, _ = measure(records, indexed, instances, 10)
        self.assertIsNone(stats['link_localization_recall'])
        self.assertEqual(stats['candidate_pairs_on_images_without_focus_links'], 12)
        self.assertNotIn('false_positive_relations', stats)


if __name__ == '__main__':
    unittest.main()
