import copy
import unittest

from audit_vcoco_upt_review import verify_native_row
from captioning.vcoco_relation_audit import audit_image
from tests.test_vcoco_relation_audit import data


class NativeReviewTests(unittest.TestCase):
    def fixture(self):
        ref, native, instances = data()
        raw = copy.deepcopy(native['claims'])
        for c in native['claims']:
            c['score'] *= native['detections'][c['subject']]['confidence']**2.8 * native['detections'][c['object']]['confidence']**2.8
        image, categories = instances['images'][0], {1: 'person', 2: 'horse'}
        row = audit_image(ref, native, instances['annotations'], image, categories)
        row.update(detection_details=native['detections'], raw_claims=raw)
        return row, ref, instances['annotations'], image, categories

    def test_saved_boxes_and_official_prior_replay(self):
        verify_native_row(*self.fixture())

    def test_raw_score_substitution_rejected(self):
        args = self.fixture()
        args[0]['candidates'][0]['score'] = args[0]['raw_claims'][0]['score']
        with self.assertRaisesRegex(ValueError, 'detector-prior'):
            verify_native_row(*args)

    def test_changed_box_rejected(self):
        args = self.fixture()
        args[0]['detection_details'][0]['box'] = [0, 0, .01, .01]
        with self.assertRaisesRegex(ValueError, 'IoU'):
            verify_native_row(*args)

    def test_missing_raw_claim_rejected(self):
        args = self.fixture()
        args[0]['raw_claims'].pop()
        with self.assertRaisesRegex(ValueError, 'inventory'):
            verify_native_row(*args)


if __name__ == '__main__':
    unittest.main()
