import copy
import unittest
import numpy as np
from hidden_data import select_pairs, validate_row, human_sample, SLOTS, WIDTH


class Checks(unittest.TestCase):
    def test_native_rows_confidence_filter_and_stable_topk(self):
        labels = np.array([0, 0] + [i + 1 for i in range(10)])
        scores = np.array([.9, .2] + [.8] * 10, dtype=np.float32)
        pairs = [(s, o) for s in range(2) for o in range(len(labels)) if s != o]
        features = np.repeat(np.arange(len(pairs), dtype=np.float32)[:, None], WIDTH, 1)
        h, o = np.array(pairs).T
        rows, values, count = select_pairs(labels, scores, h, o, features)
        self.assertEqual(count, 10)
        self.assertEqual(len(rows), SLOTS)
        self.assertEqual([(int(h[i]), int(o[i])) for i in rows], [(0, i) for i in range(2, 10)])
        np.testing.assert_array_equal(values[:, 0], rows.astype(np.float16))

    def test_no_confident_human_does_not_fabricate_evidence(self):
        rows, values, n = select_pairs([0, 3], [.1, .9], [0], [1], np.zeros((1, WIDTH), np.float32))
        self.assertEqual(n, 0)
        self.assertEqual(values.shape, (0, WIDTH))
        self.assertEqual(rows.size, 0)
        rows, values, n = select_pairs([3], [.9], [], [], np.zeros((0, WIDTH), np.float32))
        self.assertEqual(values.shape, (0, WIDTH))

    def test_nonfinite_duplicate_and_wrong_endpoint_rejected(self):
        features = np.zeros((1, WIDTH), np.float32)
        bad = features.copy(); bad[0, 0] = np.nan
        for labels, h, o, f in [([0, 3], [0], [1], bad),
                                 ([0, 3], [1], [0], features),
                                 ([0, 3], [0, 0], [1, 1], np.zeros((2, WIDTH), np.float32))]:
            with self.assertRaises(ValueError):
                select_pairs(labels, [.9, .8], h, o, f)

    def test_padding_and_pair_identity_validation(self):
        features = np.zeros((SLOTS, WIDTH), np.float16)
        valid = np.zeros(SLOTS, bool)
        pairs = np.full((SLOTS, 2), -1, np.int16)
        native = np.full(SLOTS, -1, np.int32)
        labels = np.full((SLOTS, 2), -1, np.int16)
        confidence = np.zeros((SLOTS, 2), np.float32)
        boxes = np.zeros((SLOTS, 2, 4), np.float32)
        validate_row(features, valid, pairs, native, labels, confidence, boxes)
        valid[0] = True; pairs[0] = [0, 1]; native[0] = 2; labels[0] = [0, 3]
        confidence[0] = [.8, .9]; boxes[0] = [[0, 0, .5, 1], [.3, .4, .9, 1]]
        validate_row(features, valid, pairs, native, labels, confidence, boxes)
        features[1, 0] = 1
        with self.assertRaises(ValueError):
            validate_row(features, valid, pairs, native, labels, confidence, boxes)

    def test_human_sample_is_fixed_excludes_known_and_fills_short_strata(self):
        rows = [dict(coco_id=i, filename=f'{i}.jpg') for i in range(50)]
        counts = {i: int(i >= 4) for i in range(50)}
        a = human_sample(rows, counts, [0, 45])
        b = human_sample(list(reversed(rows)), counts, [0, 45])
        self.assertEqual(a, b)
        self.assertEqual(len(a), 32)
        self.assertFalse({0, 45} & {r['coco_id'] for r in a})
        self.assertEqual(sum(counts[r['coco_id']] == 0 for r in a), 3)


if __name__ == '__main__':
    unittest.main(verbosity=2)
