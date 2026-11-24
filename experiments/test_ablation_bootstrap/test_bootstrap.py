import unittest

import numpy as np

from bootstrap import resample, statistics
from prepare_inputs import indexed, same_ids


class Checks(unittest.TestCase):
    def test_pairing_uses_ids_not_input_order(self):
        rows = [{'id': 2, 'x': 20}, {'id': 1, 'x': 10}]
        left, right = indexed(rows, 'id'), indexed(rows[::-1], 'id')
        same_ids(left, right)
        self.assertEqual([left[i] for i in sorted(left)], [right[i] for i in sorted(right)])
        with self.assertRaises(ValueError): indexed(rows+rows[:1], 'id')
        with self.assertRaises(ValueError): same_ids(left, {1: right[1]})

    def test_chairi_is_ratio_of_totals(self):
        # Per-caption ratios average to 50%; correct corpus ratio is 10%.
        rows = np.array([[[1., 1, 1]]*3, [[3., 0, 9]]*3])
        np.testing.assert_allclose(statistics(rows), [[200, 50, 10]]*3)
        with self.assertRaises(ValueError): statistics(np.zeros((2, 3, 3)))

    def test_identical_arms_have_exact_zero_differences(self):
        rows = np.array([[[1., 0, 2]]*3, [[3., 1, 4]]*3])
        np.testing.assert_array_equal(resample(rows, 37, 42), np.zeros((37, 2, 3)))

    def test_shared_draws_match_explicit_reference_and_batching(self):
        rows = np.array([[[1., 1, 2], [2., 0, 1], [4., 0, 3]],
                         [[5., 0, 4], [7., 1, 3], [9., 1, 8]],
                         [[3., 0, 6], [6., 1, 7], [2., 0, 2]]])
        rng = np.random.Generator(np.random.PCG64(42))
        expected = []
        for indices in rng.integers(3, size=(19, 3)):
            values = statistics(rows[indices])
            expected.append(values[:1]-values[1:])
        actual = resample(rows, 19, 42, batch_size=4)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual, resample(rows, 19, 42, batch_size=7))


if __name__ == '__main__':
    unittest.main(verbosity=2)
