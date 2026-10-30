"""NumPy-only construction oracles; no model or semantic-performance claims."""
import math
import unittest

import numpy as np


def read(query, keys, values):
    score = query @ keys.T / math.sqrt(keys.shape[-1])
    weight = np.exp(score - score.max(axis=-1, keepdims=True))
    weight /= weight.sum(axis=-1, keepdims=True)
    return weight @ values


class ConstructionChecks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(42)
        self.query = rng.normal(size=(4, 3))
        self.keys = rng.normal(size=(5, 3))
        self.values = rng.normal(size=(5, 7))
        self.values[-1] = 0  # Null is a singleton orbit.
        self.inverse = np.array([1, 0, 3, 2, 4])

    def test_projection_key_swap_and_control_are_distinct(self):
        projected = (self.values + self.values[self.inverse]) / 2
        np.testing.assert_allclose(projected, projected[self.inverse], atol=1e-12)
        np.testing.assert_allclose(
            projected, (projected + projected[self.inverse]) / 2, atol=1e-12
        )
        swap = np.array([1, 0, 2, 3, 4])
        np.testing.assert_allclose(
            read(self.query, self.keys, projected),
            read(self.query, self.keys[swap], projected),
            rtol=1e-12, atol=1e-12,
        )
        difference = read(self.query, self.keys, self.values) - read(
            self.query, self.keys[swap], self.values
        )
        self.assertGreater(float(np.max(np.abs(difference))), 1e-4)

    def test_joint_permutation_and_null_only(self):
        order = np.array([4, 2, 0, 3, 1])
        np.testing.assert_allclose(
            read(self.query, self.keys, self.values),
            read(self.query, self.keys[order], self.values[order]),
            rtol=1e-12, atol=1e-12,
        )
        np.testing.assert_allclose(
            read(self.query, self.keys[-1:], self.values[-1:]), 0, atol=1e-12
        )

    def test_projection_can_reduce_value_energy(self):
        projected = (self.values + self.values[self.inverse]) / 2
        self.assertLess(float(np.square(projected).sum()),
                        float(np.square(self.values).sum()))
        # Equal-valued orbits can hold entirely incorrect content.
        constant = np.full_like(self.values, 17.0)
        np.testing.assert_allclose(
            read(self.query, self.keys, constant),
            read(self.query, self.keys[self.inverse], constant), atol=1e-12,
        )

    def test_additive_encoder_collapses_opposite_orbits(self):
        rng = np.random.default_rng(7)
        subject, obj, relation, inv_relation, geom = rng.normal(size=(5, 4))
        a, b, c, d = rng.normal(size=(4, 4, 4))

        def encode(s, o, p, g):
            return a @ s + b @ o + c @ p + d @ g

        fact = (encode(subject, obj, relation, geom)
                + encode(obj, subject, inv_relation, -geom)) / 2
        opposite = (encode(subject, obj, inv_relation, geom)
                    + encode(obj, subject, relation, -geom)) / 2
        np.testing.assert_allclose(fact, opposite, rtol=1e-12, atol=1e-12)

    def test_interaction_can_avoid_collapse_but_is_not_learning(self):
        # Toy p*g changes sign for only-predicate reversal, not joint reversal.
        def toy(subject, obj, predicate, geometry):
            return np.array([subject + obj, predicate * geometry])

        first = (toy(2, 9, 1, 3) + toy(9, 2, -1, -3)) / 2
        same = (toy(9, 2, -1, -3) + toy(2, 9, 1, 3)) / 2
        opposite = (toy(2, 9, -1, 3) + toy(9, 2, 1, -3)) / 2
        np.testing.assert_array_equal(first, same)
        self.assertGreater(float(np.linalg.norm(first - opposite)), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
