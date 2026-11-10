"""Numerical fixture tests only, no models/Torch/Accelerate."""
from unittest import TestCase, main
import numpy as np
from gradient_agreement import gradient_agreement


class Checks(TestCase):
    def test_scaled_small_relative_error_and_report(self):
        expected=np.tile([.2,-.1,.05,-.03],256)
        actual=expected*(1+.0009)
        for scale in (1,1000):
            result=gradient_agreement(actual*scale,expected*scale)
            self.assertTrue(result['passed'])
            self.assertAlmostEqual(result['relative_l2'],.0009,places=10)
            self.assertGreater(result['max_abs_error'],5e-5)
            self.assertTrue(result['max_error_index'])

    def test_reject_wrong_normalization_and_localized_missing_gradient(self):
        expected=np.ones(100)
        for actual in (expected*.5,expected*2,np.r_[.8,np.ones(99)]):
            self.assertFalse(gradient_agreement(actual,expected)['passed'])

    def test_near_zero_absolute_guard(self):
        expected=np.zeros(20)
        self.assertTrue(gradient_agreement(expected,expected)['passed'])
        self.assertTrue(gradient_agreement(expected+2e-7,expected)['passed'])
        self.assertFalse(gradient_agreement(expected+1e-6,expected)['passed'])

    def test_invalid_shape_and_nonfinite_rejected(self):
        for actual,expected in [([],[]),([1],[1,2]),([float('inf')],[1]),([1],[float('nan')])]:
            with self.assertRaises(ValueError):gradient_agreement(actual,expected)


if __name__=='__main__':main(verbosity=2)
