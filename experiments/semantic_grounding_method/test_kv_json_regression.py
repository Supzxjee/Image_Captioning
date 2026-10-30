"""Regression: actual NumPy count types, not only Python-list fixtures."""
import json
import unittest
import numpy as np
from kv_data import canonical
from kv_pilot_evaluate import paired_false


class Checks(unittest.TestCase):
    def test_real_numpy_flag_arrays_produce_plain_json_counts(self):
        result = paired_false(np.array([1,1,0],dtype=np.int64),np.array([0,0,1],dtype=np.int64))
        self.assertIs(type(result['repaired']),int)
        self.assertIs(type(result['introduced']),int)
        self.assertEqual(json.loads(canonical(result))['repaired'],2)

    def test_mcnemar_denominator_does_not_overflow_numpy_int64(self):
        result = paired_false(np.ones(100,dtype=np.int64),np.zeros(100,dtype=np.int64))
        self.assertAlmostEqual(result['exact_mcnemar_p'],2./2**100,places=40)

    def test_shared_writer_supports_nested_numpy_scalars(self):
        result = json.loads(canonical(dict(count=np.int64(4),flag=np.bool_(True),
                                           nested=[np.int32(2),np.float32(.5)])))
        self.assertEqual(result,dict(count=4,flag=True,nested=[2,.5]))

    def test_writer_keeps_native_identity_and_rejects_nonfinite_or_arrays(self):
        self.assertEqual(canonical(dict(b=2,a=1)),b'{"a":1,"b":2}')
        with self.assertRaises(ValueError):canonical(dict(x=np.float32('nan')))
        with self.assertRaises(TypeError):canonical(dict(x=np.array([1,2])))


if __name__=='__main__':unittest.main(verbosity=2)
