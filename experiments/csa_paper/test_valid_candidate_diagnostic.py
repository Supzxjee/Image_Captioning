"""Pure fixtures; never execute model/official scorer."""
import unittest
from score_valid_candidate_oracle import oracle_summary
from analyze_valid_rerank import repeated_trigram


class Checks(unittest.TestCase):
    def test_oracle_and_capture(self):
        r=oracle_summary([dict(selected_rank=1),dict(selected_rank=0)],[[.1,.3,.4,.2,.1],[.5,.4,.1,.2,.3]])
        self.assertAlmostEqual(r['rank0_CIDEr_x100'],30)
        self.assertAlmostEqual(r['selected_CIDEr_x100'],40)
        self.assertAlmostEqual(r['oracle_CIDEr_x100'],45)
        self.assertAlmostEqual(r['fraction_oracle_gain_captured'],2/3)
        self.assertEqual(r['selected_is_CIDEr_best'],1)

    def test_tied_no_gain_and_invalid(self):
        r=oracle_summary([dict(selected_rank=2)],[[.2]*5])
        self.assertIsNone(r['fraction_oracle_gain_captured'])
        with self.assertRaises(ValueError):oracle_summary([dict(selected_rank=0)],[[.1]])
        with self.assertRaises(ValueError):oracle_summary([dict(selected_rank=0)],[[float('nan')]*5])

    def test_repetition_is_specific_heuristic(self):
        self.assertTrue(repeated_trigram('a cat on a cat on a mat'))
        self.assertFalse(repeated_trigram('a cat and a dog on a mat'))


if __name__=='__main__':unittest.main(verbosity=2)
