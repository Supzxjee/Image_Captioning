"""Small analytical reference fixtures for paired result summaries, no models/scorers."""
import unittest
import numpy as np
from audit_prompt_pilot_result import paired


def fixture(hallucination=False,score=0,caption='same'):
    return dict(rows={i:dict(caption=caption) for i in range(3)},cider={i:score for i in range(3)},
        chair={i:dict(mscoco_hallucinated_words=['ghost'] if hallucination else []) for i in range(3)})


class Checks(unittest.TestCase):
    def test_identical_results_have_zero_intervals_and_no_discordant_pairs(self):
        a=fixture();r=paired(a,a)
        self.assertEqual(r['CIDEr_delta'],0);self.assertEqual(r['CIDEr_paired95CI'],[0,0])
        self.assertEqual(r['CHAIRs_delta'],0);self.assertEqual(r['McNemar_exact_p'],1)
        self.assertEqual(r['changed_captions'],0)

    def test_delta_direction_and_exact_small_mcnemar(self):
        r=paired(fixture(),fixture(True,.1,'changed'))
        self.assertAlmostEqual(r['CIDEr_delta'],.1);np.testing.assert_allclose(r['CIDEr_paired95CI'],[.1,.1])
        self.assertEqual(r['CHAIRs_delta'],1);self.assertEqual(r['new_hallucinated_captions'],3)
        self.assertEqual(r['repaired_captions'],0);self.assertEqual(r['McNemar_exact_p'],.25)
        self.assertEqual(r['changed_captions'],3)


if __name__=='__main__':unittest.main(verbosity=2)
