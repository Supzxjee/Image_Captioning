"""Checks for uncertain-label handling and whole-pair alignment."""
import unittest
from analyze_panel import aggregate,paired_vectors

class Checks(unittest.TestCase):
    def test_unknown_target_is_not_counted_as_correct_or_wrong(self):
        rows=[dict(role='positive',q_claim='yes',target_relation='yes',relation_correctness='correct',confirmed_by_human=True),
              dict(role='negative',q_claim='uncertain',target_relation='uncertain',relation_correctness='wrong',confirmed_by_human=False)]
        m=aggregate(rows)
        self.assertEqual((m['correct'],m['wrong'],m['target_relation_covered']),(1,0,1))
        self.assertEqual(m['Y_uncertainty_bounds'],[0,1])
        self.assertEqual(m['coverage_uncertainty_bounds'],[.5,1])

    def test_pair_alignment_ignores_export_order(self):
        rows=[dict(pair_id='p',role='positive',q_claim='yes',target_relation='yes',relation_correctness='correct'),
              dict(pair_id='p',role='negative',q_claim='no',target_relation='no',relation_correctness='not_applicable')]
        self.assertEqual(paired_vectors(rows[::-1],['p']).tolist(),[[1,0,.5,.5,0]])

if __name__=='__main__':unittest.main()
