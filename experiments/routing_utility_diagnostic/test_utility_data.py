"""Model-free checks for selection, matched magnitude and selective utility."""
from pathlib import Path
import sys
import unittest
import numpy as np

shared=Path(__file__).resolve().parent.parent/'attention_geometry_diagnostic'
if shared.is_dir():sys.path.insert(0,str(shared))
from utility_data import *


class Checks(unittest.TestCase):
    def test_pair_is_deterministic_and_deduplicates_axis_entries(self):
        r=dict(coco_id=42,pairs=[dict(i=0,j=1,axis=0),dict(i=0,j=1,axis=1),dict(i=1,j=2,axis=0)])
        first=choose_pair(r)
        self.assertEqual(set(first) in ({0,1},{1,2}),True)
        r['pairs'].reverse();self.assertEqual(choose_pair(r),first)

    def test_raw_mass_selection_keeps_cls_information(self):
        q=np.zeros((32,197));q[:,0]=1
        # Q0 looks at the receiver only when conditioned on a tiny spatial mass.
        q[0,0]=.99;q[0,1]=.01
        q[1,0]=.2;q[1,1]=.8
        q[2,0]=.1;q[2,2]=.9
        r=dict(mask=[1]+[0]*195);d=dict(mask=[0,1]+[0]*194)
        selected=q_choice(q,r,d)
        self.assertEqual((selected['q_plus'],selected['q_minus']),(1,2))
        self.assertTrue(selected['distinguishable'])
        self.assertFalse(q_choice(q,r,r)['distinguishable'])

    def test_same_q_tie_uses_first_and_never_rescues(self):
        q=np.zeros((32,197));q[:,1:]=1/196
        selected=q_choice(q,dict(mask=[1]*196),dict(mask=[1]*196))
        self.assertEqual(selected['q_plus'],0)
        self.assertFalse(selected['distinguishable'])

    def test_matched_norms_are_convex_and_zero_safe(self):
        for a,b in [(1.,5.),(30.,.2),(9.,9.)]:
            x,y,h=coefficients(a,b)
            self.assertLessEqual(max(x,y),.1)
            self.assertAlmostEqual(x*a,y*b);self.assertAlmostEqual(h,min(a,b)*.1)
        self.assertIsNone(coefficients(0,3))
        self.assertEqual(coefficients(1,3,0),(0.,0.,0.))
        with self.assertRaises(ValueError):coefficients(1,3,.2)

    def test_global_probability_shift_is_not_selectivity(self):
        scores=dict(A=dict(receiver=-2.,donor=-3.),T=dict(receiver=-1.,donor=-2.),W=dict(receiver=-2.,donor=-3.))
        result=teacher_contrasts(scores)
        self.assertEqual(result['benefit'],1.)
        self.assertEqual(result['selectivity'],0.)
        scores['T']['donor']=-3.;self.assertEqual(teacher_contrasts(scores)['selectivity'],1.)
        scores['W']['donor']=None;self.assertIsNone(teacher_contrasts(scores))

    def test_common_primary_population_and_coverage_guard(self):
        rows=[dict(intervened=True,contrasts=dict(benefit=.2,contrast=.3,selectivity=.1)) for _ in range(100)]
        rows+=[dict(intervened=False,contrasts=dict(benefit=0.,contrast=0.,selectivity=0.)) for _ in range(38)]
        out=summarize_teacher(rows)
        self.assertEqual(out['primary_images'],100);self.assertTrue(out['positive_teacher_signal'])
        self.assertEqual(out['metrics']['benefit']['n'],out['metrics']['selectivity']['n'])
        self.assertFalse(summarize_teacher(rows[:2])['coverage_guard_pass'])
        rows[0]['contrasts']=None;self.assertEqual(summarize_teacher(rows)['primary_images'],99)


if __name__=='__main__':unittest.main(verbosity=2)
