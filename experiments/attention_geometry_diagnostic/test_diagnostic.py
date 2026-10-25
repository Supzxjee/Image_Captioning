"""CPU synthetic checks only. No model inference."""
import unittest
import numpy as np
from diagnostic_data import *


class Checks(unittest.TestCase):
    def test_derangement_preserves_complete_maps(self):
        for n in range(2,10):
            p=derangement(n,77)
            self.assertEqual(sorted(p.tolist()),list(range(n)))
            self.assertTrue(np.all(p!=np.arange(n)))

    def test_alias_subwords_and_truncation(self):
        text='cat cell phone'
        enc=dict(offset_mapping=[(0,0),(0,3),(4,8),(9,12),(12,14),(0,0)],attention_mask=[1]*6)
        aliases=[(('cat',),'cat'),(('cell','phone'),'cell phone'),(('phone',),'phone')]
        rows=phrase_positions(text,aliases,enc)
        self.assertEqual([r['positions'] for r in rows],[[1],[2,3,4]])
        enc['offset_mapping'][4]=(0,0)
        self.assertEqual(len(phrase_positions(text,aliases,enc)),1)

    def test_text_absence_never_uses_reference(self):
        self.assertIsNone(prompt_text(dict(tokens=[])))
        with self.assertRaises(ValueError):prompt_text(dict(prompt='cat',text='dog'))
        self.assertEqual(text_index({'/images/a.jpg':'cat','b.jpg':{'prompt':'dog'}}),{'a.jpg':'cat','b.jpg':'dog'})
        with self.assertRaises(ValueError):text_index({'/x/a.jpg':'cat','/y/a.jpg':'dog'})

    def test_correct_maps_beat_swapped_and_uniform_area(self):
        left=np.tile(np.r_[np.ones(4),np.zeros(10)],14);right=left.reshape(14,14)[:,::-1].ravel()
        phrases=[dict(box=[0,0,4/14,1],mask=left.tolist()),dict(box=[10/14,0,1,1],mask=right.tolist())]
        pairs=axis_pairs(phrases);self.assertEqual(pairs,[dict(i=0,j=1,axis=0,sign=1)])
        maps=np.stack([left/left.sum(),right/right.sum()])
        result=evaluate_image(maps,phrases,pairs,42)
        self.assertAlmostEqual(result['true']['mass'],1.)
        self.assertAlmostEqual(result['uniform']['excess_mass'],0.,places=10)
        self.assertEqual(result['true']['axis_accuracy'],1.)
        self.assertEqual(result['within_image_full_map_derangement']['axis_accuracy'],0.)
        self.assertEqual(result['image_mean']['axis_accuracy'],.5)
        self.assertEqual(result['image_mean']['axis_ties'],1)
        report=aggregate([dict(metrics=result) for _ in range(100)])
        self.assertEqual(report['status'],'PREMISE_SIGNAL_GO_DESIGN_ONLY')
        json.dumps(report,allow_nan=False)
        self.assertEqual(aggregate([dict(metrics=result)])['status'],'NOT_YET_GO')
        with self.assertRaises(ValueError):summaries(maps*2,phrases,pairs)

    def test_axis_gap_is_edge_gap_not_center_distance(self):
        p=[dict(box=[0,0,.8,.8]),dict(box=[.4,.4,1,1])]
        self.assertEqual(axis_pairs(p),[])

    def test_bootstrap_is_paired_image_level_and_no_caption_claim(self):
        result=paired_ci([.1]*110)
        np.testing.assert_allclose(result['ci95'],[.1,.1])
        self.assertEqual(result['n'],110)
        self.assertEqual(paired_ci([1])['ci95'],None)


if __name__=='__main__':unittest.main(verbosity=2)
