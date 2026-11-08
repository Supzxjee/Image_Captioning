"""Pure data checks only; no official scorer/model imports or execution."""
import unittest
from score_spice_subsets import clean, summarize, validate_categories
from compare_spice_subsets import paired


class Checks(unittest.TestCase):
    def test_official_singular_schema(self):
        # SPICE SpiceScorer.java uses Object/Relation/Attribute, not plural names.
        row=dict(scores=clean({'All':{'f':.4},'Object':{'f':.6},
                              'Relation':{'f':.2},'Attribute':{'f':None}}))
        validate_categories([row])
        self.assertEqual(summarize([row])['Objects']['f_x100'],60)
        self.assertIsNone(summarize([row])['Attributes']['f_x100'])
        with self.assertRaisesRegex(ValueError,'observed='):validate_categories([dict(scores={'All':{'f':.1}})])

    def test_paired_common_denominator(self):
        a=[dict(coco_id=1,scores={'Relations':{'f':None}}),dict(coco_id=2,scores={'Relations':{'f':.1}})]
        b=[dict(coco_id=1,scores={'Relations':{'f':.9}}),dict(coco_id=2,scores={'Relations':{'f':.3}})]
        r=paired(a,b)['Relations']
        self.assertEqual(r['common_finite_images'],1)
        self.assertAlmostEqual(r['prompt_minus_control_x100'],20)
        with self.assertRaises(ValueError):paired(a,b[:1])

    def test_undefined_is_not_zero(self):
        rows = [dict(scores=clean({'All': {'f': .5}, 'Relations': {'f': float('nan')}})),
                dict(scores=clean({'All': {'f': .3}, 'Relations': {'f': .2}}))]
        s = summarize(rows)
        self.assertEqual(s['All']['f_x100'], 40)
        self.assertEqual(s['Relations']['f_x100'], 20)
        self.assertEqual(s['Relations']['finite_images'], 1)
        self.assertEqual(s['Objects']['f_x100'], None)
        self.assertEqual(rows[0]['scores']['Relations']['f'], None)


if __name__ == '__main__': unittest.main(verbosity=2)
