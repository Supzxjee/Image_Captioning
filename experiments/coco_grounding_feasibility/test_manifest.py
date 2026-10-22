"""Meaningful policy checks before processing the full annotation source; no ML imports."""
import unittest
from prepare_manifest import filter_span,rectangle_mask,cap_person
from manifest_policy import context_exclusion


class Checks(unittest.TestCase):
    def test_compounds_and_real_teddy_class(self):
        self.assertEqual(context_exclusion('a pizza box on a table'.split(),1,2,'pizza'),'compound_head_changes_object')
        self.assertIsNone(context_exclusion('a pizza on a plate'.split(),1,2,'pizza'))
        self.assertIsNone(context_exclusion('a toy teddy bear'.split(),2,4,'teddy bear'))
        self.assertEqual(context_exclusion('a toy car'.split(),2,3,'car'),'artificial_object_modifier')

    def test_instance_count_before_crowd_and_size(self):
        box=dict(bbox=[20,20,30,30],iscrowd=0);image=dict(width=100,height=100);vocab={'dog':4}
        args=(['dog'],0,1,'dog','dog')
        self.assertEqual(filter_span(*args,[box,dict(box,iscrowd=1)],image,vocab)[0],'multiple_class_instances')
        self.assertEqual(filter_span(*args,[dict(box,iscrowd=1)],image,vocab)[0],'single_crowd_instance')
        self.assertIsNone(filter_span(*args,[box],image,vocab)[0])
        self.assertEqual(filter_span(*args,[dict(box,bbox=[1,1,1,1])],image,vocab)[0],'bbox_too_small')
        self.assertEqual(filter_span(*args,[dict(box,bbox=[0,0,90,90])],image,vocab)[0],'bbox_too_large')

    def test_fractional_mask_and_person_cap(self):
        mask=rectangle_mask([25,25,50,50],100,100)
        self.assertAlmostEqual(sum(mask['patch_area_fraction']),49.)
        self.assertAlmostEqual(mask['image_fraction'],.25)
        self.assertIsNone(rectangle_mask([150,150,2,2],100,100))
        rows=[dict(category='dog',span_id=str(i)) for i in range(6)]
        rows += [dict(category='person',span_id='p'+str(i)) for i in range(9)]
        kept,dropped=cap_person(rows)
        self.assertEqual((len(kept),dropped),(8,7))
        reversed_kept,_=cap_person(list(reversed(rows)))
        self.assertEqual([r['span_id'] for r in kept if r['category']=='person'],
                         [r['span_id'] for r in reversed_kept if r['category']=='person'])


if __name__=='__main__':unittest.main(verbosity=2)
