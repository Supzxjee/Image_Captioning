import unittest
from audit_regions import region_box, diagnostic_group, sample24


class Checks(unittest.TestCase):
    def test_original_region_is_not_clipped_or_expanded(self):
        m = dict(width=100, height=50)
        self.assertEqual(region_box(dict(x=10,y=5,width=20,height=10),m), [.1,.1,.3,.3])
        for values in [(-1,0,20,10),(90,0,20,10),(0,0,0,5),(0,0,float('nan'),5)]:
            self.assertIsNone(region_box(dict(zip(('x','y','width','height'),values)),m))

    def test_no_pair_images_and_spatial_descriptions_are_allowed(self):
        region = dict(phrase='a dog on the sofa', objects=[], relationships=[])
        self.assertEqual(diagnostic_group(region,0),'no_selected_UPT_pair')
        self.assertEqual(diagnostic_group(region,1),'spatial_or_other')

    def test_sample_is_stable_distinct_and_excludes_previous_images(self):
        groups = ['human_object','spatial_or_other','no_selected_UPT_pair']
        rows = [dict(coco_id=i,region_id=j,diagnostic_group=g,phrase='a man on a bench')
                for i in range(40) for j,g in enumerate(groups)]
        a=sample24(rows,{0,1,2});b=sample24(list(reversed(rows)),{0,1,2})
        self.assertEqual(a,b)
        self.assertEqual(len({r['coco_id'] for r in a}),24)
        self.assertFalse({r['coco_id'] for r in a}&{0,1,2})
        unrelated = dict(coco_id=999,region_id=0,diagnostic_group='human_object',phrase='A gold necklace')
        self.assertEqual(sample24(rows+[unrelated],{0,1,2}),a)


if __name__ == '__main__':
    unittest.main(verbosity=2)
