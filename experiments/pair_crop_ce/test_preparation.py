import unittest
from crop_geometry import crop_rectangle,patch_centers


class Checks(unittest.TestCase):
    def test_expansion_and_rounding(self):
        self.assertEqual(crop_rectangle([.2,.3,.4,.6],[.4,.5,.6,.8],100,100),[16,25,64,85])
        self.assertEqual(crop_rectangle([0,0,.2,.4],[.1,.2,1,1],100,80),[0,0,100,80])

    def test_patch_mapping_keeps_original_frame(self):
        full=patch_centers([0,0,100,200],100,200);crop=patch_centers([20,40,60,100],100,200)
        self.assertEqual(len(crop),196);self.assertAlmostEqual(full[0][0],.5/14)
        self.assertAlmostEqual(crop[0][0],.2+.4*.5/14)
        self.assertAlmostEqual(crop[-1][1],.2+.3*13.5/14)

    def test_invalid_box_rejected(self):
        with self.assertRaises(AssertionError):crop_rectangle([.8,0,.1,1],[0,0,1,1],100,100)


if __name__=='__main__':unittest.main(verbosity=2)
