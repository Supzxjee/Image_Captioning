import unittest
from audit_probe_review import geometry


class GeometryControlChecks(unittest.TestCase):
    def test_identical_boxes_have_unit_iou_and_zero_distance(self):
        scores=geometry({'bounds':[[0,0,1,1],[0,0,1,1]]})
        self.assertEqual(scores,{'box_iou':1.,'negative_center_distance':0.})

    def test_disjoint_boxes_have_zero_iou_and_order_by_distance(self):
        near=geometry({'bounds':[[0,0,.1,.1],[.2,0,.3,.1]]})
        far=geometry({'bounds':[[0,0,.1,.1],[.8,0,.9,.1]]})
        self.assertEqual(near['box_iou'],0.)
        self.assertGreater(near['negative_center_distance'],far['negative_center_distance'])


if __name__=='__main__': unittest.main()
