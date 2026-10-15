import unittest
import torch
from captioning.upt_reference import cpu_nms,prepare_image
from captioning.hoi_transfer_eval import match_inventory,node_key,aggregate_cases,exploratory_pairwise_fallback
from PIL import Image


class PretrainedHOITests(unittest.TestCase):
    def test_nms_overlap_and_threshold_boundary(self):
        boxes=torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.],[20.,20.,30.,30.]])
        self.assertEqual(cpu_nms(boxes,torch.tensor([.9,.8,.7]),.5).tolist(),[0,2])
        self.assertEqual(cpu_nms(boxes,torch.tensor([.9,.8,.7]),1.).tolist(),[0,1,2])
        self.assertEqual(cpu_nms(boxes[:0],torch.tensor([]),.5).tolist(),[])

    def test_class_consistent_one_to_one_not_two_targets_on_one_detection(self):
        rows=[{'subject':'person','subject_box':[0.,0.,.2,1.], 'object':'bicycle','object_box':[.3,.3,.8,.9]},
              {'subject':'person','subject_box':[0.,0.,.2,1.], 'object':'bicycle','object_box':[.31,.3,.81,.9]}]
        pred={'boxes':torch.tensor([[0.,0.,20.,100.],[30.,30.,80.,90.]]),'size':torch.tensor([100,100]),'labels':torch.tensor([0,1])}
        mapping,counts=match_inventory(rows,pred,['person','bicycle'])
        self.assertEqual(counts['matched'],2)
        self.assertEqual(sum(k[0]=='bicycle' for k in mapping),1)
        pred['labels'][1]=0
        mapping,_=match_inventory(rows,pred,['person','bicycle'])
        self.assertTrue(all(k[0]=='person' for k in mapping))

    def test_missing_and_unsupported_never_count_as_correct_negatives(self):
        good={'image_id':1,'same_class':True,'localized':True,'ontology_supported':True,'missing_nodes':[],
              'clip_gap':-1.,'raw_gap':1.,'official_gap':1.,'detection_confidence_gap':-1.}
        missing={**good,'image_id':2,'localized':False,'missing_nodes':['positive']}
        unsupported={**good,'image_id':3,'ontology_supported':False,'clip_gap':2.}
        result=aggregate_cases([good,missing,unsupported])
        self.assertEqual(result['localized_and_ontology_supported'],1)
        self.assertEqual(result['raw_covered_and_correct_of_all'],1)
        self.assertFalse(result['go_to_larger_audit'])
        fallback=exploratory_pairwise_fallback([good,missing,unsupported])
        self.assertEqual(fallback['all']['fallback_wins'],2)
        self.assertEqual(fallback['teacher_routed'],1)
        self.assertEqual(fallback['clip_fallback'],2)

    def test_aspect_ratio_resize_matches_upstream_rounding(self):
        tensor=prepare_image(Image.new('RGB',(640,480),color=(100,120,140)))
        self.assertEqual(tuple(tensor.shape),(3,800,1066))
        self.assertTrue(torch.isfinite(tensor).all())
        wide=prepare_image(Image.new('RGB',(2000,400)))
        # Upstream rounds the short side first: round(1333/5)=267,
        # then recomputes the long side. Preserve that exact behavior.
        self.assertEqual(tuple(wide.shape),(3,267,1335))
