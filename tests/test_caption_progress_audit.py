import unittest
from audit_caption_progress import chair_aggregate


class CaptionProgressAuditTests(unittest.TestCase):
    def payload(self):
        return {'sentences':[
            {'image_id':1,'mscoco_hallucinated_words':[],'mscoco_generated_words':['person','horse'],'mscoco_gt_words':['person','horse'],'metrics':{'CHAIRs':0}},
            {'image_id':2,'mscoco_hallucinated_words':[['cow','cow']],'mscoco_generated_words':['person','cow'],'mscoco_gt_words':['person','car'],'metrics':{'CHAIRs':1}}],
            'overall_metrics':{'CHAIRs':.5,'CHAIRi':.25,'Recall':.75}}

    def test_reaggregate_counts_and_coverage_not_mean_sentence_ratios(self):
        result,counts=chair_aggregate(self.payload())
        self.assertEqual(result,{'CHAIRs':.5,'CHAIRi':.25,'Recall':.75})
        self.assertEqual(counts['recognized_mentions'],4)
        self.assertEqual(counts['recalled_gt_objects'],3)

    def test_mismatched_report_is_rejected(self):
        value=self.payload();value['overall_metrics']['CHAIRi']=.5
        with self.assertRaises(ValueError):chair_aggregate(value)

    def test_duplicate_image_and_false_sentence_flag_rejected(self):
        value=self.payload();value['sentences'][1]['image_id']=1
        with self.assertRaises(ValueError):chair_aggregate(value)
        value=self.payload();value['sentences'][1]['metrics']['CHAIRs']=0
        with self.assertRaises(ValueError):chair_aggregate(value)
