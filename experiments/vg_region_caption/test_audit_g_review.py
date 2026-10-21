import copy
import unittest
from audit_g_review import gates,index


class Checks(unittest.TestCase):
    def test_gain_alone_cannot_pass_factual_guardrails(self):
        metrics={'original':{'CIDEr':1.18},'A':{'CIDEr':1.20,'chair':{'CHAIRs':.052,'CHAIRi':.032,'Recall':.46}},
                 'G':{'CIDEr':1.22,'chair':{'CHAIRs':.053,'CHAIRi':.033,'Recall':.459}}}
        result=gates(metrics,{'delta':.02,'ci95':[.001,.04]})
        self.assertTrue(result['cider_gain_at_least_001'])
        self.assertFalse(result['chair_s_not_increased']);self.assertFalse(result['chair_i_not_increased'])
        self.assertFalse(all(result.values()))
        clean=copy.deepcopy(metrics);clean['G']['chair']=dict(clean['A']['chair'])
        self.assertTrue(all(gates(clean,{'delta':.02,'ci95':[.001,.04]}).values()))
        self.assertFalse(gates(clean,{'delta':.02,'ci95':[0.,.04]})['paired_ci_lower_positive'])

    def test_duplicate_image_scores_are_rejected(self):
        rows=[{'image_id':i,'CIDEr':0.} for i in range(1000)]
        self.assertEqual(len(index(rows,'CIDEr')),1000)
        rows[-1]['image_id']=0
        with self.assertRaises(ValueError):index(rows,'CIDEr')


if __name__=='__main__':unittest.main(verbosity=2)
