import unittest
import torch
from captioning.instance_relation_prompts import pair_inventory,fit_gates,passes,metrics,emit_prompts


class InstancePromptTests(unittest.TestCase):
    def test_same_actor_same_class_competitor_without_cross_actor_contamination(self):
        p={'size':torch.tensor([100,100]),'labels':torch.tensor([0,0,1,1]),
           'boxes':torch.tensor([[0,0,10,50],[50,0,60,50],[0,50,10,90],[60,50,70,90]]),
           'subject_indices':torch.tensor([0,0,1]),'object_indices':torch.tensor([2,3,3]),
           'logits':torch.tensor([[2.],[1.],[100.]])}
        inv=pair_inventory(p,['person','horse'],['ride','hold','sit_on','carry','eat','talk_on'],[[],[0]])
        a=next(c for c in inv if c['subject_index']==0 and c['object_index']==2)
        self.assertEqual(a['target_margin'],1.)
        self.assertEqual(a['competitor_object_index'],3)
        self.assertEqual(a['subject_box'],[0.,0.,.1,.5])
        self.assertIsNone(next(c for c in inv if c['subject_index']==1)['target_margin'])

    def test_singleton_can_pass_absolute_gate_and_missing_never_becomes_negative(self):
        policy={'minimum_probability':.8,'minimum_margin':1.}
        self.assertTrue(passes({'probability':.9,'target_margin':None},policy))
        self.assertFalse(passes(None,policy))
        self.assertFalse(passes({'probability':.7,'target_margin':10.},policy))
        self.assertEqual(emit_prompts([],{'ride':policy}),[])

    def test_unknowns_do_not_fit_thresholds_or_count_as_false(self):
        def c(p,m):return {'verb':'hold','probability':p,'target_margin':m}
        cells=[c(.9,1.)]*5+[c(.2,-1.)]*5
        labels=[1]*5+[0]*5
        fit=fit_gates(cells,labels,local_margin=True)
        self.assertEqual(fit,fit_gates(cells+[c(.999,3.)],labels+[-1],local_margin=True))
        result=metrics(cells+[c(.999,3.),None],labels+[-1,1],{'hold':fit})
        self.assertEqual(result['accepted_unknown'],1)
        self.assertEqual(result['accepted_false'],0)
        self.assertEqual(result['full_pool_positive_recall'],5/6)

    def test_pair_gate_removes_ambiguous_negatives_and_tie_keeps_raw_policy(self):
        def c(p,m):return {'verb':'hold','probability':p,'target_margin':m}
        cells=[c(.8,1.)]*5+[c(.9,-1.)]*5+[c(.1,-1.)]*5
        labels=[1]*5+[0]*10
        self.assertIsNone(fit_gates(cells,labels,local_margin=False))
        p=fit_gates(cells,labels,local_margin=True)
        self.assertEqual(p['calibration_true'],5)
        plain=[c(.9,1.)]*5+[c(.1,-1.)]*5
        self.assertIsNone(fit_gates(plain,[1]*5+[0]*5,local_margin=True)['minimum_margin'])


if __name__=='__main__': unittest.main()
