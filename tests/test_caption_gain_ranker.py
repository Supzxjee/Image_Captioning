import unittest
import numpy as np
import torch
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.cider.cider_scorer import CiderScorer
from captioning.caption_gain_ranker import CaptionGainRanker,gain_loss,propose,calibrate
from captioning.fixed_idf_targets import fixed_scores
from run_caption_gain_pilot import split_groups
from tests.test_learned_caption_ranker import batch


class GainTests(unittest.TestCase):
    def test_initial_gain_and_anchor_exactly_zero_with_fixed_fallback(self):
        b=batch()
        for mode in ('scores','global','regional'):
            model=CaptionGainRanker(mode);g,base=model(b)
            self.assertTrue(torch.equal(g[b['candidate_mask']],torch.zeros_like(g[b['candidate_mask']])))
            self.assertEqual(propose(g,base).tolist(),[1,1])
            other={**b,'reference_targets':torch.randn(2,5)}
            self.assertTrue(torch.equal(g,model(other)[0]))

    def test_gradient_padding_and_single_candidate(self):
        b=batch();model=CaptionGainRanker('regional');opt=torch.optim.SGD(model.parameters(),lr=.05)
        for _ in range(2):
            opt.zero_grad();g,base=model(b)
            loss=gain_loss(g,base,torch.tensor([[1.,.4,0.,99.,99.]]*2),b['candidate_mask'])
            self.assertTrue(torch.isfinite(loss));loss.backward();opt.step()
        self.assertGreater(float(model.patch.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        b['candidate_mask'][:,1:]=False;g,base=model(b)
        self.assertEqual(float(gain_loss(g,base,torch.ones(2,5),b['candidate_mask']).detach()),0.)

    def test_guard_never_invents_a_third_choice_and_tie_keeps_fixed(self):
        g=torch.tensor([[0.,.3,.2],[0.,.6,.4],[0.,0.,0.]])
        base=torch.zeros(3,dtype=torch.long)
        self.assertEqual(propose(g,base).tolist(),[1,1,0])
        self.assertEqual(propose(g,base,.5).tolist(),[0,1,0])
        with self.assertRaises(ValueError):propose(g,base,-.1)

    def test_calibration_uses_selected_proposal_error_not_oracle_target(self):
        pred=np.tile([0.,.3,.2],(20,1));truth=np.tile([0.,-1.,10.],(20,1))
        policy=calibrate(pred,np.zeros(20,dtype=int),truth)
        self.assertAlmostEqual(policy['correction'],1.3)
        self.assertEqual(policy['quantile_order'],19)
        self.assertFalse(policy['selective_precision_guaranteed'])
        self.assertEqual(calibrate(np.zeros((20,3)),np.ones(20,dtype=int),truth)['correction'],0.)

    def test_calibration_and_epoch_tune_are_disjoint_reproducible(self):
        make=lambda ids:[{'image_id':i,'coco_id':i} for i in ids]
        old={'fit':make(range(1024)),'tune':make(range(1024,1280)),'validation':make(range(1280,1792))}
        result=split_groups(old)
        a={r['image_id'] for r in result['tune']};b={r['image_id'] for r in result['calibration']}
        self.assertEqual(len(a),128);self.assertEqual(len(b),128)
        self.assertFalse(a&b);self.assertEqual(a|b,set(range(1024,1280)))
        self.assertEqual(result,split_groups(old))

    def test_frozen_fit_idf_matches_standard_and_subset_is_invariant(self):
        refs={1:['a man with an umbrella','a man holding an umbrella'],
              2:['a dog in a field','a brown dog running'],3:['a bus on a street','a large bus']}
        pred={1:['a man holding an umbrella'],2:['a dog running'],3:['a bus']}
        scorer=CiderScorer()
        for i in refs:scorer+=(pred[i][0],refs[i])
        scorer.compute_doc_freq();df=dict(scorer.document_frequency)
        _,expected=Cider().compute_score(refs,pred)
        np.testing.assert_allclose(fixed_scores(refs,pred,df,3),expected,atol=1e-12)
        np.testing.assert_allclose(fixed_scores({1:refs[1]},{1:pred[1]},df,3),expected[:1],atol=1e-12)
        self.assertEqual(df,dict(scorer.document_frequency))


if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
