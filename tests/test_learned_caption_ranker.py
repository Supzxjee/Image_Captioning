import unittest
import torch
from captioning.learned_caption_ranker import CaptionRanker,MODES,ranking_loss,select_rank
from run_learned_caption_ranker_pilot import cohort


def batch():
    torch.manual_seed(7)
    return {'scores':torch.randn(2,5,6),'fixed_scores':torch.tensor([[.5,.6,.4,0,0]]*2,dtype=torch.float64),
            'candidate_mask':torch.tensor([[True,True,True,False,False]]*2),
            'image_global':torch.randn(2,512),'text_global':torch.randn(2,5,512),
            'text_tokens':torch.randn(2,5,7,512),'text_mask':torch.tensor([[[1,1,1,0,0,0,0]]*5]*2).bool(),
            'patches':torch.randn(2,9,768)}


class RankerTests(unittest.TestCase):
    def test_zero_init_reproduces_fixed_without_float32_tie_flips(self):
        b=batch();b['fixed_scores'][0,:2]=torch.tensor([.60000000001,.6],dtype=torch.float64)
        for mode in MODES:
            model=CaptionRanker(mode)
            scores,residual=model(b)
            self.assertTrue(torch.equal(scores[b['candidate_mask']],b['fixed_scores'][b['candidate_mask']]))
            self.assertEqual(select_rank(scores,b['candidate_mask']).tolist(),[0,1])
            self.assertTrue(torch.equal(residual,torch.zeros_like(residual)))

    def test_padded_candidates_do_not_produce_nan_loss_or_gradient(self):
        b=batch();model=CaptionRanker('scores');scores,residual=model(b)
        target=torch.tensor([[1.,0.,.5,100.,100.]]*2)
        loss=ranking_loss(scores,target,b['candidate_mask'],residual)
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(float(model.head[-1].weight.grad.abs().sum()),0)

    def test_no_preference_ties_and_single_candidate_are_safe(self):
        b=batch();b['candidate_mask'][:,1:]=False
        model=CaptionRanker('scores');scores,residual=model(b)
        loss=ranking_loss(scores,torch.ones(2,5),b['candidate_mask'],residual)
        loss.backward();self.assertEqual(float(loss.detach()),0.)
        self.assertEqual(select_rank(scores,b['candidate_mask']).tolist(),[0,0])

    def test_region_branch_receives_gradient_and_ignores_padded_words(self):
        b=batch();model=CaptionRanker('regional');opt=torch.optim.SGD(model.parameters(),lr=.1)
        target=torch.tensor([[1.,0.,.5,0.,0.]]*2)
        for _ in range(2):
            opt.zero_grad();scores,residual=model(b)
            ranking_loss(scores,target,b['candidate_mask'],residual).backward();opt.step()
        self.assertGreater(float(model.patch.weight.grad.abs().sum()),0)
        before=model(b)[0]
        other={**b,'text_tokens':b['text_tokens'].clone()}
        other['text_tokens'][~b['text_mask']]=1000
        self.assertTrue(torch.allclose(before,model(other)[0],equal_nan=True))
        other={**b,'reference_targets':torch.randn(2,5)}
        self.assertTrue(torch.equal(before,model(other)[0]))

    def test_train_tune_validation_image_split_excludes_test(self):
        rows=[]
        for i in range(2000):
            split='train' if i<1280 else 'val' if i<1900 else 'test'
            rows.append({'filename':f'COCO_val2014_{i:012d}.jpg','filepath':'val2014','split':split})
        groups=cohort({'images':rows})
        self.assertEqual({k:len(v) for k,v in groups.items()},{'fit':1024,'tune':256,'validation':512})
        self.assertEqual(groups,cohort({'images':rows}))
        self.assertTrue(all(r['split']!='test' for v in groups.values() for r in v))


if __name__=='__main__':
    torch.set_num_threads(2);unittest.main()
