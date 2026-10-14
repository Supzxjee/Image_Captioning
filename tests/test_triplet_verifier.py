import unittest
import torch
from captioning.triplet_verifier import TripletVerifier
from captioning.instance_contrast_alignment import supervised_alignment_loss


class TripletTests(unittest.TestCase):
    def test_text_and_actor_order_matter_union_does_not(self):
        torch.manual_seed(42);model=TripletVerifier(dim=8).eval()
        views=torch.randn(2,3,8);text=torch.randn(2,6,8);geometry=torch.randn(2,10)
        base=model(views,geometry,text);changed=views.clone();changed[:,2]=100
        self.assertTrue(torch.equal(base,model(changed,geometry,text)))
        self.assertFalse(torch.equal(base,model(views,geometry,text.roll(1,1))))
        changed=views.clone();changed[:,0],changed[:,1]=views[:,1],views[:,0]
        self.assertFalse(torch.equal(base,model(changed,geometry,text)))

    def test_unknown_scores_do_not_enter_loss(self):
        scores=torch.tensor([[.2,.3],[.1,.4]])
        labels=torch.tensor([[1,-1],[0,-1]]);edges=torch.tensor([[0,0,1,0]])
        original=supervised_alignment_loss(scores,labels,edges,temperature=1.)[0]
        changed=scores.clone();changed[:,1]=1000
        self.assertTrue(torch.equal(original,supervised_alignment_loss(changed,labels,edges,temperature=1.)[0]))


if __name__=='__main__':unittest.main()
