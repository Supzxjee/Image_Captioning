"""Kaggle synthetic CPU checks; no real checkpoint or cache. Not run locally without torch."""
import copy
import unittest
import torch
from torch import nn
from captioning.models import CaptionDecoder
from phrase_model import Evidence
from pair_model import PairModel,MappedEvidence


def synthetic(device='cpu'):
    torch.manual_seed(42);model=PairModel(CaptionDecoder(16),nn.Linear(768,512)).to(device).eval()
    p=torch.randn(4,196,768,device=device);xy=model.evidence.xy[None].expand(4,-1,-1)
    packed=torch.cat((p,xy),-1);g=torch.randn(4,32,512,device=device)
    boxes=torch.tensor([[[0,0,.5,1],[.5,0,1,1]]]*4,device=device,dtype=torch.float32);geo=torch.randn(4,8,device=device)
    targets=torch.randint(4,16,(8,8),device=device);targets[:,0]=1;targets[:,-2]=2;targets[:,-1]=0
    positive=torch.tensor([[0,1],[1,0]],device=device)
    return model,(packed,g,boxes,geo,targets,positive)


class Checks(unittest.TestCase):
    def test_uniform_coordinates_reproduce_original_encoder(self):
        torch.manual_seed(42);projection=nn.Linear(768,512);old=Evidence(projection).eval();new=MappedEvidence(projection).eval();new.load_state_dict(old.state_dict())
        p=torch.randn(2,196,768);b=torch.tensor([[[0,0,.4,.8],[.4,.4,.8,1.]]]*2);g=torch.randn(2,8)
        expected=old(p,b,g);actual=new(torch.cat((p,old.xy[None].expand(2,-1,-1)),-1),b,g)
        for a,e in zip(actual,expected):torch.testing.assert_close(a,e,rtol=0,atol=0)

    def test_mapped_coordinates_change_membership_and_nearest_fallback(self):
        model,batch=synthetic();p,g,b,geo,_,_=batch
        local=p.clone();local[:,:,768:]=torch.tensor([.8,.8])
        _,attention=model.evidence(local,b,geo)
        self.assertTrue(torch.isfinite(attention).all());torch.testing.assert_close(attention.sum(-1),torch.ones(4,6))
        self.assertTrue((torch.count_nonzero(attention[:,0],dim=1)==1).all())

    def test_ce_gradients_and_frozen_state(self):
        model,batch=synthetic();before={n:p.clone() for n,p in model.named_parameters() if not p.requires_grad}
        loss,ce,_=model(*batch,0.);torch.testing.assert_close(loss,ce);loss.backward()
        for name,p in model.named_parameters():
            if p.requires_grad:self.assertTrue(p.grad is not None and torch.isfinite(p.grad).all(),name)
            else:torch.testing.assert_close(p,before[name],rtol=0,atol=0);self.assertIsNone(p.grad)


if __name__=='__main__':torch.set_num_threads(2);unittest.main(verbosity=2)
