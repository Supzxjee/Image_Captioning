"""USER-RUN ONLY neural fixtures. Not an author-local check."""
import unittest
import torch
from model import Config,CSACaptioner
from prompt_adapter_model import PromptCSA


class Checks(unittest.TestCase):
    def fixture(self):
        torch.manual_seed(42)
        base=CSACaptioner(Config(vocab=30,dim=16,heads=4,hidden=32,prototypes=4,layers=1,raw_dim=12),torch.randn(8,16),torch.randn(5,16)).eval()
        model=PromptCSA(base).eval();x=torch.randn(2,3,12);v=torch.tensor([[True,True,True],[True,True,False]])
        p=dict(object_text=torch.randn(2,2,16),object_roi=torch.tensor([[0,1],[1,0]]),object_box=torch.rand(2,2,4),
            object_valid=torch.tensor([[True,True],[True,False]]),relation_text=torch.randn(2,1,16),
            relation_pair=torch.tensor([[[0,1]],[[0,1]]]),relation_geometry=torch.randn(2,1,8),relation_valid=torch.tensor([[True],[False]]))
        return model,x,v,p

    def test_closed_and_empty_identity_and_no_nan_rows(self):
        model,x,v,p=self.fixture()
        with torch.no_grad():
            a=model.base.encode(x,v);b=model.encode(x,v,p)
            for left,right in zip(a,b):torch.testing.assert_close(left,right,rtol=0,atol=0)
            model.object_prompt.output.weight.normal_();model.relation_prompt.output.weight.normal_()
            p['object_valid'].zero_();p['relation_valid'].zero_()
            for left,right in zip(a,model.encode(x,v,p)):torch.testing.assert_close(left,right,rtol=0,atol=0)

    def test_opened_text_path_changes_memory_and_has_finite_gradients(self):
        model,x,v,p=self.fixture()
        with torch.no_grad():
            model.object_prompt.output.weight.normal_(std=.01);model.relation_prompt.output.weight.normal_(std=.01)
        a=model.encode(x,v,p);b=model.encode(x,v,p,text_mode='zero')
        self.assertTrue(any(not torch.equal(left,right) for left,right in zip(a,b)))
        sum(z.square().sum() for z in a).backward()
        for module in (model.object_prompt,model.relation_prompt):
            grads=[t.grad for t in module.parameters() if t.grad is not None]
            self.assertTrue(grads and all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(float(module.project[0].weight.grad.abs().sum()),0)

    def test_invalid_pair_or_padding_is_rejected(self):
        model,x,v,p=self.fixture();p['relation_pair'][0,0]=0
        with self.assertRaises(ValueError):model.encode(x,v,p)
        model,x,v,p=self.fixture();p['object_roi'][1,0]=2
        with self.assertRaises(ValueError):model.encode(x,v,p)


if __name__=='__main__':unittest.main(verbosity=2)
