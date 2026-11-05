"""User-run neural guards for common fresh init and grouped image sharing."""
import copy
import unittest
import torch
from model import Config,CSACaptioner
from prompt_scratch_model import ScratchCaptioner
import test_prompt_adapter_model as adapter_tests


class Checks(unittest.TestCase):
    def test_common_init_zero_output_and_first_gradient(self):
        prompt,x,v,p=adapter_tests.Checks().fixture();common=copy.deepcopy(prompt.base)
        control=ScratchCaptioner(copy.deepcopy(common),False).eval();model=ScratchCaptioner(common,True).eval()
        self.assertEqual(control.common_digest(),model.common_digest())
        ids=torch.tensor([[1,3,4],[1,5,6]])
        torch.testing.assert_close(model(x,v,ids,p),control(x,v,ids,p),rtol=0,atol=0)
        model(x,v,ids,p).square().mean().backward()
        for module in (model.model.object_prompt,model.model.relation_prompt):
            self.assertGreater(float(module.output.weight.grad.abs().sum()),0)
            self.assertTrue(all(t.grad is not None and torch.isfinite(t.grad).all() for t in module.parameters()))
        self.assertFalse(control.model.object_prompt.output.weight.requires_grad)

    def test_shared_forward_and_gradients_with_opened_adapter(self):
        prompt,x,v,p=adapter_tests.Checks().fixture();a=ScratchCaptioner(prompt.base,True).eval()
        with torch.no_grad():
            a.model.object_prompt.output.weight.normal_(std=.005);a.model.relation_prompt.output.weight.normal_(std=.005)
        b=copy.deepcopy(a);inverse=torch.tensor([0,1,0,1,0]);ids=torch.tensor([[1,3,4]]*5)
        ordinary=a(x[inverse],v[inverse],ids,{k:z[inverse] for k,z in p.items()})
        shared=b(x,v,ids,p,inverse)
        torch.testing.assert_close(ordinary,shared,rtol=1e-4,atol=1e-5)
        ordinary.square().mean().backward();shared.square().mean().backward()
        for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
            self.assertEqual(na,nb)
            if pa.grad is not None:torch.testing.assert_close(pa.grad,pb.grad,rtol=.005,atol=3e-5)


if __name__=='__main__':unittest.main(verbosity=2)
