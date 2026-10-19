"""Synthetic CPU preflight only; this does not load the trained checkpoint."""
import unittest
import torch
from pilot_model import EvidenceMemory


class ProbeModelChecks(unittest.TestCase):
    def test_zero_visual_has_no_instance_preference(self):
        torch.manual_seed(42)
        model=EvidenceMemory().eval().requires_grad_(False)
        text=torch.randn(1,512).expand(2,-1)
        boxes=torch.tensor([[[0,0,.3,1],[.2,.2,.5,.8],[0,0,.5,1]],
                            [[.7,0,1,1],[.5,.2,.8,.8],[.5,0,1,1]]],dtype=torch.float32)
        with torch.inference_mode():
            _,scores,_=model.attend(torch.zeros(2,196,768),text,torch.full((2,),2),boxes)
        self.assertTrue(torch.isfinite(scores).all())
        self.assertLess(abs(float(scores[0]-scores[1])),1e-7)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__=='__main__': unittest.main()
