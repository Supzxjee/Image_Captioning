"""Synthetic CPU check: same base memory, image-only evidence, frozen parameters."""
import unittest
import torch
from pilot_model import PilotModel
from synthetic_fixtures import FakeBase,batch


class ModelChecks(unittest.TestCase):
    def test_only_relation_query_changes_and_no_parameter_update(self):
        torch.manual_seed(42);model=PilotModel(FakeBase(),'F').eval().requires_grad_(False)
        data=batch();data['valid'][:,3:]=False
        state={k:v.clone() for k,v in model.state_dict().items()}
        with torch.inference_mode():
            positive,pad=model.memory(data);attention=model.last_attention.clone()
            alternate=data.copy();alternate['text']=data['text'].clone();alternate['text'][:,2]=torch.randn(2,512)
            negative,pad2=model.memory(alternate)
            torch.testing.assert_close(positive[:,:-9],negative[:,:-9],rtol=0,atol=0)
            torch.testing.assert_close(positive[:,-9:-7],negative[:,-9:-7],rtol=0,atol=0)
            self.assertGreater(float((attention[:,2]-model.last_attention[:,2]).abs().sum()),0)
            self.assertGreater(float((positive[:,-7]-negative[:,-7]).abs().sum()),0)
            torch.testing.assert_close(pad,pad2)
            # Query changes cannot convey a phrase through constant image values.
            data['visual']=data['visual'][:,:1].expand_as(data['visual']).clone()
            alternate['visual']=data['visual']
            p,_=model.memory(data);n,_=model.memory(alternate)
            torch.testing.assert_close(p[:,-7],n[:,-7],atol=2e-5,rtol=2e-5)
        for key,value in state.items():torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__=='__main__':unittest.main(verbosity=2)
