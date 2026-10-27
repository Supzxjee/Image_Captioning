"""Synthetic torch checks: USER-RUN Kaggle only, not executed locally."""
import copy
import os
import unittest
import torch
from torch.nn import functional as F
from gain_model import GainHead,attributed,check_completeness


def check_distributed_head(rank,world):
    import torch.distributed as dist
    from gain_runtime import sync_gradients
    with torch.random.fork_rng(devices=[rank]):
        torch.manual_seed(817)
        h=torch.randn(8,2,512,device=f'cuda:{rank}');z=torch.randn(8,196,512,device=f'cuda:{rank}')
        vocab=torch.randn(7,512,device=f'cuda:{rank}');head=GainHead().cuda(rank)
        with torch.no_grad():head.o.weight.normal_(0,.01)
        full=copy.deepcopy(head)
        local=head(h[rank*4:(rank+1)*4],z[rank*4:(rank+1)*4],vocab).square().mean();local.backward();sync_gradients(head,world)
        full(h,z,vocab).square().mean().backward()
        a=torch.cat([p.grad.flatten() for p in head.parameters()]);b=torch.cat([p.grad.flatten() for p in full.parameters()])
        torch.testing.assert_close(a,b,atol=1e-5,rtol=1e-4)
        opt1=torch.optim.AdamW(head.parameters(),lr=1e-4,weight_decay=.01);opt2=torch.optim.AdamW(full.parameters(),lr=1e-4,weight_decay=.01)
        torch.nn.utils.clip_grad_norm_(head.parameters(),1.);torch.nn.utils.clip_grad_norm_(full.parameters(),1.);opt1.step();opt2.step()
        for p,q in zip(head.parameters(),full.parameters()):torch.testing.assert_close(p,q,atol=1e-5,rtol=1e-4)


class ModelChecks(unittest.TestCase):
    def test_zero_branch_and_two_step_gradients(self):
        torch.manual_seed(42);head=GainHead();h=torch.randn(2,2,512);z=torch.randn(2,196,512);vocab=torch.randn(11,512)
        self.assertEqual(sum(p.numel() for p in head.parameters()),262144)
        self.assertEqual(float(head(h,z,vocab).abs().max()),0)
        opt=torch.optim.AdamW(head.parameters(),lr=1e-4)
        for step in range(2):
            opt.zero_grad();loss=F.cross_entropy(head(h,z,vocab).reshape(-1,11),torch.tensor([1,2,3,4]));loss.backward()
            self.assertGreater(float(head.o.weight.grad.abs().sum()),0)
            if step:
                for p in (head.q,head.k,head.v):self.assertGreater(float(p.weight.grad.abs().sum()),0)
            opt.step()

    def test_group_redistribution_and_quadrature(self):
        vocab=torch.zeros(2,512);vocab[0,0]=1
        a=torch.zeros(1,196);a[0,0]=1
        u=torch.zeros(196,512,requires_grad=True)
        with torch.no_grad():u[0,0]=1
        correction=a@u@vocab.T
        traces=[(torch.tensor([i]),torch.zeros(1,2),a,u,correction) for i in (0,1)]
        mask=torch.zeros(196);mask[0]=1
        for ordered in (traces,traces[::-1]):
            result=attributed(ordered,vocab,mask,64);check_completeness(result)
            self.assertGreater(float(result['components'][0].abs().max()),.1)
            for key in ('delta','cg','cu','gain','cancel','direction','rank'):
                self.assertLess(abs(float(result[key])),1e-5,key)
            self.assertLess(float(result['endpoint'].abs().max()),1e-5)
            fine=attributed(ordered,vocab,mask,128)
            torch.testing.assert_close(result['components'],fine['components'],atol=1e-5,rtol=1e-4)

    def test_competitor_suppression_and_common_offsets(self):
        torch.manual_seed(83);vocab=torch.randn(5,512);a=torch.softmax(torch.randn(2,196),-1)
        u=torch.randn(196,512,requires_grad=True)*.01;base=torch.randn(2,5);target=torch.tensor([1,2])
        traces=[(target,base,a,u,a@u@vocab.T)];mask=torch.rand(196)
        r=attributed(traces,vocab,mask,64);check_completeness(r)
        grad=torch.autograd.grad(r['cg'],u)[0]
        self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.abs().sum()),0)


if __name__=='__main__':unittest.main(verbosity=2)
