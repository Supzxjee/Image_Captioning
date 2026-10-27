"""Synthetic torch checks: USER-RUN Kaggle only, not executed locally."""
import copy
import os
import unittest
import torch
from torch.nn import functional as F
from gain_model import GainHead,Captioner,attributed,check_completeness


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
        # Unequal token counts and an entire rank without auxiliary labels.
        head.zero_grad(set_to_none=True);full=copy.deepcopy(head);full.zero_grad(set_to_none=True)
        token_mask=torch.tensor([[1,0],[1,0],[1,0],[1,0],[1,1],[1,1],[1,1],[1,1]],device=h.device,dtype=torch.bool)
        eligible=torch.tensor([0,0,0,0,1,0,1,0],device=h.device,dtype=torch.bool)
        sl=slice(rank*4,(rank+1)*4);prediction=head(h[sl],z[sl],vocab).square().mean(-1)
        token_count=token_mask[sl].sum().float();aux_count=eligible[sl].sum().float()
        dist.all_reduce(token_count);dist.all_reduce(aux_count)
        loss=prediction.mean()+.1*(prediction*token_mask[sl]).sum()*world/token_count
        loss=loss+.1*prediction.mean(1)[eligible[sl]].sum()*world/aux_count
        loss.backward();sync_gradients(head,world)
        all_prediction=full(h,z,vocab).square().mean(-1)
        expected=all_prediction.mean()+.1*(all_prediction*token_mask).sum()/token_mask.sum()
        expected=expected+.1*all_prediction.mean(1)[eligible].mean();expected.backward()
        for p,q in zip(head.parameters(),full.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=1e-5,rtol=1e-4)


class EOSFixture(Captioner):
    """Deterministic toy states; exercise actual rollout/rescore and EOS code."""
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.base=torch.nn.Module();self.base.decoder=torch.nn.Module()
        self.base.decoder.fc_out=torch.nn.Linear(512,7);self.head=GainHead()
        with torch.no_grad():self.head.o.weight.normal_(0,.01)

    @torch.no_grad()
    def states(self,ids,memory,padding,tgt_padding=None):
        h=torch.sin(ids.float()[...,None]+torch.arange(512,device=ids.device))
        base=torch.full((*ids.shape,7),-4.,device=ids.device)
        for b in range(len(ids)):
            for t in range(ids.shape[1]):base[b,t,2 if b==0 or t>0 else 4]=4.
        return h,base


class ModelChecks(unittest.TestCase):
    def test_nonzero_rollout_rescore_and_first_eos_mask(self):
        from types import SimpleNamespace
        torch.manual_seed(47);model=EOSFixture().eval();z=torch.randn(2,196,512)
        memory=torch.zeros(2,1,512);padding=torch.zeros(2,1,dtype=torch.bool)
        tok=SimpleNamespace(bos_idx=1,pad_idx=0,eos_idx=2)
        seq,active,trace=model.generate(memory,padding,z,tok,sample=False,return_trace=True)
        self.assertEqual(active.tolist(),[[True,False],[True,True]])
        self.assertEqual(seq.tolist(),[[1,2,0],[1,4,2]])
        logits,_=model.logits(seq[:,:-1],memory,padding,z)
        rescored=logits.log_softmax(-1).gather(2,seq[:,1:,None]).squeeze(-1)
        torch.testing.assert_close(trace[active],rescored[active],atol=1e-5,rtol=1e-4)

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

    def test_nonzero_contribution_gradient(self):
        torch.manual_seed(83);vocab=torch.randn(5,512);a=torch.softmax(torch.randn(2,196),-1)
        u=torch.randn(196,512,requires_grad=True)*.01;base=torch.randn(2,5);target=torch.tensor([1,2])
        traces=[(target,base,a,u,a@u@vocab.T)];mask=torch.rand(196)
        r=attributed(traces,vocab,mask,64);check_completeness(r)
        grad=torch.autograd.grad(r['cg'],u)[0]
        self.assertTrue(torch.isfinite(grad).all());self.assertGreater(float(grad.abs().sum()),0)


if __name__=='__main__':unittest.main(verbosity=2)
