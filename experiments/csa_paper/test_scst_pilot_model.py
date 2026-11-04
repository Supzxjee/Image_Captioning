"""USER-RUN ONLY neural/DDP regression tests; never executed by author locally."""
import copy
from datetime import timedelta
import os
import sys
import unittest
import torch
from torch.nn import functional as F
from model import Config,CSACaptioner
from scst_pilot_model import rollout,selected_logps,policy_loss,PolicyReader


def fixture(device='cpu'):
    torch.manual_seed(21)
    c=Config(vocab=11,dim=8,heads=2,hidden=16,prototypes=3,layers=2,dropout=0,raw_dim=6)
    model=CSACaptioner(c,torch.randn(5,8),torch.randn(4,8)).to(device).eval()
    x=torch.randn(4,3,6,device=device);valid=torch.tensor([[True,True,False]]*4,device=device)
    prefix=torch.tensor([[1,4,5,2]]*4,device=device)
    return model,x,valid,prefix


class Checks(unittest.TestCase):
    def test_sample_replay_and_gradients(self):
        base,x,v,ce=fixture();vo,vr=base.encode(x,v)
        seq,mask,logps=rollout(base,vo.detach(),vr.detach(),v,True,maximum=6)
        logits=base.decode(seq[:,:-1],vo,vr,v)
        torch.testing.assert_close(logps,selected_logps(logits,seq[:,1:],mask),rtol=1e-4,atol=2e-5)
        loss=policy_loss(logits,seq[:,1:],mask,torch.ones(4))
        loss=loss+F.cross_entropy(base.decode(ce[:,:-1],vo,vr,v).flatten(0,1),ce[:,1:].flatten())
        loss.backward()
        for name,p in base.named_parameters():
            self.assertIsNotNone(p.grad,name);self.assertTrue(torch.isfinite(p.grad).all(),name)
        self.assertIsNone(base.nouns.grad);self.assertIsNone(base.verbs.grad)
        self.assertFalse(((seq[:,1:]<2)&mask).any())

    def test_eos_and_post_eos_padding(self):
        class AlwaysEOS:
            training=False
            def decode(self,seq,vo,vr,valid):
                out=torch.full((*seq.shape,5),-100.);out[...,2]=100.;return out
        seq,mask,_=rollout(AlwaysEOS(),torch.zeros(2,1,1),None,None,True)
        self.assertEqual(seq.tolist(),[[1,2],[1,2]]);self.assertTrue(mask.all())
        logits=torch.zeros(1,3,5,requires_grad=True)
        loss=policy_loss(logits,torch.tensor([[4,2,0]]),torch.tensor([[True,True,False]]),torch.tensor([1.]))
        self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertTrue((logits.grad[:,2]==0).all())

    def test_advantage_sign(self):
        logits=torch.zeros(1,1,5,requires_grad=True)
        actions=torch.tensor([[4]]);mask=torch.ones(1,1,dtype=torch.bool)
        policy_loss(logits,actions,mask,torch.tensor([1.])).backward()
        self.assertLess(float(logits.grad[0,0,4]),0)


def ddp_check():
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    dist.init_process_group('nccl',timeout=timedelta(seconds=180))
    try:
        base,x,v,seq=fixture(device);reference=copy.deepcopy(base)
        ddp=DDP(PolicyReader(base),device_ids=[rank],broadcast_buffers=False)
        sl=slice(rank*2,rank*2+2);mask=torch.ones(2,3,device=device,dtype=torch.bool)
        logits,ce=ddp(x[sl],v[sl],seq[sl,:-1],seq[sl,:-1])
        loss=.05*policy_loss(logits,seq[sl,1:],mask,torch.tensor([.5,-.2],device=device))
        loss+=F.cross_entropy(ce.flatten(0,1),seq[sl,1:].flatten())
        loss.backward()
        full=reference(x,v,seq[:,:-1]);expected=.05*policy_loss(full,seq[:,1:],mask.repeat(2,1),torch.tensor([.5,-.2,.5,-.2],device=device))
        expected+=F.cross_entropy(full.flatten(0,1),seq[:,1:].flatten());expected.backward()
        for (name,p),(_,q) in zip(base.named_parameters(),reference.named_parameters()):
            if p.grad is None or q.grad is None:raise ValueError('Missing DDP gradient:'+name)
            torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=2e-5)
        if rank==0:print('SCST two-GPU DDP gradient scaling PASS',flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':
    if '--ddp' in sys.argv:ddp_check()
    else:unittest.main()
