"""USER-RUN ONLY: tiny Torch fixtures and two-rank weighted-gradient preflight."""
import copy
import os
import sys
import unittest
import torch
from torch import nn
from grounding_model import GroundingAttention,grounding_sum,install


class Checks(unittest.TestCase):
    def setUp(self):torch.manual_seed(42)

    def test_actual_attention_forward_gradient_parity(self):
        old=nn.MultiheadAttention(8,2,dropout=0,batch_first=True)
        new=GroundingAttention(8,2,dropout=0,batch_first=True);new.load_state_dict(old.state_dict());new.collect=True
        q=torch.randn(3,5,8);k=torch.randn(3,4,8);mask=torch.tensor([[False,False,False,True]]*3)
        a=old(q,k,k,key_padding_mask=mask,need_weights=False)[0];b=new(q,k,k,key_padding_mask=mask)[0]
        torch.testing.assert_close(a,b,rtol=1e-4,atol=1e-5)
        a.square().sum().backward();b.square().sum().backward()
        for (_,p),(_,r) in zip(old.named_parameters(),new.named_parameters()):torch.testing.assert_close(p.grad,r.grad,rtol=1e-4,atol=1e-5)
        self.assertTrue(torch.isneginf(new.log_attention[:,:,:,-1]).all())
        torch.testing.assert_close(new.log_attention.exp().sum(-1),torch.ones(3,2,5))

    def test_no_parameters_added_and_original_eval_forward(self):
        from model import CSACaptioner,Config
        m=CSACaptioner(Config(vocab=12,dim=8,heads=2,hidden=16,prototypes=3,layers=2,dropout=0,raw_dim=8),torch.randn(4,8),torch.randn(5,8)).eval()
        reference=copy.deepcopy(m);state=set(m.state_dict());att=install(m)
        self.assertEqual(state,set(m.state_dict()))
        x=torch.randn(2,3,8);v=torch.ones(2,3,dtype=torch.bool);ids=torch.tensor([[1,4,5],[1,6,0]])
        with torch.no_grad():
            expected=reference(x,v,ids);att.collect=True;actual=m(x,v,ids)
            torch.testing.assert_close(expected,actual,rtol=1e-4,atol=1e-5)
            att.collect=False;torch.testing.assert_close(reference(x,v,ids),m(x,v,ids),rtol=0,atol=0)

    def test_mass_loss_multi_positive_and_empty(self):
        logits=torch.zeros(2,2,3,4,requires_grad=True);logp=logits.log_softmax(-1)
        positive=torch.zeros(2,3,4,dtype=torch.bool);positive[0,1,:2]=True
        loss,count=grounding_sum(logp,positive);self.assertEqual(count,1)
        torch.testing.assert_close(loss,torch.tensor(2.).log());loss.backward(retain_graph=True)
        self.assertTrue((logits.grad[0,:,1,:2]<0).all())
        zero,n=grounding_sum(logp,torch.zeros_like(positive));self.assertEqual(n,0);self.assertEqual(float(zero),0.)
        zero.backward();self.assertTrue(torch.isfinite(logits.grad).all())


def ddp():
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    torch.backends.cuda.matmul.allow_tf32=False
    dist.init_process_group('nccl')
    try:
        for empty_rank in (False,True):
            torch.manual_seed(51)
            ref=GroundingAttention(8,2,dropout=0,batch_first=True).to(device);ref.collect=True
            local=copy.deepcopy(ref);model=DDP(local,device_ids=[rank])
            q=torch.randn(4,3,8,device=device);k=torch.randn(4,4,8,device=device)
            positives=torch.zeros(4,3,4,dtype=torch.bool,device=device);positives[0,0,0]=True;positives[1,1,:2]=True
            if not empty_rank:positives[2,:,1]=True
            full=ref(q,k,k)[0];g,n=grounding_sum(ref.log_attention,positives)
            (full.square().sum()/full.numel()+.1*g/n).backward()
            sl=slice(rank*2,rank*2+2);value=model(q[sl],k[sl],k[sl])[0]
            lg,ln=grounding_sum(local.log_attention,positives[sl]);den=torch.tensor(ln,device=device,dtype=torch.long);dist.all_reduce(den)
            (value.square().sum()*2/full.numel()+.1*lg*2/int(den)).backward()
            for (_,p),(_,r) in zip(local.named_parameters(),ref.named_parameters()):torch.testing.assert_close(p.grad,r.grad,rtol=1e-4,atol=1e-5)
            dist.barrier(device_ids=[rank]);del model
        if rank==0:print('DDP unequal/empty grounding denominators gradient parity PASS',flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':
    if '--ddp' in sys.argv:ddp()
    else:unittest.main(verbosity=2)
