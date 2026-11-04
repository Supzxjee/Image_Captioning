"""USER-RUN ONLY: actual two-GPU uneven4/3 CE+SCST gradient scaling check."""
import copy
from datetime import timedelta
import os
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from test_scst_pilot_model import fixture
from scst_pilot_model import PolicyReader,policy_loss


def run():
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    dist.init_process_group('nccl',timeout=timedelta(seconds=180))
    try:
        base,x,valid,seq=fixture(device);reference=copy.deepcopy(base)
        x=x.repeat(2,1,1)[:7];valid=valid.repeat(2,1)[:7];seq=seq.repeat(2,1)[:7]
        seq[4:]=torch.tensor([1,4,2,0],device=device)
        advantages=torch.linspace(-.4,.6,7,device=device);sl=slice(rank*4,min(rank*4+4,7));count=len(x[sl])
        ddp=DDP(PolicyReader(base),device_ids=[rank],broadcast_buffers=False)
        logits,ce=ddp(x[sl],valid[sl],seq[sl,:-1],seq[sl,:-1])
        mask=seq[sl,1:]!=0;tokens=(seq[:,1:]!=0).sum()
        loss=.05*policy_loss(logits,seq[sl,1:],mask,advantages[sl])*(2*count/7)
        loss+=F.cross_entropy(ce.flatten(0,1),seq[sl,1:].flatten(),ignore_index=0,reduction='sum')*(2/tokens)
        loss.backward()
        full=reference(x,valid,seq[:,:-1])
        expected=.05*policy_loss(full,seq[:,1:],seq[:,1:]!=0,advantages)
        expected+=F.cross_entropy(full.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum')/tokens
        expected.backward()
        for (name,p),(_,q) in zip(base.named_parameters(),reference.named_parameters()):
            if p.grad is None or q.grad is None:raise ValueError('Missing gradient:'+name)
            torch.testing.assert_close(p.grad,q.grad,rtol=2e-4,atol=2e-5)
        if rank==0:print('FULL1 uneven4/3 CEtoken+SCSTimage DDP gradient parity PASS',flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':run()
