"""User-run two-GPU single-backward GV smoke with CE and CE+SCST."""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from captioning.scst_runtime import configure_strict_runtime
from visual_only_model import VisualOnlyModel
from test_region_model import Base,batch
from captioning.models import CaptionDecoder


def main():
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);configure_strict_runtime(True)
    dist.init_process_group('nccl')
    try:
        torch.manual_seed(42);base=Base();base.decoder=CaptionDecoder(9,d_model=32,nhead=4,num_layers=1)
        model=VisualOnlyModel(base).cuda().eval()
        ddp=DDP(model,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=5e-6)
        for weight in (0.,.05):
            torch.manual_seed(90+rank);data=batch(f'cuda:{rank}')
            optimizer.zero_grad(set_to_none=True)
            loss,_,_=ddp(data,data['captions'][:,0,:4],torch.ones(2,3,dtype=torch.bool,device=rank),
                torch.tensor([1.,-.2],device=rank),weight,0)
            assert bool(torch.isfinite(loss));loss.backward()
            assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.decoder.parameters())
            optimizer.step()
            actual=torch.cat([p.detach().flatten() for p in model.parameters()])
            expected=actual.clone();dist.broadcast(expected,0)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        if rank==0:print('TWO_GPU_GV_SMOKE_PASS',flush=True)
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
