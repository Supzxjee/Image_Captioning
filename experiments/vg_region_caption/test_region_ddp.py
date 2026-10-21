"""Two-rank synthetic caption+auxiliary CE/SCST backward; no inference on real data."""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from captioning.scst_runtime import configure_strict_runtime
from region_model import RegionCaptionModel
from test_region_model import Base,batch


def main():
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);configure_strict_runtime(True)
    dist.init_process_group('nccl')
    try:
        for arm in ('G','T'):
            torch.manual_seed(42);model=RegionCaptionModel(Base(),arm).cuda().eval()
            ddp=DDP(model,device_ids=[rank],broadcast_buffers=False,find_unused_parameters=True)
            optimizer=torch.optim.AdamW(model.parameters(),lr=5e-6)
            for weight in (0.,.05):
                torch.manual_seed(90+rank);data=batch(f'cuda:{rank}')
                optimizer.zero_grad(set_to_none=True)
                loss,_,_=ddp(data,data['captions'][:,0,:4],torch.ones(2,3,dtype=torch.bool,device=rank),
                    torch.tensor([1.,-.2],device=rank),weight,0)
                assert bool(torch.isfinite(loss));loss.backward()
                optimizer.step()
                actual=torch.cat([p.detach().flatten() for p in model.parameters()])
                expected=actual.clone();dist.broadcast(expected,0)
                torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            if rank==0: print('TWO_GPU_REGION_CAPTION_SMOKE_PASS',arm,flush=True)
            del ddp,optimizer,model
    finally:dist.destroy_process_group()


if __name__=='__main__':main()
