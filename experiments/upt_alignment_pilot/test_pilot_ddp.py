"""Two-rank gradient synchronization smoke test, not a caption training run."""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from pilot_model import PilotModel
from test_pilot_model import FakeBase, batch


def main():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl')
    try:
        for arm in 'ABCD':
            torch.manual_seed(42)
            model = PilotModel(FakeBase(),arm).cuda().eval()
            ddp = DistributedDataParallel(model,device_ids=[rank],find_unused_parameters=True,broadcast_buffers=False)
            optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
            for weight in (0., .05):
                torch.manual_seed(90+rank)
                data = batch(f'cuda:{rank}')
                optimizer.zero_grad(set_to_none=True)
                loss,_,_ = ddp(data,data['captions'][:,0,:4],torch.ones(2,3,dtype=torch.bool,device=rank),
                              torch.tensor([1.,-.3],device=rank),weight,0)
                loss.backward(); optimizer.step()
                vector = torch.cat([p.detach().flatten() for p in model.parameters()])
                other = vector.clone(); dist.broadcast(other,0)
                torch.testing.assert_close(vector,other,rtol=0,atol=0)
            if rank == 0: print(f'Two-GPU synchronized CE/SCST smoke PASS: {arm}',flush=True)
            del optimizer,ddp,model
    finally:
        dist.destroy_process_group()


if __name__ == '__main__': main()
