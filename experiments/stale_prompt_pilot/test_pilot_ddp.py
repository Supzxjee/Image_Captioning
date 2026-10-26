"""User-run synthetic two-GPU gradient normalization check; no model downloads."""
import torch
from profile_stale import setup,token_mean


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(2.,device='cuda'))
    def forward(self,x,valid):return token_mean(self.weight*x,valid)


if __name__=='__main__':
    rank=setup(True)
    try:
        model=torch.nn.parallel.DistributedDataParallel(Tiny(),device_ids=[rank])
        x=torch.tensor([[1.,2.,3.],[4.,5.,6.]],device='cuda')+rank*6
        valid=torch.tensor([[True,True,False],[True,False,False]],device='cuda') if rank==0 else torch.ones((2,3),dtype=torch.bool,device='cuda')
        model(x,valid).backward()
        total=torch.stack([(x*valid).sum(),valid.sum().to(x.dtype)]).double()
        torch.distributed.all_reduce(total)
        torch.testing.assert_close(model.module.weight.grad,(total[0]/total[1]).float(),rtol=1e-5,atol=1e-6)
        print(f'Rank{rank}: uneven-token global-mean gradient PASS; no optimizer step.',flush=True)
    finally:torch.distributed.destroy_process_group()
