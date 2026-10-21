"""Synthetic freeze/gradient/optimizer and rank synchronization checks, no real data."""
import argparse
from datetime import timedelta
import os
import torch
from torch import nn
from csa_model import CSAModel
from phase_policy import freeze_base,parameter_groups,ADAM
from captioning.scst_runtime import model_digest


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.words = nn.Embedding(12,512)
        self.output = nn.Linear(512,12)
    def forward(self,tokens,memory,tgt_key_padding_mask=None,memory_key_padding_mask=None):
        valid = (~memory_key_padding_mask).to(memory.dtype)
        pooled = (memory*valid[...,None]).sum(1)/valid.sum(1,keepdim=True)
        return self.output(self.words(tokens)+pooled[:,None])


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Linear(512,512)
        self.decoder = Decoder()
    def build_memory(self,visual,legacy,mask):
        return torch.cat((legacy,self.visual(visual)),1),torch.cat((mask == 0,torch.zeros(visual.shape[:2],device=visual.device,dtype=torch.bool)),1)


class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(512,512)
        self.register_buffer('fixed_bank',torch.ones(1))
    def forward(self,x):
        y = self.proj(x)
        return y,2*y,torch.zeros(x.shape[0],1,1,32,device=x.device,dtype=torch.bool)


def check(phase,device,distributed=False):
    torch.manual_seed(42)
    model = CSAModel(Base(),Core()).to(device)
    before = model_digest(model.base)
    core_before = model_digest(model.extra)
    freeze_base(model,phase)
    optimizer = torch.optim.AdamW(parameter_groups(model,phase),**ADAM)
    assert len(optimizer.state) == 0
    assert len(optimizer.param_groups) == phase
    wrapped = model
    if distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        wrapped = DDP(model,device_ids=[device.index],broadcast_buffers=False,find_unused_parameters=True)
    batch = dict(visual=torch.randn(2,32,512,device=device),legacy=torch.randn(2,3,512,device=device),
                 legacy_mask=torch.tensor([[1,1,0],[1,0,0]],device=device),
                 captions=torch.tensor([[[1,4,5,2,0]],[[1,6,7,2,0]]],device=device),
                 caption_mask=torch.tensor([[[False,False,False,False,True]],[[False,False,False,False,True]]],device=device))
    with torch.no_grad():
        off = model.memory(batch,'off')
        expected = model.base.build_memory(batch['visual'],batch['legacy'],batch['legacy_mask'])
        assert all(torch.equal(a,b) for a,b in zip(off,expected))
    sample = torch.tensor([[1,4,2],[1,6,2]],device=device) if phase == 2 else None
    active = torch.ones(2,2,device=device,dtype=torch.bool) if phase == 2 else None
    advantage = torch.ones(2,device=device) if phase == 2 else None
    loss,_,_ = wrapped(batch,sample,active,advantage,.05 if phase == 2 else 0.,0)
    loss.backward()
    assert torch.isfinite(loss)
    assert sum(float(p.grad.abs().sum()) for p in model.extra.parameters() if p.grad is not None) > 0
    if phase == 1:
        assert all(p.grad is None for p in model.base.parameters())
    else:
        assert sum(float(p.grad.abs().sum()) for p in model.base.parameters() if p.grad is not None) > 0
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    assert (model_digest(model.base) == before) == (phase == 1)
    assert model_digest(model.extra) != core_before
    assert torch.equal(model.extra.fixed_bank,torch.ones(1,device=device))
    # Newly constructed phase2 optimizer must not inherit moments from phase1.
    new_optimizer = torch.optim.AdamW(parameter_groups(model,phase),**ADAM)
    assert len(new_optimizer.state) == 0 and len(optimizer.state) > 0
    if distributed:
        import torch.distributed as dist
        hashes = [None,None]
        dist.all_gather_object(hashes,model_digest(model))
        assert hashes[0] == hashes[1]
    print(f'Phase{phase} freeze/decoder-through-gradient/fresh-optimizer/DDP={distributed} PASS',flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--phase',type=int,choices=(1,2))
    args = p.parse_args()
    torch.set_num_threads(2)
    if args.phase is None:
        check(1,torch.device('cpu'))
        check(2,torch.device('cpu'))
    else:
        import torch.distributed as dist
        from captioning.scst_runtime import configure_strict_runtime
        local = int(os.environ['LOCAL_RANK'])
        assert int(os.environ['WORLD_SIZE']) == 2
        torch.cuda.set_device(local)
        configure_strict_runtime(True)
        dist.init_process_group('nccl',timeout=timedelta(minutes=3))
        try:
            check(args.phase,torch.device('cuda',local),True)
        finally:
            dist.destroy_process_group()
