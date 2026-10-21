"""Two-GPU synthetic gradients: nonempty, one empty rank, both empty. User-run only."""
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from captioning.scst_runtime import configure_strict_runtime
from hidden_model import HiddenModel
from test_hidden_model import FakeBase, batch


def main():
    rank = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(rank); configure_strict_runtime(True)
    dist.init_process_group('nccl')
    try:
        for arm in ('A', 'H'):
            torch.manual_seed(42); model = HiddenModel(FakeBase(), arm).cuda().eval()
            ddp = DDP(model, device_ids=[rank], find_unused_parameters=True, broadcast_buffers=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            for case in ('nonempty', 'one_rank_empty', 'both_empty'):
                for weight in (0., .05):
                    torch.manual_seed(90 + rank)
                    empty = case == 'both_empty' or (case == 'one_rank_empty' and rank == 0)
                    data = batch(f'cuda:{rank}', no_pairs=empty)
                    optimizer.zero_grad(set_to_none=True)
                    loss, _, _ = ddp(data, data['captions'][:, 0, :4], torch.ones(2, 3, dtype=torch.bool, device=rank),
                                     torch.tensor([1., -.2], device=rank), weight, 0)
                    assert bool(torch.isfinite(loss))
                    loss.backward()
                    if arm == 'H' and case == 'both_empty':
                        assert all(p.grad is None or not bool(p.grad.any()) for p in model.extra.parameters())
                    optimizer.step()
                    actual = torch.cat([p.detach().flatten() for p in model.parameters()])
                    expected = actual.clone(); dist.broadcast(expected, 0)
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                if rank == 0:
                    print('TWO_GPU_HIDDEN_SMOKE_PASS', arm, case, flush=True)
            del optimizer, ddp, model
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
