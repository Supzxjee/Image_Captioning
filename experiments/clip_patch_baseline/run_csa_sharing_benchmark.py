"""USER-RUN FP32/DDP benchmark: same-caption ordinary vs image-shared encoder."""
import argparse
import copy
from datetime import timedelta
import os
from pathlib import Path
import time
import numpy as np
from real_bank_data import read, save
from patch_data import load_stage
from run_csa_structure_smoke import prepare
from csa_structure_data import checked_semantics
from csa_sharing_data import image_plan, benchmark_orders, local_events


def worker(cfg):
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.nn import functional as F
    from csa_shared_model import SharedCSAPatchCaptioner
    from semantic_data import SemanticContext
    from cuda_runtime_guard import check_runtime
    rank = int(os.environ['LOCAL_RANK']); out = Path(cfg['output']); check_runtime(out, rank)
    if int(os.environ['WORLD_SIZE']) != 2 or cfg['updates'] != 40 or cfg['warmup'] != 10:
        raise ValueError('Need2GPUs/40updates per pass/10warmup')
    torch.set_num_threads(1); torch.cuda.set_device(rank); device = torch.device('cuda', rank)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(seconds=300))
    deadline = time.monotonic() + cfg['worker_seconds']
    try:
        data, words, _ = load_stage(cfg['stage']); checked_semantics(cfg['stage'])
        context = SemanticContext(cfg['stage']); orders = benchmark_orders(data['caption_image'])
        banks = [torch.from_numpy(np.load(Path(cfg['stage'])/('sem_'+k+'_bank.npy'), allow_pickle=False)) for k in ('noun','verb')]

        def tensors(events, shared):
            first, inverse = image_plan(data['caption_image'], events, shared)
            images = data['caption_image'][events[first]]
            x = torch.from_numpy(np.asarray(data['patches'][images], np.float32)).to(device)
            seq = np.asarray(data['sequences'][events], np.int64)
            seq = torch.from_numpy(seq[:, :int((seq != 0).sum(1).max())]).to(device)
            p = context.images(images, device)
            return x, seq, p, None if inverse is None else torch.from_numpy(inverse).to(device)

        def build():
            torch.manual_seed(42)
            model = SharedCSAPatchCaptioner(len(words), *banks).to(device)
            result = nn.parallel.DistributedDataParallel(model, device_ids=[rank], broadcast_buffers=False)
            torch.cuda.manual_seed(4242+rank)
            return result

        def step(net, events, shared):
            x, seq, p, inverse = tensors(local_events(events, rank), shared)
            total = torch.tensor(int((seq[:,1:] != 0).sum()), device=device); dist.all_reduce(total)
            logits = net(x, seq[:,:-1], p, inverse)
            nll = F.cross_entropy(logits.flatten(0,1), seq[:,1:].flatten(), ignore_index=0, reduction='sum')
            if not torch.isfinite(nll): raise ValueError('Nonfinite loss')
            (nll * (2/int(total))).backward()
            return nll.detach(), len(x), len(seq), int((seq[:,1:] != 0).sum())

        model = build().eval(); parity = []
        for label, events in [('grouped32', orders['grouped'][:32]), ('tail27', orders['grouped'][-27:]), ('random32', orders['random'][:32])]:
            reference_grads = None; reference_nll = None
            if rank == 0:
                reference = copy.deepcopy(model.module).eval()
                x, seq, p, _ = tensors(events, False)
                logits = reference(x, seq[:,:-1], p)
                loss = F.cross_entropy(logits.flatten(0,1), seq[:,1:].flatten(), ignore_index=0, reduction='sum')/(seq[:,1:] != 0).sum()
                loss.backward(); reference_nll = float(loss.detach())
                reference_grads = {n: v.grad.detach().cpu().clone() for n,v in reference.named_parameters()}
                del reference, x, seq, p, logits, loss
            for shared in (False, True):
                model.zero_grad(set_to_none=True)
                ev = local_events(events, rank); x, seq, p, inverse = tensors(ev, shared)
                with torch.no_grad():
                    a = model.module(x, seq[:,:-1], p, inverse)
                    ox, osq, op, _ = tensors(ev, False)
                    b = model.module(ox, osq[:,:-1], op)
                    torch.testing.assert_close(a, b, rtol=1e-4, atol=5e-5)
                del x, seq, p, inverse, a, b, ox, osq, op
                encoded = []
                handle = model.module.core.project.register_forward_pre_hook(lambda _,args: encoded.append(len(args[0])))
                try: nll, image_count, caption_count, tokens = step(model, events, shared)
                finally: handle.remove()
                if encoded != [image_count]: raise ValueError('Actual encoder row count differs')
                stats = torch.tensor([float(nll), tokens, image_count, caption_count], device=device, dtype=torch.float64); dist.all_reduce(stats)
                error = [None]; max_relative = 0.; max_absolute = 0.
                if rank == 0:
                    try:
                        if abs(float(stats[0]/stats[1])-reference_nll) > 5e-5: raise ValueError('Global token loss differs')
                        for name, v in model.module.named_parameters():
                            if v.grad is None or not torch.isfinite(v.grad).all(): raise ValueError('Missing/nonfinite gradient '+name)
                            a = v.grad.detach().cpu(); b = reference_grads[name]; difference = a-b
                            absolute = float(difference.abs().max()); norm = float(b.double().norm())
                            relative = float(difference.double().norm())/max(norm, 1e-30)
                            max_absolute = max(max_absolute, absolute)
                            if norm > 1e-7:
                                max_relative = max(max_relative, relative)
                                if relative > .01 or absolute > 5e-5: raise ValueError('Gradient parity '+name)
                            elif absolute > 3e-7: raise ValueError('Nearzero gradient parity '+name)
                    except BaseException as exc: error[0] = str(exc)
                    parity.append(dict(case=label, shared=shared, passed=error[0] is None, error=error[0],
                                       global_NLL=float(stats[0]/stats[1]), reference_NLL=reference_nll,
                                       global_tokens=int(stats[1]), encoded_images_global=int(stats[2]), captions=int(stats[3]),
                                       max_relative_gradient_error=max_relative, max_absolute_gradient_error=max_absolute))
                    save(out/'parity.json', parity)
                dist.broadcast_object_list(error, src=0)
                if error[0]: raise ValueError(error[0])
            del reference_grads
        del model; torch.cuda.empty_cache(); dist.barrier(device_ids=[rank])

        summaries = []
        for order_name, order in orders.items():
            for pass_id, shared in enumerate((False, True, True, False)):
                mode = 'shared' if shared else 'ordinary'; label = f'{order_name}_{pass_id}_{mode}'
                torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
                model = build().train()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=(.9,.99), eps=1e-8, weight_decay=0)
                history = []
                for i, events in enumerate(order[:32*cfg['updates']].reshape(-1,32)):
                    stop = torch.tensor(int(time.monotonic() > deadline-60), device=device); dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                    if int(stop): raise TimeoutError('Benchmark worker cap; preserve partial logs')
                    dist.barrier(device_ids=[rank]); torch.cuda.synchronize(device); started = time.monotonic()
                    optimizer.zero_grad(set_to_none=True)
                    nll, encoded, captions, tokens = step(model, events, shared)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True); optimizer.step()
                    torch.cuda.synchronize(device)
                    elapsed = torch.tensor(time.monotonic()-started, device=device, dtype=torch.float64); dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
                    stats = torch.tensor([float(nll), tokens, encoded, captions], device=device, dtype=torch.float64); dist.all_reduce(stats)
                    history.append(dict(step=i+1, timed=i>=cfg['warmup'], seconds=float(elapsed), NLL=float(stats[0]/stats[1]),
                                        target_tokens=int(stats[1]), encoded_images_global=int(stats[2]), caption_events_global=int(stats[3]), grad_norm=float(norm)))
                    if rank == 0 and (i+1)%10 == 0:
                        save(out/(label+'_steps.json'), history)
                        print('SHARING', label, i+1, '/40;', history[-1], flush=True)
                for kind, bank in zip(('nouns','verbs'), banks):
                    if not torch.equal(getattr(model.module.core,kind).detach().cpu(),bank): raise ValueError('Frozen bank changed')
                peak = torch.tensor(torch.cuda.max_memory_allocated(device)/1024**3, device=device); dist.all_reduce(peak, op=dist.ReduceOp.MAX)
                if rank == 0:
                    timed = history[cfg['warmup']:]; seconds = [x['seconds'] for x in timed]
                    summaries.append(dict(order=order_name, pass_id=pass_id, mode=mode, updates=40, warmup=10, timed_updates=30,
                                          median_seconds=float(np.median(seconds)), p90_seconds=float(np.quantile(seconds,.9)),
                                          mean_encoded_images_global=float(np.mean([x['encoded_images_global'] for x in timed])), peak_allocated_GiB=float(peak)))
                    save(out/'pass_summaries.json', summaries)
                del model, optimizer; torch.cuda.empty_cache(); dist.barrier(device_ids=[rank])
        if rank == 0:
            by_order = {}
            for order_name in orders:
                modes = {}
                for mode in ('ordinary','shared'):
                    selected = [s for s in summaries if s['order']==order_name and s['mode']==mode]
                    times = [r['seconds'] for s in selected for r in read(out/f"{order_name}_{s['pass_id']}_{mode}_steps.json")[10:]]
                    modes[mode] = dict(median_seconds=float(np.median(times)), p90_seconds=float(np.quantile(times,.9)),
                                       peak_allocated_GiB=max(s['peak_allocated_GiB'] for s in selected),
                                       mean_encoded_images_global=float(np.mean([s['mean_encoded_images_global'] for s in selected])))
                by_order[order_name] = dict(modes=modes, speedup_ordinary_over_shared=modes['ordinary']['median_seconds']/modes['shared']['median_seconds'])
            save(out/'result.json', dict(status='CSA_FEATURE_SHARING_BENCHMARK_COMPLETE', by_order=by_order, passes=summaries,
                 parity_cases=len(parity), global_batch=32, world_size=2, precision='FP32', total_optimizer_updates=320,
                 quality_evaluated=False, full_training_complete=False, checkpoint_saved=False, test_used=False, validation_used=False,
                 limits='Matched events WITHIN each order. Shared encoder dropout is correlated across captions; grouping changes sampler. Tiny staged pool is not full IO or quality evidence. Timing excludes logging/measurement allreduces; no billing/ETA guarantee.'))
    finally:
        if dist.is_initialized(): dist.destroy_process_group()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('mode', choices=('prepare','worker')); p.add_argument('--config', required=True)
    a = p.parse_args(); globals()[a.mode](read(a.config))
