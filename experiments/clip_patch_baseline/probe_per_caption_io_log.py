"""User-run, same-session parity and timing for per-caption FP32 XE."""
import hashlib
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

from csa_full_data import order_for_epoch
from full_data import CaptionBatches, FeatureReader, learning_rate, load_stage, loader, prepare
from full_runtime import tensor_digest
from real_bank_data import read, save
from semantic_data import SemanticContext, stage_semantics

TIMED_STEPS = 256
AUDIT_STEPS = 8


class LocalProbeBatches(CaptionBatches):
    def __getitem__(self, item):
        if self.reader is None:
            self.reader = FeatureReader(self.root)
            root = Path(self.root)
            self.reader.local = np.load(root/'probe_patches.npy', mmap_mode='r', allow_pickle=False)
            self.reader.local_rows = np.load(root/'probe_rows.npy', mmap_mode='r', allow_pickle=False)
        return super().__getitem__(item)


def digest(arrays):
    h = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        h.update(str((array.shape, array.dtype)).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def difference(actual, expected):
    if actual.shape!=expected.shape:
        raise ValueError('Tensor comparison shape mismatch')
    delta=actual.astype(np.float64)-expected.astype(np.float64)
    norm=float(np.linalg.norm(expected.astype(np.float64)))
    return dict(exact=bool(np.array_equal(actual,expected)),
                max_abs=float(np.max(np.abs(delta))),
                delta_l2=float(np.linalg.norm(delta)),reference_l2=norm,
                relative_l2=float(np.linalg.norm(delta))/norm if norm else None)


def compare_records(reference, candidate):
    fields=('input_sha256','cuda_rng_sha256','local_loss','gradient_sha256','weight_sha256')
    mismatches=[dict(step=i,fields=[key for key in fields if left[key]!=right[key]])
                for i,(left,right) in enumerate(zip(reference['audit'],candidate['audit'],strict=True),1)]
    return dict(initial_equal=reference['initial_sha256']==candidate['initial_sha256'],
                mismatches=[row for row in mismatches if row['fields']],
                summed_nll_delta=candidate['summed_nll']-reference['summed_nll'],
                final_weights_equal=reference['final_sha256']==candidate['final_sha256'])


def stage(cfg):
    started = time.monotonic()
    prepare(cfg)
    stage_semantics(cfg, started+1500)
    common_seconds = time.monotonic()-started
    root = Path(cfg['stage'])
    reader = FeatureReader(root)
    try:
        order = order_for_epoch(reader.data, 1)
        steps=AUDIT_STEPS if cfg.get('diagnostic_only') else TIMED_STEPS
        selected = np.unique(reader.data['caption_image'][np.r_[order[:steps*32], order[-27:]]])
        locations = reader.data['locations'][selected]
        selected = selected[np.lexsort((locations[:, 1], locations[:, 0]))]
        row_map = np.full(len(reader.data['ids']), -1, np.int64)
        row_map[selected] = np.arange(len(selected))
        patches = np.lib.format.open_memmap(root/'probe_patches.npy', mode='w+', dtype=np.float16,
                                            shape=(len(selected),196,768))
        for ix, image in enumerate(selected):
            value = reader.get(image)
            patches[ix] = value
            if not np.array_equal(patches[ix].view(np.uint16), value.view(np.uint16)):
                raise ValueError('Local patch bytes differ')
        patches.flush()
        del patches
        np.save(root/'probe_rows.npy', row_map)
    finally:
        reader.close()
    save(Path(cfg['output'])/'stage_timing.json', dict(common_seconds=common_seconds,
         subset_copy_seconds=time.monotonic()-started-common_seconds, subset_images=len(selected)))


def benchmark(cfg):
    import torch
    import torch.distributed as dist
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
    from torch.nn import functional as F
    from csa_full_runtime import make_model

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    accelerator = Accelerator(mixed_precision='no', dynamo_backend='no', kwargs_handlers=[
        DistributedDataParallelKwargs(broadcast_buffers=False),
        InitProcessGroupKwargs(backend='nccl', timeout=timedelta(seconds=300))])
    if accelerator.num_processes!=2 or accelerator.device.type!='cuda':
        raise ValueError('Select T4 x2')
    rank, device = accelerator.process_index, accelerator.device
    root, out = Path(cfg['stage']), Path(cfg['output'])
    data, words, _ = load_stage(root, verify=rank==0)
    order = order_for_epoch(data, 1)
    context = SemanticContext(root)

    def run(mode, audit, snapshot=None):
        torch.manual_seed(42)
        raw = make_model(len(words), cfg).to(device)
        model = accelerator.prepare_model(raw)
        initial_sha=tensor_digest(raw) if audit else None
        optimizer = accelerator.prepare_optimizer(torch.optim.Adam(raw.parameters(),lr=8e-6,betas=(.9,.99),eps=1e-8,weight_decay=0))
        torch.manual_seed(4242+rank)
        torch.cuda.manual_seed(4242+rank)
        events = np.r_[order[:AUDIT_STEPS*32],order[-27:]] if audit else order[:TIMED_STEPS*32]
        dataset = (LocalProbeBatches if mode=='old' else CaptionBatches)(root, events, rank)
        pending, losses, audit_rows = [], [], []
        torch.cuda.synchronize(device)
        dist.barrier(device_ids=[rank])
        started = time.monotonic()
        for step,(x,seq,ev) in enumerate(loader(dataset),1):
            input_sha = digest((x.numpy(), seq.numpy(), ev.numpy())) if audit else None
            rng_sha=digest([torch.cuda.get_rng_state(device).cpu().numpy()]) if audit else None
            x=x.to(device,non_blocking=True);seq=seq.to(device,non_blocking=True)
            for group in optimizer.param_groups:group['lr']=learning_rate(step)
            optimizer.zero_grad(set_to_none=True)
            logits=model(x,seq[:,:-1],context.events(data,ev,device))
            count=torch.tensor([int((seq[:,1:]!=0).sum()),len(ev)],device=device)
            dist.all_reduce(count)
            nll=F.cross_entropy(logits.flatten(0,1),seq[:,1:].flatten(),ignore_index=0,reduction='sum')
            loss=nll*(2/int(count[0]))
            if not torch.isfinite(loss):raise ValueError('Nonfinite XE')
            accelerator.backward(loss)
            norm=accelerator.clip_grad_norm_(raw.parameters(),1)
            if not torch.isfinite(norm):raise ValueError('Nonfinite gradient norm')
            optimizer.step()
            if mode in ('old','h5_step'):
                total=nll.detach().double()
                dist.all_reduce(total)
                losses.append(float(total))
            else:
                pending.append(nll.detach().double())
                if step%100==0 or step==len(dataset):
                    totals=torch.stack(pending)
                    dist.all_reduce(totals)
                    losses.extend(totals.cpu().tolist())
                    pending.clear()
            if audit:
                audit_rows.append(dict(input_sha256=input_sha,cuda_rng_sha256=rng_sha, local_loss=float(nll.detach()),
                    gradient_sha256=digest(p.grad.detach().cpu().numpy() for p in raw.parameters()),
                    weight_sha256=tensor_digest(raw)))
        torch.cuda.synchronize(device)
        elapsed=torch.tensor(time.monotonic()-started,device=device,dtype=torch.float64)
        dist.all_reduce(elapsed,op=dist.ReduceOp.MAX)
        if len(losses)!=len(dataset) or pending:
            raise ValueError('Incomplete loss accounting')
        record=dict(mode=mode,seconds=float(elapsed),events=len(events),updates=len(dataset),
                    final_sha256=tensor_digest(raw),loss_sha256=digest([np.asarray(losses,np.float64)]),
                    summed_nll=sum(losses),audit=audit_rows)
        if audit:record['initial_sha256']=initial_sha
        if snapshot is not None and rank==0:
            directory=root/'diagnostic_tensors'
            directory.mkdir(exist_ok=True)
            errors={}
            for ix,(name,parameter) in enumerate(raw.named_parameters()):
                for kind,tensor in (('weight',parameter),('gradient',parameter.grad)):
                    array=tensor.detach().cpu().numpy()
                    path=directory/f'{ix}_{kind}.npy'
                    if snapshot=='write':
                        np.save(path,array)
                    else:
                        expected=np.load(path,mmap_mode='r',allow_pickle=False)
                        stats=difference(array,expected)
                        if not stats['exact']:errors[name+'/'+kind]=stats
                        del expected
            record['tensor_errors']=errors
        del model,raw,optimizer
        accelerator.free_memory()
        return record

    try:
        if cfg.get('diagnostic_only'):
            records=[run('old',True,'write'),run('old',True,'compare'),
                     run('h5_step',True,'compare'),run('optimized',True,'compare')]
            comparisons=[compare_records(records[0],row) for row in records[1:]]
            save(out/f'diagnostic_rank{rank}.json',dict(complete=True,records=records,comparisons=comparisons))
            dist.barrier(device_ids=[rank])
            if rank==0:
                save(out/'result.json',dict(complete=True,diagnostic_only=True,
                    ranks=[read(out/f'diagnostic_rank{r}.json') for r in (0,1)]))
            return
        old, optimized = run('old',True),run('optimized',True)
        parity = all(old[k]==optimized[k] for k in ('audit','loss_sha256','final_sha256'))
        save(out/f'parity_rank{rank}.json',dict(passed=parity,old=old,optimized=optimized))
        flag=torch.tensor(int(parity),device=device)
        dist.all_reduce(flag,op=dist.ReduceOp.MIN)
        if not int(flag):raise ValueError('Input/loss/gradient/weight parity failed')
        runs=[run(mode,False) for mode in ('old','optimized','optimized','old')]
        if any((r['loss_sha256'],r['final_sha256'])!=(runs[0]['loss_sha256'],runs[0]['final_sha256']) for r in runs):
            raise ValueError('Timed runs diverged')
        save(out/f'timing_rank{rank}.json',dict(passed=True,runs=runs))
        dist.barrier(device_ids=[rank])
        if rank==0:
            save(out/'result.json',dict(complete=True,runs=runs,fp32=True,feature_sharing=False,
                optimized_over_old_throughput=(runs[0]['seconds']+runs[3]['seconds'])/(runs[1]['seconds']+runs[2]['seconds'])))
    finally:
        accelerator.end_training()


if __name__=='__main__':
    {'stage':stage,'bench':benchmark}[sys.argv[1]](read(Path(sys.argv[2])))
