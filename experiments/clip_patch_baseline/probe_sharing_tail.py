"""User-run exact tail27 probe. Only selected H5 feature rows are read."""
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
from csa_full_data import order_for_epoch
from csa_sharing_data import image_plan, local_events
from full_data import FeatureReader, load_stage, prepare
from gradient_agreement import gradient_agreement
from real_bank_data import read, save
from semantic_data import SemanticContext, checked_inputs, record_arrays


def stage_tail(cfg):
    prepare(cfg)
    stage = Path(cfg['stage'])
    data, _, _ = load_stage(stage)
    events = order_for_epoch(data, 1)[-27:]
    indices, inverse = np.unique(data['caption_image'][events], return_inverse=True)
    if sum(len(image_plan(data['caption_image'], local_events(events, rank), True)[0])
           for rank in range(2)) != 7:
        raise ValueError('Tail encoder count differs from V2')
    ids = [int(x) for x in data['ids'][indices]]
    _, root, cohort, texts, vectors, _, banks, _ = checked_inputs(cfg)
    if not set(ids) <= set(cohort['train']):
        raise ValueError('Tail contains a non-TRAIN image')
    selected = {}
    wanted = set(ids)
    with (root/'prompt_links.jsonl').open(encoding='utf8') as stream:
        for line in stream:
            row = json.loads(line)
            coco = row['coco_id']
            if coco in wanted:
                if coco in selected or row['group'] != 'train':
                    raise ValueError('Wrong tail prompt provenance')
                selected[coco] = record_arrays(row, texts)
    if set(selected) != wanted:
        raise ValueError('Missing tail prompts')
    for name in selected[ids[0]]:
        np.save(stage/f'sem_{name}.npy', np.stack([selected[i][name] for i in ids]))
    np.save(stage/'sem_text_vectors.npy', vectors)
    for name, bank in banks.items():
        np.save(stage/f'sem_{name}_bank.npy', bank)
    reader = FeatureReader(stage)
    try:
        patches = np.stack([reader.get(i) for i in indices])
    finally:
        reader.close()
    sequences = data['sequences'][events]
    if int((sequences[:, 1:] != 0).sum()) != 303:
        raise ValueError('Tail token count differs from V2')
    np.savez(stage/'tail.npz', patches=patches, sequences=sequences, image=inverse)
    save(Path(cfg['output'])/'tail_inputs.json',
         dict(coco_ids=ids, event_ids=events.tolist(), feature_rows=len(ids), global_tokens=303))


def probe(cfg):
    import torch
    import torch.distributed as dist
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs, InitProcessGroupKwargs
    from csa_shared_model import SharedCSAPatchCaptioner
    from full_runtime import tensor_digest

    torch.set_num_threads(1)
    accel = Accelerator(mixed_precision='no', gradient_accumulation_steps=1,
                        dynamo_backend='no', kwargs_handlers=[
                            DistributedDataParallelKwargs(broadcast_buffers=False),
                            InitProcessGroupKwargs(backend='nccl', timeout=timedelta(seconds=180))])
    if accel.num_processes != 2 or accel.device.type != 'cuda':
        raise ValueError('Select GPU T4 x2')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    stage, out = Path(cfg['stage']), Path(cfg['output'])
    words = read(stage/'vocabulary.json')
    banks = [torch.from_numpy(np.load(stage/f'sem_{kind}_bank.npy', allow_pickle=False))
             for kind in ('noun', 'verb')]
    model = SharedCSAPatchCaptioner(len(words), *banks).to(accel.device).eval()
    wrapped = accel.prepare_model(model)
    rank = accel.process_index
    with np.load(stage/'tail.npz', allow_pickle=False) as tail:
        rows = local_events(np.arange(27), rank)
        image = tail['image'][rows]
        patches = tail['patches'].astype(np.float32)
        seq = tail['sequences'][rows].astype(np.int64)
    seq = torch.from_numpy(seq[:, :int((seq != 0).sum(1).max())]).to(accel.device)
    context = SemanticContext(stage)
    activation = []
    hook = model.core.decoder[1].ffn.net[0].register_forward_hook(
        lambda module, inputs, output: activation.append(output[..., 933].detach().cpu().numpy()))
    targets = ('core.decoder.1.ffn.net.0.weight', 'core.decoder.1.ffn.net.0.bias')

    def run(shared, synchronized):
        first, inverse = image_plan(image, np.arange(len(image)), shared)
        selected = image[first]
        x = torch.from_numpy(patches[selected]).to(accel.device)
        p = context.images(selected, accel.device)
        inverse = None if inverse is None else torch.from_numpy(inverse).to(accel.device)
        wrapped.zero_grad(set_to_none=True)
        net = wrapped if synchronized else model
        logits = net(x, seq[:, :-1], p, inverse)
        nll = torch.nn.functional.cross_entropy(logits.flatten(0, 1), seq[:, 1:].flatten(),
                                                ignore_index=0, reduction='sum')
        accel.backward(nll * (2 / 303))
        names = None if synchronized else targets
        grads = {name: p.grad.detach().cpu().numpy().copy() for name, p in model.named_parameters()
                 if names is None or name in names}
        return float(nll.detach()), grads

    ordinary_loss, ordinary_local = run(False, False)
    shared_loss, shared_local = run(True, False)
    hook.remove()
    ordinary, shared = activation
    flips = (ordinary > 0) != (shared > 0)
    valid = seq[:, 1:].ne(0).cpu().numpy()
    locations = np.argwhere(flips)
    sign_changes = [dict(event_index=int(rows[i]), token=int(j), valid_target=bool(valid[i, j]),
                         ordinary=float(ordinary[i, j]), shared=float(shared[i, j]))
                    for i, j in locations]
    save(out/f'tail_rank{rank}.json',
         dict(torch_version=torch.__version__, gpu=torch.cuda.get_device_name(accel.device),
              initial_tensor_sha256=tensor_digest(model), caption_events=len(rows),
              ordinary_nll_sum=ordinary_loss, shared_nll_sum=shared_loss,
              neuron_933_sign_changes=sign_changes,
              local_gradients={n:gradient_agreement(shared_local[n], ordinary_local[n]) for n in targets}))
    ordinary_nll, ordinary_ddp = run(False, True)
    shared_nll, shared_ddp = run(True, True)
    losses = torch.tensor([ordinary_nll, shared_nll], dtype=torch.float64, device=accel.device)
    dist.all_reduce(losses)
    if rank == 0:
        diagnostics = {n:gradient_agreement(shared_ddp[n], g) for n, g in ordinary_ddp.items()}
        nll = (losses / 303).tolist()
        failed = [n for n, v in diagnostics.items() if not v['passed']]
        save(out/'tail_ddp.json', dict(passed=not failed and abs(nll[0]-nll[1]) <= 5e-5,
             ordinary_nll=nll[0], shared_nll=nll[1], failed_parameters=failed,
             gradient_diagnostics=diagnostics, optimizer_updates=0))
    accel.end_training()


if __name__ == '__main__':
    {'prepare': stage_tail, 'probe': probe}[sys.argv[1]](read(sys.argv[2]))
