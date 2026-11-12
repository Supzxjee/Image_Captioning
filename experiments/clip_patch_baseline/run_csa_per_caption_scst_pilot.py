"""User-run 500-step SCST + XE for CSA patch captioner."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from full_data import FeatureReader, load_stage, prepare
from real_bank_data import read, save, sha
from semantic_data import SemanticContext, checked_root, record_arrays

UPDATES = 500
BATCH = 8


def prepare_train(cfg):
    local = dict(cfg, stage=cfg['train_stage'])
    prepare(local)  # Metadata only.
    stage, out = Path(local['stage']), Path(cfg['output'])
    data, _, _ = load_stage(stage)
    refs = {int(r['cocoid']): [s['raw'] for s in r['sentences'][:5]]
            for r in read(cfg['dataset_json'])['images'] if r['split'] in ('train', 'restval')}
    if len(refs) != 113287 or set(refs) != set(map(int, data['ids'][:113287])):
        raise ValueError('Wrong TRAIN reward cohort')
    selected = np.random.default_rng(42).permutation(113287)[:UPDATES*BATCH]
    ids = data['ids'][selected]
    first = np.searchsorted(data['caption_image'], selected)
    sequences = data['sequences'][first]
    inv = read(Path(__file__).with_name('FULL_PROMPT_FRESH_V175_RESULT.json'))['transport_inventory']['cache_files']
    files = {n: inv[n] for n in ('cache_receipt.json', 'cohort.json', 'prompt_links.jsonl',
                                'prompt_texts.json', 'prompt_embeddings.npy')}
    root = checked_root(cfg['prompt_test_root'], 'cache_receipt.json', files)
    cohort = read(root/'cohort.json')
    if cohort['test'] or set(cohort['train']) != set(refs):
        raise ValueError('Wrong TRAIN prompt cohort')
    texts = read(root/'prompt_texts.json')
    vectors = np.load(root/'prompt_embeddings.npy', allow_pickle=False)
    records, wanted = {}, set(map(int, ids))
    with (root/'prompt_links.jsonl').open(encoding='utf8') as stream:
        for line in stream:
            row = json.loads(line)
            cid = row['coco_id']
            if cid in wanted:
                if row['group'] != 'train' or cid in records:
                    raise ValueError('Wrong TRAIN prompt row')
                records[cid] = record_arrays(row, texts)
    if set(records) != wanted:
        raise ValueError('Missing TRAIN prompts')
    for key in records[int(ids[0])]:
        np.save(stage/f'sem_{key}.npy', np.stack([records[int(i)][key] for i in ids]))
    np.save(stage/'sem_text_vectors.npy', vectors)
    reader = FeatureReader(stage)
    patches = np.lib.format.open_memmap(stage/'pilot_patches.npy', mode='w+', dtype=np.float16,
                                        shape=(len(ids), 196, 768))
    try:
        locations = data['locations'][selected]
        order = np.lexsort((locations[:, 1], locations[:, 0]))
        for j, idx in enumerate(order):
            patches[idx] = reader.get(selected[idx])
            if (j+1) % 500 == 0:
                print('Pilot TRAIN patches', j+1, '/', len(ids), flush=True)
        patches.flush()
    finally:
        reader.close()
    np.save(stage/'pilot_ids.npy', ids)
    np.save(stage/'pilot_sequences.npy', sequences)
    save(stage/'training_references.json', refs)
    save(out/'pilot_inputs.json', dict(updates=UPDATES, image_events=len(ids),
         reward_images=len(refs), selected_ids=ids.tolist(), test_used=False))


def train(cfg):
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.nn import functional as F
    from datetime import timedelta
    from captioning.scst import TrainingCider
    from csa_patch_model import CSAPatchCaptioner
    from csa_per_caption_test_data import checkpoint, validate_checkpoint_state, configure_valid
    from scst_pilot_model import action_logits, policy_loss, captions, selected_logps

    rank = int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 2 or torch.cuda.device_count() != 2:
        raise ValueError('Select T4 x2')
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(4242+rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=900))
    out, stage = Path(cfg['output']), Path(cfg['train_stage'])
    start = time.monotonic()

    class Policy(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x, p, sample, teacher):
            memory = self.model.encode(x, p)
            return self.model.decode(memory, sample), self.model.decode(memory, teacher)

    @torch.no_grad()
    def rollout(model, memory, sample):
        seq = torch.ones((len(memory), 1), dtype=torch.long, device=device)
        active = torch.ones(len(memory), dtype=torch.bool, device=device)
        masks, selected = [], []
        for _ in range(20):
            logits = model.decode(memory, seq)[:, -1]
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite rollout logits')
            logp = F.log_softmax(action_logits(logits), -1)
            token = torch.multinomial(logp.exp(), 1).squeeze(1) if sample else logp.argmax(-1)
            selected.append(logp.gather(1, token[:, None]).squeeze(1).masked_fill(~active, 0))
            masks.append(active)
            seq = torch.cat((seq, token.masked_fill(~active, 0)[:, None]), 1)
            active = active & token.ne(2)
            if not active.any():
                break
        return seq, torch.stack(masks, 1), torch.stack(selected, 1)

    try:
        configure_valid(cfg)
        src = checkpoint(cfg['checkpoint_root'])
        ck = torch.load(src, map_location='cpu', weights_only=False)
        validate_checkpoint_state(ck)
        words = read(stage/'vocabulary.json')
        model = CSAPatchCaptioner(len(words), ck['model']['core.nouns'], ck['model']['core.verbs'])
        model.load_state_dict(ck['model'], strict=True)
        del ck
        model.to(device).eval()
        wrapped = nn.parallel.DistributedDataParallel(Policy(model).eval(), device_ids=[rank], broadcast_buffers=False)
        optimizer = torch.optim.Adam(wrapped.parameters(), lr=5e-6, betas=(.9,.99), eps=1e-8)
        refs = {int(i): v for i, v in read(stage/'training_references.json').items()}
        cider = TrainingCider(refs, reference_cache_size=2048)
        del refs
        context = SemanticContext(stage)
        patches = np.load(stage/'pilot_patches.npy', mmap_mode='r', allow_pickle=False)
        sequences = np.load(stage/'pilot_sequences.npy', allow_pickle=False)
        ids = np.load(stage/'pilot_ids.npy', allow_pickle=False)

        def checkpoint_step(step):
            if rank == 0:
                path = out/'pilot_checkpoint.bin'
                tmp = path.with_suffix('.tmp')
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                updates=step, source_sha256=sha(src)), tmp)
                os.replace(tmp, path)
                save(out/'pilot_checkpoint_receipt.json', dict(sha256=sha(path), updates=step,
                     source_sha256=sha(src)))
            dist.barrier(device_ids=[rank])

        for step in range(UPDATES):
            stop = torch.tensor(int(time.monotonic()-start > cfg['train_seconds']-120), device=device)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if int(stop):
                checkpoint_step(step)
                raise TimeoutError('Pilot cap reached; partial checkpoint preserved')
            rows = np.arange(step*BATCH+rank*4, step*BATCH+rank*4+4)
            x = torch.from_numpy(np.asarray(patches[rows], np.float32)).to(device)
            p = context.images(rows, device)
            ce = np.asarray(sequences[rows], np.int64)
            ce = torch.from_numpy(ce[:, :int((ce != 0).sum(1).max())]).to(device)
            with torch.no_grad():
                memory = model.encode(x, p)
                sample, mask, replay = rollout(model, memory, True)
                greedy, _, _ = rollout(model, memory, False)
            sampled = cider.score(ids[rows].tolist(), captions(sample, words))
            baseline = cider.score(ids[rows].tolist(), captions(greedy, words))
            advantage = torch.from_numpy(sampled-baseline).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, teacher = wrapped(x, p, sample[:, :-1], ce[:, :-1])
            if step == 0:
                rescored = selected_logps(logits.detach(), sample[:, 1:], mask)
                torch.testing.assert_close(rescored, replay, rtol=1e-4, atol=3e-4)
            rl = policy_loss(logits, sample[:, 1:], mask, advantage)
            nll = F.cross_entropy(teacher.flatten(0,1), ce[:,1:].flatten(), ignore_index=0, reduction='sum')
            tokens = ce[:,1:].ne(0).sum()
            dist.all_reduce(tokens)
            loss = nll*(2/int(tokens)) + .05*rl
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite SCST loss')
            loss.backward()
            norm = nn.utils.clip_grad_norm_(wrapped.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if (step+1) % 25 == 0:
                stats = torch.tensor([float(nll.detach()), float(rl.detach()), float(sampled.mean()),
                                      float(baseline.mean())], dtype=torch.float64, device=device)
                dist.all_reduce(stats)
                if rank == 0:
                    row = dict(updates=step+1, XE=float(stats[0])/int(tokens), SCST=float(stats[1])/2,
                               sample_reward=float(stats[2])/2, greedy_reward=float(stats[3])/2,
                               grad_norm=float(norm), seconds=time.monotonic()-start)
                    print('SCST pilot', row, flush=True)
                    with (out/'training.jsonl').open('a', encoding='utf8') as log:
                        log.write(json.dumps(row, allow_nan=False)+'\n')
            if (step+1) % 100 == 0:
                checkpoint_step(step+1)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('prepare_train','train'))
    parser.add_argument('config')
    args = parser.parse_args()
    globals()[args.mode](read(args.config))
