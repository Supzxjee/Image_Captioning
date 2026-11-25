"""USER-RUN two-GPU FP32 beam-mean SCST; no model import at module load."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from real_bank_data import read, save, sha
from beam_policy import image_losses, mechanism_guard, rank_microbatch, train_beams
from full_epoch_data import FullTrain

SOURCE_SHA = 'b0c5bde0921a4c828ac094b264f502654224aa9365ee40cdce1c15c82fe9aeef'
IMAGES, GLOBAL_BATCH, STEPS = 113287, 32, 3541


def validate_resume(receipt, identity):
    cursor, step = receipt['cursor_images'], receipt['updates']
    if receipt['identity'] != identity or not 0 <= cursor <= IMAGES or step != math.ceil(cursor/GLOBAL_BATCH):
        raise ValueError('Foreign recipe, software, cache or resume cursor')
    if cursor != IMAGES and cursor != step*GLOBAL_BATCH:
        raise ValueError('Resume cursor is not an optimizer boundary')
    if receipt['source_sha256'] != SOURCE_SHA:
        raise ValueError('Foreign XE source')


def run(cfg, mode):
    import torch
    import torch.distributed as dist
    from torch import nn
    from cuda_runtime_guard import check_runtime
    from csa_patch_model import CSAPatchCaptioner
    from csa_per_caption_test_data import checkpoint, validate_checkpoint_state, configure_valid
    from patch_beam import text_from_ids
    from scst_cache import load_reward

    rank = int(os.environ['LOCAL_RANK'])
    out = Path(cfg['output'])
    check_runtime(out, rank)
    if int(os.environ['WORLD_SIZE']) != 2 or torch.cuda.device_count() != 2:
        raise ValueError('Select GPU T4 x2')
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(4242+rank)
    torch.cuda.manual_seed_all(4242+rank)
    np.random.seed(4242+rank)
    dist.init_process_group('nccl', timeout=timedelta(seconds=1800))
    start = time.monotonic()
    deadline = start+cfg['train_seconds']
    try:
        configure_valid(cfg)
        source = checkpoint(cfg['checkpoint_root'])
        if sha(source) != SOURCE_SHA:
            raise ValueError('Need original CE best9, not an SCST checkpoint')
        ck = torch.load(source, map_location='cpu', weights_only=False)
        validate_checkpoint_state(ck)
        words = read(out/'vocabulary.json')
        if len(words) != 10000:
            raise ValueError('Wrong vocabulary')
        model = CSAPatchCaptioner(len(words), ck['model']['core.nouns'], ck['model']['core.verbs'])
        model.load_state_dict(ck['model'], strict=True)
        del ck
        if sum(p.numel() for p in model.parameters()) != 53183736:
            raise ValueError('Captioner architecture differs')
        model = model.to(device).train()
        data = FullTrain(cfg['train_roots'])
        rewarder = load_reward(cfg['train_roots'][0])
        if set(map(int, np.concatenate(data.ids))) != set(rewarder.references):
            raise ValueError('Cached TRAIN reward and feature cohorts differ')
        identity = dict(method='beam_mean_scst_epoch1_v1', source_sha256=SOURCE_SHA,
            train_images=IMAGES, global_batch=32, local_microbatch=4, accumulation=4,
            beams=5, maximum=20, denominator=20, baseline='image_mean_of_five_beam_rewards',
            mode='train', CE_weight=0, gradient_clipping=False, optimizer='Adam',
            learning_rate=4e-6, betas=[.9, .99], eps=1e-8, weight_decay=0,
            cache_manifest_sha256=cfg['cache_manifest_sha256'], source_hashes=cfg['source_hashes'],
            vocabulary_sha256=sha(out/'vocabulary.json'), torch=str(torch.__version__), numpy=np.__version__,
            seed=4242, order='original sealed cache, once, no shuffle/replacement',
            precision='FP32', TF32=False)

        class Policy(nn.Module):
            def __init__(self, captioner):
                super().__init__()
                self.captioner = captioner
            def forward(self, patches, prompts):
                return train_beams(self.captioner, self.captioner.encode(patches, prompts))

        if mode == 'preflight':
            guard = mechanism_guard(device)
            # Real model eval path agrees with the existing decoder beam policy.
            from patch_beam import beam_search
            x, prompt, _ = data.batch([0], device)
            model.eval()
            with torch.no_grad():
                memory = model.encode(x, prompt)
                seq, logps = train_beams(model, memory)
                old, scores = beam_search(model, memory)
                padded = torch.cat((old, old.new_zeros((1, 5, seq.shape[-1]-old.shape[-1]))), -1)
                torch.testing.assert_close(seq, padded, atol=0, rtol=0)
                torch.testing.assert_close(logps.sum(-1), scores, atol=3e-4, rtol=1e-4)
            del x, prompt, memory, seq, logps, old, padded, scores
            model.train()
        else:
            guard = {}
        wrapped = nn.parallel.DistributedDataParallel(Policy(model), device_ids=[rank],
            broadcast_buffers=False, find_unused_parameters=True)
        optimizer = torch.optim.Adam(wrapped.parameters(), lr=4e-6, betas=(.9, .99), eps=1e-8, weight_decay=0)
        cursor = updates = 0
        if mode == 'train' and cfg.get('resume_root'):
            root = Path(cfg['resume_root'])
            receipt = read(root/'checkpoint_receipt.json')
            validate_resume(receipt, identity)
            if sha(root/'checkpoint.bin') != receipt['sha256']:
                raise ValueError('Changed resume binary')
            prior = torch.load(root/'checkpoint.bin', map_location='cpu', weights_only=False)
            if prior['identity'] != identity or prior['updates'] != receipt['updates'] or prior['cursor_images'] != receipt['cursor_images']:
                raise ValueError('Checkpoint and resume receipt differ')
            model.load_state_dict(prior['model'], strict=True)
            optimizer.load_state_dict(prior['optimizer'])
            cursor, updates = prior['cursor_images'], prior['updates']
            if rank == 0:
                save(out/'resume_source.json', dict(source_root=str(root), receipt=receipt,
                    previous_log_sha256=sha(root/'training.jsonl') if (root/'training.jsonl').is_file() else None))
            state = prior['rng'][rank]
            torch.set_rng_state(state['cpu'])
            torch.cuda.set_rng_state(state['cuda'], device)
            np.random.set_state(state['numpy'])
            del prior
        save(out/f'identity_rank{rank}.json', identity)

        def checkpoint_now():
            rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device), numpy=np.random.get_state())
            states = [None, None]
            dist.all_gather_object(states, rng)
            if rank == 0:
                target = out/'checkpoint.bin'
                temp = out/'checkpoint.tmp'
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    updates=updates, cursor_images=cursor, source_sha256=SOURCE_SHA,
                    identity=identity, rng=states), temp)
                os.replace(temp, target)
                save(out/'checkpoint_receipt.json', dict(sha256=sha(target), updates=updates,
                    cursor_images=cursor, source_sha256=SOURCE_SHA, identity=identity,
                    complete_epoch=cursor == IMAGES, test_used=False))
            dist.barrier(device_ids=[rank])

        def update(position, guard_gradients=False):
            real = min(32, IMAGES-position)
            optimizer.zero_grad(set_to_none=True)
            summaries = torch.zeros(5, dtype=torch.float64, device=device)
            for micro in range(4):
                rows, keep = rank_microbatch(position, rank, micro)
                weights = torch.tensor(keep, dtype=torch.float32, device=device)
                x, prompt, ids = data.batch(rows, device)
                sync = wrapped.no_sync() if micro < 3 else nullcontext()
                with sync:
                    seq, logps = wrapped(x, prompt)
                    captions = [text_from_ids(tokens, words) for tokens in seq.detach().reshape(20, 21).cpu().tolist()]
                    ids5 = [image for image in ids for _ in range(5)]
                    rewards = torch.from_numpy(rewarder.score(ids5, captions).reshape(4, 5)).to(device)
                    if not torch.isfinite(rewards).all():
                        raise FloatingPointError('Nonfinite CIDEr reward')
                    per_image = image_losses(logps, rewards)
                    loss = (per_image*weights).sum()*(2/real)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite beam SCST loss')
                    loss.backward()
                summaries[0] += (per_image.detach()*weights).sum()
                summaries[1] += (rewards.mean(-1)*weights).sum()
                summaries[2] += ((rewards-rewards.mean(-1, keepdim=True)).abs().mean(-1)*weights).sum()
                summaries[3] += weights.sum()
                del x, prompt, seq, logps, loss, rewards, per_image
            norm = nn.utils.clip_grad_norm_(wrapped.parameters(), float('inf'), error_if_nonfinite=True)
            if guard_gradients:
                for prefix in ('core', 'object_prompt', 'relation_prompt'):
                    grads = [p.grad for n,p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
                    if not grads or not all(torch.isfinite(g).all() for g in grads) or not any(torch.count_nonzero(g) for g in grads):
                        raise ValueError('Missing finite/nonzero gradient group: '+prefix)
            optimizer.step()
            summaries[4] = float(norm)
            dist.all_reduce(summaries)
            return summaries.cpu().tolist()

        torch.cuda.reset_peak_memory_stats(device)
        if mode == 'preflight':
            # Probe all three cache shards; none of these updates initializes training.
            for position in (0, 32):
                update(position)
            timings = []
            for position in (64, 40000, 80000):
                torch.cuda.synchronize(device)
                timed = time.monotonic()
                update(position, guard_gradients=position == 64)
                torch.cuda.synchronize(device)
                timings.append(time.monotonic()-timed)
            update(IMAGES-7)
            elapsed = torch.tensor(max(timings), device=device)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            projected = float(elapsed)*STEPS
            save(out/f'beam_preflight_rank{rank}.json', dict(passed=True, mechanism=guard,
                real_eval_paths_match=True, real_gradient_groups_checked=['core','object_prompt','relation_prompt'],
                actual_tail7_update=True, discarded_optimizer_steps=6, shard_timing_seconds=timings,
                seconds_per_global32_update=float(elapsed), projected_training_seconds=projected,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                trainable_parameters=53183736, training_initializes_from_discarded_guard=False,
                not_a_quota_or_completion_guarantee=True))
            if projected > 14*3600:
                raise RuntimeError('Measured full-epoch projection exceeds14h; preserve preflight for review, no full train')
            return

        while cursor < IMAGES:
            stop = torch.tensor(int(time.monotonic() > deadline-180), device=device)
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            if int(stop):
                checkpoint_now()
                raise TimeoutError('Epoch partial; exact checkpoint/RNG/cursor preserved, resume without restart')
            stats = update(cursor)
            cursor = min(IMAGES, cursor+32)
            updates += 1
            if updates % 25 == 0 or cursor == IMAGES:
                if rank == 0:
                    row = dict(updates=updates, cursor_images=cursor, SCST=stats[0]/stats[3],
                        mean_beam_reward=stats[1]/stats[3], mean_absolute_advantage=stats[2]/stats[3],
                        last_real_images=int(stats[3]), mean_grad_norm=stats[4]/2, seconds=time.monotonic()-start)
                    with (out/'training.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(row, allow_nan=False)+'\n')
                    print('Beam SCST', row, flush=True)
            if updates % 100 == 0 or cursor == IMAGES:
                checkpoint_now()
        if updates != STEPS:
            raise ValueError('One-epoch endpoint mismatch')
        if rank == 0:
            # Existing VALID evaluator accepts this weights-only, receipt-sealed export.
            target, tmp = out/'pilot_checkpoint.bin', out/'pilot_checkpoint.tmp'
            torch.save(dict(model=model.state_dict(), updates=updates, source_sha256=SOURCE_SHA), tmp)
            os.replace(tmp, target)
            save(out/'pilot_checkpoint_receipt.json', dict(sha256=sha(target), updates=updates,
                source_sha256=SOURCE_SHA, cursor_images=cursor, identity=identity))
            save(out/'training_complete.json', dict(complete=True, train_images=cursor, updates=updates,
                epochs=1, source_sha256=SOURCE_SHA, test_used=False, reranking=False,
                seconds=time.monotonic()-start, peak_allocated_bytes=torch.cuda.max_memory_allocated(device)))
        dist.barrier(device_ids=[rank])
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('preflight','train'))
    parser.add_argument('config')
    args = parser.parse_args()
    run(read(args.config), args.mode)
