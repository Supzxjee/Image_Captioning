"""Four disposable optimizer steps on eight TRAIN images. No dev/test metrics."""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import time


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def prepare(cfg):
    import torch
    from hidden_data import INPUT_HASHES
    from pilot_data import digest
    from run_pilot import corpus
    from captioning.semantic_prompts import align_prompt_cache
    from captioning.visual_cache import VisualCache
    from captioning.object_reward import make_targets
    out = Path(cfg['output'])
    for key, expected in INPUT_HASHES.items():
        print('Hashing pinned input:', key, flush=True)
        if digest(cfg[key]) != expected:
            raise ValueError('Pinned input differs: ' + key)
    dataset, tokenizer = corpus(cfg)
    selected = []
    for i, row in enumerate(dataset['images']):
        if row['split'] in ('train','restval'):
            selected.append(dict(image_id=i, coco_id=int(Path(row['filename']).stem.rsplit('_',1)[1]),
                                 filename=row['filename'], split='train'))
    selected.sort(key=lambda r: hashlib.sha256(('upt-alignment-pilot-v1:train:'+r['filename']).encode()).hexdigest())
    rows = sorted(selected[:8],key=lambda r:r['image_id'])
    assert len(rows) == 8 and len({r['coco_id'] for r in rows}) == 8
    save(out/'train8.json', rows)
    targets = make_targets(dict(train=rows,val=[]),dataset,Path(cfg['chair'])/'chair.pkl',
                           json.loads(Path(cfg['detections']).read_text()),'aliases_v2','unique_count_v2')
    save(out/'object_targets.json',targets)
    print('Loading legacy cache once; staging only 8 train rows.',flush=True)
    cache = torch.load(cfg['legacy_prompt'],map_location='cpu',weights_only=False)
    names = [str(Path('/images')/dataset['images'][r['image_id']]['filepath']/r['filename']) for r in rows]
    legacy = align_prompt_cache(cache['data'],names)
    if len(legacy) != 8:
        raise ValueError('Missing legacy rows.')
    visual = VisualCache([cfg['visual_cache']],id_key='coco_id')
    try:
        visual.require_ids([r['coco_id'] for r in rows],'CSA technical smoke TRAIN8')
        samples = []
        for row,name in zip(rows,names):
            captions = [tokenizer.encode(s['raw']) for s in dataset['images'][row['image_id']]['sentences'][:5]]
            assert len(captions) == 5
            old = legacy[name]
            tokens, mask = old['tokens'].float(), old['mask'].long()
            if tokens.ndim != 2 or tokens.shape[-1] != 512 or mask.shape != tokens.shape[:1]:
                raise ValueError('Invalid legacy shape.')
            if not torch.isfinite(tokens).all() or not ((mask == 0)|(mask == 1)).all():
                raise ValueError('Invalid legacy content.')
            samples.append(dict(image_id=torch.tensor(row['image_id']), visual=torch.from_numpy(visual.read(row['coco_id'])),
                                legacy=tokens, legacy_mask=mask, captions=torch.stack([p[0] for p in captions]),
                                caption_mask=torch.stack([p[1] for p in captions])))
            print('Staged train image',row['coco_id'],flush=True)
        torch.save({k:torch.stack([s[k] for s in samples]) for k in samples[0]},out/'train8.pt')
    finally:
        visual.close()
    save(out/'stage_receipt.json',dict(train_only=True,images=8,stage_sha256=digest(out/'train8.pt'),
                                      checkpoint_sha256=INPUT_HASHES['checkpoint'],no_dev_or_test=True))


def worker(cfg):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from run_pilot import build_model, corpus, seed_all
    from csa_core import CSACore, load_banks
    from csa_model import CSAModel
    from captioning.scst import TrainingCider, rollout, decode
    from captioning.object_reward import TrainingObjectReward, combine_rewards
    from captioning.scst_runtime import configure_strict_runtime, runtime_manifest, tensor_digest
    from pilot_data import digest
    rank, local = int(os.environ['RANK']), int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 2:
        raise RuntimeError('Select T4x2; one smoke arm uses both GPUs.')
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    configure_strict_runtime(True)
    dist.init_process_group('nccl',timeout=timedelta(minutes=5))
    out, reference = Path(cfg['output']), Path(cfg['reference'])
    try:
        seed_all(42)
        dataset, tokenizer = corpus(cfg)
        base = build_model(dict(cfg,arm='A'),tokenizer).base
        core = CSACore(reference/'csa_net.py',load_banks(reference))
        model = CSAModel(base,core).cuda().eval()
        assert sum(p.numel() for p in core.parameters()) == 23161644
        for b in core.buffers():
            assert b.is_cuda and not b.requires_grad and torch.isfinite(b).all()
        receipt = json.loads((out/'stage_receipt.json').read_text())
        if digest(out/'train8.pt') != receipt['stage_sha256']:
            raise ValueError('Staged train data changed.')
        data = torch.load(out/'train8.pt',map_location='cpu',weights_only=True)
        batch = {k:v[rank::2].cuda() for k,v in data.items()}
        assert batch['visual'].shape == (4,197,768)
        with torch.no_grad():
            expected = base.build_memory(batch['visual'],batch['legacy'],batch['legacy_mask'])
            off = model.memory(batch,'off')
            for a,b in zip(expected,off):
                torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
            t = batch['captions'][:,0,:-1]
            a = model.teacher(expected[0],expected[1],t)
            b = model.teacher(off[0],off[1],t)
            torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-6)
        del a,b,expected,off
        print(f'[rank {rank}] Real checkpoint bypass/logits PASS; building full TRAIN CIDEr IDF.',flush=True)
        refs = {i:[s['raw'] for s in row['sentences'][:5]] for i,row in enumerate(dataset['images'])
                if row['split'] in ('train','restval')}
        reward = TrainingCider(refs,reference_cache_size=256)
        objects = TrainingObjectReward(json.loads((out/'object_targets.json').read_text())['data'],'aliases_v2','unique_count_v2')
        del dataset,refs
        before_banks = {n:tensor_digest(b) for n,b in core.named_buffers()}
        optimizer = torch.optim.AdamW([dict(params=[p for p in base.parameters() if p.requires_grad],lr=5e-6),
                                       dict(params=list(core.parameters()),lr=1e-4)],weight_decay=.01)
        ddp = DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=True)
        save(out/f'runtime_rank{rank}.json',runtime_manifest())
        seed_all(42+rank)
        history = []
        for step in range(1,5):
            optimizer.zero_grad(set_to_none=True)
            dist.barrier()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin = time.monotonic()
            sample = active = advantage = None
            weight = 0. if step <= 2 else .05
            if weight:
                with torch.no_grad():
                    memory,padding = model.memory(batch)
                    sample,active = rollout(model,memory,padding,tokenizer,sample=True)
                    greedy,_ = rollout(model,memory,padding,tokenizer,sample=False)
                ids = batch['image_id'].cpu().tolist()
                sampled,greedy_text = decode(sample,tokenizer),decode(greedy,tokenizer)
                sr = combine_rewards(reward.score(ids,sampled),objects.score(ids,sampled)[0],.5)
                gr = combine_rewards(reward.score(ids,greedy_text),objects.score(ids,greedy_text)[0],.5)
                advantage = torch.from_numpy(sr-gr).cuda()
                del memory,padding
            loss,ce,rl = ddp(batch,sample,active,advantage,weight,tokenizer.pad_idx)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite caption loss.')
            loss.backward()
            grad = sum(float(p.grad.detach().square().sum()) for p in core.parameters() if p.grad is not None)**.5
            if not grad > 0:
                raise AssertionError('No caption gradient into CSA.')
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            record = dict(step=step,mode='CE' if weight == 0 else 'CE_SCST_OBJECT',seconds=time.monotonic()-begin,
                          loss=float(loss.detach()),ce=float(ce),scst=float(rl),core_grad_norm=grad,
                          global_grad_norm=float(norm),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved())
            history.append(record)
            save(out/f'history_rank{rank}.json',history)
            print(f'[rank {rank}] '+json.dumps(record),flush=True)
        for name,b in core.named_buffers():
            assert b.grad is None and tensor_digest(b) == before_banks[name],name
        # Verify synchronization after real optimizer updates, not just DDP construction.
        from captioning.scst_runtime import model_digest
        hashes = [None,None]
        dist.all_gather_object(hashes,model_digest(core))
        assert hashes[0] == hashes[1], 'CSA parameters diverged across ranks.'
        save(out/f'complete_rank{rank}.json',dict(status='TECHNICAL_SMOKE_PASS',completed_updates=4,
             core_parameters=23161644,ddp_core_hash=hashes[rank],bank_buffers_unchanged=True,
             real_baseline_bypass_pass=True,train_images_global=8,dev_used=False,test_used=False,
             resource_only=True,caption_gain_not_evaluated=True,probe_lr_not_pilot_recipe=True))
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','worker'))
    parser.add_argument('--config',required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    (prepare if args.mode == 'prepare' else worker)(cfg)
