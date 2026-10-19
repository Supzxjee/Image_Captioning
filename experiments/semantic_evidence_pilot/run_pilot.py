"""Manual, one-arm, two-GPU UPT caption pilot. Importing this file never trains."""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np

from pilot_data import cohorts, digest, identity, region_slots, evidence_bounds, phrase_identity, save, SLOTS, INPUT_DIM

CLIP = 'openai/clip-vit-base-patch16'
CLIP_REVISION = '57c216476eefef5ab752ec549e440a49ae4ae5f3'
CHECKPOINT_SHA = '5f8eead3f4e960189345b1a9bf492763fe52ec71f70115b73244ffe9d03d2e75'


def prepare(config_path):
    import h5py
    import torch
    from transformers import CLIPTextModel, AutoTokenizer
    from captioning.visual_cache import VisualCache
    from captioning.semantic_prompts import align_prompt_cache
    from captioning.object_reward import make_targets, _AnnotationReader
    from captioning.scst_runtime import configure_strict_runtime
    cfg = json.loads(Path(config_path).read_text())
    out = Path(cfg['output'])
    if out.exists():
        raise FileExistsError(f'Use a new OUTPUT: {out}')
    for key in ('checkpoint', 'legacy_prompt', 'upt_prompt', 'dataset_json', 'detections'):
        if not Path(cfg[key]).is_file():
            raise FileNotFoundError(f'{key}: {cfg[key]}')
    hashes = {}
    for key in ('checkpoint', 'legacy_prompt', 'upt_prompt', 'dataset_json', 'detections'):
        print(f'Checking pinned input: {key}', flush=True)
        hashes[key] = digest(cfg[key])
    if hashes != cfg['expected_hashes']:
        raise ValueError(f'Pinned inputs differ: {hashes}. Do not silently replace the baseline/cache.')
    dataset = json.loads(Path(cfg['dataset_json']).read_text())
    groups = cohorts(dataset)
    all_rows = groups['train'] + groups['val']
    prompts = json.loads(Path(cfg['upt_prompt']).read_text())
    if len(prompts) != 123287 or any(r['filename'] not in prompts for r in all_rows):
        raise ValueError('Incomplete UPT prompt mapping.')
    for rows in groups.values():
        for row in rows:
            row['mode'] = prompts[row['filename']]['mode']
    if identity(groups) != '6bbdf3b75869927334ec6c2f868edf9f52346c0e79401aa0e118ba5a02fb711b':
        raise ValueError('Cohort differs from the locally reviewed 10000/1000 image manifest.')
    chair = Path(cfg['chair'])
    if digest(chair / 'chair.pkl') != '73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':
        raise ValueError('Pinned CHAIR annotation cache differs.')
    # Read validation annotation coverage for evaluation only, not training targets.
    with (chair / 'chair.pkl').open('rb') as stream:
        annotation = _AnnotationReader(stream).load()
    if any(r['coco_id'] not in annotation.imid_to_objects for r in groups['val']):
        raise ValueError('CHAIR validation coverage incomplete.')
    del annotation
    out.mkdir(parents=True)
    cfg.update(groups=groups, cohort_sha256=identity(groups), test_used=False,
               alignment_weight=.1 if cfg['arm']=='E' else 0., alignment_margin=.2,
               negative_policy='Within-rank same-role, other image, exclude exact known phrases in either image; deterministic first eligible; may contain false negatives',
               validation_scope='Historically reused development validation, not an independent unseen test',
               source_checkpoint_sha256=hashes['checkpoint'], supported_actions=116,
               loss='CE + scheduled 0.05*SCST(CIDEr-D+0.5*object_count_v2) + E-only 0.1*pseudo_pair_hinge_margin0.2',
               evaluation_policy='Raw greedy, all arms and start checkpoint; no reranking in this screening pilot',
               comparison_limits='A continued-training control; E text-query patch attention plus alignment loss. Whole-method comparison only; UPT pseudo-pairs and cross-image negatives can be noisy.',
               module_claim='Text-conditioned visual evidence with pseudo-pair supervision; not calibrated relation truth or established novelty')
    save(out / 'groups.json', groups)
    targets = make_targets(groups, dataset, chair / 'chair.pkl', json.loads(Path(cfg['detections']).read_text()),
                           'aliases_v2', 'unique_count_v2')
    save(out / 'train_object_targets.json', targets)
    cfg['object_targets_sha256'] = digest(out / 'train_object_targets.json')
    bundle = torch.load(cfg['legacy_prompt'], map_location='cpu', weights_only=False)
    names = [str(Path('/images') / dataset['images'][r['image_id']]['filepath'] / r['filename']) for r in all_rows]
    legacy = align_prompt_cache(bundle['data'], names)
    if len(legacy) != len(all_rows):
        raise ValueError('Missing baseline prompt entries; UPT JSON must not replace this cache.')
    first = legacy[names[0]]
    length = first['tokens'].shape[0]
    if first['tokens'].shape != (length, 512):
        raise ValueError('Legacy prompt shape differs.')
    stage = Path(cfg['stage'])
    if stage.exists():
        raise FileExistsError(f'Use a fresh stage path: {stage}')
    stage.parent.mkdir(parents=True, exist_ok=True)
    required = len(all_rows) * (197 * 768 + SLOTS * INPUT_DIM + length * 512 + SLOTS * 512) * 4
    if shutil.disk_usage(stage.parent).free < required + 2 * 1024**3:
        raise RuntimeError('Not enough disk space to stage the 11000-image subset without quantizing.')
    cache = VisualCache([cfg['visual_cache']], id_key='coco_id')
    cache.require_ids([r['coco_id'] for r in all_rows], 'pilot train/val')
    text_rows = []
    n = len(all_rows)
    try:
        with h5py.File(stage, 'w') as f:
            arrays = {'visual': ((n, 197, 768), 'f4'), 'legacy': ((n, length, 512), 'f4'),
                      'legacy_mask': ((n, length), 'i8'), 'regions': ((n, SLOTS, INPUT_DIM), 'f4'),
                      'roles': ((n, SLOTS), 'i8'), 'valid': ((n, SLOTS), '?'),
                      'text': ((n, SLOTS, 512), 'f4'), 'bounds': ((n,SLOTS,3,4),'f4'), 'phrase_ids': ((n,SLOTS),'i8')}
            for key, (shape, dtype) in arrays.items():
                f.create_dataset(key, shape=shape, dtype=dtype, chunks=(1, *shape[1:]))
            # Read source HDF5 in source-row order rather than random image order.
            order = sorted(range(n), key=lambda i: cache.index[all_rows[i]['coco_id']])
            text_rows = [None] * n
            for count, i in enumerate(order, 1):
                row = all_rows[i]
                visual = cache.read(row['coco_id'])
                features, roles, valid, texts = region_slots(prompts[row['filename']], visual)
                old = legacy[names[i]]
                if old['tokens'].shape != (length, 512) or old['mask'].shape != (length,):
                    raise ValueError('Inconsistent legacy prompt length.')
                for key, value in [('visual', visual), ('legacy', old['tokens'].float().numpy()),
                                   ('legacy_mask', old['mask'].long().numpy()), ('regions', features),
                                   ('roles', roles), ('valid', valid), ('bounds',evidence_bounds(prompts[row['filename']])),
                                   ('phrase_ids',np.asarray([phrase_identity(t) for t in texts],dtype=np.int64))]:
                    f[key][i] = value
                text_rows[i] = texts
                if count % 250 == 0 or count == n:
                    print(f'Staged spatial tokens/regions: {count}/{n}', flush=True)
    finally:
        cache.close()
    del bundle, legacy, prompts, targets
    # Frozen EOS hidden states (512D), shared across repeated phrases; no trainable text encoder.
    torch.manual_seed(42)
    configure_strict_runtime(True)
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Select GPU T4 x2. This notebook runs ONE arm using two GPUs.')
    encoder = CLIPTextModel.from_pretrained(CLIP, revision=CLIP_REVISION).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(CLIP, revision=CLIP_REVISION)
    unique = sorted({t for row in text_rows for t in row if t})
    embedding = {}
    with torch.no_grad():
        for start in range(0, len(unique), 128):
            batch = unique[start:start+128]
            tokens = tokenizer(batch, padding=True, truncation=True, max_length=77, return_tensors='pt').to('cuda')
            vectors = encoder(**tokens).pooler_output.float().cpu().numpy()
            for text, vector in zip(batch, vectors):
                embedding[text] = vector
            if start % 1024 == 0:
                print(f'Frozen text embeddings: {min(start+128, len(unique))}/{len(unique)}', flush=True)
    with h5py.File(stage, 'r+') as f:
        for i, texts in enumerate(text_rows):
            values = np.zeros((SLOTS, 512), dtype=np.float32)
            for j, text in enumerate(texts):
                if text:
                    values[j] = embedding[text]
            f['text'][i] = values
        f.attrs['cohort_sha256'] = cfg['cohort_sha256']
        f.attrs['complete'] = True
    cfg['stage_sha256'] = digest(stage)
    cfg['counts_by_mode'] = {key: dict(Counter(r['mode'] for r in rows)) for key, rows in groups.items()}
    save(out / 'slot_texts.json', [{'filename': r['filename'], 'texts': texts} for r, texts in zip(all_rows, text_rows)])
    save(out / 'protocol.json', cfg)
    print('PREPARATION PASS:', json.dumps(cfg['counts_by_mode']), 'cohort:', cfg['cohort_sha256'], flush=True)


def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg, tokenizer):
    from captioning.models import UniversalVisionEncoder, CaptionDecoder, ImageCaptioningModel
    from captioning.checkpoints import load_checkpoint
    from pilot_model import PilotModel
    base = ImageCaptioningModel(UniversalVisionEncoder(load_backbone=False, visual_adapter='qformer',
                               num_visual_queries=32, qformer_layers=2), CaptionDecoder(len(tokenizer.word2idx)))
    load_checkpoint(base, cfg['checkpoint'], tokenizer)
    return PilotModel(base, cfg['arm']).cuda().eval()


def corpus(cfg):
    from captioning.tokenizer import CaptionTokenizer
    dataset = json.loads(Path(cfg['dataset_json']).read_text())
    train = [r for r in dataset['images'] if r['split'] in ('train', 'restval')]
    tokenizer = CaptionTokenizer([s['raw'] for r in train for s in r['sentences'][:5]])
    return dataset, tokenizer


def make_loader(cfg, dataset, tokenizer, split, rank, world):
    import h5py
    import torch
    from torch.utils.data import Dataset, DataLoader, DistributedSampler
    rows = cfg['groups'][split]
    train_count = len(cfg['groups']['train'])

    class Staged(Dataset):
        def __init__(self):
            self.f = h5py.File(cfg['stage'], 'r')
            if not self.f.attrs.get('complete') or self.f.attrs['cohort_sha256'] != cfg['cohort_sha256']:
                raise ValueError('Incomplete/different staged cohort.')

        def __len__(self):
            return len(rows)

        def __getitem__(self, index):
            row = rows[index]
            i = index if split == 'train' else index + train_count
            result = {key: torch.from_numpy(self.f[key][i]) for key in
                      ('visual', 'legacy', 'legacy_mask', 'roles', 'valid', 'text', 'bounds', 'phrase_ids')}
            refs = dataset['images'][row['image_id']]['sentences'][:5]
            encoded = [tokenizer.encode(s['raw']) for s in refs]
            result['captions'] = torch.stack([x[0] for x in encoded])
            result['caption_mask'] = torch.stack([x[1] for x in encoded])
            result['image_id'] = torch.tensor(row['image_id'])
            # Same-mode deterministic donor. Missing slots are not filled with fake evidence.
            candidates = donors[row['mode']]
            pos = candidates.index(index)
            donor = candidates[(pos + 1) % len(candidates)]
            di = donor if split == 'train' else donor + train_count
            donor_valid = self.f['valid'][di]
            result['swap_visual']=torch.from_numpy(self.f['visual'][di])
            result['swap_bounds']=torch.from_numpy(self.f['bounds'][di])
            joint = self.f['valid'][i] & donor_valid
            result['swap_text'] = torch.from_numpy(np.where(joint[:, None], self.f['text'][di], self.f['text'][i]))
            result['swap_valid']=torch.from_numpy(joint)
            return result

    donors = {mode: [i for i, r in enumerate(rows) if r['mode'] == mode] for mode in set(r['mode'] for r in rows)}
    staged = Staged()
    if split == 'train':
        sampler = DistributedSampler(staged, num_replicas=world, rank=rank, shuffle=True, seed=42, drop_last=False)
        return DataLoader(staged, batch_size=cfg['global_batch_size']//world, sampler=sampler,
                          num_workers=0, pin_memory=True, drop_last=False)
    # Explicit rank slice: no padding or duplicate validation IDs from DistributedSampler.
    subset = torch.utils.data.Subset(staged, list(range(rank, len(rows), world)))
    return DataLoader(subset, batch_size=cfg['eval_batch_size'], shuffle=False, num_workers=0, pin_memory=True)


def cuda_batch(batch):
    return {key: value.cuda(non_blocking=True) for key, value in batch.items()}


def ground_truth(cfg, dataset, out):
    rows = cfg['groups']['val']
    gt = {'info': {}, 'licenses': [], 'type': 'captions', 'images': [{'id': r['image_id']} for r in rows],
          'annotations': [{'id': i*5+j, 'image_id': r['image_id'], 'caption': ref['raw']}
                          for i, r in enumerate(rows) for j, ref in enumerate(dataset['images'][r['image_id']]['sentences'][:5])]}
    save(out / 'val_gt.json', gt)


def predict(model, loader, tokenizer, intervention):
    import torch
    from captioning.scst import rollout, decode
    records, diagnostics = [], []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            batch = cuda_batch(batch)
            memory, padding = model.memory(batch, intervention)
            seq, _ = rollout(model, memory, padding, tokenizer, sample=False)
            texts = decode(seq, tokenizer)
            target = batch['captions'].flatten(0, 1)
            mask = batch['caption_mask'].flatten(0, 1)
            logits = model.teacher(memory, padding, target[:, :-1], mask[:, :-1])
            logp = logits.log_softmax(-1).gather(-1, target[:, 1:, None]).squeeze(-1)
            valid = target[:, 1:].ne(tokenizer.pad_idx)
            nll = (-(logp * valid).sum(1) / valid.sum(1).clamp_min(1)).reshape(-1, 5).mean(1).cpu().tolist()
            ids = batch['image_id'].cpu().tolist()
            for i, (image_id, caption, value) in enumerate(zip(ids, texts, nll)):
                records.append({'image_id': image_id, 'caption': caption})
                row = {'image_id': image_id, 'reference_nll_per_token': value}
                if model.extra is not None:
                    v = batch['valid'][i]
                    row['valid_evidence_slots']=int(v.sum())
                    att=model.last_attention[i]
                    row['attention_entropy']=float((-(att.clamp_min(1e-12).log()*att).sum(-1))[v].mean()) if v.any() else None
                diagnostics.append(row)
            if index % 10 == 0:
                print(f'Evaluation {intervention}: {len(records)} local images', flush=True)
    return records, diagnostics


def worker(protocol):
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from captioning.scst import TrainingCider, rollout, decode
    from captioning.object_reward import TrainingObjectReward, combine_rewards
    from captioning.scst_runtime import configure_strict_runtime, runtime_manifest, model_digest
    cfg = json.loads(Path(protocol).read_text())
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])
    if world != 2:
        raise RuntimeError('This pilot requires exactly two GPU ranks for ONE arm.')
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(2)
    configure_strict_runtime(True)
    # Both GPUs belong to this single arm; NCCL synchronizes its gradients.
    from datetime import timedelta
    dist.init_process_group('nccl', timeout=timedelta(minutes=20))
    out = Path(cfg['output'])
    save(out / f'runtime_rank{rank}.json', runtime_manifest())
    seed_all(42)
    dataset, tokenizer = corpus(cfg)
    model = build_model(cfg, tokenizer)
    initial_base_hash = model_digest(model.base)
    train = make_loader(cfg, dataset, tokenizer, 'train', rank, world)
    validation = make_loader(cfg, dataset, tokenizer, 'val', rank, world)
    if len(train) != math.ceil(len(cfg['groups']['train'])/cfg['global_batch_size']):
        raise ValueError('Unexpected DDP training step count.')
    # Inspect the frozen starting system with extra tokens masked off, on the same GPU/profile.
    records, detail = predict(model, validation, tokenizer, 'off')
    save(out / f'origin_rank{rank}.json', records)
    save(out / f'origin_diagnostic_rank{rank}.json', detail)
    # Every rank uses identical source initialization; different sampling RNG streams are explicit.
    seed_all(42 + rank)
    if digest(out / 'train_object_targets.json') != cfg['object_targets_sha256']:
        raise ValueError('Object reward targets changed.')
    targets = json.loads((out / 'train_object_targets.json').read_text())['data']
    object_reward = TrainingObjectReward(targets, 'aliases_v2', 'unique_count_v2')
    # Retain full TRAIN IDF like the successful full-train baseline, even in this 10k pilot.
    references = {i: [s['raw'] for s in row['sentences'][:5]] for i, row in enumerate(dataset['images'])
                  if row['split'] in ('train', 'restval')}
    print(f'[rank {rank}] Building fixed TRAIN CIDEr-D IDF: {len(references)} images; CPU work, please wait.', flush=True)
    reward = TrainingCider(references, reference_cache_size=256)
    print(f'[rank {rank}] TRAIN reward ready; starting synchronized updates.', flush=True)
    params = [{'params': [p for p in model.base.parameters() if p.requires_grad], 'lr': cfg['base_lr']}]
    if model.extra is not None:
        params.append({'params': list(model.extra.parameters()), 'lr': cfg['branch_lr']})
    optimizer = torch.optim.AdamW(params)
    # No collective calls during sampled/greedy decoding: those use the unwrapped model in no_grad.
    # All ranks execute exactly ONE synchronized loss backward per update.
    ddp = DDP(model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False,
              find_unused_parameters=True)
    history = []
    begin = time.monotonic()
    for step, batch in enumerate(train, 1):
        batch = cuda_batch(batch)
        optimizer.zero_grad(set_to_none=True)
        weight = 0. if step <= cfg['warmup_updates'] else cfg['scst_weight']
        sample = active = advantage = None
        record = {'step': step, 'global_images_seen': min(step*cfg['global_batch_size'], 10000),
                  'scst_weight': weight}
        if weight:
            with torch.no_grad():
                memory, padding = model.memory(batch)
                sample, active = rollout(model, memory, padding, tokenizer, sample=True)
                greedy, _ = rollout(model, memory, padding, tokenizer, sample=False)
            ids = batch['image_id'].cpu().tolist()
            sampled_text, greedy_text = decode(sample, tokenizer), decode(greedy, tokenizer)
            sample_cider, greedy_cider = reward.score(ids, sampled_text), reward.score(ids, greedy_text)
            sample_object, _ = object_reward.score(ids, sampled_text)
            greedy_object, _ = object_reward.score(ids, greedy_text)
            sample_reward = combine_rewards(sample_cider, sample_object, .5)
            greedy_reward = combine_rewards(greedy_cider, greedy_object, .5)
            advantage = torch.from_numpy(sample_reward - greedy_reward).cuda()
            record.update(sample_reward=float(sample_reward.mean()), greedy_reward=float(greedy_reward.mean()),
                          sample_cider=float(sample_cider.mean()), greedy_cider=float(greedy_cider.mean()),
                          sample_object=float(sample_object.mean()), greedy_object=float(greedy_object.mean()))
        loss, ce, rl = ddp(batch, sample, active, advantage, weight, tokenizer.pad_idx)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss, rank {rank}, step {step}.')
        loss.backward()
        branch_grad = (sum(float(p.grad.detach().square().sum()) for p in model.extra.parameters() if p.grad is not None)**.5
                       if model.extra is not None else 0.)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        record.update(alignment=model.last_alignment,
                      submodule_grad_norms={name:sum(float(p.grad.detach().square().sum()) for p in part.parameters() if p.grad is not None)**.5 for name,part in model.extra.named_children()} if model.extra is not None else {},
                      loss=float(loss.detach()), ce_loss=float(ce), scst_loss=float(rl),
                      branch_grad_norm=branch_grad, grad_norm=float(grad_norm), seconds=time.monotonic()-begin)
        history.append(record)
        if step == 1 or step % 25 == 0 or step == len(train):
            save(out / f'history_rank{rank}.json', history)
            print(f'[{cfg["arm"]} rank {rank}] {json.dumps(record)}', flush=True)
    if rank == 0:
        state = {'format': 'semantic_evidence_pilot_v1',
                 'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                 'arm': cfg['arm'], 'caption_word2idx': tokenizer.word2idx, 'cohort_sha256': cfg['cohort_sha256'],
                 'source_checkpoint_sha256': cfg['source_checkpoint_sha256'],
                 'protocol_sha256': digest(protocol), 'completed_updates': len(train)}
        temporary = out / 'checkpoint.pth.partial'
        # Legacy serialization avoids Kaggle treating a .pth checkpoint as a ZIP to unpack.
        torch.save(state, temporary, _use_new_zipfile_serialization=False)
        temporary.replace(out / 'checkpoint.pth')
    # Evaluation has no DDP communication; different rank runtimes cannot stall loss collectives.
    conditions = ['on'] if cfg['arm'] == 'A' else ['on', 'off', 'text_swap', 'region_swap']
    for condition in conditions:
        records, detail = predict(model, validation, tokenizer, condition)
        save(out / f'{condition}_rank{rank}.json', records)
        save(out / f'{condition}_diagnostic_rank{rank}.json', detail)
    save(out / f'worker_complete_rank{rank}.json', {'status': 'COMPLETE', 'updates': len(train),
         'initial_base_sha256': initial_base_hash, 'rank': rank, 'source_arm': cfg['arm'],
         'trainable_base_parameters': sum(p.numel() for p in model.base.parameters() if p.requires_grad),
         'trainable_branch_parameters': sum(p.numel() for p in model.extra.parameters()) if model.extra is not None else 0,
         'test_used': False})
    dist.destroy_process_group()


def evaluate(protocol):
    from captioning.metrics import score_files
    cfg = json.loads(Path(protocol).read_text())
    out = Path(cfg['output'])
    dataset = json.loads(Path(cfg['dataset_json']).read_text())
    ground_truth(cfg, dataset, out)
    expected = {r['image_id']: r for r in cfg['groups']['val']}
    conditions = ['origin', 'on'] + ([] if cfg['arm'] == 'A' else ['off', 'text_swap', 'region_swap'])
    results, predictions, details = {}, {}, {}
    for condition in conditions:
        rows, detail = [], []
        for rank in (0, 1):
            completion = json.loads((out / f'worker_complete_rank{rank}.json').read_text())
            if completion['status'] != 'COMPLETE':
                raise RuntimeError('A GPU worker did not finish.')
            rows.extend(json.loads((out / f'{condition}_rank{rank}.json').read_text()))
            detail.extend(json.loads((out / f'{condition}_diagnostic_rank{rank}.json').read_text()))
        rows.sort(key=lambda r: r['image_id'])
        if len(rows) != len(expected) or {r['image_id'] for r in rows} != expected.keys():
            raise ValueError('Missing/duplicated validation predictions.')
        path = out / f'{condition}_predictions.json'
        save(path, rows)
        predictions[condition] = {r['image_id']: r['caption'] for r in rows}
        details[condition] = {r['image_id']: r for r in detail}
        if len(details[condition]) != len(expected):
            raise ValueError('Incomplete validation diagnostics.')
        # Score branch removal as well: a caption change alone is not evidence of a useful branch.
        if condition not in ('origin', 'on', 'off'):
            continue
        caption = score_files(path, out / 'val_gt.json', out / f'{condition}_metrics.json')
        chair_input = out / f'{condition}_chair_input.json'
        chair_output = out / f'{condition}_chair_output.json'
        save(chair_input, [{'image_id': expected[r['image_id']]['coco_id'], 'caption': r['caption']} for r in rows])
        subprocess.run([sys.executable, '-u', str(Path(cfg['chair'])/'chair.py'), '--cap_file', str(chair_input),
                        '--cache', str(Path(cfg['chair'])/'chair.pkl'), '--save_path', str(chair_output)],
                       cwd=cfg['chair'], check=True)
        chair = json.loads(chair_output.read_text())
        if {r['image_id'] for r in chair['sentences']} != {r['coco_id'] for r in expected.values()}:
            raise ValueError('CHAIR IDs differ.')
        # Per-image CIDEr uses the SAME full validation IDF for all mode subsets.
        from pycocotools.coco import COCO
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        from pycocoevalcap.cider.cider import Cider
        gt = COCO(str(out / 'val_gt.json'))
        tokenizer = PTBTokenizer()
        ids = [r['image_id'] for r in rows]
        refs = tokenizer.tokenize({i: gt.imgToAnns[i] for i in ids})
        hyp = tokenizer.tokenize({r['image_id']: [{'caption': r['caption']}] for r in rows})
        _, per_image = Cider().compute_score(refs, hyp)
        grouped = {}
        for mode in ('object_relation', 'object_only', 'no_objects'):
            indices = [i for i, row in enumerate(rows) if expected[row['image_id']]['mode'] == mode]
            grouped[mode] = {'images': len(indices), 'CIDEr_full_val_IDF': float(np.mean(np.asarray(per_image)[indices])) if indices else None}
        save(out / f'{condition}_per_image_cider.json', [{'image_id': r['image_id'], 'CIDEr': float(v)} for r, v in zip(rows, per_image)])
        results[condition] = {'caption': caption, 'chair': chair['overall_metrics'], 'by_mode': grouped}
    sensitivity = {}
    for condition in conditions:
        if condition in ('origin', 'on'):
            continue
        sensitivity[condition] = {}
        for mode in ('all', 'object_relation', 'object_only', 'no_objects'):
            ids = [i for i, row in expected.items() if mode == 'all' or row['mode'] == mode]
            sensitivity[condition][mode] = {'images': len(ids),
                'changed_captions': sum(predictions[condition][i] != predictions['on'][i] for i in ids),
                'mean_reference_nll_delta': float(np.mean([details[condition][i]['reference_nll_per_token'] -
                                                          details['on'][i]['reference_nll_per_token'] for i in ids])) if ids else None}
    summary = {'status': 'PILOT_COMPLETE', 'arm': cfg['arm'], 'metrics': results, 'sensitivity': sensitivity,
               'cohort_sha256': cfg['cohort_sha256'], 'stage_sha256': cfg['stage_sha256'],
               'protocol_sha256': digest(protocol), 'checkpoint_sha256': digest(out / 'checkpoint.pth'),
               'test_used': False, 'selection': 'End checkpoint only, fixed before metrics; no automatic promotion',
               'policy': cfg['evaluation_policy'], 'limits': [cfg['validation_scope'], cfg['comparison_limits'],
               'Greedy pilot metrics must not be compared directly to historical beam5/fixed test metrics.',
               'Text/region swaps are sensitivity probes, not verified false relations.',
               'CHAIR measures objects, not predicate correctness; contrastive scores are not calibrated probabilities.',
               'Different accounts/GPUs may differ numerically; compare input/cohort/runtime manifests.']}
    save(out / 'run_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)


def archive(protocol):
    cfg = json.loads(Path(protocol).read_text())
    out = Path(cfg['output'])
    with zipfile.ZipFile(str(out) + '_review.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.log', '.py'):
                z.write(path, path.relative_to(out))
    print('Return:', str(out) + '_review.zip', flush=True)
    print('Keep checkpoint separately in saved notebook output:', out / 'checkpoint.pth', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'worker', 'evaluate', 'archive'])
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    globals()[args.mode](args.config)
