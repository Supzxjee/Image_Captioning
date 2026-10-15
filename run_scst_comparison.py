"""Manual Kaggle CE/SCST experiment. No training runs on import."""
import argparse
import json
import os
import random
import subprocess
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from run_hoi_caption_validation import sha, save, command, run_workers


def cohorts(dataset, train_count=20000, val_count=1000):
    groups = {'train': [], 'val': []}
    for i, r in enumerate(dataset['images']):
        split = 'train' if r['split'] in ('train', 'restval') else r['split']
        if split in groups:
            groups[split].append({'image_id': i, 'coco_id': int(r['filename'].split('_')[-1].split('.')[0]),
                                  'filename': r['filename'], 'filepath': r['filepath']})
    for name, count in [('train', train_count), ('val', val_count)]:
        if count < 2 or count > len(groups[name]):
            raise ValueError(f'Invalid {name} cohort size {count}.')
        random.Random(20261016 if name == 'train' else 20261017).shuffle(groups[name])
        groups[name] = sorted(groups[name][:count], key=lambda r: r['image_id'])
    if len({r['coco_id'] for rows in groups.values() for r in rows}) != train_count+val_count:
        raise ValueError('Train/validation identities overlap.')
    return groups


def training_arms(cfg):
    return ('scst', 'scst_ce') if cfg.get('ce_anchor_comparison', False) else ('ce', 'scst')


def objective_weights(cfg, arm):
    if arm not in training_arms(cfg):
        raise ValueError(f'Unexpected training arm: {arm}')
    return (float(cfg.get('scst_weight', 1.)) if arm != 'ce' else 0.,
            float(cfg.get('ce_weight', 1.)) if arm in ('ce', 'scst_ce') else 0.)


def caption_ce(model, images, prompt, prompt_mask, captions, masks, pad_idx):
    target = captions.reshape(-1, captions.size(-1))
    logits = model(images, prompt, prompt_mask, target[:, :-1],
                   tgt_key_padding_mask=masks.reshape_as(target)[:, :-1])
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target[:, 1:].reshape(-1),
                                            ignore_index=pad_idx)


def fit(config_path, arm):
    from torch.utils.data import DataLoader, Dataset
    from captioning.config import Config
    from captioning.data import load_data, CocoPromptDataset
    from captioning.models import UniversalVisionEncoder, CaptionDecoder, ImageCaptioningModel
    from captioning.checkpoints import load_checkpoint
    from captioning.scst import TrainingCider, rollout, decode, policy_loss
    cfg = json.loads(Path(config_path).read_text(encoding='utf-8'))
    rl_weight, ce_weight = objective_weights(cfg, arm)
    torch.set_num_threads(2)
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires a CUDA GPU.')
    c = Config(mode='train', device='cuda', visual_adapter='qformer', num_visual_queries=32,
               qformer_layers=2, visual_cache=[cfg['visual_cache']],
               dataset_json_path=cfg['dataset_json_path'], base_path=cfg['base_path'],
               prompt_cache_path=cfg['prompt_cache_path'], num_workers=0)
    data = load_data(c)
    ids = [r['image_id'] for r in cfg['groups']['train']]
    frame = data.train_df.set_index('eval_id', drop=False).loc[ids].reset_index(drop=True)
    references = {int(r['eval_id']): r['captions'] for _, r in frame.iterrows()}
    reward = TrainingCider(references) if rl_weight > 0 else None
    dataset = CocoPromptDataset(frame, data.transform, data.tokenizer, data.prompt_cache, data.visual_cache)

    class Indexed(Dataset):
        def __len__(self): return len(dataset)
        def __getitem__(self, index): return index, dataset[index]

    loader = DataLoader(Indexed(), batch_size=cfg['batch_size'], shuffle=True, drop_last=False,
                        num_workers=0, generator=torch.Generator().manual_seed(42))
    model = ImageCaptioningModel(UniversalVisionEncoder(load_backbone=False, visual_adapter='qformer',
                                num_visual_queries=32, qformer_layers=2), CaptionDecoder(data.vocab_size)).cuda()
    payload = load_checkpoint(model, cfg['checkpoint'], data.tokenizer)
    # Disable dropout in BOTH arms. eval() does not disable gradients.
    # This makes rollout and teacher-forced sequence likelihood the SAME policy.
    model.eval()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg['lr'])
    output = Path(cfg['output'])/arm
    output.mkdir(parents=True, exist_ok=True)
    if (output/'checkpoint.pth').exists():
        raise FileExistsError('Completed checkpoint already exists; choose a fresh output directory.')
    history, started = [], time.monotonic()
    print(f'{arm}: {len(frame)} images | {len(loader)} updates | FP32 | dropout disabled', flush=True)
    try:
        for step, (indices, batch) in enumerate(loader, 1):
            images, captions, masks, prompt, prompt_mask = [v.cuda() for v in batch]
            optimizer.zero_grad(set_to_none=True)
            record = {'step': step, 'images_seen': min(step*cfg['batch_size'], len(frame))}
            if arm == 'ce':
                loss = ce_weight*caption_ce(model, images, prompt, prompt_mask, captions, masks, data.tokenizer.pad_idx)
            else:
                with torch.no_grad():
                    memory, padding = model.build_memory(images, prompt, prompt_mask)
                    sampled, active = rollout(model, memory, padding, data.tokenizer, sample=True)
                    greedy, _ = rollout(model, memory, padding, data.tokenizer, sample=False)
                image_ids = [ids[i] for i in indices.tolist()]
                sampled_reward = reward.score(image_ids, decode(sampled, data.tokenizer))
                greedy_reward = reward.score(image_ids, decode(greedy, data.tokenizer))
                advantage = torch.from_numpy(sampled_reward-greedy_reward).cuda()
                # Recompute with gradients, avoiding retention of 29 rollout graphs.
                logits = model(images, prompt, prompt_mask, sampled[:, :-1])
                rl_loss = policy_loss(logits, sampled[:, 1:], active, advantage)
                loss = rl_weight*rl_loss
                record['scst_loss'] = float(rl_loss.detach())
                record.update(sample_reward=float(sampled_reward.mean()), greedy_reward=float(greedy_reward.mean()),
                              advantage_mean=float(advantage.mean()), advantage_std=float(advantage.std(unbiased=False)),
                              sampled_eos_fraction=float((sampled[:, 1:] == data.tokenizer.eos_idx).any(1).float().mean()))
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{arm}: nonfinite loss at step {step}')
            loss.backward()
            if arm == 'scst_ce':
                # Fresh CE graph after RL backward: accumulate BOTH gradients,
                # then clip once and take exactly one optimizer step.
                ce_loss = caption_ce(model, images, prompt, prompt_mask, captions, masks, data.tokenizer.pad_idx)
                if not torch.isfinite(ce_loss):
                    raise FloatingPointError(f'Nonfinite CE anchor at step {step}')
                (ce_weight*ce_loss).backward()
                record['ce_loss'] = float(ce_loss.detach())
                loss = loss.detach()+ce_weight*ce_loss.detach()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            record.update(loss=float(loss.detach()), grad_norm=float(grad_norm), seconds=time.monotonic()-started)
            history.append(record)
            if step == 1 or step % 25 == 0 or step == len(loader):
                save(output/'history.json', history)
                print(arm, json.dumps(record), flush=True)
        payload.update(encoder_state_dict={k: v.detach().cpu() for k, v in model.encoder.state_dict().items()},
                       decoder_state_dict={k: v.detach().cpu() for k, v in model.decoder.state_dict().items()},
                       caption_word2idx=data.tokenizer.word2idx, epoch=1, optimization=arm,
                       scst_weight=rl_weight, ce_weight=ce_weight,
                       scst_protocol_sha256=sha(config_path), source_checkpoint_sha256=cfg['input_sha256']['checkpoint'])
        payload.pop('optimizer_state_dict', None)
        torch.save(payload, output/'checkpoint.pth.partial')
        (output/'checkpoint.pth.partial').replace(output/'checkpoint.pth')
        save(output/'training_summary.json', {'arm': arm, 'steps': len(history), 'train_images': len(frame),
             'seconds': time.monotonic()-started, 'checkpoint_sha256': sha(output/'checkpoint.pth'),
             'source_checkpoint_sha256': cfg['input_sha256']['checkpoint'],
             'protocol_sha256': sha(config_path), 'dropout': False, 'precision': 'fp32',
             'scst_weight': rl_weight, 'ce_weight': ce_weight})
    finally:
        data.visual_cache.close()


def train_pair(cfg, protocol, repo):
    processes, streams = [], []
    try:
        for gpu, arm in enumerate(training_arms(cfg)):
            stream = (Path(cfg['output'])/f'{arm}.log').open('w', encoding='utf-8'); streams.append(stream)
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONUNBUFFERED': '1', 'HDF5_USE_FILE_LOCKING': 'FALSE'}
            processes.append(subprocess.Popen([sys.executable, '-u', str(repo/'run_scst_comparison.py'),
                '--worker', arm, '--protocol', str(protocol)], cwd=repo, env=env, stdout=stream, stderr=subprocess.STDOUT))
        positions = [0, 0]; last_heartbeat = time.monotonic()
        while True:
            for i, arm in enumerate(training_arms(cfg)):
                with (Path(cfg['output'])/f'{arm}.log').open(encoding='utf-8', errors='replace') as f:
                    f.seek(positions[i]); text = f.read(); positions[i] = f.tell()
                if text: print(f'[{arm}] {text}', end='', flush=True)
                if processes[i].poll() not in (None, 0):
                    raise RuntimeError(f'{arm} worker failed; see {arm}.log')
            if all(p.poll() == 0 for p in processes): break
            if time.monotonic()-last_heartbeat > 30:
                print('CE/SCST workers active; independent GPU jobs, no DDP collectives.', flush=True)
                last_heartbeat = time.monotonic()
            time.sleep(2)
    finally:
        for p in processes:
            if p.poll() is None: p.terminate()
        for p in processes:
            try: p.wait(timeout=10)
            except subprocess.TimeoutExpired: p.kill(); p.wait()
        for f in streams: f.close()


def evaluate(cfg, repo):
    from captioning.metrics import score_files
    from captioning.multiscore_reranking import rerank_multiscore
    from captioning.object_consistency import index_detections
    output = Path(cfg['output']); chair = Path(cfg['chair_directory'])
    detections = index_detections(json.loads(Path(cfg['detections']).read_text(encoding='utf-8')))
    dataset = json.loads(Path(cfg['dataset_json_path']).read_text(encoding='utf-8'))['images']
    images = cfg['groups']['val']
    gt = {'info': {}, 'licenses': [], 'type': 'captions', 'images': [{'id': r['image_id']} for r in images],
          'annotations': [{'id': n*5+j, 'image_id': r['image_id'], 'caption': s['raw']}
                          for n, r in enumerate(images) for j, s in enumerate(dataset[r['image_id']]['sentences'][:5])]}
    save(output/'val_gt.json', gt)
    metrics = {}
    for arm in ('original', *training_arms(cfg)):
        folder = output/arm; folder.mkdir(parents=True, exist_ok=True)
        checkpoint = cfg['checkpoint'] if arm == 'original' else str(folder/'checkpoint.pth')
        worker_cfg = {**cfg, 'output': str(folder), 'checkpoint': checkpoint, 'images': images,
                      'workers': 2, 'split': 'val', 'checkpoint_sha256': sha(checkpoint)}
        records = run_workers(worker_cfg, 'beams', repo)
        raw, scored = folder/'beams.json', folder/'clip_beams.json'
        save(raw, {'metadata': {'split': 'val', 'count': len(records), 'checkpoint_sha256': sha(checkpoint)}, 'data': records})
        command([sys.executable, '-u', repo/'score_clip_candidates.py', '--candidates', raw,
                 '--visual-cache', cfg['visual_cache'], '--output', scored, '--device', 'cuda',
                 '--revision', cfg['clip_revision']])
        records = json.loads(scored.read_text(encoding='utf-8'))['data']
        for policy in ('rank0', 'fixed'):
            chosen = [r['candidates'][0] if policy == 'rank0' else
                      rerank_multiscore(r, detections[r['filename']], .4, .3)[0] for r in records]
            name = f'{arm}_{policy}'; pred, ci, co = (folder/f'{policy}_{s}.json' for s in ('predictions', 'chair_input', 'chair_output'))
            save(pred, [{'image_id': r['image_id'], 'caption': c['caption']} for r, c in zip(records, chosen)])
            caption = score_files(pred, output/'val_gt.json', folder/f'{policy}_metrics.json')
            save(ci, [{'image_id': r['coco_id'], 'caption': c['caption']} for r, c in zip(records, chosen)])
            command([sys.executable, '-u', chair/'chair.py', '--cap_file', ci, '--cache', chair/'chair.pkl', '--save_path', co], cwd=chair)
            evaluated = json.loads(co.read_text(encoding='utf-8'))
            if {r['image_id'] for r in evaluated['sentences']} != {r['coco_id'] for r in images}:
                raise ValueError('CHAIR image IDs differ.')
            metrics[name] = {'caption': caption, 'chair': evaluated['overall_metrics']}
    return metrics


def run(a):
    repo = Path(__file__).resolve().parent; output = Path(a.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output/'protocol.json').exists():
        raise FileExistsError('Use a fresh OUTPUT directory; do not overwrite a prior comparison.')
    paths = {k: str(Path(getattr(a, k)).resolve()) for k in ('checkpoint', 'prompt_cache_path', 'dataset_json_path',
             'base_path', 'visual_cache', 'detections', 'chair_directory', 'output')}
    for k in ('checkpoint', 'prompt_cache_path', 'dataset_json_path', 'detections'):
        if not Path(paths[k]).is_file(): raise FileNotFoundError(paths[k])
    if not (Path(paths['chair_directory'])/'chair.pkl').is_file():
        raise FileNotFoundError('Pinned CHAIR evaluator cache required before training.')
    payload = torch.load(paths['checkpoint'], map_location='cpu', weights_only=False)
    if (payload.get('epoch'), payload.get('visual_adapter'), payload.get('num_visual_queries'), payload.get('qformer_layers')) != (10, 'qformer', 32, 2):
        raise ValueError('Expected original QFormer epoch10/32q/2l checkpoint.')
    if any(payload.get(k, False) for k in ('prompt_conditioned_qformer', 'object_semantic_alignment',
                                         'cascade_semantic_alignment', 'grounded_relation_alignment')) or payload.get('itc_weight', 0):
        raise ValueError('This comparison requires the unmodified QFormer baseline.')
    del payload
    dataset = json.loads(Path(paths['dataset_json_path']).read_text(encoding='utf-8'))
    groups = cohorts(dataset, a.train_images, a.validation_images)
    input_hashes = {k: sha(paths[k]) for k in ('checkpoint', 'prompt_cache_path', 'dataset_json_path', 'detections')}
    expected = json.loads((repo/'audit/hoi_validation_input_identity.json').read_text())['input_sha256']
    if any(value != expected[key] for key, value in input_hashes.items()):
        raise ValueError('Input bytes differ from the previously verified QFormer baseline inputs.')
    cfg = {**paths, 'groups': groups, 'batch_size': a.batch_size, 'lr': a.lr, 'epochs': 1, 'seed': 42,
           'ce_anchor_comparison': a.ce_anchor_comparison, 'scst_weight': a.scst_weight, 'ce_weight': a.ce_weight,
           'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
           'input_sha256': input_hashes,
           'clip_revision': '57c216476eefef5ab752ec549e440a49ae4ae5f3',
           'reward': 'CIDEr-D x10; Treebank normalization; IDF from selected training images only',
           'dropout': False, 'precision': 'fp32', 'test_used': False,
           'budget_matching': 'Same images, order, optimizer/LR and updates; not equal FLOPs or number of captions.',
           'evaluation': 'original plus two training arms; same beam5 and fixed .3 decoder/.4 CLIP/.3 OCC; no UPT; no tuning'}
    if a.ce_anchor_comparison:
        cfg['variant'] = 'scst_ce_anchor_v1'
        cfg['anchor_hypothesis'] = 'Add CE only; same scaled SCST coefficient in both arms; no object/relation rewards.'
    from captioning.visual_cache import VisualCache
    cache = VisualCache([paths['visual_cache']], id_key='coco_id')
    try: cache.require_ids([r['coco_id'] for rows in groups.values() for r in rows], 'SCST comparison')
    finally: cache.close()
    protocol = output/'protocol.json'; save(protocol, cfg)
    # Verify CHAIR coverage before any expensive training.
    check = "import sys,json,pickle;sys.path.insert(0,sys.argv[1]);from chair import CHAIR;c=pickle.load(open(sys.argv[2],'rb'));p=json.load(open(sys.argv[3]));assert all(r['coco_id'] in c.imid_to_objects for r in p['groups']['val']);print('CHAIR coverage PASS')"
    command([sys.executable, '-c', check, paths['chair_directory'], Path(paths['chair_directory'])/'chair.pkl', protocol])
    train_pair(cfg, protocol, repo)
    metrics = evaluate(cfg, repo)
    treatment, control = ('scst_ce', 'scst') if a.ce_anchor_comparison else ('scst', 'ce')
    scst, ce, original = (metrics[k+'_rank0'] for k in (treatment, control, 'original'))
    better = scst['caption']['CIDEr'] > max(ce['caption']['CIDEr'], original['caption']['CIDEr'])
    safe = all(scst['chair'][key] <= min(ce['chair'][key], original['chair'][key]) for key in ('CHAIRs', 'CHAIRi'))
    recall = scst['chair']['Recall'] >= max(ce['chair']['Recall'], original['chair']['Recall'])-.01
    summary = {'status': 'COMPARISON_COMPLETE', 'metrics': metrics, 'test_used': False,
               'train_images': a.train_images, 'validation_images': a.validation_images,
               'raw_SCST_beats_both_CIDEr': better, 'raw_CHAIR_not_worse_than_both': safe,
               'raw_recall_within_1pp_of_best': recall,
               'expansion_gate_passed': bool(better and safe and recall), 'protocol_sha256': sha(protocol),
               'limits': ['One seed; validation has been inspected in prior work; exploratory, not confirmatory.',
                          'SCST is an established training method, not our claimed contribution.',
                          'CHAIR measures object hallucination, not relation correctness.',
                          'Reward normalization differs from official PTB evaluation; no test-based selection.']}
    if a.ce_anchor_comparison:
        # The primary gate is assessed before any test or checkpoint selection.
        fixed, baseline = metrics['scst_ce_fixed'], metrics['original_fixed']
        fixed_safe = (fixed['caption']['CIDEr'] >= baseline['caption']['CIDEr'] and
                      all(fixed['chair'][k] <= baseline['chair'][k] for k in ('CHAIRs', 'CHAIRi')) and
                      fixed['chair']['Recall'] >= baseline['chair']['Recall']-.01)
        summary.update(variant=cfg['variant'], treatment=treatment, control=control,
                       raw_anchor_beats_both_CIDEr=better, raw_anchor_CHAIR_not_worse_than_both=safe,
                       raw_anchor_recall_within_1pp_of_best=recall,
                       fixed_not_worse_than_original=fixed_safe,
                       expansion_gate_passed=bool(better and safe and recall and fixed_safe))
        for key in ('raw_SCST_beats_both_CIDEr', 'raw_CHAIR_not_worse_than_both', 'raw_recall_within_1pp_of_best'):
            summary.pop(key)
    save(output/'run_summary.json', summary)
    with zipfile.ZipFile(str(output)+'_review.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for file in sorted(output.rglob('*')):
            if file.is_file() and file.suffix in ('.json', '.jsonl', '.log'):
                z.write(file, file.relative_to(output))
    print(json.dumps(summary, indent=2), flush=True)
    print('RETURN FILE:', str(output)+'_review.zip', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker', choices=['ce', 'scst', 'scst_ce']); p.add_argument('--protocol')
    p.add_argument('--ce-anchor-comparison', action='store_true')
    p.add_argument('--scst-weight', type=float, default=1.); p.add_argument('--ce-weight', type=float, default=1.)
    for k in ('checkpoint', 'prompt-cache-path', 'dataset-json-path', 'base-path', 'visual-cache',
              'detections', 'chair-directory', 'output'): p.add_argument('--'+k)
    p.add_argument('--train-images', type=int, default=20000)
    p.add_argument('--validation-images', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=8); p.add_argument('--lr', type=float, default=5e-6)
    args = p.parse_args()
    if args.worker: fit(args.protocol, args.worker)
    else:
        if args.batch_size < 1 or not 0 < args.lr < 1: p.error('Invalid batch size or LR.')
        if not (0 < args.scst_weight < float('inf') and 0 < args.ce_weight < float('inf')):
            p.error('Objective weights must be finite and positive.')
        required = ('checkpoint', 'prompt_cache_path', 'dataset_json_path', 'base_path', 'visual_cache', 'detections', 'chair_directory', 'output')
        if any(not getattr(args, k) for k in required): p.error('All explicit input paths and --output are required.')
        run(args)
