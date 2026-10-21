"""Manual A/H caption pilot. Reuses the audited caption optimizer/SCST/DDP loop."""
import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import numpy as np

import run_pilot as shared
from hidden_data import INPUT_HASHES, COHORT_SHA, SLOTS, WIDTH, check_cache, locked_groups, validate_row, human_sample
from pilot_data import digest, identity, save

LIVE_MODEL = None


def prepare(config):
    import h5py
    import torch
    from captioning.visual_cache import VisualCache
    from captioning.semantic_prompts import align_prompt_cache
    from captioning.object_reward import make_targets, _AnnotationReader
    cfg = json.loads(Path(config).read_text())
    out, stage = Path(cfg['output']), Path(cfg['stage'])
    if out.exists() or stage.exists():
        raise FileExistsError('Preserve existing output/stage; use a fresh session or names.')
    if cfg['arm'] not in ('A', 'H'):
        raise ValueError('Expected one A or H arm.')
    for key, expected in INPUT_HASHES.items():
        print('Hashing pinned input:', key, flush=True)
        if digest(cfg[key]) != expected:
            raise ValueError('Pinned input differs: ' + key)
    for name, expected in cfg['implementation_sha256'].items():
        if digest(Path(__file__).parent / name) != expected:
            raise ValueError('Runtime source differs: ' + name)
    print('Checking complete hidden-pair cache and file hashes.', flush=True)
    receipt, groups = check_cache(cfg['hidden_cache'])
    for name in ('extract_hidden.py', 'hidden_data.py', 'pilot_data.py', 'captioning/upt_reference.py'):
        if receipt['implementation_sha256'][name] != cfg['implementation_sha256'][name]:
            raise ValueError('Extraction source policy differs from this notebook: ' + name)
    dataset = json.loads(Path(cfg['dataset_json']).read_text())
    if groups != locked_groups(dataset):
        raise ValueError('Cache/COCO cohort mapping differs.')
    rows = groups['train'] + groups['val']
    hidden_files = [h5py.File(Path(cfg['hidden_cache']) / f'features_rank{r}.h5', 'r') for r in range(2)]
    positions = {}
    try:
        for rank, f in enumerate(hidden_files):
            expected = rows[rank::2]
            if not f.attrs.get('complete') or f['image_ids'][:].tolist() != [r['coco_id'] for r in expected]:
                raise ValueError('Hidden HDF5 identity/order differs.')
            if f['dataset_indices'][:].tolist() != [r['image_id'] for r in expected]:
                raise ValueError('Caption-index/COCO-ID mismatch.')
            for index, row in enumerate(expected):
                validate_row(*(f[k][index] for k in ('features', 'valid', 'pairs', 'native_rows', 'labels', 'confidence', 'boxes')))
                positions[row['coco_id']] = (rank, index)
        counts = {iid: int(hidden_files[r]['valid'][i].sum()) for iid, (r, i) in positions.items()}
        chair = Path(cfg['chair'])
        if digest(chair / 'chair.pkl') != '73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':
            raise ValueError('CHAIR cache differs.')
        with (chair / 'chair.pkl').open('rb') as stream:
            annotation = _AnnotationReader(stream).load()
        if any(r['coco_id'] not in annotation.imid_to_objects for r in groups['val']):
            raise ValueError('CHAIR validation coverage incomplete.')
        del annotation
        bundle = torch.load(cfg['legacy_prompt'], map_location='cpu', weights_only=False)
        names = [str(Path('/images') / dataset['images'][r['image_id']]['filepath'] / r['filename']) for r in rows]
        legacy = align_prompt_cache(bundle['data'], names)
        if len(legacy) != len(rows):
            raise ValueError('Legacy prompts missing. Hidden memory does not replace them.')
        length = legacy[names[0]]['tokens'].shape[0]
        required = len(rows) * ((197 * 768 + length * 512) * 4 + SLOTS * WIDTH * 2)
        stage.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(stage.parent).free < required + 2 * 1024**3:
            raise RuntimeError('Need space for FP32 CLIP/legacy subset plus2GiB.')
        out.mkdir(parents=True)
        sources = out / 'source'
        sources.mkdir()
        for name in cfg['implementation_sha256']:
            dest = sources / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(Path(__file__).parent / name, dest)
        targets = make_targets(groups, dataset, chair / 'chair.pkl',
                               json.loads(Path(cfg['detections']).read_text()), 'aliases_v2', 'unique_count_v2')
        save(out / 'train_object_targets.json', targets)
        cache = VisualCache([cfg['visual_cache']], id_key='coco_id')
        try:
            cache.require_ids([r['coco_id'] for r in rows], 'hidden-pair pilot')
            with h5py.File(stage, 'w') as f:
                specs = dict(visual=((len(rows), 197, 768), 'f4'), legacy=((len(rows), length, 512), 'f4'),
                    legacy_mask=((len(rows), length), 'i8'), hidden=((len(rows), SLOTS, WIDTH), 'f2'),
                    valid=((len(rows), SLOTS), '?'))
                for key, (shape, dtype) in specs.items():
                    f.create_dataset(key, shape=shape, dtype=dtype, chunks=(1, *shape[1:]))
                f.create_dataset('image_ids', data=np.asarray([r['coco_id'] for r in rows]))
                order = sorted(range(len(rows)), key=lambda i: cache.index[rows[i]['coco_id']])
                for count, i in enumerate(order, 1):
                    old = legacy[names[i]]
                    if old['tokens'].shape != (length, 512) or old['mask'].shape != (length,):
                        raise ValueError('Inconsistent legacy prompt shape.')
                    tokens, mask = old['tokens'].float().numpy(), old['mask'].long().numpy()
                    if not np.isfinite(tokens).all() or not np.isin(mask, [0, 1]).all():
                        raise ValueError('Nonfinite legacy tokens/invalid mask.')
                    rank, index = positions[rows[i]['coco_id']]
                    for key, value in [('visual', cache.read(rows[i]['coco_id'])), ('legacy', tokens),
                                       ('legacy_mask', mask), ('hidden', hidden_files[rank]['features'][index]),
                                       ('valid', hidden_files[rank]['valid'][index])]:
                        f[key][i] = value
                    if count % 250 == 0 or count == 1:
                        print(f'Staged exact CLIP/legacy + cached pair features: {count}/{len(rows)}', flush=True)
                f.attrs.update(complete=True, cohort_sha256=COHORT_SHA,
                    hidden_receipt_sha256=digest(Path(cfg['hidden_cache']) / 'cache_receipt.json'))
        finally:
            cache.close()
    finally:
        for f in hidden_files:
            f.close()
    excluded = json.loads((Path(__file__).parent / 'excluded_phrase_ids.json').read_text())
    review = human_sample(groups['val'], counts, excluded)
    save(out / 'human_review_cohort.json', dict(scope='Development only; fixed before model outputs; no known phrase IDs',
        policy='Up to16 no-pair and remaining pair images by fixed hash; hash-fill short strata; never choose by captions.',
        rows=[dict(r, pair_count=counts[r['coco_id']]) for r in review]))
    save(out / 'groups.json', groups)
    save(out / 'hidden_cache_receipt.json', receipt)
    cfg.update(groups=groups, cohort_sha256=COHORT_SHA, test_used=False,
        object_targets_sha256=digest(out / 'train_object_targets.json'),
        source_checkpoint_sha256=INPUT_HASHES['checkpoint'], stage_sha256=digest(stage),
        hidden_receipt_sha256=digest(Path(cfg['hidden_cache']) / 'cache_receipt.json'),
        human_review_cohort_sha256=digest(out / 'human_review_cohort.json'),
        conditions=['on'] if cfg['arm'] == 'A' else ['on', 'off', 'pair_swap'],
        checkpoint_name='checkpoint.bin', checkpoint_format='upt_hidden_pair_caption_v1',
        validation_scope='Reused development cohort, not independent confirmatory evidence',
        comparison_limits='H adds pretrained UPT hidden-pair information and parameters; A does not isolate extra-encoder/context/capacity effects.',
        evaluation_policy='Raw greedy, fixed development IDs/IDF, original + end; no reranking or checkpoint selection.',
        loss='CE including EOS +.05SCST after200 updates; reward CIDErD+.5object_count_v2; no UPT-label reward.',
        pair_coverage={s: dict(images=len(part), with_pairs=sum(counts[r['coco_id']] > 0 for r in part),
            count_histogram=dict(Counter(counts[r['coco_id']] for r in part))) for s, part in groups.items()})
    save(out / 'protocol.json', cfg)
    print('PREPARE PASS:', cfg['pair_coverage'], flush=True)


def build_model(cfg, tokenizer):
    from captioning.models import UniversalVisionEncoder, CaptionDecoder, ImageCaptioningModel
    from captioning.checkpoints import load_checkpoint
    from hidden_model import HiddenModel
    from captioning.scst_runtime import model_digest
    global LIVE_MODEL
    base = ImageCaptioningModel(UniversalVisionEncoder(load_backbone=False, visual_adapter='qformer',
                               num_visual_queries=32, qformer_layers=2), CaptionDecoder(len(tokenizer.word2idx)))
    load_checkpoint(base, cfg['checkpoint'], tokenizer)
    model = HiddenModel(base, cfg['arm']).cuda().eval()
    LIVE_MODEL = model
    model.initial_full_sha256 = model_digest(model)
    return model


def make_loader(cfg, dataset, tokenizer, split, rank, world):
    import h5py
    import torch
    from torch.utils.data import Dataset, DataLoader, DistributedSampler
    rows, offset = cfg['groups'][split], 0 if split == 'train' else len(cfg['groups']['train'])
    with h5py.File(cfg['stage']) as f:
        counts = f['valid'][offset:offset + len(rows)].sum(1).tolist()
    buckets = defaultdict(list)
    for i, count in enumerate(counts):
        buckets[count].append(i)
    donors = {i: bucket[(position + 1) % len(bucket)] for bucket in buckets.values() for position, i in enumerate(bucket)}
    class Staged(Dataset):
        def __init__(self):
            self.f = h5py.File(cfg['stage'], 'r')
            if not self.f.attrs['complete'] or self.f.attrs['cohort_sha256'] != COHORT_SHA:
                raise ValueError('Incomplete/different caption stage.')
        def __len__(self):
            return len(rows)
        def __getitem__(self, index):
            i, donor = offset + index, offset + donors[index]
            row = rows[index]
            if int(self.f['image_ids'][i]) != row['coco_id']:
                raise ValueError('HDF5 caption row identity differs.')
            result = {k: torch.from_numpy(self.f[k][i]) for k in ('visual', 'legacy', 'legacy_mask', 'hidden', 'valid')}
            result['hidden'] = result['hidden'].float()
            result.update(swap_hidden=torch.from_numpy(self.f['hidden'][donor]).float(),
                          swap_valid=torch.from_numpy(self.f['valid'][donor]))
            encoded = [tokenizer.encode(s['raw']) for s in dataset['images'][row['image_id']]['sentences'][:5]]
            if len(encoded) != 5:
                raise ValueError('Expected five natural caption references.')
            result.update(captions=torch.stack([e[0] for e in encoded]),
                          caption_mask=torch.stack([e[1] for e in encoded]), image_id=torch.tensor(row['image_id']),
                          swap_donor_image_id=torch.tensor(rows[donors[index]]['image_id']))
            return result
    staged = Staged()
    if split == 'train':
        sampler = DistributedSampler(staged, num_replicas=world, rank=rank, shuffle=True, seed=42, drop_last=False)
        return DataLoader(staged, batch_size=cfg['global_batch_size'] // world, sampler=sampler,
                          num_workers=0, pin_memory=True, drop_last=False)
    subset = torch.utils.data.Subset(staged, list(range(rank, len(rows), world)))
    return DataLoader(subset, batch_size=cfg['eval_batch_size'], num_workers=0, pin_memory=True)


def predict(model, loader, tokenizer, intervention):
    import torch
    from captioning.scst import rollout, decode
    records, diagnostics = [], []
    with torch.no_grad():
        for index, batch in enumerate(loader):
            batch = shared.cuda_batch(batch)
            memory, padding = model.memory(batch, intervention)
            sequence, active = rollout(model, memory, padding, tokenizer, sample=False)
            captions = decode(sequence, tokenizer)
            for i, (image_id, caption) in enumerate(zip(batch['image_id'].cpu().tolist(), captions)):
                records.append(dict(image_id=image_id, caption=caption))
                diagnostics.append(dict(image_id=image_id, valid_pairs=int(batch['valid'][i].sum()),
                    generated_eos=bool(sequence[i].eq(tokenizer.eos_idx).any()),
                    truncated=not bool(sequence[i].eq(tokenizer.eos_idx).any()),
                    swap_donor_image_id=int(batch['swap_donor_image_id'][i]),
                    caption_word_count=len(caption.split())))
            if index % 10 == 0:
                print(f'Evaluation {intervention}: {len(records)} local images', flush=True)
    return records, diagnostics


def worker(protocol):
    from captioning.scst_runtime import model_digest
    cfg = json.loads(Path(protocol).read_text())
    for name, expected in cfg['implementation_sha256'].items():
        if digest(Path(__file__).parent / name) != expected:
            raise ValueError('Worker source differs: ' + name)
    print('Checking staged tensors before DDP:', cfg['arm'], flush=True)
    if digest(cfg['checkpoint']) != INPUT_HASHES['checkpoint']:
        raise ValueError('Baseline checkpoint changed after preparation.')
    if digest(cfg['stage']) != cfg['stage_sha256']:
        raise ValueError('Stage bytes changed.')
    shared.build_model, shared.make_loader, shared.predict = build_model, make_loader, predict
    shared.worker(protocol)
    rank = int(os.environ['RANK'])
    out = Path(cfg['output'])
    path = out / f'worker_complete_rank{rank}.json'
    done = json.loads(path.read_text())
    done.update(initial_full_sha256=LIVE_MODEL.initial_full_sha256,
        final_full_sha256=model_digest(LIVE_MODEL), final_base_sha256=model_digest(LIVE_MODEL.base))
    save(path, done)


def evaluate(protocol):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    from pycocoevalcap.cider.cider import Cider
    cfg = json.loads(Path(protocol).read_text())
    out = Path(cfg['output'])
    done = [json.loads((out / f'worker_complete_rank{r}.json').read_text()) for r in (0, 1)]
    if any(d['status'] != 'COMPLETE' or d['updates'] != 1250 for d in done):
        raise ValueError('Both ranks must complete exactly1250 updates.')
    for key in ('initial_base_sha256', 'initial_full_sha256', 'final_full_sha256', 'final_base_sha256'):
        if done[0][key] != done[1][key]:
            raise ValueError('Rank state identity differs: ' + key)
    dataset = json.loads(Path(cfg['dataset_json']).read_text())
    expected = {r['image_id']: r for r in cfg['groups']['val']}
    tokenizer = PTBTokenizer()
    refs = tokenizer.tokenize({iid: [{'caption': s['raw']} for s in dataset['images'][iid]['sentences'][:5]] for iid in sorted(expected)})
    metrics, all_predictions = {}, {}
    for condition in ['origin'] + cfg['conditions']:
        rows = sum((json.loads((out / f'{condition}_rank{r}.json').read_text()) for r in (0, 1)), [])
        rows.sort(key=lambda r: r['image_id'])
        if len(rows) != 1000 or {r['image_id'] for r in rows} != set(expected):
            raise ValueError('Missing/duplicate validation predictions.')
        save(out / f'{condition}_predictions.json', rows)
        all_predictions[condition] = {r['image_id']: r['caption'] for r in rows}
        if condition == 'pair_swap':
            continue
        hypothesis = tokenizer.tokenize({r['image_id']: [{'caption': r['caption']}] for r in rows})
        score, per_image = Cider().compute_score(refs, hypothesis)
        save(out / f'{condition}_per_image_cider.json', [dict(image_id=r['image_id'], CIDEr=float(v)) for r, v in zip(rows, per_image)])
        chair_input, chair_output = out / f'{condition}_chair_input.json', out / f'{condition}_chair_output.json'
        save(chair_input, [dict(image_id=expected[r['image_id']]['coco_id'], caption=r['caption']) for r in rows])
        subprocess.run([sys.executable, '-u', str(Path(cfg['chair']) / 'chair.py'), '--cap_file', str(chair_input),
            '--cache', str(Path(cfg['chair']) / 'chair.pkl'), '--save_path', str(chair_output)], cwd=cfg['chair'], check=True)
        chair = json.loads(chair_output.read_text())
        if len(chair['sentences']) != 1000 or {s['image_id'] for s in chair['sentences']} != {r['coco_id'] for r in expected.values()}:
            raise ValueError('CHAIR IDs/count differ.')
        sentences = chair['sentences']
        hallucinated = sum(len(s['mscoco_hallucinated_words']) for s in sentences)
        mentions = sum(len(s['mscoco_generated_words']) for s in sentences)
        correct = sum(len(set(s['mscoco_generated_words']) & set(s['mscoco_gt_words'])) for s in sentences)
        total = sum(len(set(s['mscoco_gt_words'])) for s in sentences)
        # Reaggregate numerators/denominators, rather than hiding denominator changes.
        reaggregated = dict(CHAIRs=sum(bool(s['mscoco_hallucinated_words']) for s in sentences) / 1000,
            CHAIRi=hallucinated / mentions if mentions else 0., Recall=correct / total if total else 0.)
        if any(abs(reaggregated[k] - chair['overall_metrics'][k]) > 1e-12 for k in reaggregated):
            raise ValueError('CHAIR aggregation differs from evaluator output.')
        metrics[condition] = dict(CIDEr=float(score), chair=chair['overall_metrics'],
            counts=dict(images=1000, hallucinated_captions=sum(bool(s['mscoco_hallucinated_words']) for s in sentences),
                hallucinated_mentions=hallucinated, object_mentions=mentions,
                recalled_unique_objects=correct, reference_unique_objects=total),
            mean_caption_words=float(np.mean([len(r['caption'].split()) for r in rows])),
            mean_CHAIR_tokenized_words=float(np.mean([len(s['words']) for s in sentences])))
    summary = dict(status='CAPTION_PILOT_COMPLETE_WAIT_MATCHED_COMPARISON', arm=cfg['arm'], metrics=metrics,
        sensitivity={c: dict(changed_captions=sum(all_predictions[c][i] != all_predictions['on'][i] for i in expected))
                     for c in cfg['conditions'] if c != 'on'},
        cohort_sha256=COHORT_SHA, human_review_cohort_sha256=cfg['human_review_cohort_sha256'],
        hidden_receipt_sha256=cfg['hidden_receipt_sha256'], protocol_sha256=digest(protocol),
        checkpoint_sha256=digest(out / 'checkpoint.bin'), initial_base_sha256=done[0]['initial_base_sha256'],
        final_full_sha256=done[0]['final_full_sha256'], test_used=False,
        evaluation_reference_tokens_sha256=identity(refs),
        selection='End only; no promotion before matched comparison and human factual review.',
        limits=[cfg['validation_scope'], cfg['comparison_limits'],
                'CHAIR is object hallucination, not relation truth. No verb text or UPT verb-score reward.',
                'Pair swaps are sensitivity only, not verified false relation labels. One seed does not measure seed variance.'])
    save(out / 'run_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def archive(protocol):
    cfg = json.loads(Path(protocol).read_text())
    out = Path(cfg['output'])
    with zipfile.ZipFile(str(out) + '_review.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.jsonl', '.log', '.py'):
                z.write(path, path.relative_to(out))
    print('SEND:', str(out) + '_review.zip', '\nKEEP:', out / 'checkpoint.bin', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['prepare', 'worker', 'evaluate', 'archive'])
    p.add_argument('--config', required=True)
    a = p.parse_args()
    globals()[a.mode](a.config)
