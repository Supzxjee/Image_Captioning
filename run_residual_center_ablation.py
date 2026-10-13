"""Fixed 2x2 architecture/centering development ablation on saved V2 features."""
import argparse
import hashlib
import io
import json
import statistics
import time
import zipfile
from pathlib import Path
import torch
from captioning.instance_contrast_alignment import (InstanceContrastAlignment,
    ResidualInstanceAlignment, supervised_alignment_loss)
from captioning.supervised_relations import contrast_edges
from run_instance_alignment_diagnostic import metrics, fit_thresholds, edge_metrics


def load_source(source):
    with zipfile.ZipFile(source) as archive:
        def read(base):
            matches = [n for n in archive.namelist() if n.endswith(base)]
            if len(matches) != 1:
                raise ValueError(f'Expected one {base}.')
            return archive.read(matches[0])
        summary = json.loads(read('diagnostic_summary.json'))
        provenance = json.loads(read('data/pair_summary.json'))
        if summary['variant'] != 'instance_contrast_alignment_diagnostic_v2':
            raise ValueError('Expected cached V2 source.')
        splits = []
        for split in ('train', 'val'):
            raw = read(f'data/{split}_pairs.json')
            if hashlib.sha256(raw).hexdigest() != provenance[split]['pair_json_sha256']:
                raise ValueError('Pair hash mismatch.')
            splits.append(json.loads(raw))
        records = []
        for shard in range(2):
            records += torch.load(io.BytesIO(read(f'features_{shard}.pt')), weights_only=True, map_location='cpu')
        predictions = json.loads(read('validation_predictions.json'))
    train, val = splits
    rows = train + val
    if {r['image_id'] for r in train} & {r['image_id'] for r in val}:
        raise ValueError('Image overlap.')
    if any(r['caption_split'] not in {'train', 'restval'} for r in rows):
        raise ValueError('Caption heldout included.')
    if any(len(r['labels']) != 6 or any(v not in (-1, 0, 1) for v in r['labels']) for r in rows):
        raise ValueError('Label schema invalid.')
    records.sort(key=lambda r: r['index'])
    if [r['index'] for r in records] != list(range(len(rows))):
        raise ValueError('Feature coverage invalid.')
    views, text = torch.stack([r['views'] for r in records]), torch.stack([r['text'] for r in records])
    if views.shape[1:] != (3, 512) or text.shape[1:] != (6, 512):
        raise ValueError('Feature shape invalid.')
    if not torch.isfinite(views).all() or not torch.isfinite(text).all():
        raise ValueError('Nonfinite features.')
    if not torch.allclose(views.norm(dim=-1), torch.ones(views.shape[:2]), atol=1e-3):
        raise ValueError('Image features not normalized.')
    if [(r['subject_id'], r['object_id']) for r in predictions] != [(r['subject_id'], r['object_id']) for r in val]:
        raise ValueError('Prediction instance order invalid.')
    base = torch.einsum('bd,bkd->bk', views[:, 2], torch.nn.functional.normalize(text, dim=-1))
    if not torch.allclose(base[len(train):], torch.tensor([r['cosine_baseline'] for r in predictions]), atol=1e-6):
        raise ValueError('CLIP baseline mismatch.')
    return train, val, views, text, base


def flips_vs_base(scores, base, rows):
    result = {'actor': {}, 'predicate': {}}
    for i, r, j, c, kind in contrast_edges(rows):
        old, new = bool(base[i, r] > base[j, c]), bool(scores[i, r] > scores[j, c])
        key = 'both_correct' if old and new else 'both_wrong' if not old and not new else 'fixed' if new else 'regressed'
        result[kind][key] = result[kind].get(key, 0) + 1
    return result


def train_variant(architecture, centered, seed, views, text, labels, n, edges, kinds,
                  center, epochs, scale):
    torch.manual_seed(seed)
    model = (ResidualInstanceAlignment(scale=scale) if architecture == 'residual'
             else InstanceContrastAlignment())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(views[:n], text[:n])
        scores = output[0] if architecture == 'residual' else output
        loss, bce, ranking = supervised_alignment_loss(
            scores - (center if centered else 0), labels[:n], edges, edge_kinds=kinds)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        history.append({'epoch': epoch+1, 'total': float(loss.detach()),
                        'bce': float(bce.detach()), 'ranking': float(ranking.detach())})
    with torch.inference_mode():
        output = model(views, text)
        scores = output[0] if architecture == 'residual' else output
    return model, scores, history, output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 7, 123])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--scale', type=float, default=.1)
    args = parser.parse_args()
    if args.epochs < 1 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError('Invalid epochs/seeds.')
    torch.set_num_threads(2)
    train, val, views, text, base = load_source(args.source)
    n = len(train)
    labels = torch.tensor([r['labels'] for r in train + val])
    known = labels[:n] >= 0
    prior = labels[:n][known].float().mean().clamp(.01, .99)
    center = base[:n][known].mean() - .1 * torch.logit(prior)
    pairs = contrast_edges(train)
    edges = torch.tensor([e[:4] for e in pairs], dtype=torch.long).reshape(-1, 4)
    kinds = torch.tensor([1 if e[-1] == 'actor' else 0 for e in pairs])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    plan = {'seeds': args.seeds, 'epochs': args.epochs, 'scale': args.scale,
            'variants': ['projected_raw', 'projected_centered', 'residual_raw', 'residual_centered'],
            'selection': 'Fixed final epoch, no dev hyperparameter or seed selection',
            'train_only_center': float(center), 'scope': 'Same repeatedly inspected V2 development set'}
    (output / 'run_plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    runs = []
    for seed in args.seeds:
        for architecture in ('projected', 'residual'):
            for centered in (False, True):
                name = architecture + ('_centered' if centered else '_raw')
                started = time.monotonic()
                print(f'Start seed={seed} {name}', flush=True)
                model, scores, history, model_output = train_variant(
                    architecture, centered, seed, views, text, labels, n, edges, kinds,
                    center, args.epochs, args.scale)
                thresholds = fit_thresholds(scores[:n], labels[:n])
                thresholds = [2.0 if t == 1.01 else t for t in thresholds]
                run = {'seed': seed, 'variant': name, 'thresholds': thresholds,
                       'validation': metrics(scores[n:], labels[n:], thresholds),
                       'validation_edges': edge_metrics(scores[n:], val),
                       'flips_vs_clip': flips_vs_base(scores[n:], base[n:], val), 'history': history}
                if architecture == 'residual':
                    run['mean_abs_dev_correction'] = float(model_output[2][n:].abs().mean())
                runs.append(run)
                (output / f'run_{seed}_{name}.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
                torch.save({'model': model.state_dict(), 'variant': name, 'scale': args.scale,
                            'logit_center': float(center) if centered else 0}, output / f'head_{seed}_{name}.pt')
                (output / f'predictions_{seed}_{name}.json').write_text(json.dumps([
                    {**row, 'scores': scores[n+i].tolist()} for i, row in enumerate(val)], indent=2), encoding='utf-8')
                print(f'Done {name} seed{seed}: AP={run["validation"]["macro_AP_on_known_labels"]:.4f}; '
                      f'actor={run["validation_edges"]["actor"].get("wins",0)}/'
                      f'{run["validation_edges"]["actor"]["edges"]}; '
                      f'{time.monotonic()-started:.1f}s', flush=True)
    aggregate = {}
    for name in plan['variants']:
        selected = [r for r in runs if r['variant'] == name]
        ap = [r['validation']['macro_AP_on_known_labels'] for r in selected]
        actors = [r['validation_edges']['actor']['wins'] for r in selected]
        aggregate[name] = {'macro_AP_mean': statistics.mean(ap),
                           'macro_AP_seed_stdev': statistics.stdev(ap) if len(ap)>1 else None,
                           'AP_by_seed': ap, 'actor_wins_by_seed': actors,
                           'actor_wins_mean': statistics.mean(actors)}
    summary = {'plan': plan, 'train_pairs': n, 'dev_pairs': len(val),
               'clip_baseline': {'validation_edges': edge_metrics(base[n:], val)},
               'aggregate': aggregate, 'runs': runs,
               'notes': 'Seed variation is not image-sampling uncertainty. No new heldout images. '
                        'Center changes training objective; alone a shared inference shift does not change ordering.'}
    digest = hashlib.sha256()
    with Path(args.source).open('rb') as handle:
        while block := handle.read(4*1024**2):
            digest.update(block)
    summary['source_zip_sha256'] = digest.hexdigest()
    (output / 'ablation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == '__main__':
    main()
