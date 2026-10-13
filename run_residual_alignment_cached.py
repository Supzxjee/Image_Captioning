"""CPU diagnostic from an existing V2 review ZIP. No COCO/model downloads."""
import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path
import torch
from captioning.supervised_relations import RELATIONS, contrast_edges
from captioning.instance_contrast_alignment import ResidualInstanceAlignment, PairOnlyResidualAlignment, supervised_alignment_loss
from run_instance_alignment_diagnostic import metrics, fit_thresholds, edge_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--scale', type=float, default=.1)
    parser.add_argument('--pair-only', action='store_true', help='Correction uses A/B only; frozen union CLIP prior retained.')
    args = parser.parse_args()
    if args.epochs < 1:
        raise ValueError('epochs must be positive.')
    torch.set_num_threads(2)
    torch.manual_seed(42)
    with zipfile.ZipFile(args.source) as archive:
        def read(base):
            matches = [n for n in archive.namelist() if n.endswith(base)]
            if len(matches) != 1:
                raise ValueError(f'Expected exactly one {base}.')
            return archive.read(matches[0])
        train_bytes, val_bytes = read('data/train_pairs.json'), read('data/val_pairs.json')
        train, val = json.loads(train_bytes), json.loads(val_bytes)
        provenance = json.loads(read('data/pair_summary.json'))
        original = json.loads(read('diagnostic_summary.json'))
        predictions = json.loads(read('validation_predictions.json'))
        if original['variant'] != 'instance_contrast_alignment_diagnostic_v2':
            raise ValueError('Require the V2 diagnostic archive.')
        for split, content in [('train', train_bytes), ('val', val_bytes)]:
            if hashlib.sha256(content).hexdigest() != provenance[split]['pair_json_sha256']:
                raise ValueError('Pair metadata hash mismatch.')
        records = []
        for shard in range(2):
            records += torch.load(io.BytesIO(read(f'features_{shard}.pt')), map_location='cpu', weights_only=True)
    rows = train + val
    if {r['image_id'] for r in train} & {r['image_id'] for r in val}:
        raise ValueError('Image leakage.')
    if any(r['caption_split'] not in {'train', 'restval'} for r in rows):
        raise ValueError('Caption heldout included.')
    records.sort(key=lambda r: r['index'])
    if [r['index'] for r in records] != list(range(len(rows))):
        raise ValueError('Feature coverage invalid.')
    if [(r['subject_id'], r['object_id']) for r in predictions] != [(r['subject_id'], r['object_id']) for r in val]:
        raise ValueError('Prediction order invalid.')
    views, text = torch.stack([r['views'] for r in records]), torch.stack([r['text'] for r in records])
    if views.shape[1:] != (3, 512) or text.shape[1:] != (6, 512):
        raise ValueError('Unexpected frozen features.')
    if not torch.isfinite(views).all() or not torch.isfinite(text).all():
        raise ValueError('Nonfinite features.')
    if not torch.allclose(views.norm(dim=-1), torch.ones(views.shape[:2]), atol=1e-3):
        raise ValueError('Expected normalized image features.')
    labels = torch.tensor([r['labels'] for r in rows])
    n = len(train)
    edges = contrast_edges(train)
    tensor_edges = torch.tensor([e[:4] for e in edges], dtype=torch.long).reshape(-1, 4)
    kinds = torch.tensor([1 if e[-1] == 'actor' else 0 for e in edges])
    model = (PairOnlyResidualAlignment if args.pair_only else ResidualInstanceAlignment)(scale=args.scale)
    with torch.inference_mode():
        initial, base, _ = model(views, text)
    if not torch.equal(initial, base):
        raise ValueError('Residual must start at CLIP exactly.')
    stored_clip = torch.tensor([r['cosine_baseline'] for r in predictions])
    if not torch.allclose(base[n:], stored_clip, atol=1e-6):
        raise ValueError('Cached CLIP scores differ from recomputed scores.')
    # A single train-only centering constant fixes raw cosine/logit offset without
    # changing any action/actor ordering. This is not fitted from dev labels.
    known = labels[:n] >= 0
    prior = labels[:n][known].float().mean().clamp(.01, .99)
    center = base[:n][known].mean() - .1 * torch.logit(prior)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    history = []
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        scores, _, delta = model(views[:n], text[:n])
        loss, bce, ranking = supervised_alignment_loss(scores - center, labels[:n], tensor_edges,
                                                       edge_kinds=kinds)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite loss.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        history.append({'epoch': epoch+1, 'total': float(loss.detach()), 'bce': float(bce.detach()),
                        'ranking': float(ranking.detach()), 'mean_abs_correction': float(delta.detach().abs().mean())})
        print(history[-1], flush=True)
    with torch.inference_mode():
        scores, base, delta = model(views, text)
    summary = {'variant': 'clip_prior_pair_only_residual_v1' if args.pair_only else 'clip_prior_bounded_residual_v1', 'seed': 42, 'epochs': args.epochs,
               'scale': args.scale, 'train_pairs': n, 'dev_pairs': len(val),
               'source_code_commit': original.get('code_commit'), 'history': history,
               'train_only_logit_center': float(center),
               'mean_abs_dev_correction': float(delta[n:].abs().mean()),
               'max_abs_dev_correction': float(delta[n:].abs().max()),
               'initial_exact_clip': True, 'baselines_from_same_cached_v2': original,
               'notes': 'Same repeatedly inspected development set, GT boxes/classes; no heldout/novelty/caption claim. '
                        'Shared train-only logit centering is used for BCE, raw scores for ranking and AP. '
                        'No dev tuning or epoch selection in this run; fixed scale0.1 and20 epochs.'}
    thresholds = fit_thresholds(scores[:n], labels[:n])
    thresholds = [2.0 if t == 1.01 else t for t in thresholds]  # Cosine+delta can exceed1.01.
    summary['residual'] = {'thresholds': thresholds, 'validation': metrics(scores[n:], labels[n:], thresholds),
                            'validation_edges': edge_metrics(scores[n:], val)}
    flips = {'actor': {}, 'predicate': {}}
    for kind in flips:
        for i, r, j, c, edge_kind in contrast_edges(val):
            if kind != edge_kind:
                continue
            old = bool(base[n+i, r] > base[n+j, c])
            new = bool(scores[n+i, r] > scores[n+j, c])
            key = 'both_correct' if old and new else 'both_wrong' if not old and not new else 'fixed' if new else 'regressed'
            flips[kind][key] = flips[kind].get(key, 0) + 1
    summary['flips_vs_clip'] = flips
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with Path(args.source).open('rb') as handle:
        while block := handle.read(4*1024**2):
            digest.update(block)
    summary['source_zip_sha256'] = digest.hexdigest()
    (output / 'residual_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'validation_predictions.json').write_text(json.dumps([
        {**row, 'scores': scores[n+i].tolist(), 'clip': base[n+i].tolist(),
         'correction': delta[n+i].tolist()} for i, row in enumerate(val)], indent=2), encoding='utf-8')
    torch.save({'model': model.state_dict(), 'architecture': 'pair_only' if args.pair_only else 'union_conditioned', 'scale': args.scale, 'relations': RELATIONS,
                'thresholds': thresholds, 'logit_center': float(center)}, output / 'residual_head.pt')
    print('RESIDUAL RESULT', json.dumps(summary['residual'], indent=2), flush=True)
    print('FLIPS', json.dumps(flips), flush=True)


if __name__ == '__main__':
    main()
