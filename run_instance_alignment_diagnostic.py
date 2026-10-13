"""Two independent frozen-CLIP extractors, followed by a small supervised head on CPU."""
import argparse
import json
from pathlib import Path
import torch
from torch.nn import functional as F
from captioning.supervised_relations import RELATIONS, contrast_edges
from captioning.instance_contrast_alignment import InstanceContrastAlignment, supervised_alignment_loss


def extract(args, rows):
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    from captioning.relation_candidates import union_crop_box
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for feature extraction.')
    torch.set_num_threads(2)
    model = CLIPModel.from_pretrained(args.model, local_files_only=True).to('cuda').eval()
    processor = CLIPProcessor.from_pretrained(args.model, local_files_only=True)
    records = []
    for index, row in enumerate(rows):
        if index % args.shards != args.shard:
            continue
        path = Path(args.images) / row['filepath'] / row['filename']
        with Image.open(path) as source:
            image = source.convert('RGB')
            sb, ob = row['subject_box'], row['object_box']
            views = [image.crop(union_crop_box(sb, sb, *image.size, padding=0)),
                     image.crop(union_crop_box(ob, ob, *image.size, padding=0)),
                     image.crop(union_crop_box(sb, ob, *image.size, padding=.1))]
            # Resize complete crops before CLIP processing so center cropping cannot cut off B.
            views = [view.resize((224, 224), Image.Resampling.BICUBIC) for view in views]
            obj = row['object']
            article = 'an' if obj[:1].lower() in 'aeiou' else 'a'
            texts = [f'a photo of a person {r.replace("_", " ")} {article} {obj}' for r in RELATIONS]
            with torch.inference_mode():
                visual = model.get_image_features(**processor(images=views, return_tensors='pt').to('cuda'))
                text = model.get_text_features(**processor(text=texts, padding=True, return_tensors='pt').to('cuda'))
            records.append({'index': index, 'views': F.normalize(visual.float(), dim=-1).cpu(),
                            'text': F.normalize(text.float(), dim=-1).cpu()})
        if len(records) % 10 == 0:
            print(f'GPU worker {args.shard}: {len(records)} pairs extracted', flush=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(records, output / f'features_{args.shard}.pt')
    print(f'GPU worker {args.shard} complete: {len(records)} pairs', flush=True)


def binary_average_precision(truth, scores):
    """Non-interpolated AP, grouping tied scores before counting a recall increment."""
    positives = sum(truth)
    if not positives or positives == len(truth):
        return None
    ranked = sorted(zip(scores, truth), reverse=True)
    tp = seen = 0
    total = 0.0
    index = 0
    while index < len(ranked):
        end = index
        added = 0
        while end < len(ranked) and ranked[end][0] == ranked[index][0]:
            added += ranked[end][1]
            end += 1
        tp += added
        seen += end - index
        total += (added / positives) * (tp / seen)
        index = end
    return total


def metrics(scores, labels, thresholds):
    result = {}
    for c, name in enumerate(RELATIONS):
        known = labels[:, c] >= 0
        y = labels[known, c].numpy()
        s = scores[known, c].numpy()
        if not len(y):
            result[name] = {'known': 0, 'AP': None, 'precision': None, 'recall': None}
            continue
        predictions = s >= thresholds[c]
        tp = int(((y == 1) & predictions).sum())
        result[name] = {'known': len(y), 'positive': int((y == 1).sum()),
                        'AP': binary_average_precision(y.tolist(), s.tolist()),
                        'precision': tp / int(predictions.sum()) if predictions.sum() else None,
                        'recall': tp / int(y.sum()) if y.sum() else None,
                        'accepted_known': int(predictions.sum())}
    aps = [x['AP'] for x in result.values() if x['AP'] is not None]
    return {'per_relation': result, 'macro_AP_on_known_labels': sum(aps)/len(aps) if aps else None}


def fit_thresholds(scores, labels):
    # Training-only diagnostic thresholds. Require >=5 known accepts at >=90% train precision.
    thresholds = []
    for c in range(len(RELATIONS)):
        known = labels[:, c] >= 0
        s, y = scores[known, c], labels[known, c]
        threshold = 1.01  # Cosines cannot reach this: abstain for unsupported classes.
        for value in sorted(set(s.tolist())):
            accepted = s >= value
            if accepted.sum() >= 5 and (y[accepted] == 1).float().mean() >= .9:
                threshold = value
                break
        thresholds.append(threshold)
    return thresholds


def train(args, rows, train_rows, val_rows):
    from collections import Counter
    torch.set_num_threads(2)
    torch.manual_seed(42)
    records = []
    for shard in range(args.shards):
        records += torch.load(Path(args.output) / f'features_{shard}.pt', weights_only=True)
    records.sort(key=lambda r: r['index'])
    if [r['index'] for r in records] != list(range(len(rows))):
        raise ValueError('Missing/duplicate features.')
    views = torch.stack([r['views'] for r in records])
    text = torch.stack([r['text'] for r in records])
    labels = torch.tensor([r['labels'] for r in rows])
    n = len(train_rows)
    edges = contrast_edges(train_rows)
    edge_tensor = torch.tensor([e[:4] for e in edges], dtype=torch.long).reshape(-1, 4)
    model = InstanceContrastAlignment()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    history = []
    # Fixed full-batch diagnostic preserves same-image wrong-actor edges.
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        scores = model(views[:n], text[:n])
        loss, bce, ranking = supervised_alignment_loss(scores, labels[:n], edge_tensor)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite diagnostic loss.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        history.append({'epoch': epoch+1, 'total': loss.item(), 'bce': bce.item(), 'ranking': ranking.item()})
        print('Head diagnostic:', history[-1], flush=True)
    model.eval()
    with torch.inference_mode():
        scores = model(views, text)
        cosine = torch.einsum('bd,bkd->bk', views[:, 2], text)
        # Same supervised architecture without ranking, for a controlled comparison.
    torch.manual_seed(42)
    plain = InstanceContrastAlignment()
    plain_optimizer = torch.optim.AdamW(plain.parameters(), lr=1e-3, weight_decay=.01)
    for _ in range(args.epochs):
        plain_optimizer.zero_grad()
        plain_loss, _, _ = supervised_alignment_loss(
            plain(views[:n], text[:n]), labels[:n], torch.empty((0, 4), dtype=torch.long))
        plain_loss.backward()
        torch.nn.utils.clip_grad_norm_(plain.parameters(), 1)
        plain_optimizer.step()
    with torch.inference_mode():
        plain_scores = plain(views, text)
    summary = {'variant': 'instance_contrast_alignment_diagnostic_v1',
               'train_pairs': n, 'val_pairs': len(val_rows), 'epochs': args.epochs,
               'contrast_edges': dict(Counter(e[-1] for e in edges)), 'history': history,
               'threshold_protocol': 'Training-only 90% empirical precision, >=5 known accepts; not calibrated',
               'notes': 'GT boxes and diagnostic subsampling; metrics on known labels only, '
                        'not official V-COCO role AP or caption performance. Unknown claims are not negatives.'}
    predictions = []
    for name, values in [('clip_union_cosine', cosine), ('supervised_without_contrast', plain_scores),
                         ('supervised_with_contrast', scores)]:
        thresholds = fit_thresholds(values[:n], labels[:n])
        summary[name] = {'thresholds': thresholds,
                         'validation': metrics(values[n:], labels[n:], thresholds)}
    for i, row in enumerate(val_rows, n):
        predictions.append({**row, 'scores': scores[i].tolist(),
                            'cosine_baseline': cosine[i].tolist(),
                            'without_contrast': plain_scores[i].tolist()})
    output = Path(args.output)
    torch.save({'model': model.state_dict(), 'relations': RELATIONS,
                'thresholds': summary['supervised_with_contrast']['thresholds'],
                'scope': 'diagnostic_gt_boxes'}, output / 'alignment_head.pt')
    (output / 'diagnostic_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'validation_predictions.json').write_text(json.dumps(predictions, indent=2), encoding='utf-8')
    print('INSTANCE ALIGNMENT DIAGNOSTIC COMPLETE\n' + json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--images', default='')
    parser.add_argument('--model', default='')
    parser.add_argument('--output', required=True)
    parser.add_argument('--shards', type=int, default=2)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards or args.epochs < 1:
        raise ValueError('Invalid diagnostic configuration.')
    data = Path(args.data)
    train_rows = json.loads((data / 'train_pairs.json').read_text())
    val_rows = json.loads((data / 'val_pairs.json').read_text())
    if not train_rows or not val_rows:
        raise ValueError('Empty split.')
    if {r['image_id'] for r in train_rows} & {r['image_id'] for r in val_rows}:
        raise ValueError('Image leakage between train and val.')
    if any(r['caption_split'] not in {'train', 'restval'} for r in train_rows + val_rows):
        raise ValueError('Caption heldout image in diagnostic data.')
    rows = train_rows + val_rows
    if args.train:
        train(args, rows, train_rows, val_rows)
    else:
        extract(args, rows)


if __name__ == '__main__':
    main()
