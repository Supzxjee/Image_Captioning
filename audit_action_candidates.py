"""Audit competing actions on fixed instance pairs; do not replace production cache."""
import argparse
import csv
import html
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageOps
from captioning.detection_audit import dataset_records
from captioning.grounded_relations import ACTION_PREDICATES
from captioning.relation_candidates import candidate_phrases, select_candidate, union_crop_box


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grounded-cache', required=True)
    parser.add_argument('--dataset-json', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--priority-filenames', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pair-count', type=int, default=100)
    parser.add_argument('--model', default='openai/clip-vit-base-patch16')
    parser.add_argument('--min-gap', type=float, default=.01)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.pair_count < 1:
        parser.error('pair-count must be positive.')
    records = dataset_records(args.dataset_json)
    priority = set(Path(args.priority_filenames).read_text(encoding='utf-8').splitlines())
    payload = torch.load(args.grounded_cache, map_location='cpu', weights_only=False)
    pools, selected = [], []
    for filename, entry in payload['data'].items():
        filename = Path(filename).name
        record = records.get(filename)
        if not record or record['split'] != 'val':
            continue
        for slot, metadata in enumerate(entry.get('relations', [])):
            if metadata.get('relation') not in ACTION_PREDICATES or metadata.get('subject') != 'person':
                continue
            if not bool(entry['mask'][slot]):
                continue
            pair = {'filename': filename, 'slot': slot, 'old_predicate': metadata['relation'],
                    'subject': metadata['subject'], 'object': metadata['object'],
                    'subject_box': entry['subject_boxes'][slot].tolist(),
                    'object_box': entry['object_boxes'][slot].tolist(),
                    'confidence': float(entry['confidences'][slot]),
                    'image_path': str(Path(args.images) / record.get('filepath', '') / filename)}
            (selected if filename in priority else pools).append(pair)
    del payload
    selected.sort(key=lambda x: (x['filename'], x['slot']))
    random.Random(args.seed).shuffle(pools)
    selected = (selected + pools)[:args.pair_count]
    if len(selected) != args.pair_count:
        raise ValueError(f'Only {len(selected)} eligible action pairs in validation.')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    print(f'Selected {len(selected)} person-object pairs; loading CLIP: {args.model}', flush=True)
    from transformers import CLIPModel, CLIPProcessor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CLIPModel.from_pretrained(args.model).float().to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.model)
    results, cards = [], []
    with torch.inference_mode():
        for index, pair in enumerate(selected):
            candidates = candidate_phrases(pair['subject'], pair['object'])
            with Image.open(pair['image_path']) as original:
                full = original.convert('RGB')
                crop_box = union_crop_box(pair['subject_box'], pair['object_box'], *full.size)
                crop = full.crop(crop_box)
                # Letterbox to square so CLIP center crop cannot remove either instance.
                views = [ImageOps.pad(view, (224, 224), color=(128, 128, 128)) for view in (crop, full)]
                inputs = processor(text=[c['text'] for c in candidates], images=views,
                                   return_tensors='pt', padding=True, truncation=True).to(device)
                values = model(**inputs)
                scores = (values.image_embeds @ values.text_embeds.T).cpu().tolist()
                decision = select_candidate(candidates, scores[0], scores[1], args.min_gap)
                ranking = [{**candidate, 'crop_cosine': scores[0][i], 'full_cosine': scores[1][i]}
                           for i, candidate in enumerate(candidates)]
                result = {**pair, **decision, 'candidates': ranking}
                results.append(result)
                thumbnail = full.copy()
                thumbnail.thumbnail((640, 480))
                draw = ImageDraw.Draw(thumbnail)
                for key, color in [('subject_box', 'red'), ('object_box', 'blue')]:
                    box = pair[key]
                    draw.rectangle(tuple(v * (thumbnail.width if j % 2 == 0 else thumbnail.height)
                                         for j, v in enumerate(box)), outline=color, width=2)
                thumbnail.save(output / f'{index:03d}_full.jpg')
                crop.thumbnail((640, 480))
                crop.save(output / f'{index:03d}_crop.jpg')
            (output / 'candidate_records.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
            esc = lambda v: html.escape(str(v))
            rows = ''.join(f'<tr><td>{esc(r["predicate"])}</td><td>{r["crop_cosine"]:.4f}</td>'
                           f'<td>{r["full_cosine"]:.4f}</td></tr>' for r in sorted(
                               ranking, key=lambda r: r['crop_cosine'], reverse=True))
            cards.append(f'<article><h3>{index}: {esc(pair["filename"])} / slot {pair["slot"]}</h3>'
                         f'<img src="{index:03d}_full.jpg"><img src="{index:03d}_crop.jpg">'
                         f'<p>Old: {esc(pair["old_predicate"])}; proposed: {esc(decision["proposed_predicate"])}; '
                         f'decision: {esc(decision["decision"])}</p>'
                         f'<table><tr><th>Candidate</th><th>Crop cosine</th><th>Full cosine</th></tr>{rows}</table></article>')
            print(f'Pair {index+1}/{len(selected)} | {pair["old_predicate"]} -> '
                  f'{decision["proposed_predicate"]} | {decision["decision"]}', flush=True)
    with (output / 'manual_review.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        fields = ['filename', 'slot', 'subject', 'object', 'old_predicate', 'proposed_predicate',
                  'decision', 'crop_top', 'full_top', 'crop_gap', 'full_gap',
                  'human_old_status', 'human_proposed_status', 'human_supported_predicates', 'notes']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key, '') for key in fields})
    summary = {'variant': 'multi_candidate_crop_full_clip_audit_v1', 'count': len(results),
               'split': 'val', 'seed': args.seed, 'model': args.model, 'min_gap': args.min_gap,
               'decision_counts': dict(Counter(r['decision'] for r in results)),
               'proposed_predicates': dict(Counter(r['proposed_predicate'] for r in results if r['proposed_predicate'])),
               'changed_proposal_count': sum(r['proposed_predicate'] is not None and
                                             r['proposed_predicate'] != r['old_predicate'] for r in results),
               'notes': 'CLIP cosine ranks are not calibrated probabilities. Human review required. '
                        'Alternatives may coexist; lower rank does not make a valid training negative. '
                        'The near description is a neutral baseline; abstention is a selection rule. '
                        'Only existing heuristic person-object pairs are audited; this does not detect new pairs.'}
    (output / 'candidate_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Action candidate audit</title>'
        '<style>body{font:16px Arial;margin:24px}article{border-bottom:1px solid #ccc;padding:20px}'
        'img{max-width:45%;max-height:400px;margin:8px}td,th{padding:8px;text-align:left}</style>'
        '<h1>Action candidates: review required</h1><p>Red: subject; blue: object. '
        'Left: full image; right: pair crop. Review without relying on original captions.</p>'
        + ''.join(cards), encoding='utf-8')
    archive = shutil.make_archive(str(output), 'zip', output)
    print('ACTION CANDIDATE AUDIT COMPLETE\n' + json.dumps(summary, indent=2), flush=True)
    print('Download:', archive, flush=True)


if __name__ == '__main__':
    main()
