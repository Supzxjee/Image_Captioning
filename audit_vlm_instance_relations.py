"""Run one GPU shard or merge a two-GPU audit on a pinned set of 100 pairs."""
import argparse
import csv
import html
import json
import time
from collections import Counter
from pathlib import Path

from captioning.vlm_relation_audit import audit_prompt, parse_teacher_answer, shard_pairs
from captioning.relation_candidates import union_crop_box


def merge(output, shard_count, expected_count):
    results = []
    for shard in range(shard_count):
        results += json.loads((output / f'shard_{shard}.json').read_text(encoding='utf-8'))
    results.sort(key=lambda r: r['pair_index'])
    if [r['pair_index'] for r in results] != list(range(expected_count)):
        raise ValueError('Missing or duplicate pair indices; do not report a partial audit as complete.')
    summary = {
        'variant': 'instance_marked_qwen25vl3b_teacher_audit_v1', 'count': len(results),
        'decision_counts': dict(Counter(r['decision'] for r in results)),
        'proposed_relations': dict(Counter(p for r in results for p in r['proposed_relations'])),
        'old_label_retained_count': sum(r['old_predicate'] in r['proposed_relations'] for r in results),
        'object_label_mismatch_count': sum(r['decision'] == 'object_label_mismatch' for r in results),
        'notes': 'VLM proposals require human visual review, not ground truth or calibrated confidence. '
                 'No reference captions, old predicate or CLIP scores were shown to the model. '
                 'This audits fixed existing pairs, not a complete relation detector.',
    }
    (output / 'vlm_records.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    (output / 'vlm_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    fields = ['pair_index', 'filename', 'slot', 'object', 'old_predicate', 'clip_proposal',
              'decision', 'proposed_relations', 'object_description', 'evidence',
              'human_object_status', 'human_relation_status', 'human_supported_relations', 'notes']
    with (output / 'manual_review.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, '') for key in fields})
    esc = lambda x: html.escape(str(x))
    cards = []
    for row in results:
        i = row['pair_index']
        cards.append(f'<article><h3>{i}: {esc(row["filename"])} / slot {row["slot"]}</h3>'
                     f'<img src="{i:03d}_full.jpg"><img src="{i:03d}_pair.jpg">'
                     f'<img src="{i:03d}_object.jpg"><p>Heuristic: {esc(row["old_predicate"])}; '
                     f'CLIP: {esc(row.get("clip_proposal"))}; VLM: {esc(row["proposed_relations"])}; '
                     f'decision: {esc(row["decision"])}</p><pre>{esc(row["raw_answer"])}</pre></article>')
    (output / 'review.html').write_text('<!doctype html><meta charset="utf-8">'
        '<title>Instance relation teacher audit</title><style>body{font:16px Arial;margin:24px}'
        'article{border-bottom:1px solid #ccc;padding:20px}img{max-width:32%;max-height:380px}'
        'pre{white-space:pre-wrap}</style><h1>Review required: instance relation proposals</h1>'
        '<p>Red A: subject; blue B: object. Last image: object detail. Model output is not ground truth.</p>'
        + ''.join(cards), encoding='utf-8')
    print('VLM INSTANCE RELATION AUDIT SUMMARY\n' + json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--dataset-json', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=2)
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()
    pairs = json.loads(Path(args.pairs).read_text(encoding='utf-8'))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge(output, args.shard_count, len(pairs))
        return
    import torch
    from PIL import Image, ImageDraw
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from captioning.detection_audit import dataset_records
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required for this teacher audit.')
    torch.manual_seed(42)
    dataset = dataset_records(args.dataset_json)
    print(f'Shard {args.shard_index}: loading Qwen onto visible GPU0...', flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map={'': 0}, attn_implementation='sdpa').eval()
    processor = AutoProcessor.from_pretrained(args.model, min_pixels=256 * 28 * 28,
                                            max_pixels=256 * 28 * 28)
    assigned = shard_pairs(pairs, args.shard_index, args.shard_count)
    path = output / f'shard_{args.shard_index}.json'
    previous = json.loads(path.read_text(encoding='utf-8')) if path.is_file() else []
    completed = {r['pair_index'] for r in previous}
    results = list(previous)
    for index, pair in assigned:
        if index in completed:
            continue
        started = time.monotonic()
        image_path = Path(args.images) / dataset[pair['filename']].get('filepath', '') / pair['filename']
        with Image.open(image_path) as original:
            full = original.convert('RGB')
            marked = full.copy()
            draw = ImageDraw.Draw(marked)
            for key, color, label in [('subject_box', 'red', 'A'), ('object_box', 'blue', 'B')]:
                box = pair[key]
                pixels = tuple(v * (full.width if j % 2 == 0 else full.height) for j, v in enumerate(box))
                draw.rectangle(pixels, outline=color, width=max(2, int(full.width / 250)))
                draw.text((pixels[0] + 3, pixels[1] + 3), label, fill=color)
            pair_crop = marked.crop(union_crop_box(pair['subject_box'], pair['object_box'], *full.size))
            obj_crop = full.crop(union_crop_box(pair['object_box'], pair['object_box'], *full.size, padding=.1))
            views = [marked, pair_crop, obj_crop]
            messages = [{'role': 'user', 'content': [
                {'type': 'image'}, {'type': 'image'}, {'type': 'image'},
                {'type': 'text', 'text': audit_prompt(pair)}]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=views, padding=True, return_tensors='pt').to('cuda:0')
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            raw = processor.batch_decode(generated[:, inputs['input_ids'].shape[1]:],
                                         skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            try:
                answer = parse_teacher_answer(raw)
            except (ValueError, TypeError) as error:
                answer = {'decision': 'parse_error', 'proposed_relations': [], 'parse_error': str(error)}
            row = {**pair, **answer, 'pair_index': index, 'raw_answer': raw,
                   'seconds': time.monotonic() - started}
            for suffix, view in zip(['full', 'pair', 'object'], views):
                preview = view.copy()
                preview.thumbnail((640, 480))
                preview.save(output / f'{index:03d}_{suffix}.jpg')
            results.append(row)
            temporary = path.with_suffix('.json.partial')
            temporary.write_text(json.dumps(results, indent=2), encoding='utf-8')
            temporary.replace(path)
            print(f'Shard {args.shard_index} pair {index}: {row["decision"]} '
                  f'{row["proposed_relations"]} | {row["seconds"]:.1f}s', flush=True)
    print(f'Shard {args.shard_index} complete: {len(results)} pairs', flush=True)


if __name__ == '__main__':
    main()
