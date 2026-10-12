"""Validation-only inspection of E3 relation ranking, gates and caption sensitivity.

Hooks observe the existing checkpoint without changing its saved architecture.
The off condition bypasses only the pair-grounded residual, retaining prompt V2.1.
"""
import argparse
import csv
import html
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from captioning.cli import parse_args
from captioning.checkpoints import load_checkpoint
from captioning.data import load_data, load_visual_input
from captioning.inference import generate_caption_beam_search
from captioning.models import UniversalVisionEncoder, CaptionDecoder, ImageCaptioningModel
from captioning.grounded_relations import ACTION_PREDICATES


def ranking(diagnostics):
    valid = diagnostics['mask'].bool()
    appearance = 'appearance_features' in diagnostics
    pair = F.normalize(diagnostics['appearance_features' if appearance else 'pair_features'][valid], dim=-1)
    positive = F.normalize(diagnostics['fixed_positive_features' if appearance else 'positive_relation_features'][valid], dim=-1)
    negative = F.normalize(diagnostics['fixed_negative_features' if appearance else 'negative_relation_features'][valid], dim=-1)
    return (pair * positive).sum(-1), (pair * negative).sum(-1)


def mean(values):
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--sample-size', type=int, default=100)
    parser.add_argument('--expected-epoch', type=int, default=3)
    args, remaining = parser.parse_known_args()
    config = parse_args(remaining)
    if config.mode != 'predict' or config.split != 'val' or not config.visual_cache:
        raise ValueError('Require --mode predict --split val and a visual cache.')
    if not config.grounded_relation_alignment or config.limit:
        raise ValueError('Require grounded relation alignment and --limit 0.')
    if args.sample_size < 2:
        raise ValueError('Sample size must be at least two.')
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.is_main_process = True
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    data = load_data(config)
    try:
        encoder = UniversalVisionEncoder(
            load_backbone=False, visual_adapter='qformer',
            num_visual_queries=config.num_visual_queries, qformer_layers=config.qformer_layers,
            grounded_relation_alignment=True,
            grounded_relation_version=config.grounded_relation_version,
        ).to(config.device)
        model = ImageCaptioningModel(encoder, CaptionDecoder(data.vocab_size)).to(config.device)
        checkpoint = load_checkpoint(model, config.checkpoint, data.tokenizer)
        if checkpoint.get('epoch') != args.expected_epoch:
            raise ValueError(f'This diagnostic expects epoch {args.expected_epoch}.')
        checkpoint_metadata = {key: checkpoint.get(key) for key in (
            'epoch', 'visual_adapter', 'num_visual_queries', 'qformer_layers',
            'grounded_relation_alignment', 'grounded_relation_version',
            'relation_alignment_weight', 'relation_alignment_margin')}
        del checkpoint
        model.eval()

        # A random sample represents validation; this is not a quality benchmark.
        selected = data.val_df.sample(n=args.sample_size, random_state=config.seed)
        rows = [row for _, row in selected.iterrows()]
        visuals = [load_visual_input(row, data.transform, data.visual_cache) for row in rows]
        module = encoder.relation_alignment
        state = {'bypass': False}

        def query_hook(_module, _inputs, logits):
            state['query_gate'] = torch.sigmoid(logits).detach()

        def relation_hook(_module, inputs, result):
            state['inputs'] = inputs
            state['diagnostics'] = result[1]
            state['residual_ratio'] = float(
                (result[0] - inputs[0]).norm() / inputs[0].norm().clamp_min(1e-8))
            if state['bypass']:
                return inputs[0], result[1]

        query_handle = module.query_gate.register_forward_hook(query_hook)
        relation_handle = module.register_forward_hook(relation_hook)
        records, relations = [], []
        try:
            with torch.inference_mode():
                for index, (row, image) in enumerate(zip(rows, visuals)):
                    entry = data.grounded_relation_cache[row['image']]
                    prompt = data.prompt_cache[row['image']]
                    kwargs = dict(
                        model=model, image=image, cached_prompt_tokens=prompt['tokens'],
                        prompt_mask=prompt['mask'], tokenizer=data.tokenizer, beam_size=5,
                        device=config.device, relation_entry=entry,
                    )
                    state['bypass'] = False
                    caption_on = generate_caption_beam_search(**kwargs)
                    diagnostics = state['diagnostics']
                    positive, negative = ranking(diagnostics)
                    positive, negative = positive.tolist(), negative.tolist()
                    evidence = diagnostics['evidence_gate'][diagnostics['mask']].tolist()
                    query_gate = float(state['query_gate'].mean())
                    residual_ratio = state['residual_ratio']
                    inputs = state['inputs']

                    # Replace the visual evidence with a different sampled image,
                    # while preserving the original relation text and box geometry.
                    other = visuals[(index + 1) % len(visuals)].unsqueeze(0).to(config.device)
                    other_projected = encoder.dropout(encoder.relu(encoder.vis_projection(other.float())))
                    _, wrong_diagnostics = module(inputs[0], other_projected, *inputs[2:])
                    wrong_positive, wrong_negative = ranking(wrong_diagnostics)
                    wrong_positive, wrong_negative = wrong_positive.tolist(), wrong_negative.tolist()

                    state['bypass'] = True
                    caption_off = generate_caption_beam_search(**kwargs)
                    state['bypass'] = False
                    count = len(positive)
                    for slot in range(count):
                        metadata = entry.get('relations', [])[slot]
                        relations.append({
                            'filename': row['filename'], 'slot': slot,
                            'subject': metadata.get('subject'), 'predicate': metadata.get('relation'),
                            'object': metadata.get('object'),
                            'counterfactual': metadata.get('counterfactual_relation'),
                            'action_predicate': metadata.get('relation') in ACTION_PREDICATES,
                            'positive_score': positive[slot], 'negative_score': negative[slot],
                            'margin': positive[slot] - negative[slot],
                            'wrong_image_positive_score': wrong_positive[slot],
                            'wrong_image_negative_score': wrong_negative[slot],
                            'wrong_image_margin': wrong_positive[slot] - wrong_negative[slot],
                            'evidence_gate': evidence[slot],
                            'confidence': float(entry['confidences'][slot]),
                            'subject_box': entry['subject_boxes'][slot].tolist(),
                            'object_box': entry['object_boxes'][slot].tolist(),
                        })
                    record = {
                        'filename': row['filename'], 'image_path': row['image'],
                        'image_id': int(row['eval_id']), 'relation_count': count,
                        'caption_on': caption_on, 'caption_off': caption_off,
                        'caption_changed': caption_on != caption_off,
                        'query_gate_mean': query_gate if count else None,
                        'residual_ratio': residual_ratio, 'references': row['captions'],
                        'relations': relations[-count:] if count else [],
                    }
                    records.append(record)
                    (output / 'diagnostic_records.json').write_text(
                        json.dumps(records, indent=2, ensure_ascii=False), encoding='utf-8')
                    print(f"Diagnostic {index + 1}/{len(rows)} | relations={count} | "
                          f"caption changed={record['caption_changed']}", flush=True)
        finally:
            relation_handle.remove()
            query_handle.remove()

        def stats(items):
            return {
                'count': len(items), 'mean_margin': mean([x['margin'] for x in items]),
                'positive_above_negative_rate': mean([x['margin'] > 0 for x in items]),
                'margin_satisfied_rate': mean([x['margin'] >= config.relation_alignment_margin for x in items]),
                'wrong_image_mean_margin': mean([x['wrong_image_margin'] for x in items]),
                'wrong_image_positive_above_negative_rate': mean([x['wrong_image_margin'] > 0 for x in items]),
                'correct_minus_wrong_positive_score': mean([
                    x['positive_score'] - x['wrong_image_positive_score'] for x in items]),
                'evidence_gate_mean': mean([x['evidence_gate'] for x in items]),
            }
        grouped = defaultdict(list)
        for relation in relations:
            grouped[relation['predicate']].append(relation)
        with_relations = [r for r in records if r['relation_count']]
        without_relations = [r for r in records if not r['relation_count']]
        summary = {
            'checkpoint': config.checkpoint, 'checkpoint_metadata': checkpoint_metadata,
            'split': 'val', 'sampling': 'random_without_replacement', 'seed': config.seed,
            'sample_count': len(records), 'images_with_relations': len(with_relations),
            'score_source': ('appearance_only_fixed_text' if config.grounded_relation_version == 'appearance_v2'
                             else 'geometry_and_appearance_trainable_text'),
            'caption_changed_count': sum(r['caption_changed'] for r in records),
            'caption_changed_rate_with_relations': mean([r['caption_changed'] for r in with_relations]),
            'no_relation_caption_changed_count': sum(r['caption_changed'] for r in without_relations),
            'query_gate_mean': mean([r['query_gate_mean'] for r in with_relations]),
            'residual_ratio_mean': mean([r['residual_ratio'] for r in with_relations]),
            'ranking': stats(relations), 'by_predicate': {k: stats(v) for k, v in grouped.items()},
            'action_ranking': stats([r for r in relations if r['action_predicate']]),
            'interpretation': 'Cached positives are heuristic labels, not human ground truth. '
                'Wrong-image scores test sensitivity to visual evidence while preserving geometry. '
                'Off bypasses only the E3 residual; relation information remains in prompt V2.1. '
                'Caption changes do not measure quality or causal credit for individual words.',
        }
        (output / 'diagnostic_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        if relations:
            with (output / 'relation_scores.csv').open('w', encoding='utf-8-sig', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(relations[0]))
                writer.writeheader()
                writer.writerows(relations)
        cards = []
        from PIL import Image, ImageDraw
        for i, record in enumerate(records):
            image_name = f'{i:03d}.jpg'
            with Image.open(record['image_path']) as original:
                thumbnail = original.convert('RGB')
                thumbnail.thumbnail((640, 480))
                draw = ImageDraw.Draw(thumbnail)
                for relation in record['relations']:
                    for key, color in [('subject_box', 'red'), ('object_box', 'blue')]:
                        box = relation[key]
                        draw.rectangle(tuple(value * (thumbnail.width if n % 2 == 0 else thumbnail.height)
                                             for n, value in enumerate(box)), outline=color, width=2)
                thumbnail.save(output / image_name)
            esc = lambda value: html.escape(str(value))
            cards.append(f'<article><h3>{esc(record["filename"])}</h3><img src="{image_name}">'
                f'<p><b>Relation ON:</b> {esc(record["caption_on"])}</p>'
                f'<p><b>Relation OFF:</b> {esc(record["caption_off"])}</p>'
                f'<p>Reference: {esc(record["references"])}</p>'
                f'<pre>{esc(json.dumps(record["relations"], indent=2, ensure_ascii=False))}</pre></article>')
        (output / 'review.html').write_text('<!doctype html><meta charset="utf-8">'
            '<title>E3 relation diagnostic</title><style>body{font:16px Arial;margin:24px;max-width:1100px}'
            'article{border-bottom:1px solid #aaa;padding:20px}img{max-width:640px}pre{white-space:pre-wrap}</style>'
            '<h1>E3 validation diagnostic</h1><p>Red: subject; blue: object. '
            'ON/OFF changes only the pair-grounded residual. These labels need visual review.</p>'
            + ''.join(cards), encoding='utf-8')
        print('\nE3 DIAGNOSTIC SUMMARY\n' + json.dumps(summary, indent=2), flush=True)
        print('Output:', output, flush=True)
    finally:
        if data.visual_cache is not None:
            data.visual_cache.close()


if __name__ == '__main__':
    main()
