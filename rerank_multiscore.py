"""Tune/apply decoder + CLIPScore + OCC candidate re-ranking."""
import argparse
import json
from itertools import product
from pathlib import Path

from captioning.metrics import compute_coco_metrics
from captioning.multiscore_reranking import rerank_multiscore
from captioning.object_consistency import index_detections


def _tag(value):
    return f'{value:.3f}'.replace('.', 'p')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--ground-truth', required=True)
    parser.add_argument('--detections', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--clip-weight', type=float)
    parser.add_argument('--object-weight', type=float)
    parser.add_argument('--clip-weights', default='0,0.1,0.2,0.3')
    parser.add_argument('--object-weights', default='0,0.05,0.1,0.2')
    parser.add_argument('--min-confidence', type=float, default=0.5)
    parser.add_argument('--hallucination-penalty', type=float, default=1.0)
    parser.add_argument('--objects-field', default='objects')
    parser.add_argument('--name-key', default='label')
    parser.add_argument('--confidence-key', default='conf')
    args = parser.parse_args(argv)

    fixed = args.clip_weight is not None or args.object_weight is not None
    if fixed and (args.clip_weight is None or args.object_weight is None):
        parser.error('Fixed apply mode requires both --clip-weight and --object-weight.')
    if fixed:
        combinations = [(args.clip_weight, args.object_weight)]
        mode = 'apply'
    else:
        clip_values = [float(value) for value in args.clip_weights.split(',')]
        object_values = [float(value) for value in args.object_weights.split(',')]
        combinations = [(clip, obj) for clip, obj in product(clip_values, object_values)
                        if clip >= 0 and obj >= 0 and clip + obj <= 1]
        mode = 'tune'
    if not combinations or any(c < 0 or o < 0 or c + o > 1 for c, o in combinations):
        parser.error('Weights must be nonnegative and clip + object <= 1.')

    bundle = json.loads(Path(args.candidates).read_text(encoding='utf-8'))
    records, metadata = bundle.get('data', []), bundle.get('metadata', {})
    if not records or metadata.get('count') != len(records):
        raise ValueError('Candidate bundle is empty or incomplete.')
    if not metadata.get('clipscore_model'):
        raise ValueError('Candidates have not been processed by score_clip_candidates.py.')
    detection_source = json.loads(Path(args.detections).read_text(encoding='utf-8'))
    detections = index_detections(
        detection_source, args.objects_field, args.name_key, args.confidence_key,
        args.min_confidence)
    missing = [record['filename'] for record in records if record['filename'] not in detections]
    if missing:
        raise ValueError(f'{len(missing)} images lack detections: {missing[:10]}')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for clip_weight, object_weight in combinations:
        predictions, details = [], []
        changed = mentioned = supported = hallucinated = 0
        for record in records:
            selected, enriched = rerank_multiscore(
                record, detections[record['filename']], clip_weight, object_weight,
                args.hallucination_penalty)
            consistency = selected['object_consistency']
            changed += selected['original_rank'] != 0
            mentioned += len(consistency['mentioned'])
            supported += len(consistency['supported'])
            hallucinated += len(consistency['hallucinated'])
            predictions.append({'image_id': int(record['image_id']),
                                'caption': selected['caption']})
            details.append({'image_id': int(record['image_id']),
                            'coco_id': int(record['coco_id']),
                            'filename': record['filename'],
                            'selected_rank': selected['original_rank'],
                            'selected_caption': selected['caption'],
                            'candidates': enriched})

        decoder_weight = 1.0 - clip_weight - object_weight
        prefix = (f'{metadata.get("split", "split")}_{len(records)}_multi_'
                  f'd{_tag(decoder_weight)}_c{_tag(clip_weight)}_o{_tag(object_weight)}')
        predictions_path = output_dir / f'{prefix}_predictions.json'
        details_path = output_dir / f'{prefix}_details.json'
        predictions_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2),
                                    encoding='utf-8')
        details_path.write_text(json.dumps(details, ensure_ascii=False, indent=2),
                                encoding='utf-8')
        metrics = compute_coco_metrics(predictions_path, args.ground_truth)
        result = {
            'decoder_weight': decoder_weight,
            'clip_weight': clip_weight,
            'object_weight': object_weight,
            'metrics': metrics,
            'changed_images': changed,
            'recognized_mentions': mentioned,
            'supported_mentions': supported,
            'hallucinated_mentions': hallucinated,
            'predictions': str(predictions_path),
            'details': str(details_path),
        }
        results.append(result)
        print(f'decoder={decoder_weight:.2f} clip={clip_weight:.2f} '
              f'object={object_weight:.2f} | changed={changed}/{len(records)} | '
              f'CIDEr={metrics["CIDEr"]:.4f} | BLEU-4={metrics["Bleu_4"]:.4f}',
              flush=True)

    best = max(results, key=lambda item: (item['metrics']['CIDEr'],
                                           item['metrics']['Bleu_4'],
                                           item['metrics']['METEOR']))
    summary = {
        'mode': mode,
        'candidate_metadata': metadata,
        'selection_metric': 'CIDEr, then BLEU-4, then METEOR',
        'best_weights': {key: best[key] for key in
                         ('decoder_weight', 'clip_weight', 'object_weight')},
        'best_metrics': best['metrics'],
        'results': results,
    }
    summary_path = output_dir / f'{metadata.get("split", "split")}_multiscore_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nBest weights:', summary['best_weights'])
    print('Best metrics:', json.dumps(best['metrics'], indent=2))
    print('Summary:', summary_path)


if __name__ == '__main__':
    main()
