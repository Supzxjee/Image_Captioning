"""Tune/apply YOLO object-consistency reranking on saved beam candidates."""
import argparse
import json
from pathlib import Path

from captioning.metrics import compute_coco_metrics
from captioning.object_consistency import index_detections, rerank_record


def _weight_name(weight):
    return f'{weight:.3f}'.replace('.', 'p')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--ground-truth', required=True)
    parser.add_argument('--detections', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--weight', type=float, default=None,
                        help='Fixed weight for test. Omit to tune weights on validation.')
    parser.add_argument('--weights', default='0,0.05,0.1,0.2,0.3,0.4')
    parser.add_argument('--min-confidence', type=float, default=0.5)
    parser.add_argument('--hallucination-penalty', type=float, default=1.0)
    parser.add_argument('--objects-field', default='objects')
    parser.add_argument('--name-key', default='label')
    parser.add_argument('--confidence-key', default='conf')
    args = parser.parse_args(argv)
    if not 0 <= args.min_confidence <= 1 or args.hallucination_penalty < 0:
        parser.error('Invalid confidence or hallucination penalty.')

    candidate_bundle = json.loads(Path(args.candidates).read_text(encoding='utf-8'))
    records = candidate_bundle.get('data', [])
    metadata = candidate_bundle.get('metadata', {})
    if not records or metadata.get('count') != len(records):
        raise ValueError('Candidate bundle is empty or incomplete.')
    detection_source = json.loads(Path(args.detections).read_text(encoding='utf-8'))
    detections = index_detections(
        detection_source, args.objects_field, args.name_key, args.confidence_key,
        args.min_confidence)
    missing = [record['filename'] for record in records if record['filename'] not in detections]
    if missing:
        raise ValueError(f'{len(missing)} candidate images lack detections: {missing[:10]}')

    if args.weight is None:
        weights = [float(value) for value in args.weights.split(',')]
        if not weights or any(not 0 <= value <= 1 for value in weights):
            parser.error('Every tuning weight must be in [0, 1].')
        mode = 'tune'
    else:
        if not 0 <= args.weight <= 1:
            parser.error('weight must be in [0, 1].')
        weights, mode = [args.weight], 'apply'

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for weight in weights:
        predictions, details = [], []
        changed = mentioned = supported = hallucinated = 0
        for record in records:
            selected, enriched = rerank_record(
                record, detections[record['filename']], weight,
                args.hallucination_penalty)
            consistency = selected['object_consistency']
            changed += selected['original_rank'] != 0
            mentioned += len(consistency['mentioned'])
            supported += len(consistency['supported'])
            hallucinated += len(consistency['hallucinated'])
            predictions.append({'image_id': int(record['image_id']),
                                'caption': selected['caption']})
            details.append({'image_id': int(record['image_id']),
                            'filename': record['filename'],
                            'selected_rank': selected['original_rank'],
                            'selected_caption': selected['caption'],
                            'candidates': enriched})

        prefix = f'{metadata.get("split", "split")}_{len(records)}_occ_w{_weight_name(weight)}'
        predictions_path = output_dir / f'{prefix}_predictions.json'
        details_path = output_dir / f'{prefix}_details.json'
        metrics_path = output_dir / f'{prefix}_metrics.json'
        predictions_path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2),
                                    encoding='utf-8')
        details_path.write_text(json.dumps(details, ensure_ascii=False, indent=2),
                                encoding='utf-8')
        metrics = compute_coco_metrics(predictions_path, args.ground_truth)
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        result = {
            'weight': weight,
            'metrics': metrics,
            'changed_images': changed,
            'recognized_mentions': mentioned,
            'supported_mentions': supported,
            'hallucinated_mentions': hallucinated,
            'predictions': str(predictions_path),
            'details': str(details_path),
            'metrics_path': str(metrics_path),
        }
        results.append(result)
        print(f'weight={weight:.3f} | changed={changed}/{len(records)} | '
              f'CIDEr={metrics["CIDEr"]:.4f} | BLEU-4={metrics["Bleu_4"]:.4f} | '
              f'hallucinated={hallucinated}/{mentioned}', flush=True)

    best = max(results, key=lambda item: (item['metrics']['CIDEr'],
                                           item['metrics']['Bleu_4'],
                                           item['metrics']['METEOR']))
    summary = {
        'mode': mode,
        'candidate_metadata': metadata,
        'min_confidence': args.min_confidence,
        'hallucination_penalty': args.hallucination_penalty,
        'selection_metric': 'CIDEr, then BLEU-4, then METEOR',
        'best_weight': best['weight'],
        'best_metrics': best['metrics'],
        'results': results,
    }
    summary_path = output_dir / f'{metadata.get("split", "split")}_occ_summary.json'
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('\nBest weight:', best['weight'])
    print('Best metrics:', json.dumps(best['metrics'], indent=2))
    print('Summary:', summary_path)


if __name__ == '__main__':
    main()
