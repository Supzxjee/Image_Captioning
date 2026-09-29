"""Prepare matched Q-Former and OCC captions for COCO CHAIR evaluation."""
import argparse
import json
from pathlib import Path


def _read_json(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8'))


def prepare(candidates_path, occ_predictions_path, output_dir):
    """Write matched prediction files whose image_id is the real COCO image id."""
    candidate_bundle = _read_json(candidates_path)
    records = candidate_bundle.get('data')
    metadata = candidate_bundle.get('metadata', {})
    if not isinstance(records, list) or not records:
        raise ValueError('Candidate bundle has no data records.')
    if metadata.get('count') != len(records):
        raise ValueError('Candidate bundle count does not match its records.')

    by_eval_id = {}
    baseline = []
    for record in records:
        eval_id = int(record['image_id'])
        coco_id = int(record['coco_id'])
        candidates = record.get('candidates', [])
        if eval_id in by_eval_id:
            raise ValueError(f'Duplicate evaluation image id: {eval_id}')
        if not candidates or not isinstance(candidates[0].get('caption'), str):
            raise ValueError(f'Image {eval_id} has no rank-0 caption.')
        by_eval_id[eval_id] = coco_id
        baseline.append({'image_id': coco_id, 'caption': candidates[0]['caption']})

    occ_source = _read_json(occ_predictions_path)
    if not isinstance(occ_source, list) or not occ_source:
        raise ValueError('OCC predictions must be a non-empty JSON list.')
    if len(occ_source) != len(records):
        raise ValueError('OCC prediction count differs from candidate count.')
    occ_by_eval_id = {}
    for prediction in occ_source:
        eval_id = int(prediction['image_id'])
        if eval_id in occ_by_eval_id:
            raise ValueError(f'Duplicate OCC evaluation image id: {eval_id}')
        if eval_id not in by_eval_id:
            raise ValueError(f'OCC image id {eval_id} is absent from candidates.')
        caption = prediction.get('caption')
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f'OCC image {eval_id} has an empty caption.')
        occ_by_eval_id[eval_id] = caption
    if set(occ_by_eval_id) != set(by_eval_id):
        missing = sorted(set(by_eval_id) - set(occ_by_eval_id))
        raise ValueError(f'OCC predictions omit evaluation ids: {missing[:10]}')

    occ = [
        {'image_id': by_eval_id[eval_id], 'caption': occ_by_eval_id[eval_id]}
        for eval_id in by_eval_id
    ]
    baseline_ids = [item['image_id'] for item in baseline]
    if len(set(baseline_ids)) != len(baseline_ids):
        raise ValueError('Candidate records map to duplicate COCO image ids.')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / 'qformer_baseline_chair_input.json'
    occ_path = output_dir / 'qformer_occ_w010_chair_input.json'
    manifest_path = output_dir / 'chair_input_manifest.json'
    baseline_path.write_text(json.dumps(baseline, ensure_ascii=False, indent=2),
                             encoding='utf-8')
    occ_path.write_text(json.dumps(occ, ensure_ascii=False, indent=2),
                        encoding='utf-8')
    changed = sum(a['caption'] != b['caption'] for a, b in zip(baseline, occ))
    manifest = {
        'count': len(records),
        'split': metadata.get('split'),
        'changed_captions': changed,
        'image_id': 'COCO image id (coco_id from candidate bundle)',
        'baseline': str(baseline_path),
        'occ': str(occ_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', required=True)
    parser.add_argument('--occ-predictions', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args(argv)
    manifest = prepare(args.candidates, args.occ_predictions, args.output_dir)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
