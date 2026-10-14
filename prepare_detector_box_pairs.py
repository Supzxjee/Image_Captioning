"""Match cached YOLO boxes to the locked 500-pair new-image cohort."""
import argparse
import hashlib
import json
from collections import defaultdict, Counter
from pathlib import Path, PurePosixPath
from PIL import Image
from captioning.relation_prompts import select_detections
from captioning.detector_pair_matching import normalized_detection_box, match_instances, matched_pairs
from captioning.supervised_relations import RELATIONS, pair_statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', required=True)
    parser.add_argument('--detections', required=True)
    parser.add_argument('--images', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--min-confidence', type=float, default=.5)
    parser.add_argument('--min-iou', type=float, default=.5)
    parser.add_argument('--coordinate-format', choices=['pixel_xyxy', 'normalized_xyxy'], default='pixel_xyxy')
    parser.add_argument('--reference-count',type=int,default=500,help='Explicitly locked input cohort size; legacy default500.')
    parser.add_argument('--max-objects',type=int,default=100,help='Legacy default100; training domain protocol uses10.')
    args = parser.parse_args()
    if not 0 <= args.min_confidence <= 1 or not 0 < args.min_iou <= 1:
        raise ValueError('Confidence outside [0,1].')
    raw = Path(args.pairs).read_bytes()
    rows = json.loads(raw)
    if len(rows) != args.reference_count or args.reference_count<1 or args.max_objects<1:
        raise ValueError('Require the explicitly locked reference pair count and positive object cap.')
    source_path = Path(args.detections)
    source = json.loads(source_path.read_text())
    if not isinstance(source, dict):
        raise ValueError('Detection cache must be a mapping.')
    payload = source.get('data', source)
    if not isinstance(payload, dict):
        raise ValueError('Detection cache must map filenames to entries.')
    indexed = {}
    for key, entry in payload.items():
        name = PurePosixPath(str(key).replace('\\', '/')).name
        if name in indexed:
            raise ValueError(f'Duplicate detection filename {name}.')
        indexed[name] = entry
    groups = defaultdict(list)
    for row in rows:
        groups[row['filename']].append(row)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    selected, missing_counts, issues, image_reports = [], Counter(), Counter(), []
    for filename, pairs in groups.items():
        if filename not in indexed:
            issues['missing_cache_images'] += 1
            issues['pairs_without_cache_record'] += len(pairs)
            image_reports.append({'filename': filename, 'status': 'missing_cache_record'})
            continue
        path = Path(args.images) / pairs[0]['filepath'] / filename
        with Image.open(path) as image:
            width, height = image.size
        raw_boxes = select_detections(indexed[filename], min_confidence=args.min_confidence,
                                      max_objects=args.max_objects, duplicate_iou=.8)
        detections = []
        for item in raw_boxes:
            if not 0 <= item.confidence <= 1:
                issues['confidence_out_of_range'] += 1
                continue
            try:
                box = normalized_detection_box(item.box, width, height, args.coordinate_format)
            except ValueError:
                issues['invalid_or_empty_boxes'] += 1
                continue
            detections.append({'label': item.label, 'box': box, 'confidence': item.confidence})
        gt = {}
        for row in pairs:
            for role in ('subject', 'object'):
                aid = row[role+'_id']
                value = {'label': row[role], 'box': row[role+'_box']}
                if aid in gt and gt[aid] != value:
                    raise ValueError('Inconsistent GT instance across pairs.')
                gt[aid] = value
        assigned = match_instances(gt, detections, args.min_iou)
        matched, missing = matched_pairs(pairs, assigned)
        selected += matched
        missing_counts.update(missing)
        image_reports.append({'filename': filename, 'status': 'matched', 'detections': len(detections),
                               'gt_instances_referenced': len(gt), 'matched_instances': len(assigned),
                               'reference_pairs': len(pairs), 'matched_pairs': len(matched)})
    order = {(r['subject_id'], r['object_id']): i for i, r in enumerate(rows)}
    selected.sort(key=lambda r: order[(r['subject_id'], r['object_id'])])
    path = output / 'evaluation_pairs.json'
    path.write_text(json.dumps(selected, indent=2), encoding='utf-8')
    summary = {'variant': 'frozen_residual_detector_matched_pairs_v1', 'relations': RELATIONS,
               'reference_pairs': args.reference_count, 'matched_pairs': len(selected), 'pair_coverage': len(selected)/len(rows),
               'max_objects':args.max_objects,
               'min_confidence': args.min_confidence, 'min_iou': args.min_iou,
               'coordinate_format': args.coordinate_format, 'missing_pairs': dict(missing_counts),
               'cache_issues': dict(issues), 'matching': 'class-aware IoU-descending one-to-one greedy; same-class NMS0.8',
               'source_pairs_sha256': hashlib.sha256(raw).hexdigest(),
               'pair_json_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'reference_positive_counts': {name: sum(r['labels'][i] == 1 for r in rows) for i, name in enumerate(RELATIONS)},
               'matched_positive_counts': {name: sum(r['labels'][i] == 1 for r in selected) for i, name in enumerate(RELATIONS)},
               'evaluation': pair_statistics(selected),
               'notes': 'GT used only for matching/labels; scoring crops use detector boxes. Conditional matched-pair '
                        'diagnostic, not full HOI detection AP; unmatched detector proposals not labelled false.'}
    digest = hashlib.sha256()
    with source_path.open('rb') as handle:
        while block := handle.read(4*1024**2):
            digest.update(block)
    summary['detection_cache_sha256'] = digest.hexdigest()
    (output / 'detector_pair_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (output / 'detector_image_reports.json').write_text(json.dumps(image_reports, indent=2), encoding='utf-8')
    print('DETECTOR BOX MATCHING\n' + json.dumps(summary, indent=2), flush=True)
    if not selected:
        raise ValueError('No matched pairs; inspect coordinate format/cache labels before feature extraction.')


if __name__ == '__main__':
    main()
