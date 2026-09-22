"""Audit a saved YOLO detection cache without rerunning YOLO."""
import argparse
import json
from pathlib import Path

from captioning.detection_audit import audit_detections, dataset_records, write_csv


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--dataset-json-path', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--base-path', default='', help='COCO images root; enables image-bound checks.')
    parser.add_argument('--objects-field', default='objects')
    parser.add_argument('--name-key', default='label')
    parser.add_argument('--bbox-key', default='bbox')
    parser.add_argument('--confidence-key', default='conf')
    parser.add_argument('--duplicate-iou', type=float, default=0.8)
    args = parser.parse_args(argv)
    if not 0 < args.duplicate_iou <= 1:
        parser.error('--duplicate-iou must be in (0, 1].')

    source = json.loads(Path(args.source).read_text(encoding='utf-8'))
    records = dataset_records(args.dataset_json_path)
    size_reader = None
    if args.base_path:
        from PIL import Image
        root = Path(args.base_path)

        def size_reader(filename, record):
            path = root / record.get('filepath', '') / filename
            with Image.open(path) as image:
                return image.size

    report, rows, label_counts = audit_detections(
        source, records, args.objects_field, args.name_key, args.bbox_key,
        args.confidence_key, size_reader, args.duplicate_iou)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'yolo_audit_summary.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    write_csv(output / 'yolo_audit_images.csv', rows)
    write_csv(output / 'yolo_audit_labels.csv', [
        {'label': label, 'detections': count} for label, count in label_counts.most_common()
    ])
    print(json.dumps({
        'scope': report['scope'],
        'detections': report['detections'],
        'confidence': report['confidence'],
        'issues': report['issues'],
        'flagged_images': report['flagged_images'],
        'outputs': str(output),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()

