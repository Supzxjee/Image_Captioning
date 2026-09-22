"""Build normalized YOLO boxes and frozen CLIP label prototypes for region loss."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image

from captioning.detection_audit import dataset_records, index_source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--dataset-json-path', required=True)
    parser.add_argument('--base-path', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--objects-field', default='objects')
    parser.add_argument('--name-key', default='label')
    parser.add_argument('--bbox-key', default='bbox')
    parser.add_argument('--confidence-key', default='conf')
    parser.add_argument('--min-confidence', type=float, default=0.0)
    parser.add_argument('--max-regions', type=int, default=10)
    args = parser.parse_args(argv)
    if not 0 <= args.min_confidence <= 1 or args.max_regions <= 0:
        parser.error('min-confidence must be in [0,1]; max-regions must be positive.')

    source_path, dataset_path = Path(args.source), Path(args.dataset_json_path)
    source = index_source(json.loads(source_path.read_text(encoding='utf-8')))
    records = dataset_records(dataset_path)
    base = Path(args.base_path)

    labels = set()
    for entry in source.values():
        detections = entry if isinstance(entry, list) else entry.get(args.objects_field, [])
        for detection in detections:
            label = detection.get(args.name_key) if isinstance(detection, dict) else None
            if isinstance(label, str) and label.strip():
                labels.add(' '.join(label.lower().strip().split()))
    label_names = sorted(labels)
    label_to_id = {label: index for index, label in enumerate(label_names)}
    if not label_names:
        raise ValueError('No valid object labels found.')

    data, skipped, clipped, truncated = {}, 0, 0, 0
    for index, (filename, record) in enumerate(records.items(), 1):
        entry = source.get(filename)
        if entry is None:
            raise ValueError(f'Missing detections for {filename}')
        detections = entry if isinstance(entry, list) else entry.get(args.objects_field)
        if not isinstance(detections, list):
            raise ValueError(f'Expected {args.objects_field!r} list for {filename}')
        image_path = base / record.get('filepath', '') / filename
        with Image.open(image_path) as image:
            image_width, image_height = image.size
        targets = []
        for detection in detections:
            if not isinstance(detection, dict):
                skipped += 1
                continue
            label, bbox = detection.get(args.name_key), detection.get(args.bbox_key)
            confidence = detection.get(args.confidence_key, 1.0)
            if (not isinstance(label, str) or not label.strip() or
                    not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or
                    not all(isinstance(value, (int, float)) for value in bbox) or
                    not isinstance(confidence, (int, float)) or confidence < args.min_confidence):
                skipped += 1
                continue
            x1, y1, x2, y2 = map(float, bbox)
            bounded = (max(0.0, min(x1, image_width)), max(0.0, min(y1, image_height)),
                       max(0.0, min(x2, image_width)), max(0.0, min(y2, image_height)))
            clipped += bounded != (x1, y1, x2, y2)
            x1, y1, x2, y2 = bounded
            if x2 <= x1 or y2 <= y1:
                skipped += 1
                continue
            normalized_label = ' '.join(label.lower().strip().split())
            targets.append((float(confidence), [x1 / image_width, y1 / image_height,
                                                x2 / image_width, y2 / image_height],
                            label_to_id[normalized_label]))
        targets.sort(key=lambda item: item[0], reverse=True)
        truncated += max(0, len(targets) - args.max_regions)
        targets = targets[:args.max_regions]
        data[filename] = {
            'boxes': [item[1] for item in targets],
            'labels': [item[2] for item in targets],
            'confidences': [item[0] for item in targets],
        }
        if index == 1 or index % 10000 == 0:
            print(f'Region targets: {index}/{len(records)}', flush=True)

    from transformers import CLIPTextModel, CLIPTokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
    text_model = CLIPTextModel.from_pretrained(
        'openai/clip-vit-base-patch16').float().to(device).eval()
    encoded = tokenizer(label_names, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.inference_mode():
        hidden = text_model(**encoded).last_hidden_state
        eos_positions = encoded['attention_mask'].sum(dim=1) - 1
        prototypes = hidden[torch.arange(len(label_names), device=device), eos_positions].cpu().half()
    if prototypes.shape != (len(label_names), 512) or not torch.isfinite(prototypes).all():
        raise ValueError(f'Invalid CLIP label prototypes: {prototypes.shape}')

    metadata = {
        'complete': len(data) == len(records),
        'count': len(data),
        'labels': label_names,
        'label_count': len(label_names),
        'coordinate_format': 'normalized_xyxy',
        'model': 'openai/clip-vit-base-patch16',
        'prototype_feature': 'CLIPTextModel.last_hidden_state at EOS',
        'prototype_extraction': 'fp32',
        'prototype_storage': 'fp16',
        'min_confidence': args.min_confidence,
        'max_regions': args.max_regions,
        'skipped_detections': skipped,
        'clipped_boxes': clipped,
        'truncated_detections': truncated,
        'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
        'dataset_sha256': hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
    }
    destination = Path(args.output)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.partial')
    torch.save({'data': data, 'label_prototypes': prototypes, 'metadata': metadata}, temporary)
    temporary.replace(destination)
    destination.with_suffix('.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
