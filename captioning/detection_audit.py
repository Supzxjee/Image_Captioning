"""Statistical and structural audit for saved object-detector outputs."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from statistics import mean, median


def _filename(value):
    return PurePosixPath(str(value).replace('\\', '/')).name


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _summary(values):
    return {
        'count': len(values),
        'mean': mean(values) if values else None,
        'median': median(values) if values else None,
        'p10': _percentile(values, 0.10),
        'p90': _percentile(values, 0.90),
        'min': min(values) if values else None,
        'max': max(values) if values else None,
    }


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def dataset_records(path):
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    images = payload['images'] if isinstance(payload, dict) else payload
    records = {}
    for image in images:
        name = image['filename']
        if name in records:
            raise ValueError(f'Duplicate dataset filename: {name}')
        records[name] = image
    return records


def index_source(source):
    if isinstance(source, dict) and isinstance(source.get('data'), dict):
        source = source['data']
    if not isinstance(source, dict):
        raise ValueError('Detection source must map image paths or filenames to entries.')
    indexed = {}
    for key, value in source.items():
        name = _filename(key)
        if name in indexed:
            raise ValueError(f'Ambiguous source filename: {name}')
        indexed[name] = value
    return indexed


def _detections(entry, objects_field):
    detections = entry if isinstance(entry, list) else entry.get(objects_field)
    if not isinstance(detections, list):
        raise ValueError(f'Expected a detection list in field {objects_field!r}.')
    return detections


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def audit_detections(source, records=None, objects_field='objects', name_key='label',
                     bbox_key='bbox', confidence_key='conf', image_size=None,
                     duplicate_iou=0.8):
    """Return aggregate report and per-image rows without modifying detections.

    ``image_size`` may be a callable accepting ``(filename, dataset_record)`` and
    returning ``(width, height)``. Bounds and normalized-area checks are skipped
    when it is omitted.
    """
    indexed = index_source(source)
    records = records or {name: {'filename': name, 'split': ''} for name in indexed}
    label_counts, confidences, detections_per_image = Counter(), [], []
    issue_counts, split_coverage = Counter(), defaultdict(lambda: Counter(total=0, present=0))
    per_image = []

    for name, record in records.items():
        split = record.get('split', 'unknown') or 'unknown'
        split_coverage[split]['total'] += 1
        entry = indexed.get(name)
        row_issues = Counter()
        if entry is None:
            row_issues['missing_source'] += 1
            detections = []
        else:
            split_coverage[split]['present'] += 1
            try:
                detections = _detections(entry, objects_field)
            except (AttributeError, ValueError):
                detections = []
                row_issues['invalid_entry'] += 1

        size = None
        if image_size is not None:
            try:
                size = image_size(name, record)
                if (not isinstance(size, (tuple, list)) or len(size) != 2 or
                        not all(_number(x) and x > 0 for x in size)):
                    raise ValueError('invalid size')
            except (OSError, ValueError, TypeError):
                row_issues['image_size_unavailable'] += 1
                size = None

        valid_boxes = defaultdict(list)
        image_confidences = []
        for detection in detections:
            if not isinstance(detection, dict):
                row_issues['invalid_detection'] += 1
                continue
            label = detection.get(name_key)
            if not isinstance(label, str) or not label.strip():
                row_issues['invalid_label'] += 1
                label = '<invalid>'
            else:
                label = ' '.join(label.lower().strip().split())
                label_counts[label] += 1

            confidence = detection.get(confidence_key)
            if confidence is None:
                row_issues['missing_confidence'] += 1
            elif not _number(confidence) or not 0 <= confidence <= 1:
                row_issues['invalid_confidence'] += 1
            else:
                confidence = float(confidence)
                confidences.append(confidence)
                image_confidences.append(confidence)
                if confidence < 0.25:
                    row_issues['confidence_below_0.25'] += 1
                if confidence < 0.50:
                    row_issues['confidence_below_0.50'] += 1

            bbox = detection.get(bbox_key)
            if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or
                    not all(_number(value) for value in bbox)):
                row_issues['invalid_bbox'] += 1
                continue
            box = tuple(float(value) for value in bbox)
            width, height = box[2] - box[0], box[3] - box[1]
            if width <= 0 or height <= 0:
                row_issues['nonpositive_bbox'] += 1
                continue
            valid_boxes[label].append(box)
            aspect = max(width / height, height / width)
            if aspect > 10:
                row_issues['extreme_aspect_bbox'] += 1
            if size:
                image_width, image_height = size
                if box[0] < 0 or box[1] < 0 or box[2] > image_width or box[3] > image_height:
                    row_issues['out_of_bounds_bbox'] += 1
                area_ratio = width * height / (image_width * image_height)
                if area_ratio < 0.001:
                    row_issues['tiny_bbox'] += 1
                if area_ratio > 0.90:
                    row_issues['near_full_image_bbox'] += 1

        duplicate_pairs = 0
        for boxes in valid_boxes.values():
            for i, first in enumerate(boxes):
                duplicate_pairs += sum(_iou(first, second) >= duplicate_iou
                                       for second in boxes[i + 1:])
        if duplicate_pairs:
            row_issues['same_label_duplicate_pairs'] += duplicate_pairs

        detections_per_image.append(len(detections))
        issue_counts.update(row_issues)
        per_image.append({
            'filename': name,
            'split': split,
            'detections': len(detections),
            'unique_labels': len(valid_boxes),
            'mean_confidence': mean(image_confidences) if image_confidences else '',
            'issues': sum(row_issues.values()),
            'issue_types': ';'.join(sorted(row_issues)),
            **dict(row_issues),
        })

    extras = sorted(set(indexed) - set(records))
    flagged = sorted((row for row in per_image if row['issues']),
                     key=lambda row: (-row['issues'], row['filename']))
    report = {
        'scope': {
            'dataset_images': len(records),
            'source_images': len(indexed),
            'missing_source_images': issue_counts['missing_source'],
            'extra_source_images': len(extras),
            'extra_source_examples': extras[:20],
        },
        'coverage_by_split': {split: dict(counts) for split, counts in sorted(split_coverage.items())},
        'detections': {
            'total': sum(detections_per_image),
            'images_without_detections': sum(count == 0 for count in detections_per_image),
            'per_image': _summary(detections_per_image),
            'unique_labels': len(label_counts),
            'top_labels': label_counts.most_common(30),
        },
        'confidence': {
            **_summary(confidences),
            'below_0.25': sum(value < 0.25 for value in confidences),
            'below_0.50': sum(value < 0.50 for value in confidences),
            'at_least_0.75': sum(value >= 0.75 for value in confidences),
        },
        'issues': dict(sorted(issue_counts.items())),
        'flagged_images': len(flagged),
        'most_flagged_images': flagged[:50],
        'interpretation': [
            'This audit finds coverage, schema and statistical anomalies; it does not prove visual correctness.',
            'Caption references omit visible objects, so they are not used as detector ground truth.',
            'Review a stratified sample of flagged and unflagged images before changing thresholds.',
        ],
    }
    return report, per_image, label_counts


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ['filename']
    preferred = ['filename', 'split', 'detections', 'unique_labels',
                 'mean_confidence', 'issues', 'issue_types']
    fields = [key for key in preferred if key in fields] + [key for key in fields if key not in preferred]
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

