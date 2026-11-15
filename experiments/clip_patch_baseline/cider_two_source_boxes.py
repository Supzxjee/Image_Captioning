from pathlib import Path
import hashlib
import json
import math

INPUT = Path('/kaggle/input/datasets/ducanh2403')
SOURCES = {
    'upt': (INPUT/'upt-prompts-full-all-actions-v2/prompt_cache.json',
            '24e7861a4aa0e7e731d67689dce195688096232bcbef1b4264034cf520d6957d'),
    'detector': (INPUT/'objectdetectionecache/objectdetectioncache.json',
                 '498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c'),
}
TARGETS = {239894: ('teddy bear', 426, 640), 487438: ('bed', 640, 480)}
OUTPUT = Path('/kaggle/working/cider_error_two_source_boxes.json')

def checked_json(path, expected):
    print('Checking', path.name, flush=True)
    with path.open('rb') as stream:
        if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
            raise ValueError(f'Unexpected input bytes: {path}')
        stream.seek(0)
        return json.load(stream)

def normalized_box(values, width, height):
    limits = (width, height, width, height)
    box = [min(max(float(value), 0), limit) / limit
           for value, limit in zip(values, limits, strict=True)]
    if not (box[0] < box[2] and box[1] < box[3]):
        raise ValueError('Invalid clipped ROI')
    return box

def iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = iw * ih
    return intersection / ((a[2]-a[0])*(a[3]-a[1]) +
                           (b[2]-b[0])*(b[3]-b[1]) - intersection)

def label(text):
    return ' '.join(text.lower().replace('_', ' ').split())

def trace(image_id, target, width, height, detected, upt, linked):
    if isinstance(detected, dict):
        detected = detected['objects']
    ordered = sorted(enumerate(detected), key=lambda x: (-float(x[1]['conf']), x[0]))[:150]
    rois = [dict(roi=i, label=label(obj['label']), confidence=float(obj['conf']),
                 bbox=normalized_box(obj['bbox'], width, height))
            for i, (_, obj) in enumerate(ordered)]
    if len(rois) != linked['roi_count']:
        raise ValueError(f'Reconstructed ROI count differs for {image_id}')
    for obj in linked['objects']:
        roi = rois[obj['roi']]
        if roi['label'] != obj['label'] or not all(
            math.isclose(a, b, abs_tol=1e-6, rel_tol=0)
            for a, b in zip(roi['bbox'], obj['roi_bbox'], strict=True)
        ):
            raise ValueError(f'Reconstructed box differs for {image_id} ROI {roi["roi"]}')
    target_rois = [r for r in rois if r['label'] == target]
    target_upt = [o for o in upt['objects'] if label(o['label']) == target]
    overlaps = [dict(upt_index=o['index'], roi=r['roi'], iou=iou(o['bbox'], r['bbox']))
                for o in target_upt for r in target_rois]
    return dict(coco_id=image_id, target=target, width=width, height=height, detector_rois=rois,
                upt_mode=upt['mode'], upt_objects=upt['objects'], upt_relations=upt['relations'],
                target_rois=target_rois, target_upt_objects=target_upt, target_pair_ious=overlaps,
                linked_prompt=linked)

def main():
    links = LINKED_ROWS
    wanted = {row['filename']: row for row in links.values()}
    detector = checked_json(*SOURCES['detector'])
    if 'data' in detector:
        detector = detector['data']
    selected = {}
    for path, record in detector.items():
        name = Path(path).name
        if name in wanted:
            if name in selected:
                raise ValueError(f'Duplicate detector filename {name}')
            selected[name] = record
    del detector
    upt = checked_json(*SOURCES['upt'])
    rows = []
    for image_id, config in TARGETS.items():
        linked = links[image_id]
        name = linked['filename']
        rows.append(trace(image_id, *config, selected[name], upt[name], linked))
    result = dict(sources={name: expected for name, (_, expected) in SOURCES.items()}, rows=rows)
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)+'\n', encoding='utf-8')
    print(OUTPUT)
