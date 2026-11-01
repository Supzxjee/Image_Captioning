"""User-run CPU asset inventory; no Torch, neural models or feature arrays read."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time
import traceback

import numpy as np

from data import vocabulary

DATA_SHA = '2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
DETECTION_SHA = '498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def text(value):
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def coco_id(filename):
    match = re.fullmatch(r'COCO_(?:train|val|test)2014_(\d+)\.jpg', filename)
    if not match:
        raise ValueError('Invalid COCO filename: ' + filename)
    return int(match[1])


def detection_index(source):
    if isinstance(source, dict) and 'data' in source:
        source = source['data']
    if not isinstance(source, dict):
        raise ValueError('Detector source must map filenames to records')
    result = {}
    for name, record in source.items():
        filename = Path(name).name
        if filename in result:
            raise ValueError('Duplicate detector basename: ' + filename)
        objects = record if isinstance(record, list) else record['objects']
        if not isinstance(objects, list):
            raise ValueError('Missing/malformed objects: ' + filename)
        for obj in objects:
            box = np.asarray(obj['bbox'], dtype=float)
            score = float(obj['conf'])
            if box.shape != (4,) or not np.isfinite(box).all() or not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Nonfinite/invalid detector value: ' + filename)
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError('Nonpositive xyxy area: ' + filename)
        result[filename] = objects
    return result


def inspect(config, out, report):
    import h5py
    from PIL import Image
    paths = {k: Path(config[k]) for k in ('dataset_json', 'detections', 'visual_cache', 'image_root')}
    for key, expected in [('dataset_json', DATA_SHA), ('detections', DETECTION_SHA)]:
        print('Hashing input:', key, flush=True)
        actual = sha(paths[key])
        if actual != expected:
            raise ValueError('Pinned input differs: ' + key)
        report[key + '_sha256'] = actual
    dataset = json.loads(paths['dataset_json'].read_text(encoding='utf-8'))
    images = dataset['images']
    mapping = {}
    for row in images:
        identity = coco_id(row['filename'])
        if identity != int(row['cocoid']) or identity in mapping:
            raise ValueError('Dataset duplicate/mismatched COCO ID')
        mapping[identity] = row
    split_counts = dict(Counter(row['split'] for row in images))
    if split_counts != dict(train=82783, restval=30504, val=5000, test=5000):
        raise ValueError('Unexpected Karpathy split inventory')
    report['split_counts'] = split_counts
    report['shards'] = []
    seen, shared_profile = set(), None
    shards = sorted(paths['visual_cache'].rglob('*.h5'))
    if not shards:
        raise FileNotFoundError('No raw visual H5 shards')
    for path in shards:
        print('Reading H5 metadata and ID tables:', path.name, flush=True)
        with h5py.File(path, 'r') as f:
            if not bool(f.attrs.get('complete', False)) or text(f.attrs.get('id_key', '')) != 'coco_id':
                raise ValueError('H5 must explicitly declare complete and coco_id')
            meta = json.loads(text(f.attrs['metadata_json']))
            expected = dict(model_id='openai/clip-vit-base-patch16', output='vision_model.last_hidden_state',
                            preprocessing='bilinear', image_size=[224, 224], antialias=True,
                            dataset_json_sha256=DATA_SHA)
            for key, value in expected.items():
                if meta.get(key) != value:
                    raise ValueError('H5 profile differs: ' + key)
            for key, expected_values in [('mean', [.48145466, .4578275, .40821073]),
                                         ('std', [.26862954, .26130258, .27577711])]:
                np.testing.assert_allclose(meta[key], expected_values, atol=1e-8, rtol=0)
            profile = {k: meta.get(k) for k in (*expected, 'model_revision', 'mean', 'std', 'compute_dtype', 'storage_dtype', 'autocast')}
            if shared_profile is not None and profile != shared_profile:
                raise ValueError('Mixed feature extraction profiles')
            shared_profile = profile
            features, ids, names = f['features'], f['imgids'], f['filename']
            if features.shape != (len(ids), 197, 768) or features.dtype.kind != 'f' or features.dtype.itemsize not in (2, 4):
                raise ValueError('Expected raw CLIP [rows,197,768] fp16/fp32')
            if ids.ndim != 1 or ids.dtype.kind not in 'iu' or len(names) != len(ids):
                raise ValueError('Invalid ID/filename tables')
            current = [int(i) for i in ids[:]]
            if len(set(current)) != len(current) or seen.intersection(current):
                raise ValueError('Duplicate feature IDs')
            for identity, filename in zip(current, names[:]):
                if identity not in mapping or text(filename) != mapping[identity]['filename']:
                    raise ValueError('H5 ID/filename differs from source dataset')
            seen.update(current)
            report['shards'].append(dict(name=path.name, bytes=path.stat().st_size,
                                         shape=list(features.shape), dtype=str(features.dtype)))
    if seen != set(mapping):
        raise ValueError('Raw cache does not cover exactly Karpathy images')
    report['visual_ids_verified'] = len(seen)
    report['feature_profile'] = shared_profile
    print('Reading detector records', flush=True)
    detections = detection_index(json.loads(paths['detections'].read_text(encoding='utf-8')))
    missing = [row['filename'] for row in images if row['filename'] not in detections]
    if missing:
        raise ValueError(f'Missing detector records: {len(missing)}; examples {missing[:5]}')
    counts = Counter(len(detections[row['filename']]) for row in images)
    report['detection_count_histogram'] = dict(sorted(counts.items()))
    report['empty_detector_records'] = counts[0]
    report['images_above_cap150'] = sum(v for n, v in counts.items() if n > 150)
    report['dimension_samples'] = []
    # Header-only reads; this sample is not all-image geometry validation.
    for split in ('train', 'restval', 'val', 'test'):
        chosen = sorted((r for r in images if r['split'] == split), key=lambda r: hashlib.sha256(r['filename'].encode()).hexdigest())[:3]
        for row in chosen:
            directory = row.get('filepath', 'train2014' if 'train2014' in row['filename'] else 'val2014')
            candidates = [paths['image_root']/directory/row['filename'], paths['image_root']/row['filename']]
            existing = [p for p in candidates if p.is_file()]
            if not existing:
                raise FileNotFoundError('RGB sample missing; set IMAGE_ROOT to images directory: ' + row['filename'])
            with Image.open(existing[0]) as image:
                width, height = image.size
            boxes = detections[row['filename']]
            outside = sum(any(v < 0 for v in o['bbox'][:2]) or o['bbox'][2] > width or o['bbox'][3] > height for o in boxes)
            report['dimension_samples'].append(dict(filename=row['filename'], width=width, height=height,
                detections=len(boxes), boxes_needing_boundary_clip=outside))
    words = vocabulary(dataset)
    vocabulary_path = out/'vocabulary.json'
    vocabulary_path.write_text(json.dumps(words, ensure_ascii=False) + '\n', encoding='utf-8')
    report['vocabulary'] = dict(size=len(words), sha256=sha(vocabulary_path), train_only=True,
                                 caption_events=sum(len(r['sentences']) for r in images if r['split'] in ('train', 'restval')))
    report['status'] = 'ASSET_INVENTORY_COMPLETE'


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    args = parser.parse_args(); config = json.loads(Path(args.config).read_text())
    out = Path(config['output']); out.mkdir(parents=True, exist_ok=True)
    report = dict(status='INCOMPLETE', feature_arrays_read=False, full_h5_payload_hashes_checked=False,
                  model_executed=False, training_ready=False, pos_banks_created=False,
                  detector_producer_provenance='UNKNOWN: coordinate semantics/threshold/NMS/top-K not recovered',
                  all_image_dimensions_verified=False)
    start = time.monotonic()
    try:
        inspect(config, out, report)
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['seconds'] = time.monotonic() - start
        (out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
