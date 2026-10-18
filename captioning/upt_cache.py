"""Small, split-separated, immutable cache shards for frozen UPT inference."""
import gzip
import hashlib
import json
import math
from pathlib import Path

from .inference_journal import InferenceJournal


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def image_plan(dataset, shard_size=500):
    if shard_size < 1:
        raise ValueError('Invalid shard size.')
    groups = {name: [] for name in ('train', 'val', 'test')}
    ids, filenames = set(), set()
    for row in dataset['images']:
        iid, filename = int(row['cocoid']), row['filename']
        if iid in ids or filename in filenames or int(Path(filename).stem.rsplit('_', 1)[1]) != iid:
            raise ValueError('Duplicate or inconsistent COCO identity.')
        split = row['split']
        if split not in ('train', 'restval', 'val', 'test'):
            raise ValueError('Unknown caption split.')
        # Do not consume reference captions or any object/relation labels.
        groups['train' if split in ('train', 'restval') else split].append({
            'image_id': iid, 'filename': filename, 'filepath': row['filepath'], 'source_split': split})
        ids.add(iid); filenames.add(filename)
    tasks = []
    for split, rows in groups.items():
        rows.sort(key=lambda row: row['image_id'])
        for start in range(0, len(rows), shard_size):
            tasks.append({'name': f'{split}_{start // shard_size:04d}', 'split': split,
                          'images': rows[start:start + shard_size]})
    return tasks


def pack_prediction(row, prediction, vocabulary, image_sha256, original_size):
    """Store FP32 values as JSON numbers; sparse columns follow the HICO ontology."""
    boxes, labels, scores = prediction['boxes'], prediction['labels'], prediction['scores']
    if len(boxes) != len(labels) or len(labels) != len(scores):
        raise ValueError('Detection column lengths differ.')
    height, width = prediction['size']
    if min(height, width, *original_size) <= 0:
        raise ValueError('Invalid image dimensions.')
    if not all(len(box) == 4 and all(math.isfinite(x) for x in box) and box[0] < box[2] and box[1] < box[3] for box in boxes):
        raise ValueError('Invalid predicted box.')
    if not all(0 <= label < len(vocabulary['objects']) for label in labels) or not all(math.isfinite(s) and 0 <= s <= 1 for s in scores):
        raise ValueError('Invalid predicted label/score.')
    subjects, objects, logits = prediction['subject_indices'], prediction['object_indices'], prediction['logits']
    if len(subjects) != len(objects) or len(objects) != len(logits):
        raise ValueError('Pair column lengths differ.')
    official = prediction['official_output']
    native = {(s, o, verb): score for s, o, verb, score in zip(
        official['pairing'][0], official['pairing'][1], official['labels'], official['scores'])}
    pairs, raw, native_scores = [], [], []
    seen = set()
    for index, (s, o) in enumerate(zip(subjects, objects)):
        if (s, o) in seen or s == o or not 0 <= s < len(labels) or not 0 <= o < len(labels):
            raise ValueError('Invalid or duplicate native pair.')
        if vocabulary['objects'][labels[s]] != 'person' or len(logits[index]) != len(vocabulary['verbs']):
            raise ValueError('Invalid actor or verb dimension.')
        seen.add((s, o))
        verbs = vocabulary['verbs_by_object'][labels[o]]
        values = [logits[index][verb] for verb in verbs]
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Nonfinite logit.')
        ns = [native.get((s, o, verb), 0.) for verb in verbs]
        if not all(math.isfinite(value) and 0 <= value <= 1 for value in ns):
            raise ValueError('Invalid native score.')
        pairs.append([s, o]); raw.append(values); native_scores.append(ns)
    return {**row, 'image_sha256': image_sha256, 'original_size_wh': list(original_size),
            'resized_size_hw': [height, width], 'boxes_resized_xyxy': boxes,
            'labels': labels, 'object_scores': scores, 'pairs': pairs,
            'pair_logits': raw, 'native_scores': native_scores,
            'status': 'no_human_object_pairs' if not pairs else 'evaluated'}


def shard_protocol(task, signature):
    return {'signature': signature, 'name': task['name'], 'split': task['split'],
            'image_ids': [row['image_id'] for row in task['images']]}


def read_shard(path, task, signature):
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        header = json.loads(next(stream))
        if header != {'cache_shard': shard_protocol(task, signature)}:
            raise ValueError('Shard identity differs.')
        rows = [json.loads(line) for line in stream]
    if [r['image_id'] for r in rows] != [r['image_id'] for r in task['images']]:
        raise ValueError('Incomplete/reordered shard.')
    for actual, expected in zip(rows, task['images']):
        if any(actual[key] != value for key, value in expected.items()):
            raise ValueError('Shard image metadata differs.')
    return rows


def open_partial(output, task, signature):
    return InferenceJournal(Path(output) / f'{task["name"]}.partial.jsonl',
                            shard_protocol(task, signature), [row['image_id'] for row in task['images']])


def finish_shard(output, task, signature, rows):
    destination = Path(output) / f'{task["name"]}.jsonl.gz'
    if destination.exists():
        raise FileExistsError(destination)
    if [row['image_id'] for row in rows] != [row['image_id'] for row in task['images']]:
        raise ValueError('Cannot finish an incomplete shard.')
    temporary = destination.with_suffix('.gz.tmp')
    with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=3) as stream:
        stream.write(json.dumps({'cache_shard': shard_protocol(task, signature)}) + '\n')
        for row in rows:
            stream.write(json.dumps(row, separators=(',', ':'), allow_nan=False) + '\n')
    read_shard(temporary, task, signature)
    temporary.replace(destination)
    for path in (Path(output) / f'{task["name"]}.partial.jsonl', Path(output) / f'{task["name"]}.partial.protocol.json'):
        if path.exists():
            path.unlink()
    return destination
