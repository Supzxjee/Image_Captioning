"""Manual GPU shard: frozen UPT native detections -> human-role audit journal."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time


def run(args):
    import torch
    from PIL import Image
    from captioning.inference_journal import InferenceJournal
    from captioning.upt_reference import load_reference, prepare_image, infer
    from captioning.vcoco_relation_audit import audit_image
    cfg = json.loads(Path(args.config).read_text())
    torch.set_num_threads(2); torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; no silent CPU inference.')
    device = 'cuda'
    records = cfg['images'][args.rank::2]
    output = Path(cfg['output'])
    journal = InferenceJournal(output / f'upt_{args.rank}.jsonl',
                               {'rank': args.rank, 'configuration': cfg}, [r['image_id'] for r in records])
    print(f'Shard {args.rank}: {len(journal.records)}/{len(records)} records ready', flush=True)
    if len(journal.records) == len(records):
        return
    metadata = json.loads((Path(cfg['upt']) / 'hicodet/instances_train2015.json').read_text())
    correspondence = [[] for _ in metadata['objects']]
    for _, obj, verb in metadata['correspondence']:
        correspondence[obj].append(verb)
    model, runtime = load_reference(cfg['upt'], cfg['pocket'], cfg['checkpoint'], correspondence)
    model.to(device)
    print('Runtime:', json.dumps(runtime), flush=True)
    instances = json.loads(Path(cfg['instances']).read_text())
    images = {r['id']: r for r in instances['images']}
    categories = {r['id']: r['name'] for r in instances['categories']}
    annotations = defaultdict(list)
    for row in instances['annotations']:
        annotations[row['image_id']].append(row)
    started = time.monotonic()
    for row in records[len(journal.records):]:
        path = Path(cfg['base_path']) / row['filepath'] / row['filename']
        with Image.open(path) as image:
            if image.size != (row['width'], row['height']):
                raise ValueError('Image size differs from annotation.')
            prediction = infer(model, prepare_image(image).to(device))
        height, width = prediction['size'].tolist()
        detections = [{'label': metadata['objects'][int(label)].replace('_', ' '),
                       'confidence': float(score),
                       'box': [max(0., min(1., float(v) / scale)) for v, scale in zip(box, [width, height, width, height])]}
                      for box, label, score in zip(prediction['boxes'], prediction['labels'], prediction['scores'])]
        claims = []
        for index, (s, o) in enumerate(zip(prediction['subject_indices'].tolist(), prediction['object_indices'].tolist())):
            obj = int(prediction['labels'][o])
            for predicate, verb in [('holding', 'hold'), ('riding', 'ride')]:
                column = metadata['verbs'].index(verb)
                if column in correspondence[obj]:
                    claims.append({'subject': s, 'object': o, 'predicate': predicate,
                                   'score': float(prediction['logits'][index, column].sigmoid())})
        native = {'detections': detections, 'claims': claims, 'native_pairs': len(prediction['subject_indices'])}
        audited = audit_image(row, native, annotations[row['image_id']], images[row['image_id']], categories)
        journal.append({**audited, 'group': row['group']})
        if len(journal.records) % 10 == 0 or len(journal.records) == len(records):
            print(f'Shard {args.rank}: {len(journal.records)}/{len(records)} | {time.monotonic()-started:.1f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--rank', required=True, type=int, choices=[0, 1])
    run(parser.parse_args())
