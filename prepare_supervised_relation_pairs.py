"""Download official labels (not images), mask unknown links and exclude caption heldout."""
import argparse
import hashlib
import json
import time
import urllib.request
import zipfile
from pathlib import Path
from captioning.supervised_relations import (RELATIONS, build_pairs, diagnostic_subset,
                                             pair_statistics)

VCOCO_REVISION = '489cc4db74f2f10ab4b134f67da3874afbf245ab'
COCO_URL = 'https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip'


def download(url, path):
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    print('Downloading:', url, flush=True)
    started = last = time.monotonic()
    total = 0
    with urllib.request.urlopen(url, timeout=90) as response, temporary.open('wb') as writer:
        while block := response.read(4 * 1024**2):
            writer.write(block)
            total += len(block)
            now = time.monotonic()
            if now - last >= 10:
                print(f'Downloaded {total/1024**2:.1f} MiB | {now-started:.0f}s', flush=True)
                last = now
    temporary.replace(path)
    print(f'Download complete: {total/1024**2:.1f} MiB', flush=True)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while block := handle.read(4 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption-json', required=True)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--instances', default='', help='Optional explicit combined COCO instance JSON')
    parser.add_argument('--train-pairs', type=int, default=512)
    parser.add_argument('--val-pairs', type=int, default=128)
    args = parser.parse_args()
    work, output = Path(args.work_dir), Path(args.output)
    work.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    caption = json.loads(Path(args.caption_json).read_text(encoding='utf-8'))['images']
    annotations, inputs = {}, {'caption_json_sha256': file_hash(args.caption_json)}
    for split in ('train', 'val'):
        path = work / f'vcoco_{split}.json'
        url = f'https://raw.githubusercontent.com/s-gupta/v-coco/{VCOCO_REVISION}/data/vcoco/vcoco_{split}.json'
        download(url, path)
        annotations[split] = json.loads(path.read_text(encoding='utf-8'))
        inputs[f'vcoco_{split}_sha256'] = file_hash(path)
    ids = {iid for actions in annotations.values() for a in actions for iid in a['image_id']}
    if args.instances:
        instances = json.loads(Path(args.instances).read_text(encoding='utf-8'))
        inputs['instances_sha256'] = file_hash(args.instances)
    else:
        archive = work / 'annotations_trainval2014.zip'
        download(COCO_URL, archive)
        inputs['coco_archive_sha256'] = file_hash(archive)
        instances = {'images': [], 'annotations': [], 'categories': []}
        with zipfile.ZipFile(archive) as handle:
            for split in ('train', 'val'):
                print('Reading COCO instance annotations:', split, flush=True)
                data = json.loads(handle.read(f'annotations/instances_{split}2014.json'))
                instances['images'] += [x for x in data['images'] if x['id'] in ids]
                instances['annotations'] += [
                    {k: x[k] for k in ('id', 'image_id', 'category_id', 'bbox', 'iscrowd')}
                    for x in data['annotations'] if x['image_id'] in ids]
                if instances['categories'] and instances['categories'] != data['categories']:
                    raise ValueError('COCO category vocabularies differ.')
                instances['categories'] = data['categories']
                del data
    summary = {'variant': 'instance_contrast_alignment_diagnostic_v1',
               'relations': RELATIONS, 'vcoco_revision': VCOCO_REVISION, 'inputs': inputs,
               'unknown_label': -1, 'source': 'V-COCO train/val roles + COCO 2014 instances',
               'scope': 'GT boxes, six action roles; no wearing; no detector performance claim',
               'seed': 42}
    splits = {}
    for split, limit in [('train', args.train_pairs), ('val', args.val_pairs)]:
        rows, excluded = build_pairs(annotations[split], instances, caption)
        splits[split] = diagnostic_subset(rows, limit)
        if not splits[split]:
            raise ValueError(f'Empty {split} after caption-heldout exclusion.')
        path = output / f'{split}_pairs.json'
        path.write_text(json.dumps(splits[split], indent=2), encoding='utf-8')
        summary[split] = {'full': pair_statistics(rows), 'diagnostic': pair_statistics(splits[split]),
                          'excluded': excluded, 'pair_json_sha256': file_hash(path)}
    train_ids = {r['image_id'] for r in splits['train']}
    val_ids = {r['image_id'] for r in splits['val']}
    if train_ids & val_ids:
        raise ValueError('Relation train and validation image overlap.')
    summary['image_overlap'] = 0
    summary['caption_heldout_overlap'] = 0
    (output / 'pair_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('SUPERVISED PAIR DATA READY\n' + json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
