"""Prepare and check inputs for the frozen CSA-only rerank replay."""
import argparse
import hashlib
import json
from pathlib import Path

from captioning.object_consistency import index_detections
from patch_beam import text_from_ids
from real_bank_data import read, save, sha
from test199_data import DATA_SHA

HERE = Path(__file__).resolve().parent
POLICIES = ('decoder_rank0', 'clip_object_fixed')
DETECTION_SHA = '498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c'


def prepare(cfg):
    out = Path(cfg['output'])
    test = Path(cfg['test_output'])
    source = read(test/'test_lock.json')
    policy = read(HERE/'RERANK_TEST_INPUTS.json')
    replay = read(HERE/'REPLAY_RECEIPTS.json')
    if source['checkpoint_sha256'] != replay['checkpoint_sha256']:
        raise ValueError('Changed CSA checkpoint')
    for rank, digest in enumerate(replay['rank_caption_sha256']):
        if sha(test/f'captions_rank{rank}.json') != digest:
            raise ValueError('Beam rank 0 replay differs from original TEST')
    if sha(cfg['dataset_json']) != DATA_SHA or sha(cfg['detections']) != DETECTION_SHA:
        raise ValueError('Changed COCO metadata or detector')
    records = sorted((r for r in read(cfg['dataset_json'])['images'] if r['split'] == 'test'),
                     key=lambda r: int(r['cocoid']))
    if [int(r['cocoid']) for r in records] != source['expected_test_ids']:
        raise ValueError('Wrong TEST image cohort')
    detections = index_detections(read(cfg['detections']))
    root = Path(cfg['dataset_json']).parent.resolve()
    names = {r['filename'] for r in records}
    paths = {}
    for p in root.rglob('*.jpg'):
        if p.name in names:
            if p.name in paths or not p.resolve().is_relative_to(root):
                raise ValueError('Ambiguous TEST JPEG: '+p.name)
            paths[p.name] = p
    if set(paths) != names or not names <= detections.keys():
        raise ValueError('Missing TEST JPEGs or detector rows')
    inputs = []
    for row in records:
        name = row['filename']
        path = paths[name]
        inputs.append(dict(coco_id=int(row['cocoid']), image_path=str(path),
                           image_sha256=sha(path), detected=detections[name]))
    save(out/'inference_inputs.json', inputs)
    (out/'test_references.json').write_bytes((test/'test_references.json').read_bytes())
    save(out/'test_lock.json', dict(
        checkpoint_sha256=replay['checkpoint_sha256'], images=5000,
        clip=policy['clip'], selector=policy['frozen_selector'],
        inference_sha256=sha(out/'inference_inputs.json'),
        references_sha256=sha(out/'test_references.json')))


def checked_lock(out):
    out = Path(out)
    lock = read(out/'test_lock.json')
    policy = read(HERE/'RERANK_TEST_INPUTS.json')
    replay = read(HERE/'REPLAY_RECEIPTS.json')
    if (lock['clip'], lock['selector'], lock['checkpoint_sha256'], lock['images']) != (
            policy['clip'], policy['frozen_selector'], replay['checkpoint_sha256'], 5000):
        raise ValueError('Changed rerank policy or cohort')
    if sha(out/'inference_inputs.json') != lock['inference_sha256']:
        raise ValueError('Changed inference inputs')
    return lock


def read_beams(out, lock):
    out = Path(out)
    test = out.parent
    inputs = read(out/'inference_inputs.json')
    words = read(test/'vocabulary.json')
    by_id = {}
    for rank in (0, 1):
        path = test/f'beams_rank{rank}.json'
        receipt = read(test/f'worker{rank}.json')
        if (not receipt['complete'] or receipt['images'] != 2500
                or receipt['checkpoint_sha256'] != lock['checkpoint_sha256']
                or sha(path) != receipt['beams_sha256']):
            raise ValueError('Incomplete CSA beam replay')
        rows = read(path)
        original = read(test/f'captions_rank{rank}.json')
        if len(original) != 2500 or [r['coco_id'] for r in rows] != [r['coco_id'] for r in inputs[rank::2]]:
            raise ValueError('Wrong TEST beam coverage')
        for row, top in zip(rows, original):
            beams = row['candidates']
            if (len(beams) != 5 or row['coco_id'] != top['coco_id']
                    or beams[0] != dict(ids=top['ids'], caption=top['caption'], logprob=top['score'])):
                raise ValueError('Top-1 does not match beam candidates')
            if any(c['caption'] != text_from_ids(c['ids'], words) for c in beams):
                raise ValueError('Changed beam text')
            by_id[row['coco_id']] = row
    return [by_id[row['coco_id']] for row in inputs]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    prepare(read(parser.parse_args().config))
