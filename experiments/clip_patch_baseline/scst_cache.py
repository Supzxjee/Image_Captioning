"""CPU-build and GPU-consume reusable SCST input cache."""
from collections import Counter, OrderedDict
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import tarfile

from real_bank_data import read, save, sha

SCHEMA = 'csa_scst_inputs_v1'
SOURCE_SHA = 'b0c5bde0921a4c828ac094b264f502654224aa9365ee40cdce1c15c82fe9aeef'
VALID_META = ('test_lock.json', 'stage_receipt.json', 'test_references.json', 'vocabulary.json')
ASSETS = ('java8.tar.gz', 'chair.py', 'chair.pkl', 'reference_ptb.json')
REWARD = ('reward_refs.jsonl.gz', 'reward_df.jsonl.gz')


def build_reward(stage):
    from captioning.scst import TrainingCider

    stage = Path(stage)
    refs = {int(i): v for i, v in read(stage/'training_references.json').items()}
    scorer = TrainingCider(refs, reference_cache_size=2048)
    for name, values in zip(REWARD, (scorer.references.items(), scorer.df.items())):
        with gzip.open(stage/name, 'wt', encoding='utf8') as out:
            for row in values:
                out.write(json.dumps(row, ensure_ascii=False)+'\n')


def load_reward(stage):
    from captioning.scst import TrainingCider

    stage = Path(stage)
    with gzip.open(stage/REWARD[0], 'rt', encoding='utf8') as stream:
        refs = {int(i): tokens for i, tokens in map(json.loads, stream)}
    with gzip.open(stage/REWARD[1], 'rt', encoding='utf8') as stream:
        df = Counter({tuple(key): count for key, count in map(json.loads, stream)})
    if len(refs) != 113287 or not df:
        raise ValueError('Wrong cached TRAIN reward')
    scorer = TrainingCider.__new__(TrainingCider)
    scorer.references, scorer.df = refs, df
    scorer.log_count = math.log(len(refs))
    scorer.reference_cache_size = 2048
    scorer.vectors = OrderedDict()
    return scorer


def seal(root, train, valid, meta, images):
    root, train, valid, meta = map(Path, (root, train, valid, meta))
    if images not in (4000, 20000, 40000):
        raise ValueError('Unsupported cache schedule size')
    (root/'train').mkdir()
    for name in ('pilot_ids.npy', 'pilot_sequences.npy', 'pilot_patches.npy', *REWARD):
        shutil.move(train/name, root/'train'/name)
    for path in train.glob('sem_*.npy'):
        shutil.move(path, root/'train'/path.name)
    (root/'valid').mkdir()
    for path in valid.glob('*.npy'):
        shutil.move(path, root/'valid'/path.name)
    for name in VALID_META:
        shutil.copyfile(meta/name, root/name)
    lock = read(root/'test_lock.json')
    lock.pop('checkpoint', None)  # The consumer binds the actual mounted binary.
    if lock['checkpoint_sha256'] != SOURCE_SHA:
        raise ValueError('Wrong checkpoint lock')
    save(root/'test_lock.json', lock)
    files = {p.relative_to(root).as_posix(): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}
    save(root/'scst_cache_receipt.json', dict(schema=SCHEMA, source_sha256=SOURCE_SHA,
         train_images=images, train_start=0, train_stop=images, validation_images=5000, test_used=False, files_sha256=files))


def mount(cfg, root, destination):
    root, destination = Path(root), Path(destination)
    manifest = read(root/'scst_cache_receipt.json')
    if (manifest['schema'] != SCHEMA or manifest['source_sha256'] != SOURCE_SHA
            or manifest['train_images'] not in (4000, 20000, 40000)
            or manifest.get('train_start', 0) != 0
            or manifest['validation_images'] != 5000 or manifest['test_used']):
        raise ValueError('Wrong SCST cache identity')
    required = set((*VALID_META, *ASSETS, *(f'train/{n}' for n in REWARD),
                    'train/pilot_patches.npy', 'train/pilot_ids.npy', 'train/pilot_sequences.npy',
                    'train/sem_text_vectors.npy', 'valid/patches.npy'))
    if not required <= manifest['files_sha256'].keys():
        raise ValueError('Incomplete SCST cache')
    for name, digest in manifest['files_sha256'].items():
        path = root/name
        if not path.resolve().is_relative_to(root.resolve()) or sha(path) != digest:
            raise ValueError('Changed cache file: '+name)
    for name in VALID_META:
        shutil.copyfile(root/name, destination/name)
    lock = read(destination/'test_lock.json')
    lock['checkpoint'] = cfg['checkpoint_root']
    save(destination/'test_lock.json', lock)
    cfg.update(stage=str(root/'valid'), train_stage=str(root/'train'),
               cached_reference_ptb=str(root/'reference_ptb.json'))
    save(destination/'scst_cache_receipt.json', manifest)
    chair = Path(cfg['chair'])
    chair.mkdir()
    for name in ('chair.py', 'chair.pkl'):
        shutil.copyfile(root/name, chair/name)
    jre = destination/'jre'
    with tarfile.open(root/'java8.tar.gz') as t:
        t.extractall(jre, filter='data')
    java, = jre.rglob('bin/java')
    save(destination/'environment.json', dict(java=str(java), cached=True))


def seal_train_shard(root, train, start, stop):
    if (start, stop) not in ((40000,80000),(80000,113287)):
        raise ValueError('Wrong full-cache shard range')
    root, train = Path(root), Path(train)
    dest = root/'train'
    dest.mkdir(parents=True)
    for name in ('pilot_ids.npy','pilot_sequences.npy','pilot_patches.npy'):
        shutil.move(train/name,dest/name)
    for path in train.glob('sem_*.npy'):
        shutil.move(path,dest/path.name)
    files = {p.relative_to(root).as_posix():sha(p) for p in sorted(dest.iterdir())}
    save(root/'scst_train_shard_receipt.json',dict(schema=SCHEMA,source_sha256=SOURCE_SHA,
         train_images=stop-start,train_start=start,train_stop=stop,validation_images=0,
         test_used=False,files_sha256=files))
