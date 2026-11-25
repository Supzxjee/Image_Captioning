"""Sealed metadata staging and readonly patch access; no model import."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from real_bank_data import read, save, sha


def id_digest(ids):
    return hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest()


def prepare(cfg):
    from scst_dataset_mount import materialize_part0
    from scst_cache import mount
    from semantic_data import ARRAYS
    expected = read(Path(__file__).with_name('CACHE_PARTS.json'))
    dataset, out = Path(cfg['cache_root']).resolve(), Path(cfg['output'])
    spec = read(dataset/'cache_manifest.json')
    base = Path(cfg['metadata_root'])
    base.mkdir(parents=True)
    part0 = base/'part0'
    links = materialize_part0(dataset, part0, expected['parts'], expected['ids_sha256'])
    mount(cfg, part0, out, linked_patches=links)
    roots = [part0/'train']
    with zipfile.ZipFile(dataset/'cache_metadata_zip.bin') as archive:
        for part, exp in zip(spec['parts'][1:], expected['parts'][1:], strict=True):
            number = exp['part']
            dest = base/f'part{number}'
            dest.mkdir()
            raw = archive.read(f"part{number}/{exp['receipt_name']}")
            if hashlib.sha256(raw).hexdigest() != exp['receipt_sha256']:
                raise ValueError('Changed original shard receipt')
            receipt = json.loads(raw)
            if receipt['test_used'] or receipt['source_sha256'] != expected['source_sha256'] or receipt['train_start'] != exp['train_start'] or receipt['train_stop'] != exp['train_stop']:
                raise ValueError('Wrong shard scope')
            files = receipt['files_sha256']
            if set(files) != set(part['files']):
                raise ValueError('Incomplete shard mapping')
            for name, digest in files.items():
                entry, target = part['files'][name], dest/name
                if not target.resolve().is_relative_to(dest.resolve()) or entry['sha256'] != digest:
                    raise ValueError('Invalid shard member')
                target.parent.mkdir(parents=True, exist_ok=True)
                if name == 'train/pilot_patches.npy':
                    filename = f'part{number}_train_pilot_patches.npy'
                    if entry.get('file') != filename:
                        raise ValueError('Wrong patch mapping')
                    source = (dataset/filename).resolve(strict=True)
                    if not source.is_relative_to(dataset):
                        raise ValueError('Patch escapes input dataset')
                    target.symlink_to(source)
                else:
                    member = f'part{number}/{name}'
                    if entry.get('bundle') != 'cache_metadata_zip.bin' or entry.get('member') != member:
                        raise ValueError('Wrong metadata mapping')
                    with archive.open(member) as src, target.open('xb') as dst:
                        shutil.copyfileobj(src, dst, length=8*1024**2)
                    if sha(target) != digest:
                        raise ValueError('Changed shard metadata')
            roots.append(dest/'train')
    all_ids = []
    vector_digest = sha(roots[0]/'sem_text_vectors.npy')
    for root, exp in zip(roots, expected['parts'], strict=True):
        size = exp['train_stop']-exp['train_start']
        ids = np.load(root/'pilot_ids.npy', allow_pickle=False)
        patches = np.load(root/'pilot_patches.npy', mmap_mode='r', allow_pickle=False)
        if ids.shape != (size,) or ids.dtype.kind not in 'iu' or patches.shape != (size, 196, 768) or patches.dtype != np.float16:
            raise ValueError('Wrong TRAIN shard headers')
        if patches.offset+patches.nbytes != (root/'pilot_patches.npy').stat().st_size or id_digest(ids.tolist()) != exp['ids_sha256']:
            raise ValueError('Wrong patch size or ordered IDs')
        if sha(root/'sem_text_vectors.npy') != vector_digest:
            raise ValueError('Text dictionaries differ between shards')
        for name, (shape, dtype) in ARRAYS.items():
            array = np.load(root/f'sem_{name}.npy', mmap_mode='r', allow_pickle=False)
            if array.shape != (size, *shape) or array.dtype != dtype:
                raise ValueError('Wrong semantic array: '+name)
        all_ids.extend(ids.tolist())
    valid = read(out/'test_lock.json')['expected_test_ids']
    if len(all_ids) != 113287 or len(set(all_ids)) != 113287:
        raise ValueError('Wrong full TRAIN uniqueness')
    if len(all_ids) != 113287 or id_digest(all_ids) != expected['ids_sha256'] or set(all_ids) & set(valid):
        raise ValueError('Wrong TRAIN order or TRAIN/VALID overlap')
    cfg['train_roots'] = list(map(str, roots))
    cfg['train_stage'] = str(roots[0])
    cfg['cache_manifest_sha256'] = sha(dataset/'cache_manifest.json')
    save(out/'full_epoch_cache.json', dict(train_images=113287, valid_images=5000, test_used=False,
        cache_manifest_sha256=cfg['cache_manifest_sha256'], ordered_train_ids_sha256=id_digest(all_ids),
        metadata_payloads_SHA_checked=True, original_source_receipts_checked=True,
        patch_full_content_SHA_rehashed=False, patch_shape_size_and_readonly_dataset_mapping_checked=True,
        patch_copies=False, train_roots=cfg['train_roots']))


class FullTrain:
    def __init__(self, roots):
        from semantic_data import SemanticContext
        self.roots = list(map(Path, roots))
        self.patches = [np.load(p/'pilot_patches.npy', mmap_mode='r', allow_pickle=False) for p in self.roots]
        self.ids = [np.load(p/'pilot_ids.npy', allow_pickle=False) for p in self.roots]
        self.context = [SemanticContext(p) for p in self.roots]
        self.stops = np.cumsum([len(i) for i in self.ids])

    def batch(self, rows, device):
        import torch
        p, x, ids = [], [], []
        # Batch4; preserve row order even when crossing a shard boundary.
        for row in rows:
            part = int(np.searchsorted(self.stops, row, side='right'))
            offset = int(row-(self.stops[part-1] if part else 0))
            x.append(np.asarray(self.patches[part][offset], np.float32))
            ids.append(int(self.ids[part][offset]))
            p.append(self.context[part].images([offset], device))
        prompt = {name: torch.cat([item[name] for item in p]) for name in p[0]}
        return torch.from_numpy(np.stack(x)).to(device), prompt, ids
