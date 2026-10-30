"""User-run CPU metadata/feature copy only. Never constructs a captioner/encoder."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback
import urllib.request
import zipfile

import numpy as np
import kv_data as kd

EXPECTED = dict(
    checkpoint='5d92cd5cb48077c7332bb3ff9b295cad1df5d76ede818adabef28035f055fa9d',
    dataset_json='2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d',
    legacy_prompt='d7910e9bf87e3aad86d806c77ec336b34793e3bef68a3f662fdcb2e9456ca38d',
    detections='498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c')


def source_rows(work):
    receipt = json.loads((kd.ROOT/'experiments/relation_frames_inventory/sources/source_receipt.json').read_bytes())
    result = {}
    for split, spec in receipt['files'].items():
        target = work/(split+'.jsonl')
        if target.exists():
            raw = target.read_bytes()
        else:
            print('Fetching pinned VSR metadata:', split, flush=True)
            with urllib.request.urlopen(spec['url'], timeout=60) as response:
                raw = response.read(spec['bytes']+1)
            target.write_bytes(raw)
        if len(raw) != spec['bytes'] or hashlib.sha256(raw).hexdigest() != spec['sha256']:
            raise ValueError('VSR source identity mismatch: '+split)
        result[split] = [json.loads(line) for line in raw.splitlines() if line.strip()]
    return result, receipt


def coco_instances(wanted, presence, work):
    import ijson
    from fetch_vcoco_coco_instances import RangeFile, URL
    spec = presence['source_instances']
    if spec['source']['url'] != URL:
        raise ValueError('Unexpected annotation source')
    images, annotations, categories, members = {}, defaultdict(lambda: defaultdict(list)), {}, []
    remote = RangeFile(URL)
    if remote.length != spec['source']['bytes'] or remote.etag != spec['source']['etag']:
        raise ValueError('COCO archive ETag/size changed; review source before retry')
    with remote, zipfile.ZipFile(remote) as archive:
        for member in spec['members']:
            info = archive.getinfo(member['member'])
            if (info.CRC != int(member['crc32'], 16) or info.file_size != member['uncompressed_bytes']
                    or info.compress_size != member['compressed_bytes']):
                raise ValueError('COCO member changed')
            target = work/Path(member['member']).name
            print('Reading official instance metadata:', info.filename, flush=True)
            h = hashlib.sha256()
            with archive.open(info) as reader, target.open('xb') as writer:
                while block := reader.read(1024**2):
                    h.update(block)
                    writer.write(block)
            if h.hexdigest() != member['sha256']:
                raise ValueError('COCO uncompressed member SHA differs')
            with target.open('rb') as f:
                new_categories = {int(r['id']):r['name'] for r in ijson.items(f, 'categories.item', use_float=True)}
            if categories and categories != new_categories:
                raise ValueError('COCO category maps differ')
            categories = new_categories
            with target.open('rb') as f:
                for r in ijson.items(f, 'images.item', use_float=True):
                    if int(r['id']) in wanted:
                        if int(r['id']) in images:
                            raise ValueError('Duplicate GT image identity')
                        images[int(r['id'])] = {k:r[k] for k in ('id', 'file_name', 'width', 'height')}
            with target.open('rb') as f:
                for r in ijson.items(f, 'annotations.item', use_float=True):
                    if int(r['image_id']) in wanted:
                        annotations[int(r['image_id'])][categories[int(r['category_id'])]].append(
                            {k:r[k] for k in ('id', 'bbox', 'iscrowd')})
            members.append(member)
            # Only this invocation's newly created temporary file is removed.
            target.unlink()
    if not wanted.issubset(images):
        raise ValueError('GT metadata lacks wanted COCO images')
    return images, annotations, dict(url=URL, etag=remote.etag, members=members,
                                     actual_network_bytes=remote.transferred,
                                     full_archive_sha_computed=False, image_pixels_downloaded=False)


def detections(path, wanted, images):
    import ijson
    banks, seen = {}, set()
    with Path(path).open('rb') as f:
        for filename, row in ijson.kvitems(f, '', use_float=True):
            name = Path(filename.replace('\\', '/')).name
            iid = int(Path(name).stem.rsplit('_', 1)[-1])
            if iid not in wanted:
                continue
            if iid in seen or name != images[iid]['file_name']:
                raise ValueError('Detector duplicate/COCO filename mismatch')
            seen.add(iid)
            im = images[iid]
            banks[iid] = kd.bank(row['objects'], im['width'], im['height'])
    for iid in wanted:
        banks.setdefault(iid, [])  # Missing bank is unknown; not a negative image.
    return banks


def copy_features(cfg, out, rows, banks, main_count):
    import h5py
    import torch  # Trusted legacy tensor cache only; no model/checkpoint load.
    from captioning.semantic_prompts import align_prompt_cache
    from captioning.visual_cache import VisualCache
    from experiments.relation_frame_pretrain.frame_data import areas, geometry
    if shutil.disk_usage(out).free < 5*1024**3:
        raise RuntimeError('Need at least5GiB free for prepared cache')
    bundle = torch.load(cfg['legacy_prompt'], map_location='cpu', weights_only=False)
    prompts = align_prompt_cache(bundle['data'], [r['filename'] for r in rows[:main_count]])
    del bundle
    cache = VisualCache([cfg['visual_cache']])
    cache.require_ids([r['coco_id'] for r in rows], 'K/V main+aux CPU preparation')
    grouped = defaultdict(list)
    for index, row in enumerate(rows):
        file_index, position = cache.index[row['coco_id']]
        grouped[file_index].append((position, index))
    expected = dict(dataset_json_sha256=EXPECTED['dataset_json'], image_size=[224,224],
                    preprocessing='bilinear', model_id='openai/clip-vit-base-patch16',
                    output='vision_model.last_hidden_state', mean=[.48145466,.4578275,.40821073],
                    std=[.26862954,.26130258,.27577711], antialias=True,
                    compute_dtype='float32', autocast=False)
    count, start, sources = 0, time.monotonic(), []
    pairs = [(i,j) for i in range(6) for j in range(6) if i != j]
    try:
        with h5py.File(out/'inputs.h5', 'x') as target:
            target.attrs['complete'] = False
            n = len(rows)
            shapes = [('visual',(197,768),'f2'), ('prompt',(20,512),'f4'), ('mask',(20,),'u1'),
                      ('roi',(6,196),'f4'), ('union',(30,196),'f4'),
                      ('geometry',(30,8),'f4'), ('pair_valid',(30,),'u1')]
            for name, shape, dtype in shapes:
                target.create_dataset(name, (n,*shape), dtype=dtype)
            target.create_dataset('coco_ids', data=[r['coco_id'] for r in rows])
            target.create_dataset('caption_valid', data=np.arange(n)<main_count)
            journal = target.create_dataset('row_sha256', (n,), dtype='S64')
            for file_index, positions in sorted(grouped.items()):
                with h5py.File(cache.paths[file_index], 'r') as source:
                    info = json.loads(source.attrs['metadata_json'])
                    features = source['features']
                    if str(features.dtype) != 'float16' or any(info.get(k) != v for k,v in expected.items()):
                        raise ValueError('Native visual-cache metadata/dtype differs')
                    sources.append(dict(filename=Path(cache.paths[file_index]).name,
                                        bytes=Path(cache.paths[file_index]).stat().st_size, metadata=info))
                    for position, index in sorted(positions):
                        row = rows[index]
                        data = {name:np.zeros(shape,dtype=dtype) for name,shape,dtype in shapes}
                        data['visual'] = features[position]
                        if index < main_count:
                            p = prompts[row['filename']]
                            data['prompt'], data['mask'] = p['tokens'].float().numpy(), p['mask'].numpy()
                            if not data['mask'].any() or not np.isin(data['mask'], [0,1]).all():
                                raise ValueError('Invalid original prompt mask')
                        boxes = []
                        for slot, obj in enumerate(banks[row['coco_id']]):
                            b = np.asarray(obj['bbox_xyxy'], dtype='float64') / np.array([row['width'],row['height']]*2)
                            boxes.append(b)
                            data['roi'][slot] = areas(b)
                        for slot, (i,j) in enumerate(pairs):
                            if max(i,j) >= len(boxes):
                                continue
                            a,b = boxes[i],boxes[j]
                            data['union'][slot] = areas(np.r_[np.minimum(a[:2],b[:2]),np.maximum(a[2:],b[2:])])
                            data['geometry'][slot] = geometry(a,b)
                            data['pair_valid'][slot] = 1
                        h = hashlib.sha256()
                        for name, shape, dtype in shapes:
                            value = np.asarray(data[name], dtype=dtype)
                            if value.shape != shape or not np.isfinite(value).all():
                                raise ValueError('Bad staged tensor: '+name)
                            target[name][index] = value
                            actual = target[name][index]
                            if not np.array_equal(actual, value):
                                raise ValueError('Tensor copy differs: '+name)
                            h.update(actual.tobytes())
                        journal[index] = h.hexdigest().encode('ascii')
                        count += 1
                        if count == 1 or count % 250 == 0 or count == n:
                            target.flush()
                            print(f'Copied and verified {count}/{n}; {time.monotonic()-start:.1f}s', flush=True)
            target.attrs['complete'] = True
    finally:
        cache.close()
    return sources


def prepare(cfg):
    from captioning.object_consistency import COCO_LABELS
    start = time.monotonic()
    out, work = Path(cfg['output']), Path(cfg['work'])/'sources'
    if out.exists():
        raise FileExistsError('Preserve existing output; select a fresh output/session')
    out.mkdir(parents=True)
    work.mkdir(parents=True, exist_ok=True)
    kd.save(out/'protocol.json', dict(config=cfg, policy=kd.POLICY))
    try:
        for name, expected in EXPECTED.items():
            print('Hashing pinned input:', name, flush=True)
            if kd.digest(cfg[name]) != expected:
                raise ValueError('Wrong pinned input: '+name)
        meta = json.loads(gzip.decompress((kd.ROOT/'experiments/semantic_grounding_method/data/gain_metadata.json.gz').read_bytes()))
        presence = json.loads(gzip.decompress((kd.ROOT/'experiments/concept_mass_grounding/TRAIN_PRESENCE.json.gz').read_bytes()))
        cap = json.loads(Path(cfg['dataset_json']).read_bytes())['images']
        train = presence['train_rows']
        dev = meta['groups']['val']
        if len(train) != 10000 or len(dev) != 1000:
            raise ValueError('Changed parent cohort')
        selected = [train[int(i)] for i in np.random.default_rng(42).permutation(10000)[:4000]]
        main = []
        for row in selected+dev:
            original = cap[row['image_id']]
            if original['cocoid'] != row['coco_id'] or original['filename'] != row['filename']:
                raise ValueError('Caption identity mapping differs')
            if len(main) < 4000 and original['split'] not in ('train','restval'):
                raise ValueError('Caption TRAIN leakage')
            sentences = original['sentences'][:5]
            if len(sentences) != 5:
                raise ValueError('Need five native references')
            main.append(dict(row, references=[s['raw'] for s in sentences],
                             reference_tokens=[s['tokens'] for s in sentences]))
        caption_train_ids = {r['coco_id'] for r in main[:4000]}
        if len(caption_train_ids) != 4000 or caption_train_ids & {r['coco_id'] for r in dev}:
            raise ValueError('Duplicated/leaking caption cohort')
        vsr, vsr_receipt = source_rows(work)
        old = json.loads(gzip.decompress((kd.ROOT/'experiments/relation_frames_inventory/V2_MANIFEST.json.gz').read_bytes()))
        excluded = {kd.image_id(r) for split in ('dev','test') for r in vsr[split]}
        excluded |= {int(p[role]['image_id']) for p in old['heldout'] for role in ('positive','negative')}
        candidates, rejected = kd.candidates(vsr['train'], {r['cocoid']:r['split'] for r in cap},
                                               excluded, set(COCO_LABELS))
        wanted = {r['image_id'] for r in candidates} | {r['coco_id'] for r in main}
        print('Candidate original assertions:',len(candidates),'wanted images:',len(wanted),flush=True)
        images, annotations, coco_receipt = coco_instances(wanted, presence, work)
        banks = detections(cfg['detections'], wanted, images)
        matched = []
        for row in candidates:
            result, why = kd.match_record(row, images[row['image_id']],
                                          annotations[row['image_id']], banks[row['image_id']])
            if result is None:
                rejected[why] = rejected.get(why,0)+1
            else:
                matched.append(result)
        facts, quarantined = kd.merge_facts(matched)
        parts = kd.partition(facts, caption_train_ids)
        ready = kd.readiness(parts)
        kd.save(out/'auxiliary_facts.json', dict(partitions=parts,
                 multipositive={s:kd.multipositive_groups(r) for s,r in parts.items()},
                 quarantined=quarantined))
        kd.save(out/'statistics.json', dict(candidates=len(candidates), matched=len(matched),
                 rejected=rejected, quarantined=len(quarantined), readiness=ready))
        kd.save(out/'source_receipts.json', dict(vsr=vsr_receipt, coco=coco_receipt,
                 native_inputs=EXPECTED, image_pixels_downloaded=False,
                 official_held_truth_used=False, checkpoint_loaded=False))
        if not ready['passes_operational_gates']:
            kd.save(out/'cpu_preparation.json', dict(status='NEEDS_SOURCE_REVIEW_NO_TRAINING',
                     failures=ready['failures'], neural_forward=False, feature_extraction=False,
                     seconds=time.monotonic()-start))
            print('Data gates not ready; SEND REVIEW. No thresholds were relaxed.',flush=True)
            return
        auxiliary_ids = sorted({r['image_id'] for r in facts} - {r['coco_id'] for r in main})
        rows = [dict(r, width=images[r['coco_id']]['width'], height=images[r['coco_id']]['height']) for r in main]
        rows += [dict(coco_id=iid, filename=images[iid]['file_name'], width=images[iid]['width'],
                      height=images[iid]['height']) for iid in auxiliary_ids]
        kd.save(out/'metadata.json', dict(schema='relation-kv-preparation-v1', rows=rows,
                 train_indices=list(range(4000)), dev_indices=list(range(4000,5000)),
                 word2idx=meta['word2idx'], coco_to_row={str(r['coco_id']):i for i,r in enumerate(rows)},
                 predicates=list(kd.PREDICATES), inverse=kd.INVERSE, policy=kd.POLICY,
                 auxiliary_source='VSR observed original truth; equivalent view is transported not independent'))
        kd.save(out/'detector_boxes.json', {str(r['coco_id']):banks[r['coco_id']] for r in rows})
        sources = copy_features(cfg, out, rows, banks, 5000)
        payloads = {p.name:kd.digest(p) for p in sorted(out.iterdir()) if p.is_file()}
        kd.save(out/'cpu_preparation.json', dict(status='READY_FOR_MODEL_PREFLIGHT',
                 rows=len(rows), caption_train=4000, caption_dev=1000, auxiliary=ready['counts'],
                 policy=kd.POLICY, source_commit=cfg['source_commit'], visual_sources=sources,
                 neural_forward=False, feature_extraction=False, checkpoint_loaded=False,
                 payload_sha256=payloads, seconds=time.monotonic()-start))
        print('CPU data preparation complete. Model/CUDA/training have NOT run.',flush=True)
    except Exception as exc:
        kd.save(out/'cpu_failure.json', dict(error=repr(exc), traceback=traceback.format_exc()[-20000:],
                                            neural_forward=False, seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    prepare(json.loads(parser.parse_args().config.read_bytes()))
