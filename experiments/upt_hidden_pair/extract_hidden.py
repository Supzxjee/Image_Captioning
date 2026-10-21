"""User-run, two independent GPU workers; hooks native UPT before its classifier."""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile
import numpy as np

from hidden_data import WIDTH, SLOTS, SCHEMA, DATASET_SHA, UPT_SHA, BOX_THRESHOLD, locked_groups, select_pairs, validate_row
from pilot_data import digest, identity, save


def infer_hidden(model, pixels):
    import torch
    from captioning.upt_reference import infer
    captured = {}
    def hook(module, inputs, output):
        if captured:
            raise RuntimeError('Expected one classifier call per image.')
        captured.update(features=inputs[0].detach().float().cpu(), logits=output.detach().float().cpu())
    handle = model.interaction_head.box_pair_predictor.register_forward_hook(hook)
    try:
        prediction = infer(model, pixels)
    finally:
        handle.remove()
    if not captured or captured['features'].shape != (len(prediction['subject_indices']), WIDTH):
        raise ValueError('Classifier hook/native pair inventory differs.')
    torch.testing.assert_close(captured['logits'], prediction['logits'], rtol=0, atol=0)
    labels = prediction['labels']
    n_human = int((labels == 0).sum())
    if not bool((labels[:n_human] == 0).all()) or bool((labels[n_human:] == 0).any()):
        raise ValueError('UPT proposals were reordered; endpoint mapping would be wrong.')
    # Native head enumerates all (human, other instance) pairs, including human-human.
    expected = [(s, o) for s in range(n_human) for o in range(len(labels)) if s != o]
    actual = list(zip(prediction['subject_indices'].tolist(), prediction['object_indices'].tolist()))
    if actual != expected:
        raise ValueError('Native head pair order differs from the pinned implementation.')
    return prediction, captured['features'].numpy()


def arrays(file, rows):
    n = len(rows)
    specs = dict(features=((n, SLOTS, WIDTH), 'f2', 0), valid=((n, SLOTS), '?', False),
        pairs=((n, SLOTS, 2), 'i2', -1), native_rows=((n, SLOTS), 'i4', -1),
        labels=((n, SLOTS, 2), 'i2', -1), confidence=((n, SLOTS, 2), 'f4', 0),
        boxes=((n, SLOTS, 2, 4), 'f4', 0), native_pair_count=((n,), 'i4', 0),
        human_nonhuman_pair_count=((n,), 'i4', 0),
        eligible_pair_count=((n,), 'i4', 0), original_wh=((n, 2), 'i4', 0), resized_hw=((n, 2), 'i4', 0),
        image_sha256=((n,), 'S64', b''))
    for key, (shape, dtype, fill) in specs.items():
        file.create_dataset(key, shape=shape, dtype=dtype, chunks=(1, *shape[1:]), fillvalue=fill)
    file.create_dataset('image_ids', data=np.asarray([r['coco_id'] for r in rows], dtype=np.int64))
    file.create_dataset('dataset_indices', data=np.asarray([r['image_id'] for r in rows], dtype=np.int64))
    file.attrs.update(schema=SCHEMA, complete=False, completed_rows=0)


def worker(config, rank):
    import h5py
    import torch
    from PIL import Image
    from captioning.upt_reference import load_reference, prepare_image
    from captioning.scst_runtime import model_digest
    cfg = json.loads(Path(config).read_text())
    rows = cfg['rows'][rank::2]
    out = Path(cfg['output'])
    torch.set_num_threads(2)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if torch.cuda.device_count() != 1:
        raise RuntimeError('Each cache process must see exactly its assigned GPU.')
    model, runtime = load_reference(cfg['upt'], cfg['pocket'], cfg['upt_checkpoint'], cfg['vocabulary']['verbs_by_object'])
    model.to('cuda').eval()
    before = model_digest(model)
    save(out / f'runtime_rank{rank}.json', dict(reference=runtime, torch=str(torch.__version__),
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), model_sha256=before))
    with h5py.File(out / f'features_rank{rank}.h5', 'w') as f, (out / f'pairs_rank{rank}.jsonl').open('w', encoding='utf-8') as log:
        arrays(f, rows)
        for i, row in enumerate(rows):
            path = Path(cfg['images']) / row['filepath'] / row['filename']
            raw = path.read_bytes()
            with Image.open(io.BytesIO(raw)) as image:
                image.load()
                if image.getexif().get(274, 1) != 1:
                    raise ValueError('Nontrivial EXIF orientation; do not silently remap boxes.')
                original = image.size
                pixels = prepare_image(image).to('cuda')
            prediction, hidden = infer_hidden(model, pixels)
            plain = {key: prediction[key].numpy() for key in ('boxes', 'scores', 'labels', 'subject_indices', 'object_indices')}
            selected, values, eligible = select_pairs(plain['labels'], plain['scores'], plain['subject_indices'], plain['object_indices'], hidden)
            n = len(selected)
            pairs = np.stack((plain['subject_indices'][selected], plain['object_indices'][selected]), -1)
            hw = np.asarray(pixels.shape[-2:])
            boxes = plain['boxes'][pairs] / np.asarray([hw[1], hw[0], hw[1], hw[0]], dtype=np.float32)
            f['features'][i, :n] = values
            f['valid'][i, :n] = True
            f['pairs'][i, :n] = pairs
            f['native_rows'][i, :n] = selected
            f['labels'][i, :n] = plain['labels'][pairs]
            f['confidence'][i, :n] = plain['scores'][pairs]
            f['boxes'][i, :n] = boxes
            f['native_pair_count'][i] = len(hidden)
            human_nonhuman = int((plain['labels'][plain['object_indices']] != 0).sum())
            f['human_nonhuman_pair_count'][i] = human_nonhuman
            f['eligible_pair_count'][i] = eligible
            f['original_wh'][i], f['resized_hw'][i] = original, hw
            f['image_sha256'][i] = hashlib.sha256(raw).hexdigest().encode()
            validate_row(*(f[key][i] for key in ('features', 'valid', 'pairs', 'native_rows', 'labels', 'confidence', 'boxes')))
            record = dict(**row, image_sha256=hashlib.sha256(raw).hexdigest(), original_wh=original,
                resized_hw=hw.tolist(), native_pair_count=len(hidden), human_nonhuman_pair_count=human_nonhuman,
                eligible_pair_count=eligible, selected_pair_count=n,
                selected_native_rows=selected.tolist(), selected_pairs=pairs.tolist(),
                selected_labels=plain['labels'][pairs].tolist(), selected_confidence=plain['scores'][pairs].tolist(),
                boxes_original_normalized=boxes.tolist(), features_fp16_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                classifier_hook_parity=True, native_order_checked=True)
            log.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            log.flush()
            f.attrs['completed_rows'] = i + 1
            if (i + 1) % 10 == 0 or i == 0:
                f.flush()
                print(f'GPU {rank}: {i+1}/{len(rows)} images; {n} selected/{eligible} human-nonhuman pairs', flush=True)
        f.attrs['complete'] = True
    after = model_digest(model)
    if before != after:
        raise ValueError('Frozen UPT changed during extraction.')
    save(out / f'complete_rank{rank}.json', dict(status='COMPLETE', images=len(rows), model_sha256=after))


def run(config):
    import h5py
    import torch
    from captioning.upt_reference import UPT_REVISION, POCKET_REVISION
    cfg = json.loads(Path(config).read_text())
    if digest(cfg['dataset_json']) != DATASET_SHA or digest(cfg['upt_checkpoint']) != UPT_SHA:
        raise ValueError('Wrong COCO JSON or UPT HICO checkpoint.')
    for directory, sha in [(cfg['upt'], UPT_REVISION), (cfg['pocket'], POCKET_REVISION),
                           (str(Path(cfg['upt']) / 'detr'), '820d17abc697e8dd5f49d4cd1fdbde412545bac5'),
                           (str(Path(cfg['upt']) / 'hicodet'), '9dd59c1f6ee2ddc011567765d0bc644858391fea')]:
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=directory, text=True).strip() != sha:
            raise ValueError('Upstream reference revision differs.')
        if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=directory, text=True).strip():
            raise ValueError('Tracked upstream edits detected.')
    if torch.cuda.device_count() != 2:
        raise RuntimeError('Select GPU T4x2.')
    dataset = json.loads(Path(cfg['dataset_json']).read_text())
    groups = locked_groups(dataset)
    metadata_path = Path(cfg['upt']) / 'hicodet/instances_train2015.json'
    metadata = json.loads(metadata_path.read_text())
    correspondence = [[] for _ in metadata['objects']]
    for _, obj, verb in metadata['correspondence']:
        if verb not in correspondence[obj]:
            correspondence[obj].append(verb)
    rows = []
    for split in ('train', 'val'):
        for r in groups[split]:
            row = dict(r, filepath=dataset['images'][r['image_id']]['filepath'])
            if not (Path(cfg['images']) / row['filepath'] / row['filename']).is_file():
                raise FileNotFoundError(row['filename'])
            rows.append(row)
    out = Path(cfg['output'])
    if out.exists():
        raise FileExistsError('Use a fresh output; preserve the prior extraction.')
    if shutil.disk_usage(out.parent).free < 2 * 1024**3:
        raise RuntimeError('Need at least2GiB free for the bounded cache.')
    out.mkdir(parents=True)
    for name in cfg['implementation_sha256']:
        source = Path(__file__).parent / name
        if digest(source) != cfg['implementation_sha256'][name]:
            raise ValueError('Extraction source changed: ' + name)
        destination = out / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    cfg.update(rows=rows, vocabulary=dict(objects=metadata['objects'], verbs_by_object=correspondence))
    save(out / 'cohort.json', groups)
    save(out / 'extraction_protocol.json', cfg)
    processes, handles = [], []
    try:
        for rank in range(2):
            handle = (out / f'worker_rank{rank}.log').open('w')
            handles.append(handle)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank))
            processes.append(subprocess.Popen([sys.executable, '-u', __file__, 'worker', '--config',
                str(out / 'extraction_protocol.json'), '--rank', str(rank)], env=env, stdout=handle, stderr=subprocess.STDOUT))
        positions = [0, 0]
        while True:
            for rank in range(2):
                with (out / f'worker_rank{rank}.log').open(errors='replace') as stream:
                    stream.seek(positions[rank]); text = stream.read(); positions[rank] = stream.tell()
                if text:
                    print(text, end='', flush=True)
            codes = [p.poll() for p in processes]
            if any(code not in (None, 0) for code in codes):
                raise RuntimeError('A cache worker failed; see its log.')
            if all(code == 0 for code in codes):
                break
            print('UPT hidden-feature workers active; extraction only.', flush=True)
            time.sleep(20)
        counts = {}
        for rank in range(2):
            expected = rows[rank::2]
            with h5py.File(out / f'features_rank{rank}.h5') as f:
                if not f.attrs['complete'] or int(f.attrs['completed_rows']) != len(expected):
                    raise ValueError('Incomplete hidden-feature shard.')
                if f['image_ids'][:].tolist() != [r['coco_id'] for r in expected]:
                    raise ValueError('Cache image order differs.')
                for i, row in enumerate(expected):
                    validate_row(*(f[k][i] for k in ('features', 'valid', 'pairs', 'native_rows', 'labels', 'confidence', 'boxes')))
                    counts[row['coco_id']] = int(f['valid'][i].sum())
        complete = [json.loads((out / f'complete_rank{rank}.json').read_text()) for rank in range(2)]
        if complete[0]['model_sha256'] != complete[1]['model_sha256']:
            raise ValueError('Workers used different UPT weights.')
        filenames = ['cohort.json', 'features_rank0.h5', 'features_rank1.h5', 'pairs_rank0.jsonl', 'pairs_rank1.jsonl']
        receipt = dict(status='CACHE_COMPLETE', schema=SCHEMA, slots=SLOTS, width=WIDTH,
            cohort_sha256=identity(groups), dataset_sha256=DATASET_SHA, upt_checkpoint_sha256=UPT_SHA,
            upt_revision=UPT_REVISION, pocket_revision=POCKET_REVISION, metadata_sha256=digest(metadata_path),
            model_sha256=complete[0]['model_sha256'], files={n: digest(out / n) for n in filenames},
            implementation_sha256=cfg['implementation_sha256'],
            selection_box_threshold=BOX_THRESHOLD,
            selection='All native interactions first; human-nonhuman pairs with BOTH endpoint detection scores >.2; top8 by confidence product; stable(s,o,native_row) ties; no verb threshold/text/reference.',
            coordinates='Boxes normalized to original image after inverting UPT resize; native out-of-frame extents preserved, not clipped.',
            provenance='Frozen official UPT-R50 HICO checkpoint; detector/HOI external pretraining may overlap COCO; not overlap-audited independent test.',
            consumed_caption_or_relation_references=False, test_used=False,
            coverage={split: dict(images=len(part), with_pairs=sum(counts[r['coco_id']] > 0 for r in part),
                pair_count_histogram=dict(Counter(counts[r['coco_id']] for r in part))) for split, part in groups.items()})
        save(out / 'cache_receipt.json', receipt)
        print(json.dumps(receipt['coverage'], indent=2), flush=True)
        print('CACHE COMPLETE. Preserve entire folder:', out, flush=True)
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate(); p.wait()
        for handle in handles:
            handle.close()
        with zipfile.ZipFile(str(out) + '_review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(out.rglob('*')):
                if file.suffix in ('.json', '.jsonl', '.log', '.py'):
                    archive.write(file, file.relative_to(out))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['run', 'worker'])
    parser.add_argument('--config', required=True)
    parser.add_argument('--rank', type=int)
    args = parser.parse_args()
    worker(args.config, args.rank) if args.mode == 'worker' else run(args.config)
