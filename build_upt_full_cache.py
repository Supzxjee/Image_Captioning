"""Manual full-COCO UPT cache, two independent GPUs, bounded resumable sessions."""
import argparse
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

from audit_vcoco_alignment_data import CAPTION_SHA256, digest
from captioning.upt_cache import identity, image_plan, pack_prediction, read_shard, open_partial, finish_shard
from run_vcoco_upt_full_audit import UPT, POCKET, CHECKPOINT


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def worker(config, rank):
    import torch
    from PIL import Image
    from captioning.upt_reference import load_reference, prepare_image, infer
    cfg = json.loads(Path(config).read_text())
    output = Path(cfg['output'])
    tasks = cfg['tasks'][rank::2]
    signature = cfg['signature']
    torch.set_num_threads(2); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError('GPU required; no CPU inference fallback.')
    correspondence = cfg['vocabulary']['verbs_by_object']
    model = None
    completed, skipped = 0, 0
    for task in tasks:
        path = output / (task['name'] + '.jsonl.gz')
        if path.exists():
            read_shard(path, task, signature)
            skipped += len(task['images'])
            continue
        journal = open_partial(output, task, signature)
        if time.time() >= cfg['deadline']:
            break
        if model is None:
            model, runtime = load_reference(cfg['upt'], cfg['pocket'], cfg['checkpoint'], correspondence)
            model.to('cuda')
            save(output / f'worker_{rank}_runtime.json', runtime)
        for row in task['images'][len(journal.records):]:
            if time.time() >= cfg['deadline']:
                break
            image_path = Path(cfg['images']) / row['filepath'] / row['filename']
            raw = image_path.read_bytes()
            with Image.open(io.BytesIO(raw)) as image:
                original_size = image.size
                prediction = infer(model, prepare_image(image).to('cuda'))
            plain = {key: prediction[key].tolist() for key in
                     ('boxes', 'labels', 'scores', 'size', 'subject_indices', 'object_indices', 'logits')}
            plain['official_output'] = {key: prediction['official_output'][key].tolist()
                                        for key in ('pairing', 'labels', 'scores')}
            record = pack_prediction(row, plain, cfg['vocabulary'], hashlib.sha256(raw).hexdigest(), original_size)
            journal.append(record)
            completed += 1
            if completed % 10 == 0:
                print(f'GPU {rank}: +{completed} new images | {task["name"]} {len(journal.records)}/{len(task["images"])}', flush=True)
        if len(journal.records) == len(task['images']):
            finish_shard(output, task, signature, journal.records)
            print(f'GPU {rank}: saved {task["name"]}', flush=True)
        else:
            break
    save(output / f'worker_{rank}_status.json', {'new_images': completed, 'skipped_complete_images': skipped,
                                               'budget_reached': time.time() >= cfg['deadline']})


def restore(previous, output, tasks, signature):
    """Copy a previous self-contained cache; never modify attached inputs."""
    prior = json.loads((previous / 'cache_identity.json').read_text())
    if prior['signature'] != signature:
        raise ValueError('Previous cache model/data/schema/runtime identity differs.')
    manifest_path = previous / 'shard_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {'shards': {}}
    expected = {task['name'] + '.jsonl.gz': task for task in tasks}
    for name, info in manifest['shards'].items():
        if name not in expected or digest(previous / name) != info['sha256']:
            raise ValueError('Previous shard name/hash differs.')
    # A killed session may not have reached the final manifest write. Complete
    # gzip members still carry their source signature and exact ordered IDs.
    for name, task in expected.items():
        if (previous / name).exists():
            read_shard(previous / name, task, signature)
            shutil.copyfile(previous / name, output / name)
    for task in tasks:
        name = task['name']
        if (output / f'{name}.jsonl.gz').exists():
            continue
        partial = previous / f'{name}.partial.jsonl'
        if partial.exists():
            stamp = previous / f'{name}.partial.protocol.json'
            if not stamp.is_file():
                raise ValueError('Previous partial journal is missing its provenance.')
            shutil.copyfile(partial, output / partial.name)
            shutil.copyfile(stamp, output / stamp.name)
            open_partial(output, task, signature)


def scan(output, tasks, signature):
    shards, counts, partial_counts = {}, {'train': 0, 'val': 0, 'test': 0}, {'train': 0, 'val': 0, 'test': 0}
    for task in tasks:
        path = output / (task['name'] + '.jsonl.gz')
        if path.exists():
            rows = read_shard(path, task, signature)
            shards[path.name] = {'sha256': digest(path), 'images': len(rows), 'split': task['split'], 'bytes': path.stat().st_size}
            counts[task['split']] += len(rows)
        elif (output / f'{task["name"]}.partial.jsonl').exists():
            partial_counts[task['split']] += len(open_partial(output, task, signature).records)
    return {'shards': shards, 'complete_shard_images': counts, 'partial_images': partial_counts,
            'complete': len(shards) == len(tasks), 'signature': signature}


def run(args):
    repo = Path(__file__).resolve().parent
    if digest(args.dataset_json) != CAPTION_SHA256:
        raise ValueError('Dataset identity differs from the pinned full COCO123287.')
    if digest(args.checkpoint) != CHECKPOINT:
        raise ValueError('Wrong UPT HICO checkpoint.')
    for path, expected in [(args.upt, UPT), (args.pocket, POCKET),
                           (args.upt / 'detr', '820d17abc697e8dd5f49d4cd1fdbde412545bac5'),
                           (args.upt / 'hicodet', '9dd59c1f6ee2ddc011567765d0bc644858391fea')]:
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=path, text=True).strip() != expected:
            raise ValueError('Reference revision differs.')
        if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=path, text=True).strip():
            raise ValueError('Tracked reference changes detected.')
    tasks = image_plan(json.loads(args.dataset_json.read_text()), 500)
    counts = {split: sum(len(t['images']) for t in tasks if t['split'] == split) for split in ('train', 'val', 'test')}
    if counts != {'train': 113287, 'val': 5000, 'test': 5000}:
        raise ValueError('Unexpected split sizes.')
    for task in tasks:
        for row in task['images']:
            if not (args.images / row['filepath'] / row['filename']).is_file():
                raise FileNotFoundError(row['filename'])
    if not 0 < args.max_hours <= 10:
        raise ValueError('Session budget must be >0 and <=10 hours.')
    import torch
    import torchvision
    if torch.cuda.device_count() < 2:
        raise RuntimeError('Select GPU T4 x2.')
    metadata_path = args.upt / 'hicodet/instances_train2015.json'
    metadata = json.loads(metadata_path.read_text())
    correspondence = [[] for _ in metadata['objects']]
    for _, obj, verb in metadata['correspondence']:
        if verb not in correspondence[obj]:
            correspondence[obj].append(verb)
    vocabulary = {'objects': metadata['objects'], 'verbs': metadata['verbs'], 'verbs_by_object': correspondence}
    cache_identity = {'schema': 'upt_full_coco_supported_logits_v1', 'dataset_sha256': CAPTION_SHA256,
                      'checkpoint_sha256': CHECKPOINT, 'upt_revision': UPT, 'pocket_revision': POCKET,
                      'metadata_sha256': digest(metadata_path), 'vocabulary': vocabulary,
                      'implementation_sha256': {p: digest(repo / p) for p in
                           ('build_upt_full_cache.py', 'captioning/upt_cache.py', 'captioning/upt_reference.py', 'captioning/inference_journal.py')},
                      'runtime': {'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
                                  'cuda': torch.version.cuda},
                      'image_plan_sha256': identity(tasks), 'split_images': counts,
                      'shard_size': 500, 'score_precision': 'source FP32 -> exact JSON float round trip; no fp16 quantization',
                      'columns': 'Each pair_logits/native_scores row follows verbs_by_object[labels[pairs[row][1]]]. Unsupported verbs omitted, not negative.',
                      'scope': 'Native retained UPT detections/pairs, all supported HICO verbs; not all objects visible in an image.',
                      'training': False, 'references_or_relation_gt_consumed': False}
    signature = identity(cache_identity)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a fresh OUTPUT; attach prior cache via --previous-cache.')
    output.mkdir(parents=True)
    save(output / 'cache_identity.json', {'signature': signature, 'identity': cache_identity})
    processes, handles, positions = [], [], [0, 0]
    try:
        if args.previous_cache:
            restore(args.previous_cache, output, tasks, signature)
        cfg = {'tasks': tasks, 'signature': signature, 'vocabulary': vocabulary,
               'upt': str(args.upt.resolve()), 'pocket': str(args.pocket.resolve()),
               'checkpoint': str(args.checkpoint.resolve()), 'images': str(args.images.resolve()),
               'output': str(output), 'deadline': time.time() + args.max_hours * 3600}
        save(output / 'session_config.json', cfg)
        for rank in range(2):
            handle = (output / f'worker_{rank}.log').open('wb'); handles.append(handle)
            env = os.environ.copy(); env.update(CUDA_VISIBLE_DEVICES=str(rank), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2')
            processes.append(subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                                               '--worker-config', str(output / 'session_config.json'), '--rank', str(rank)],
                                              cwd=repo, env=env, stdout=handle, stderr=subprocess.STDOUT))
        while True:
            statuses = [p.poll() for p in processes]
            for rank in range(2):
                with (output / f'worker_{rank}.log').open('rb') as stream:
                    stream.seek(positions[rank]); data = stream.read(); positions[rank] = stream.tell()
                if data:
                    print(data.decode('utf-8', errors='replace'), flush=True)
            if any(code is not None and code != 0 for code in statuses):
                raise RuntimeError(f'Cache worker failed: {statuses}; keep completed shards and partial journals.')
            if all(code == 0 for code in statuses):
                break
            print('UPT cache workers active; no training or caption evaluation.', flush=True)
            time.sleep(15)
    except Exception as error:
        save(output / 'failure.json', {'status': 'INCOMPLETE', 'error': repr(error)})
        raise
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        for process in processes:
            process.wait()
        for handle in handles:
            handle.close()
        result = scan(output, tasks, signature)
        save(output / 'shard_manifest.json', result)
        save(output / 'run_summary.json', {'status': 'CACHE_COMPLETE' if result['complete'] else 'CACHE_PARTIAL',
                                         **{k: result[k] for k in ('complete_shard_images', 'partial_images', 'signature')},
                                         'completed_shards': len(result['shards']), 'expected_shards': len(tasks),
                                         'score_precision': cache_identity['score_precision'], 'training': False,
                                         'note': 'Preserve this entire cache directory, not only the small review ZIP.'})
        with zipfile.ZipFile(str(output) + '_review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.glob('*.json')):
                if path.name != 'session_config.json':
                    archive.write(path, path.name)
            for path in output.glob('*.log'):
                archive.write(path, path.name)
        print('CACHE:', output, '\nREVIEW:', str(output) + '_review.zip', flush=True)
        print(json.dumps(json.loads((output / 'run_summary.json').read_text()), indent=2), flush=True)


if __name__ == '__main__':
    if '--worker-config' in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument('--worker-config', required=True); parser.add_argument('--rank', type=int, required=True, choices=[0, 1])
        parsed = parser.parse_args(); worker(parsed.worker_config, parsed.rank)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        for name in ('dataset-json', 'images', 'upt', 'pocket', 'checkpoint', 'output'):
            parser.add_argument('--' + name, type=Path, required=True)
        parser.add_argument('--previous-cache', type=Path)
        parser.add_argument('--max-hours', type=float, default=8)
        run(parser.parse_args())
