import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CHECKPOINT = Path('/kaggle/input/datasets/ducanh2403/qformer-epoch10/qformer_epoch10/model_h1_2_crossattn_epoch_10.pth')
PROMPT_CACHE = Path('/kaggle/input/datasets/ducanh2403/prompt-conditioned-qformer-32q-2l/relation_prompts_v2_1_full/prompt_clip_tokens_relation_v2_1.pt')
GROUNDED_CACHE = Path('/kaggle/input/notebooks/ducanh2403/build-grounded-relation-v1/grounded_relation_v1_full/grounded_relation_cache.pt')
VISUAL_CACHE_SOURCE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')

REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = '8c44631'
VISUAL_CACHE_LOCAL = Path('/tmp/visual-cache-e3-valid')
EXPERIMENT = 'e3_appearance_v2_32q_2l_dual_gpu_1ep'
OUTPUT = Path('/kaggle/working') / EXPERIMENT


def run(args, cwd=None):
    args = list(map(str, args))
    print('\nRunning:', ' '.join(args), flush=True)
    started = time.monotonic()
    with subprocess.Popen(args, cwd=cwd) as process:
        while True:
            try:
                code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print(f'Process still running: {time.monotonic() - started:.0f}s', flush=True)
        if code:
            raise subprocess.CalledProcessError(code, args)


def stage(source):
    destination = VISUAL_CACHE_LOCAL / source.name
    total = source.stat().st_size
    if destination.is_file() and destination.stat().st_size == total:
        print('Already staged:', source.name, flush=True)
        return
    temporary = destination.with_suffix('.h5.partial')
    copied, started, last_log = 0, time.monotonic(), 0
    with source.open('rb') as reader, temporary.open('wb') as writer:
        while True:
            block = reader.read(64 * 1024 * 1024)
            if not block:
                break
            writer.write(block)
            copied += len(block)
            elapsed = time.monotonic() - started
            if elapsed - last_log >= 10 or copied == total:
                print(f'{source.name}: {copied / total:.1%} | {copied / max(elapsed, 0.001) / 1024**2:.1f} MiB/s', flush=True)
                last_log = elapsed
    assert temporary.stat().st_size == total, temporary
    temporary.replace(destination)


print('E3 Appearance V2 FULL 1-EPOCH bootstrap started.', flush=True)
for path in (CHECKPOINT, PROMPT_CACHE, GROUNDED_CACHE, COCO_JSON):
    assert path.is_file(), f'Missing file: {path}'
assert COCO_IMAGES.is_dir(), COCO_IMAGES
assert VISUAL_CACHE_SOURCE.is_dir(), VISUAL_CACHE_SOURCE

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TQDM_MININTERVAL'] = '30'
REPO.parent.mkdir(parents=True, exist_ok=True)
if not REPO.exists():
    run(['git', 'clone', 'https://github.com/Supzxjee/Image_Captioning.git', REPO])
assert (REPO / '.git').is_dir(), REPO
run(['git', 'fetch', '--depth', '30', 'origin', 'main'], cwd=REPO)
run(['git', 'checkout', '--detach', COMMIT], cwd=REPO)
run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], cwd=REPO)

VISUAL_CACHE_LOCAL.mkdir(parents=True, exist_ok=True)
parts = [VISUAL_CACHE_SOURCE / f'visual_part_{i:02d}_of_03.h5' for i in range(1, 4)]
for source in parts:
    assert source.is_file(), source
needed = sum(p.stat().st_size for p in parts if not (VISUAL_CACHE_LOCAL / p.name).is_file()
             or (VISUAL_CACHE_LOCAL / p.name).stat().st_size != p.stat().st_size)
assert shutil.disk_usage('/tmp').free > needed + 5 * 1024**3, 'Not enough space in /tmp.'
for source in parts:
    stage(source)


# Train 1 full epoch on two GPUs. Global batch 32 -> local batch 16 per GPU.
import torch
import zipfile

assert torch.cuda.device_count() >= 2, 'Select accelerator GPU T4 x2.'
for index in range(2):
    print(f'GPU {index}: {torch.cuda.get_device_name(index)}', flush=True)

common = [
    '--dataset-json-path', COCO_JSON, '--base-path', COCO_IMAGES,
    '--prompt-cache-path', PROMPT_CACHE,
    '--visual-cache', VISUAL_CACHE_LOCAL, '--visual-cache-id-key', 'coco_id',
    '--visual-adapter', 'qformer', '--num-visual-queries', '32', '--qformer-layers', '2',
    '--grounded-relation-alignment', '--grounded-relation-cache-path', GROUNDED_CACHE,
    '--grounded-relation-version', 'appearance_v2', '--relation-alignment-margin', '0.2',
    '--seed', '42', '--num-workers', '0',
]
V2_CHECKPOINT = OUTPUT / 'checkpoints/model_h1_2_crossattn_epoch_1.pth'

# If training completed but diagnostic failed, rerun without training again.
if not V2_CHECKPOINT.is_file():
    warm = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    assert warm.get('epoch') == 10 and warm.get('visual_adapter') == 'qformer'
    assert warm.get('num_visual_queries') == 32 and warm.get('qformer_layers') == 2
    assert not warm.get('grounded_relation_alignment', False), 'Use Q-Former baseline epoch10.'
    del warm
    run([
        sys.executable, '-u', '-m', 'accelerate.commands.launch',
        '--multi_gpu', '--num_processes', '2', '--num_machines', '1',
        '--mixed_precision', 'no', '--dynamo_backend', 'no',
        '--num_cpu_threads_per_process', '2', '--main_process_port', '29501',
        'train_h1_2_gated.py', '--mode', 'train', '--checkpoint', CHECKPOINT,
        '--epochs', '1', '--max-train-batches', '0', '--batch-size', '32',
        '--relation-alignment-weight', '0.1', '--relation-wrong-image-weight', '0.5',
        '--experiment-name', EXPERIMENT,
    ] + common, cwd=REPO)
else:
    print('Found full-epoch checkpoint; skip training and rerun diagnostic.', flush=True)

payload = torch.load(V2_CHECKPOINT, map_location='cpu', weights_only=False)
assert payload.get('epoch') == 1, 'Wrong epoch.'
assert payload.get('grounded_relation_version') == 'appearance_v2', 'Wrong relation version.'
assert payload.get('max_train_batches') == 0, 'This is a smoke checkpoint, not full training.'
del payload

history_path = OUTPUT / 'train_history_h1_2.json'
history = json.loads(history_path.read_text(encoding='utf-8'))
last = history[-1]
assert last['relation_actions_per_batch'] > 0, 'No action relations trained.'
assert last['relation_wrong_images_per_batch'] > 0, 'No eligible cross-image negatives.'
print('E3 V2 FULL TRAINING CHECK:')
print(json.dumps(last, indent=2), flush=True)

manifest = {
    'code_commit': COMMIT, 'experiment': EXPERIMENT, 'warm_start': str(CHECKPOINT),
    'checkpoint': str(V2_CHECKPOINT), 'epochs': 1, 'max_train_batches': 0,
    'gpu_processes': 2, 'global_batch_size': 32, 'local_batch_size': 16,
    'negative_pool': 'local GPU batch; duplicate positive phrases filtered',
    'train_history_scope': 'rank-0 local-batch means, not cross-rank averaged losses',
    'prompt_cache': str(PROMPT_CACHE), 'grounded_cache': str(GROUNDED_CACHE),
    'training': last,
}
manifest_path = OUTPUT / 'manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

REVIEW = OUTPUT / 'validation_diagnostic_100'
run([
    sys.executable, '-u', 'diagnose_grounded_relations.py',
    '--mode', 'predict', '--split', 'val', '--limit', '0',
    '--checkpoint', V2_CHECKPOINT, '--expected-epoch', '1',
    '--sample-size', '100', '--output', REVIEW,
] + common, cwd=REPO)

summary_path = REVIEW / 'diagnostic_summary.json'
summary = json.loads(summary_path.read_text(encoding='utf-8'))
summary['code_commit'] = COMMIT
summary['warm_start'] = str(CHECKPOINT)
summary['training_scope'] = '1 full epoch, 2 GPUs, global batch32'
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')

# Small review ZIP includes diagnostics/history/manifest, but not the checkpoint.
archive = Path('/kaggle/working') / (EXPERIMENT + '_review.zip')
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
    for path in REVIEW.rglob('*'):
        if path.is_file():
            zipped.write(path, path.relative_to(OUTPUT).as_posix())
    zipped.write(history_path, history_path.name)
    zipped.write(manifest_path, manifest_path.name)

print('E3 APPEARANCE V2 FULL 1-EPOCH COMPLETE')
print(json.dumps(summary, indent=2, ensure_ascii=False))
print('Review ZIP (no checkpoint):', archive)
print('Checkpoint kept in notebook output:', V2_CHECKPOINT)
