import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = '74aab6b'
MODEL_ID = 'Qwen/Qwen2.5-VL-3B-Instruct'
MODEL_REVISION = '66285546d2b821cf421d4f5eb2576359d3770cd3'
# Set this to an existing standard Hugging Face model folder to skip download.
LOCAL_MODEL_PATH = ''
MODEL_DIR = Path(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH else Path('/tmp/qwen25vl3b_66285546')
OUTPUT = Path('/kaggle/working/vlm_instance_relation_audit_100')

print('Instance VLM relation audit bootstrap started.', flush=True)
assert COCO_JSON.is_file(), COCO_JSON
assert COCO_IMAGES.is_dir(), COCO_IMAGES
import torch
assert torch.cuda.device_count() >= 2, 'Select GPU T4 x2 for two parallel workers.'
for index in range(2):
    print(f'GPU {index}: {torch.cuda.get_device_name(index)}', flush=True)

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '30'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['OMP_NUM_THREADS'] = '2'

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
                print(f'Process still running: {time.monotonic()-started:.0f}s', flush=True)
        if code:
            raise subprocess.CalledProcessError(code, args)

REPO.parent.mkdir(parents=True, exist_ok=True)
if not REPO.exists():
    run(['git', 'clone', 'https://github.com/Supzxjee/Image_Captioning.git', REPO])
run(['git', 'fetch', '--depth', '30', 'origin', 'main'], cwd=REPO)
run(['git', 'checkout', '--detach', COMMIT], cwd=REPO)
run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers>=4.51,<5',
     'accelerate>=1.2,<2', 'huggingface_hub>=0.34,<1', 'Pillow'], cwd=REPO)

if not LOCAL_MODEL_PATH:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    present = sum(p.stat().st_size for p in MODEL_DIR.glob('*.safetensors'))
    needed = max(0, 7509337976 - present)
    assert shutil.disk_usage('/tmp').free > needed + 1024**3, 'Not enough /tmp space for model (~7.5 GB).'
    download_code = (
        'from huggingface_hub import snapshot_download; '
        f'snapshot_download(repo_id={MODEL_ID!r}, revision={MODEL_REVISION!r}, '
        f'local_dir={str(MODEL_DIR)!r}, max_workers=2, '
        "allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.jinja'])"
    )
    print('Download model once; both GPU workers share these local weight files.', flush=True)
    run([sys.executable, '-u', '-c', download_code])
else:
    assert MODEL_DIR.is_dir(), MODEL_DIR
assert (MODEL_DIR / 'config.json').is_file(), MODEL_DIR

PAIRS = REPO / 'audit/action_pair_selection_100.json'
parameters = {'code_commit': COMMIT, 'model': MODEL_ID, 'model_revision': MODEL_REVISION,
              'local_model_path': str(MODEL_DIR), 'pair_count': 100, 'gpu_workers': 2,
              'selection_sha256': hashlib.sha256(PAIRS.read_bytes()).hexdigest()}
OUTPUT.mkdir(parents=True, exist_ok=True)
manifest_path = OUTPUT / 'manifest.json'
if manifest_path.is_file():
    old = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert old['parameters'] == parameters, 'Output contains another audit; use a new OUTPUT directory.'
manifest_path.write_text(json.dumps({'parameters': parameters}, indent=2), encoding='utf-8')

common = [
    sys.executable, '-u', 'audit_vlm_instance_relations.py',
    '--pairs', PAIRS, '--dataset-json', COCO_JSON, '--images', COCO_IMAGES,
    '--model', MODEL_DIR, '--output', OUTPUT, '--shard-count', '2',
]
workers = []
started = time.monotonic()
try:
    for index in range(2):
        environment = os.environ.copy()
        environment['CUDA_VISIBLE_DEVICES'] = str(index)
        command = list(map(str, common + ['--shard-index', str(index)]))
        workers.append(subprocess.Popen(command, cwd=REPO, env=environment))
    while True:
        statuses = [worker.poll() for worker in workers]
        if any(code is not None and code != 0 for code in statuses):
            raise RuntimeError(f'A GPU worker failed: {statuses}. Read preceding worker log.')
        if all(code == 0 for code in statuses):
            break
        print(f'GPU workers running: {statuses} | elapsed {time.monotonic()-started:.0f}s', flush=True)
        time.sleep(30)
finally:
    for worker in workers:
        if worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()

run(common + ['--merge'], cwd=REPO)
archive = shutil.make_archive(str(OUTPUT), 'zip', OUTPUT)
print('\nVLM INSTANCE RELATION AUDIT COMPLETE')
print((OUTPUT / 'vlm_summary.json').read_text(encoding='utf-8'))
print('Review ZIP (no model weights):', archive)
