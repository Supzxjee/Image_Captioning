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
DETECTIONS = Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
COORDINATE_FORMAT = 'pixel_xyxy'

REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = 'f9bd244'
MODEL_ID = 'openai/clip-vit-base-patch16'
MODEL_REVISION = '57c216476eefef5ab752ec549e440a49ae4ae5f3'
LOCAL_MODEL_PATH = ''  # Optional standard Hugging Face CLIP folder; not a Q-Former checkpoint.
MODEL_DIR = Path(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH else Path('/tmp/clip_instance_alignment')
OUTPUT = Path('/kaggle/working/residual_detector_full_candidates_20_features')
DATA = OUTPUT / 'data'
IMAGE_COUNT, SELECTION_SEED = 20, 20261014
IMAGE_LIST = REPO / 'audit/residual_full_candidate_images_20.json'
EXCLUSIONS = REPO / 'audit/instance_alignment_seen_images.json'
PROTOCOL = REPO / 'audit/residual_frozen_evaluation_protocol.json'

print('Detector-only full-candidate audit20 feature extraction started.', flush=True)
assert COCO_JSON.is_file(), COCO_JSON
assert COCO_IMAGES.is_dir(), COCO_IMAGES
assert DETECTIONS.is_file(), DETECTIONS
import torch
assert torch.cuda.device_count() >= 2, 'Select GPU T4 x2.'
os.environ.update(PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false',
                  HF_HUB_DISABLE_XET='1', OMP_NUM_THREADS='2')


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
                print(f'Process running: {time.monotonic()-started:.0f}s', flush=True)
        if code:
            raise subprocess.CalledProcessError(code, args)


REPO.parent.mkdir(parents=True, exist_ok=True)
if not REPO.exists():
    run(['git', 'clone', 'https://github.com/Supzxjee/Image_Captioning.git', REPO])
assert (REPO / '.git').is_dir(), REPO
run(['git', 'fetch', '--depth', '30', 'origin', 'main'], cwd=REPO)
run(['git', 'checkout', '--detach', COMMIT], cwd=REPO)
run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers>=4.51,<5',
     'huggingface_hub>=0.34,<1', 'Pillow', 'numpy'], cwd=REPO)
OUTPUT.mkdir(parents=True, exist_ok=True)
configuration = {'code_commit': COMMIT, 'model': MODEL_ID, 'model_revision': MODEL_REVISION,
                 'audit_images': IMAGE_COUNT, 'selection_seed': SELECTION_SEED,
                 'image_list_sha256': hashlib.sha256(IMAGE_LIST.read_bytes()).hexdigest(),
                 'training': False, 'variant': 'detector_full_candidate_audit_20_v1',
                 'detections': str(DETECTIONS), 'min_confidence': 0.5,
                 'max_objects': 10, 'coordinate_format': COORDINATE_FORMAT, 'gpu_workers': 2, 'seed': 42,
                 'caption_json_sha256': hashlib.sha256(COCO_JSON.read_bytes()).hexdigest(),
                 'coco_images': str(COCO_IMAGES), 'local_model_path': LOCAL_MODEL_PATH}
manifest = OUTPUT / 'manifest.json'
if manifest.is_file():
    assert json.loads(manifest.read_text())['configuration'] == configuration, 'Use a new OUTPUT for changed config.'
manifest.write_text(json.dumps({'configuration': configuration}, indent=2), encoding='utf-8')

# Only image metadata is used for selection; no GT pairs/labels/IoU matching.
run([sys.executable, '-u', 'prepare_detector_full_candidates.py',
     '--image-list', IMAGE_LIST, '--detections', DETECTIONS,
     '--images', COCO_IMAGES, '--output', DATA,
     '--coordinate-format', COORDINATE_FORMAT], cwd=REPO)
shutil.copyfile(PROTOCOL, OUTPUT / 'evaluation_protocol.json')
if not LOCAL_MODEL_PATH:
    assert shutil.disk_usage('/tmp').free > 2 * 1024**3, 'Need free /tmp space for CLIP.'
    download_code = ('from huggingface_hub import snapshot_download; '
        f'snapshot_download(repo_id={MODEL_ID!r}, revision={MODEL_REVISION!r}, '
        f'local_dir={str(MODEL_DIR)!r}, max_workers=2, '
        "allow_patterns=['*.json','*.txt','*.bin','*.safetensors'])")
    run([sys.executable, '-u', '-c', download_code])
assert (MODEL_DIR / 'config.json').is_file(), MODEL_DIR

common = [sys.executable, '-u', 'run_instance_alignment_diagnostic.py',
          '--data', DATA, '--images', COCO_IMAGES, '--model', MODEL_DIR,
          '--output', OUTPUT, '--shards', '2', '--evaluation-only']
workers = []
started = time.monotonic()
try:
    for index in range(2):
        environment = os.environ.copy()
        environment['CUDA_VISIBLE_DEVICES'] = str(index)
        workers.append(subprocess.Popen(list(map(str, common + ['--shard', index])), cwd=REPO, env=environment))
    while True:
        statuses = [worker.poll() for worker in workers]
        if any(code is not None and code != 0 for code in statuses):
            raise RuntimeError(f'Feature worker failed: {statuses}; see preceding log.')
        if all(code == 0 for code in statuses):
            break
        print(f'Feature workers: {statuses} | elapsed {time.monotonic()-started:.0f}s', flush=True)
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

summary = json.loads((DATA / 'full_candidate_summary.json').read_text())
assert summary['images'] == 20
print('FULL CANDIDATE FEATURE EXTRACTION COMPLETE; no scorer training performed.', flush=True)
print(json.dumps(summary, indent=2), flush=True)
archive = shutil.make_archive(str(OUTPUT) + '_review', 'zip', OUTPUT)
print('Send review ZIP:', archive, flush=True)
