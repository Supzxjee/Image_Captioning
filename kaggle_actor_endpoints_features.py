import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = '7eb37b7'
MODEL_ID, MODEL_REVISION = 'openai/clip-vit-base-patch16', '57c216476eefef5ab752ec549e440a49ae4ae5f3'
MODEL_DIR = Path('/tmp/clip_actor_endpoints')
OUTPUT = Path('/kaggle/working/actor_all_endpoints_features_v1')
DATA = OUTPUT / 'data'
WORK = Path('/tmp/actor_official_annotations')
EXCLUSIONS = REPO / 'audit/actor_verifier_dev_exclusions.json'

print('Actor endpoint feature extraction started.', flush=True)
assert COCO_JSON.is_file(), COCO_JSON
assert COCO_IMAGES.is_dir(), COCO_IMAGES
import torch
assert torch.cuda.device_count() >= 2, 'Select GPU T4 x2.'
os.environ.update(PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_XET='1', OMP_NUM_THREADS='2')


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


if not REPO.exists():
    run(['git', 'clone', 'https://github.com/Supzxjee/Image_Captioning.git', REPO])
assert (REPO / '.git').is_dir(), REPO
run(['git', 'fetch', '--depth', '30', 'origin', 'main'], cwd=REPO)
run(['git', 'checkout', '--detach', COMMIT], cwd=REPO)
run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers>=4.51,<5', 'huggingface_hub>=0.34,<1', 'Pillow', 'numpy'], cwd=REPO)
OUTPUT.mkdir(parents=True, exist_ok=True)
configuration = {'code_commit': COMMIT, 'training': False, 'variant': 'actor_compatibility_all_endpoints_v1',
    'model': MODEL_ID, 'model_revision': MODEL_REVISION, 'gpu_workers': 2,
    'caption_json_sha256': hashlib.sha256(COCO_JSON.read_bytes()).hexdigest(),
    'actor_dev_exclusions_sha256': hashlib.sha256(EXCLUSIONS.read_bytes()).hexdigest()}
manifest = OUTPUT / 'manifest.json'
if manifest.is_file():
    assert json.loads(manifest.read_text())['configuration'] == configuration, 'Use a new output for changed protocol.'
manifest.write_text(json.dumps({'configuration': configuration}, indent=2), encoding='utf-8')
run([sys.executable, '-u', 'prepare_supervised_relation_pairs.py', '--caption-json', COCO_JSON,
     '--work-dir', WORK, '--output', DATA, '--all-actor-endpoints', '--actor-dev-exclusions', EXCLUSIONS], cwd=REPO)
shutil.copyfile(EXCLUSIONS, OUTPUT / 'actor_dev_exclusions.json')
download = ('from huggingface_hub import snapshot_download; '
    f'snapshot_download(repo_id={MODEL_ID!r}, revision={MODEL_REVISION!r}, local_dir={str(MODEL_DIR)!r}, max_workers=2, '
    "allow_patterns=['*.json','*.txt','*.bin','*.safetensors'])")
run([sys.executable, '-u', '-c', download])
assert (MODEL_DIR / 'config.json').is_file(), MODEL_DIR
common = [sys.executable, '-u', 'run_instance_alignment_diagnostic.py', '--data', DATA,
          '--images', COCO_IMAGES, '--model', MODEL_DIR, '--output', OUTPUT, '--shards', '2']
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
            raise RuntimeError(f'Feature worker failed: {statuses}')
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
summary = json.loads((DATA / 'pair_summary.json').read_text())
print('ACTOR FEATURES READY; no head training performed.', flush=True)
print(json.dumps(summary, indent=2), flush=True)
archive = shutil.make_archive(str(OUTPUT) + '_review', 'zip', OUTPUT)
print('Send ZIP:', archive, flush=True)
