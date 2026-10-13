# 500 cặp trên ảnh mới: trích feature để đánh giá model đã khóa

Model khóa: R50 residual seed42,20epochs,scale0.1; SHA256 checkpoint và thresholds
được ghi trong audit/residual_frozen_evaluation_protocol.json. Không chọn seed tốt nhất
của lượt3seed, không huấn luyện lại hay fit threshold bằng nhãn mới.

Danh sách1000image IDs loại trừ từ diagnostic V1/V2 và VLM audits đã nhận, có SHA256
nguồn trong audit/instance_alignment_seen_images.json. Chọn đúng500pairs với seed20261013
trong phần còn lại của V-COCO val. Label thiếu vẫnunknown, GT boxes/classes.
Chọn mẫu theo diagnostic positive/negative/actor counterpart, không natural prevalence.
"Mới" nghĩa là chưa dùng trong các scorer runs/audits được liệt kê; vẫn caption train/restval,
không đảm bảo CLIP pretrained chưa thấy ảnh. Không dùng caption validation/test.

## Kaggle chỉ trích feature

Chọn T4 x2, Internet on, chỉ input COCO theo2đường dẫn đã điền. Không cần head checkpoint,
feature ZIP cũ, prompt cache hoặc VLM input. Danh sách loại trừ nằm trong repo.
Chờ Adding data hoàn tất trước khi cell chạy. Script tải official annotations và CLIP như trước.
Hai GPU trích A/B/union image features và6prompt embeddings; không tính residual predictions,
không gọi train hoặc fit threshold. Output gồm cặp/nhãn,provenance,feature0/1 và protocol.

Copy cell hoặc import RESIDUAL_NEW_IMAGES_500_FEATURES.ipynb:

```python
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
COMMIT = '910fda9'
MODEL_ID = 'openai/clip-vit-base-patch16'
MODEL_REVISION = '57c216476eefef5ab752ec549e440a49ae4ae5f3'
LOCAL_MODEL_PATH = ''  # Optional standard Hugging Face CLIP folder; not a Q-Former checkpoint.
MODEL_DIR = Path(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH else Path('/tmp/clip_instance_alignment')
OUTPUT = Path('/kaggle/working/residual_new_images_500_features')
DATA = OUTPUT / 'data'
PAIR_COUNT, SELECTION_SEED = 500, 20261013
EXCLUSIONS = REPO / 'audit/instance_alignment_seen_images.json'
PROTOCOL = REPO / 'audit/residual_frozen_evaluation_protocol.json'

print('Frozen residual new-image feature extraction started.', flush=True)
assert COCO_JSON.is_file(), COCO_JSON
assert COCO_IMAGES.is_dir(), COCO_IMAGES
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
                 'evaluation_pairs': PAIR_COUNT, 'selection_seed': SELECTION_SEED,
                 'exclusion_sha256': hashlib.sha256(EXCLUSIONS.read_bytes()).hexdigest(),
                 'training': False, 'variant': 'frozen_newimage_features_v1', 'gpu_workers': 2, 'seed': 42,
                 'caption_json_sha256': hashlib.sha256(COCO_JSON.read_bytes()).hexdigest(),
                 'coco_images': str(COCO_IMAGES), 'local_model_path': LOCAL_MODEL_PATH}
manifest = OUTPUT / 'manifest.json'
if manifest.is_file():
    assert json.loads(manifest.read_text())['configuration'] == configuration, 'Use a new OUTPUT for changed config.'
manifest.write_text(json.dumps({'configuration': configuration}, indent=2), encoding='utf-8')

# Official annotation ZIP (~241 MiB); no second download of the COCO image dataset.
run([sys.executable, '-u', 'prepare_supervised_relation_pairs.py',
     '--caption-json', COCO_JSON, '--work-dir', '/tmp/vcoco_alignment_annotations', '--output', DATA,
     '--exclude-images', EXCLUSIONS, '--evaluation-pairs', PAIR_COUNT,
     '--selection-seed', SELECTION_SEED], cwd=REPO)
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

summary = json.loads((DATA / 'evaluation_pair_summary.json').read_text())
assert summary['seen_image_overlap'] == 0
print('NEW IMAGE FEATURE EXTRACTION COMPLETE; no scorer training performed.', flush=True)
print(json.dumps(summary['evaluation'], indent=2), flush=True)
archive = shutil.make_archive(str(OUTPUT) + '_review', 'zip', OUTPUT)
print('Send review ZIP:', archive, flush=True)

```

Gửi `residual_new_images_500_features_review.zip` và Kaggle version.
Đánh giá sẽ chạy CPU bằng checkpoint đã khóa trên máy, không cần lượt GPU thứ hai.

## Đánh giá CPU sau khi nhận ZIP

```python
import subprocess, sys
subprocess.run([sys.executable, 'evaluate_frozen_residual.py',
    '--source', 'PATH_TO/residual_new_images_500_features_review.zip',
    '--checkpoint', 'PATH_TO/clip_prior_residual_v1/residual_head.pt',
    '--output', 'PATH_TO/residual_new_images_500_evaluation'], check=True)
```

Checkpoint/exclusion checksum sai, pair thiếu/thừa, ảnh đã dùng hoặc feature thiếu đều
fail. Model eval+requires_grad false; inference_mode. So sánh CLIP/residual cùng cặp,
AP trên known labels,precision-recall với frozen thresholds,actor/predicate gap/wins,
fixed/regressed claims. Accepted unknown claims báo riêng, không coi chúng là đúng/sai.
Không chỉnh threshold hoặc tune model trên kết quả lượt này để rồi báo lại như heldout.

15tests PASS; selection loại toàn bộ image IDs, deterministic và fail nếu thiếu quota;
syntax của cell/notebook/CPU evaluator checked. Chưa trích feature500cặp mới hoặc có metrics mới.
