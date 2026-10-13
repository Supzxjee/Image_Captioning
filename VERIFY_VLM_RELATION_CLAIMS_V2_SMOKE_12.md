# Claim verifier V2: kiểm tra 12 cặp trước khi chạy lại 100

Lượt trước: 94 claims; 81 parse errors do identity field chép chuỗi enum trong mẫu prompt;
13 claims parse được, 8 eligible và 5 có binding flags. 6 cặp không có claim.
Không diễn giải 8/94 như hiệu quả lọc: 81 lỗi schema không phải semantic rejections.
Raw answers: 81 supported, 5 contradicted, 8 chép nguyên enum verdict.
Các lỗi tie/hot dog, skateboard, skis/chairlift vẫn được raw answer xác nhận sai.
Không suy đoán identity supported để cứu 81 output: trường đó chưa được model chọn.

V2 bỏ chuỗi enum có dấu | trong JSON mẫu; liệt kê lựa chọn ngoài JSON, yêu cầu chọn đúng một.
Parser vẫn strict, không mặc định identity đúng. Mẫu dùng giá trị abstention thay vì positive.
Đây chỉ là sửa giao thức output, không chứng minh cải thiện reasoning/grounding.

Chạy smoke 12 cặp gồm lỗi đã thấy và vài quan hệ hợp lý: original pair indices
4,6,8,9,26,31,55,64,67,68,85,96. Không phải held-out benchmark.
Hai GPU chia 6 cặp/GPU. Giữ original_pair_index để đối chiếu với audit cũ.
Chỉ cần COCO, không cần ZIP hay checkpoint. T4 x2, Internet on.

Copy cell dưới đây hoặc import VLM_CLAIM_VERIFICATION_V2_SMOKE_12.ipynb.

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
COMMIT = 'bf5825a'
MODEL_ID = 'Qwen/Qwen2.5-VL-3B-Instruct'
MODEL_REVISION = '66285546d2b821cf421d4f5eb2576359d3770cd3'
# Set this to an existing standard Hugging Face model folder to skip download.
LOCAL_MODEL_PATH = ''
MODEL_DIR = Path(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH else Path('/tmp/qwen25vl3b_66285546')
OUTPUT = Path('/kaggle/working/vlm_claim_verification_v2_smoke_12')

print('Instance claim verification bootstrap started.', flush=True)
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

SOURCE_PAIRS = REPO / 'audit/vlm_claim_selection_100.json'
SMOKE_INDICES = [4, 6, 8, 9, 26, 31, 55, 64, 67, 68, 85, 96]
all_pairs = json.loads(SOURCE_PAIRS.read_text(encoding='utf-8'))
selected = [{**all_pairs[i], 'original_pair_index': i} for i in SMOKE_INDICES]
PAIRS = Path('/tmp/vlm_claim_smoke_12.json')
PAIRS.write_text(json.dumps(selected, indent=2), encoding='utf-8')
parameters = {'code_commit': COMMIT, 'model': MODEL_ID, 'model_revision': MODEL_REVISION,
              'local_model_path': str(MODEL_DIR), 'pair_count': 12, 'claim_count': 12, 'gpu_workers': 2, 'task': 'verify_claims_v2_smoke12',
              'selection_sha256': hashlib.sha256(PAIRS.read_bytes()).hexdigest(), 'original_indices': SMOKE_INDICES}
OUTPUT.mkdir(parents=True, exist_ok=True)
manifest_path = OUTPUT / 'manifest.json'
if manifest_path.is_file():
    old = json.loads(manifest_path.read_text(encoding='utf-8'))
    assert old['parameters'] == parameters, 'Output contains another audit; use a new OUTPUT directory.'
manifest_path.write_text(json.dumps({'parameters': parameters}, indent=2), encoding='utf-8')

common = [
    sys.executable, '-u', 'audit_vlm_instance_relations.py',
    '--pairs', PAIRS, '--dataset-json', COCO_JSON, '--images', COCO_IMAGES,
    '--model', MODEL_DIR, '--output', OUTPUT, '--shard-count', '2', '--verify-claims',
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
print('\nVLM CLAIM VERIFICATION COMPLETE')
print((OUTPUT / 'vlm_summary.json').read_text(encoding='utf-8'))
print('Review ZIP (no model weights):', archive)

```

Gửi ZIP vlm_claim_verification_v2_smoke_12.zip và Kaggle version.
Chỉ mở lại lượt 100 sau khi schema ổn và review cho thấy có khả năng bác bỏ claim sai;
nếu vẫn xác nhận lỗi cũ thì dừng dùng same-model self-verification, chuyển sang nhãn quan hệ
đã kiểm chứng và predictor/verifier có supervision theo cặp instance.

8 tests local PASS, cell kiểm tra cú pháp; chưa chạy GPU V2.
