# Kiểm chứng claim quan hệ A–predicate–B: 100 cặp

Audit teacher trước có 87 đề xuất, 7 lỗi schema và 6 no_interaction. Sửa offline đúng lỗi
`relation_status=<predicate>` chỉ khi `relations=[<cùng predicate>]` khôi phục 7 record;
giữ raw answer và ghi schema_repair. Sau sửa có 94 claim, không có phép đo accuracy.

## Thay đổi

- Mỗi claim được hỏi riêng với full image marked A/B, crop pair marked và object detail.
- Không đưa teacher evidence, heuristic predicate, CLIP proposal hoặc caption tham chiếu vào prompt.
- Kiểm tra identity của A/B, actor đúng A, target đúng B, bằng chứng có thực ra thuộc đối tượng khác không.
- Chỉ giữ candidate khi verdict supported, hai identity supported, actor/target đúng và không có other-instance evidence. Parse error/unclear/contradicted đều không giữ.
- Chỉ kiểm chứng 94 đề xuất; không tự tìm quan hệ thay thế cho các claim bị loại. 6 cặp không có claim vẫn được lưu.
- Cùng Qwen2.5-VL-3B thực hiện lần hỏi riêng, không phải verifier độc lập: sai số có thể tương quan. Cần review ảnh trước khi dùng nhãn huấn luyện.
- 100 cặp là development/regression set đã quan sát. Không coi tỉ lệ giữ lại là accuracy, hoặc kết quả held-out.

## Chạy trên Kaggle

Chọn T4 x2, bật Internet, chỉ Add Input COCO. Bộ 100 cặp và claim được lưu trong repo,
không phụ thuộc output version cũ. Đường dẫn COCO đã điền; LOCAL_MODEL_PATH tùy chọn.
Hai worker mỗi worker 50 cặp; heartbeat 30 giây, lưu sau mỗi pair và resume cùng OUTPUT/config.
Weights khoảng 7.5 GB tải vào /tmp, không nằm trong ZIP review.

Copy cell dưới đây hoặc import `VLM_CLAIM_VERIFICATION_100.ipynb`:

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
COMMIT = '7e35905'
MODEL_ID = 'Qwen/Qwen2.5-VL-3B-Instruct'
MODEL_REVISION = '66285546d2b821cf421d4f5eb2576359d3770cd3'
# Set this to an existing standard Hugging Face model folder to skip download.
LOCAL_MODEL_PATH = ''
MODEL_DIR = Path(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH else Path('/tmp/qwen25vl3b_66285546')
OUTPUT = Path('/kaggle/working/vlm_claim_verification_100')

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

PAIRS = REPO / 'audit/vlm_claim_selection_100.json'
parameters = {'code_commit': COMMIT, 'model': MODEL_ID, 'model_revision': MODEL_REVISION,
              'local_model_path': str(MODEL_DIR), 'pair_count': 100, 'claim_count': 94, 'gpu_workers': 2, 'task': 'verify_claims_v1',
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

Output: `/kaggle/working/vlm_claim_verification_100.zip`. Summary gồm verdict counts,
eligible claims, identity/binding flags; records giữ raw answers từng claim. CSV/HTML có ảnh
và các trường human review. Gửi ZIP và version Kaggle để phân tích rồi cập nhật sheet.

## Kiểm tra trước khi dùng

Review ưu tiên các pair_index 4 (tie/hot dog), 6 (skateboard), 55 (skis/chairlift),
85 (tie/cake), 31 (frisbee của người khác); đối chiếu 64 (wearing tie), 68 (sitting on surfboard).
Đo cả lỗi được loại và quan hệ đúng bị loại. Chỉ tính precision/recall trên nhãn con người
đã kiểm tra; không dựa vào self-report của teacher.

## Validation local

8 tests parser/repair/binding/sharding PASS; sửa offline đủ 7 record; manifest đối chiếu
filename/slot/classes/boxes với selection cũ đủ 100 cặp. Notebook/cell kiểm tra cú pháp.
Chưa chạy inference mới trên Kaggle, chưa có bằng chứng bộ lọc nâng chất lượng quan hệ.
