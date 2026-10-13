# Diagnostic V2: cân bằng đối chứng sai hành động và sai người

V1 có supervision tốt hơn cosine, nhưng wrong-actor win rate7/10 không tăng và mẫu lệch
positive (461/512,118/128). V2 sửa các điểm này; chưa chứng minh nó cải thiện kết quả.

## Thay đổi có kiểm soát

- Sampler lấp chỗ trống ưu tiên negatives; giữ positive anchors và confirmed actor counterparts.
  Tối đa1/8 budget là actor-connected positive anchors; remaining positive anchors lấy ngẫu nhiên seed42.
  Buddy có thể positive với hành động khác, nên không đảm bảo chính xác50% positive-containing;
  phải đọc pair_summary. Đây là diagnostic được chọn mẫu, không phản ánh prevalence toàn bộ.
- Tối đa1024 train,256 dev; cùng6actions,GT boxes,CLIP frozen; loại caption val/test.
- Ranking = trung bình hai mean-loss actor/predicate, nếu có đủ2type; thiếu một type thì dùng
  mean type hiện có. Tổng loss vẫn BCE +0.2ranking, margin0.2. Không tự tạo missing negatives.
- Báo edges/images/wins/ties/mean_gap/margin-pass riêng theo actor/predicate trên dev.
- Bốn mốc trên chính dữ liệu V2 và cùng initialization/optimizer/head20epochs:
  CLIP cosine; supervision BCE; BCE+edge-mean contrast cũ; BCE+type-balanced contrast mới.
  So sánh mốc3 với4 để xem tác động riêng của balancing; không chỉ so với AP69.53 của mẫu V1.
- Chưa thêm wrong-object negatives, wearing, detector boxes hoặc caption fusion.

## Chạy Kaggle

T4 x2, Internet on; chỉ input COCO theo đường dẫn đã điền. Không checkpoint/prompt cache cũ.
Chờ Kaggle gắn input xong; sau đó chạy cell dưới đây hoặc import notebook
INSTANCE_ALIGNMENT_V2_DIAGNOSTIC_1024.ipynb. Dùng OUTPUT mới, không trộn feature/mẫu V1.
Annotation ZIP COCO (~241MiB) và CLIP tải vào /tmp; không nằm trong review ZIP.
Hai GPU trích frozen features, head train CPU. Dữ liệu V2 thay đổi nên không dùng features V1.

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
COMMIT = 'c8c303f'
MODEL_ID = 'openai/clip-vit-base-patch16'
MODEL_REVISION = '57c216476eefef5ab752ec549e440a49ae4ae5f3'
LOCAL_MODEL_PATH = ''  # Optional standard Hugging Face CLIP folder; not a Q-Former checkpoint.
MODEL_DIR = Path(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH else Path('/tmp/clip_instance_alignment')
OUTPUT = Path('/kaggle/working/instance_contrast_alignment_v2_diagnostic_1024')
DATA = OUTPUT / 'data'
TRAIN_PAIRS, VAL_PAIRS, HEAD_EPOCHS = 1024, 256, 20

print('Instance contrast alignment V2 diagnostic started.', flush=True)
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
                 'train_pairs_limit': TRAIN_PAIRS, 'val_pairs_limit': VAL_PAIRS,
                 'head_epochs': HEAD_EPOCHS, 'variant': 'type_balanced_v2', 'gpu_workers': 2, 'seed': 42,
                 'caption_json_sha256': hashlib.sha256(COCO_JSON.read_bytes()).hexdigest(),
                 'coco_images': str(COCO_IMAGES), 'local_model_path': LOCAL_MODEL_PATH}
manifest = OUTPUT / 'manifest.json'
if manifest.is_file():
    assert json.loads(manifest.read_text())['configuration'] == configuration, 'Use a new OUTPUT for changed config.'
manifest.write_text(json.dumps({'configuration': configuration}, indent=2), encoding='utf-8')

# Official annotation ZIP (~241 MiB); no second download of the COCO image dataset.
run([sys.executable, '-u', 'prepare_supervised_relation_pairs.py',
     '--caption-json', COCO_JSON, '--work-dir', '/tmp/vcoco_alignment_annotations', '--output', DATA,
     '--train-pairs', TRAIN_PAIRS, '--val-pairs', VAL_PAIRS], cwd=REPO)
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
          '--output', OUTPUT, '--shards', '2', '--epochs', HEAD_EPOCHS]
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

# Only the small pair projection is trained, on CPU; CLIP and text targets remain frozen.
run(common + ['--train'], cwd=REPO)
summary = json.loads((OUTPUT / 'diagnostic_summary.json').read_text())
summary['code_commit'] = COMMIT
(OUTPUT / 'diagnostic_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
archive = shutil.make_archive(str(OUTPUT) + '_review', 'zip', OUTPUT)
print('Download review ZIP (no CLIP weights):', archive, flush=True)

```

Gửi `instance_contrast_alignment_v2_diagnostic_1024_review.zip` và Kaggle version.
Đọc support từng lớp, tỷ lệ positive pairs và số ảnh có actor edges trước khi diễn giải metrics.
Mục tiêu xem type-balanced có cải thiện wrong-actor gap/ranking so với edge-mean trên cùng mẫu,
đồng thời không mất chất lượng relation classification nghiêm trọng. Không giả định kết quả tốt.
Threshold90% fit từ train vẫn không đảm bảo precision90% trên dev; unknown không có ground truth
để tính precision. AP chỉ trên known labels, không official V-COCO role AP hay caption score.

Validation local:12 tests PASS, gồm regression sampler, bỏ unknown khỏi loss,
loss cân bằng không bị tăng số predicate edges lấn át, missing-type handling,
image-support/tie metrics. Synthetic 2epoch pipeline với4mốc hoàn tất;
không phải quality experiment. Chưa chạy V2 real-data inference/training.
