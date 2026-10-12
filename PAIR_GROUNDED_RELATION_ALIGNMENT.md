# Pair-grounded relation alignment V1

Thí nghiệm mới không dùng object vector gộp của E2 Cascade. Mỗi triplet giữ
`subject_id`, `object_id` và hai bounding box. Mô hình pool CLIP patches trong
subject box, object box và union box, rồi căn chỉnh visual-pair này với relation
embedding. Một predicate bị tráo được dùng làm hard counterfactual.

Code version: `276e39a`.

## 1. Tạo grounded relation cache

Notebook này cần Internet, MS COCO 2014 và output `relation_prompts_v2_1_full`.
Không cần GPU đôi.

```python
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = '276e39a'
RELATIONS = Path(
    '/kaggle/input/datasets/ducanh2403/prompt-conditioned-qformer-32q-2l/'
    'relation_prompts_v2_1_full/promptcache_relation_v2_1.json'
)
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
OUTPUT_DIR = Path('/kaggle/working/grounded_relation_v1_full')
OUTPUT = OUTPUT_DIR / 'grounded_relation_cache.pt'

def run(args, cwd=None):
    print('Running:', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)

assert RELATIONS.is_file(), RELATIONS
assert COCO_JSON.is_file(), COCO_JSON
assert COCO_IMAGES.is_dir(), COCO_IMAGES
if REPO.exists():
    shutil.rmtree(REPO)
run(['git', 'clone', 'https://github.com/Supzxjee/Image_Captioning.git', REPO])
run(['git', 'checkout', '--detach', COMMIT], cwd=REPO)
run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], cwd=REPO)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
run([
    sys.executable, '-u', 'build_grounded_relation_cache.py',
    '--relations', RELATIONS,
    '--dataset-json-path', COCO_JSON,
    '--base-path', COCO_IMAGES,
    '--output', OUTPUT,
    '--max-relations', '3',
    '--batch-size', '256',
], cwd=REPO)

import torch
bundle = torch.load(OUTPUT, map_location='cpu', weights_only=False)
metadata = bundle['metadata']
assert metadata['variant'] == 'instance_pair_grounded_relation_v1'
assert metadata['complete'] is True
assert metadata['count'] == 123287
assert metadata['max_relations'] == 3
print(metadata)
print('Save Version output:', OUTPUT_DIR)
```

Sau khi hoàn tất, Save Version và kiểm tra output có:

```text
grounded_relation_v1_full/
  grounded_relation_cache.pt
  grounded_relation_cache.json
```

## 2. Diagnostic 500 ảnh

Add Input output ở bước 1, checkpoint Q-Former epoch 10, visual-cache, prompt
token V2.1 và MS COCO. Đường dẫn cache mới có thể khác theo tên notebook output;
chỉ sửa `GROUNDED_CACHE` nếu cần.

```python
import subprocess
import sys
from pathlib import Path

REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = '276e39a'
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
PROMPT_CACHE = Path(
    '/kaggle/input/datasets/ducanh2403/prompt-conditioned-qformer-32q-2l/'
    'relation_prompts_v2_1_full/prompt_clip_tokens_relation_v2_1.pt'
)
GROUNDED_CACHE = Path(
    '/kaggle/input/notebooks/ducanh2403/kltn-mscoco/'
    'grounded_relation_v1_full/grounded_relation_cache.pt'
)
CHECKPOINT_ROOT = Path('/kaggle/input/datasets/ducanh2403/qformer-epoch10')
VISUAL_CACHE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')

checkpoint_matches = sorted(
    CHECKPOINT_ROOT.rglob('model_h1_2_crossattn_epoch_10.pth'))
assert len(checkpoint_matches) == 1, checkpoint_matches
CHECKPOINT = checkpoint_matches[0]
for path in (COCO_JSON, PROMPT_CACHE, GROUNDED_CACHE, CHECKPOINT):
    assert path.is_file(), path
assert COCO_IMAGES.is_dir(), COCO_IMAGES
assert VISUAL_CACHE.is_dir(), VISUAL_CACHE
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run([
    'git', 'clone', 'https://github.com/Supzxjee/Image_Captioning.git', REPO
], check=True)
subprocess.run(['git', 'checkout', '--detach', COMMIT], cwd=REPO, check=True)
subprocess.run([
    sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'
], cwd=REPO, check=True)

args = [
    sys.executable, '-u', 'train_h1_2_gated.py',
    '--mode', 'train',
    '--checkpoint', CHECKPOINT,
    '--dataset-json-path', COCO_JSON,
    '--base-path', COCO_IMAGES,
    '--prompt-cache-path', PROMPT_CACHE,
    '--visual-cache', VISUAL_CACHE,
    '--visual-cache-id-key', 'coco_id',
    '--visual-adapter', 'qformer',
    '--num-visual-queries', '32',
    '--qformer-layers', '2',
    '--grounded-relation-alignment',
    '--grounded-relation-cache-path', GROUNDED_CACHE,
    '--relation-alignment-weight', '0.1',
    '--relation-alignment-margin', '0.2',
    '--epochs', '1',
    '--max-train-batches', '16',
    '--batch-size', '32',
    '--num-workers', '0',
    '--seed', '42',
    '--experiment-name', 'pair_grounded_relation_diagnostic_500',
]
print('Running:', ' '.join(map(str, args)), flush=True)
subprocess.run(list(map(str, args)), cwd=REPO, check=True)
```

Diagnostic đạt yêu cầu khi:

- `relation_loss` hữu hạn và thay đổi qua batch;
- checkpoint ghi `grounded_relation_alignment=True`;
- metadata cache là `instance_pair_grounded_relation_v1`;
- không có NaN hoặc lỗi DDP.

Sau diagnostic mới chạy pilot 3 epoch bằng T4 x2. Pilot phải khởi tạo lại từ
Q-Former epoch 10, đặt `--max-train-batches 0`, giữ seed 42 và dùng đúng cache
trên. Không warm-start từ E2 V2.1 vì kiến trúc cascade cũ và pair-grounded mới
khác nhau.

## 3. Pilot 3 epoch: stage visual cache trước khi chạy dual GPU

Không đọc ngẫu nhiên các shard HDF5 trực tiếp từ `/kaggle/input` bằng hai rank.
Trong một lần thử, cả hai rank đã hoàn thành `optimizer.step()` ở batch 101 nhưng
treo khi lấy batch tiếp theo. Copy ba shard sang ổ cục bộ `/tmp` trước khi launch,
rồi truyền thư mục cục bộ cho `--visual-cache`.

```python
import shutil
import time
from pathlib import Path

VISUAL_CACHE_INPUT = Path(
    '/kaggle/input/datasets/ducanh2403/visual-cache'
)
VISUAL_CACHE_LOCAL = Path('/tmp/visual-cache')

source_shards = [
    VISUAL_CACHE_INPUT / f'visual_part_{part:02d}_of_03.h5'
    for part in range(1, 4)
]
for shard in source_shards:
    assert shard.is_file(), shard

required_bytes = sum(shard.stat().st_size for shard in source_shards)
free_bytes = shutil.disk_usage('/tmp').free
assert free_bytes > required_bytes + 5 * 1024**3, (
    f'Khong du /tmp: can {required_bytes / 1024**3:.1f} GiB, '
    f'con trong {free_bytes / 1024**3:.1f} GiB.'
)

if VISUAL_CACHE_LOCAL.exists():
    shutil.rmtree(VISUAL_CACHE_LOCAL)
VISUAL_CACHE_LOCAL.mkdir(parents=True)

def copy_with_progress(source, destination, chunk_size=64 * 1024**2):
    total = source.stat().st_size
    copied = 0
    next_report = 1024**3
    started = time.monotonic()
    with source.open('rb') as src, destination.open('wb') as dst:
        while True:
            chunk = src.read(chunk_size)
            if not chunk:
                break
            dst.write(chunk)
            copied += len(chunk)
            if copied >= next_report or copied == total:
                elapsed = max(time.monotonic() - started, 1e-6)
                print(
                    f'{source.name}: {copied / 1024**3:.1f}/'
                    f'{total / 1024**3:.1f} GiB | '
                    f'{copied / 1024**2 / elapsed:.1f} MiB/s',
                    flush=True,
                )
                next_report += 1024**3

for source in source_shards:
    destination = VISUAL_CACHE_LOCAL / source.name
    copy_with_progress(source, destination)
    assert destination.stat().st_size == source.stat().st_size

print('Local visual cache ready:', VISUAL_CACHE_LOCAL, flush=True)
```

Trong lệnh train, dùng:

```python
'--visual-cache', VISUAL_CACHE_LOCAL,
```

và giữ `--batch-size 32`, Accelerate `--num_processes 2`, relation weight `0.1`,
margin `0.2`, seed `42`, `--epochs 3`, `--max-train-batches 0`.
