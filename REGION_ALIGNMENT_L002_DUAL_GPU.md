# Thực nghiệm Region Alignment λ=0.02 với hai GPU bằng Accelerate

Pipeline dùng Hugging Face Accelerate để chạy DistributedDataParallel trên hai GPU T4. Batch toàn cục vẫn là 32: mỗi GPU nhận 16 ảnh, gradient được đồng bộ trước khi cập nhật mô hình. Chỉ tiến trình chính ghi log, checkpoint và history.

Chạy cell với `SMOKE = True` trước. Khi thấy đủ 20 batch và có checkpoint, đổi thành `False` rồi Save Version để train 10 epoch và đánh giá 5.000 ảnh validation.

```python
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = '4613a0f'

DETECTIONS = Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
PROMPT_CACHE = Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
VISUAL_CACHE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')
REGION_TARGETS = Path('/kaggle/input/datasets/ducanh2403/region-target/region_targets_yolo.pt')

SMOKE = True  # Thành công 20 batch thì đổi thành False để chạy đủ 10 epoch.
EPOCHS = 1 if SMOKE else 10
EXPERIMENT = ('gated_region_align_l002_dual_gpu_smoke' if SMOKE
              else 'gated_region_align_l002_dual_gpu')


def run(args):
    args = list(map(str, args))
    print('\nRunning:', ' '.join(args), flush=True)
    subprocess.run(args, check=True)


# 1. Kiểm tra hai GPU và toàn bộ Input.
print('PyTorch:', torch.__version__)
print('CUDA devices:', torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(f'GPU {index}:', torch.cuda.get_device_name(index))
assert torch.cuda.device_count() == 2, 'Notebook phải chọn Accelerator GPU T4 x2.'

for path in (DETECTIONS, COCO_JSON, PROMPT_CACHE):
    assert path.is_file(), f'Thiếu Input: {path}'
assert COCO_IMAGES.is_dir(), f'Sai thư mục ảnh: {COCO_IMAGES}'
for part in range(1, 4):
    shard = VISUAL_CACHE / f'visual_part_{part:02d}_of_03.h5'
    assert shard.is_file(), f'Thiếu visual cache: {shard}'


# 2. Clone và cố định phiên bản có Accelerate DDP.
if not REPO.exists():
    run([
        'git', 'clone',
        'https://github.com/Supzxjee/Image_Captioning.git',
        REPO,
    ])

os.chdir(REPO)
run(['git', 'fetch', 'origin'])
run(['git', 'checkout', '--detach', COMMIT])
run(['git', 'rev-parse', '--short', 'HEAD'])
run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])


# 3. Dùng region targets đã được tạo sẵn và upload thành Kaggle Input.
# /kaggle/input là read-only, vì vậy không tạo cache mới tại đây.
assert REGION_TARGETS.is_file(), f'Thiếu region target cache: {REGION_TARGETS}'
print('Dùng lại region target cache:', REGION_TARGETS, flush=True)

# File JSON chỉ là metadata tùy chọn; training chỉ bắt buộc file .pt.
metadata_path = REGION_TARGETS.with_suffix('.json')
if metadata_path.is_file():
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    print('Region metadata:', json.dumps(metadata, indent=2), flush=True)
    assert metadata['complete'], 'Region target cache chưa hoàn chỉnh.'
    assert metadata['min_confidence'] == 0.5
    assert metadata['min_area_ratio'] == 0.001
    assert metadata['max_area_ratio'] == 0.9


# 4. Cấu hình dùng chung. batch-size=32 là GLOBAL batch;
# Accelerate split_batches=True sẽ cấp 16 ảnh cho mỗi GPU.
common = [
    '--dataset-json-path', COCO_JSON,
    '--base-path', COCO_IMAGES,
    '--prompt-cache-path', PROMPT_CACHE,
    '--visual-cache', VISUAL_CACHE,
    '--visual-cache-id-key', 'coco_id',
    '--visual-preprocessing', 'bilinear',
    '--visual-precision', 'fp32',
    '--experiment-name', EXPERIMENT,
]

train_args = [
    'train_h1_2_gated.py',
    '--mode', 'train',
    '--epochs', EPOCHS,
    '--seed', '42',
    '--batch-size', '32',
    '--num-workers', '0',
    '--region-targets-path', REGION_TARGETS,
    '--alignment-weight', '0.02',
    '--alignment-temperature', '0.07',
    '--max-regions', '10',
] + common

if SMOKE:
    train_args += ['--max-train-batches', '20']


# 5. Hai process, mỗi process điều khiển một GPU.
launch_command = [
    sys.executable, '-m', 'accelerate.commands.launch',
    '--multi_gpu',
    '--num_processes', '2',
    '--num_machines', '1',
    '--mixed_precision', 'no',
    '--num_cpu_threads_per_process', '1',
] + train_args
run(launch_command)


# 6. Kiểm tra checkpoint và history do rank 0 ghi.
experiment_dir = Path('/kaggle/working') / EXPERIMENT
history_path = experiment_dir / 'train_history_h1_2.json'
assert history_path.is_file(), f'Thiếu history: {history_path}'
history = json.loads(history_path.read_text(encoding='utf-8'))
print('\nHistory:', json.dumps(history, indent=2), flush=True)

checkpoint = experiment_dir / f'checkpoints/model_h1_2_crossattn_epoch_{EPOCHS}.pth'
assert checkpoint.is_file(), f'Thiếu checkpoint: {checkpoint}'


# 7. Chỉ bản full mới đánh giá validation 5.000 ảnh bằng một GPU.
if not SMOKE:
    run([
        sys.executable, '-u', 'train_h1_2_gated.py',
        '--mode', 'evaluate',
        '--checkpoint', checkpoint,
        '--split', 'val',
        '--limit', '0',
        '--batch-size', '32',
        '--num-workers', '0',
    ] + common)

    metrics_path = experiment_dir / 'evaluation/val_5000_metrics_h1_2_gated.json'
    assert metrics_path.is_file(), f'Thiếu metrics validation: {metrics_path}'
    print('\nValidation metrics:')
    print(metrics_path.read_text(encoding='utf-8'), flush=True)

print('\nHoàn tất:', experiment_dir)
```

Log đầu phiên full phải có dạng:

```text
CUDA devices: 2
Using Accelerate: 2 process(es) | global batch size: 32 | device: cuda:0
Epoch 1: first batch loaded; starting GPU forward/backward.
Epoch 1: batch 1/3540 ...
Epoch 1: batch 100/3540 ...
```

Trong `nvidia-smi`, cả GPU 0 và GPU 1 phải có process Python và bộ nhớ GPU được sử dụng. Số `3540` vẫn là số bước của một epoch vì Accelerate chia từng batch 32 thành hai nửa, thay vì tăng global batch lên 64.


