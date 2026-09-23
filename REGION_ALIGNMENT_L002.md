# Kaggle: Region alignment lambda 0.02

Thí nghiệm này giữ nguyên seed, dữ liệu, Gate, region target và temperature của
bản lambda 0.1; chỉ giảm `alignment_weight` xuống `0.02`. Sau train chỉ đánh giá
toàn bộ validation 5.000 ảnh. Không chạy test cho đến khi chốt cấu hình.

Copy **phần bên trong** khối code sau vào một cell Kaggle. Không copy dòng mở
` ```python ` hoặc dòng đóng ` ``` `.

```python
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = '9e53ef5'

DETECTIONS = Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
PROMPT_CACHE = Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
VISUAL_CACHE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')
REGION_TARGETS = Path('/kaggle/working/region_targets_yolo.pt')

SMOKE = True  # Chạy 20 batch trước; thành công thì đổi thành False.
EPOCHS = 1 if SMOKE else 10
EXPERIMENT = 'gated_region_align_l002_smoke' if SMOKE else 'gated_region_align_l002'


def run(args):
    args = list(map(str, args))
    print('\nRunning:', ' '.join(args), flush=True)
    subprocess.run(args, check=True)


# 1. Kiểm tra Input.
for path in (DETECTIONS, COCO_JSON, PROMPT_CACHE):
    assert path.is_file(), f'Thiếu Input: {path}'
assert COCO_IMAGES.is_dir(), f'Sai thư mục ảnh: {COCO_IMAGES}'
for part in range(1, 4):
    shard = VISUAL_CACHE / f'visual_part_{part:02d}_of_03.h5'
    assert shard.is_file(), f'Thiếu visual cache: {shard}'


# 2. Clone và cố định code đã vector hóa region pooling.
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


# 3. Tạo region target đã lọc nếu phiên hiện tại chưa có.
if not REGION_TARGETS.exists():
    run([
        sys.executable, '-u', 'build_region_targets.py',
        '--source', DETECTIONS,
        '--dataset-json-path', COCO_JSON,
        '--base-path', COCO_IMAGES,
        '--objects-field', 'objects',
        '--name-key', 'label',
        '--bbox-key', 'bbox',
        '--confidence-key', 'conf',
        '--min-confidence', '0.5',
        '--min-area-ratio', '0.001',
        '--max-area-ratio', '0.9',
        '--max-regions', '10',
        '--output', REGION_TARGETS,
    ])
else:
    print('Dùng lại region target cache:', REGION_TARGETS, flush=True)

assert REGION_TARGETS.is_file(), f'Không tạo được: {REGION_TARGETS}'
metadata_path = REGION_TARGETS.with_suffix('.json')
if metadata_path.is_file():
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    print('\nRegion metadata:', json.dumps(metadata, indent=2), flush=True)
    assert metadata['complete'], 'Region target cache chưa hoàn chỉnh.'
    assert metadata['min_confidence'] == 0.5
    assert metadata['min_area_ratio'] == 0.001
    assert metadata['max_area_ratio'] == 0.9


# 4. Cấu hình dùng chung.
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


# 5. Train lambda 0.02.
train_command = [
    sys.executable, '-u', 'train_h1_2_gated.py',
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
    train_command += ['--max-train-batches', '20']

run(train_command)


# 6. Kiểm tra history đúng experiment.
experiment_dir = Path('/kaggle/working') / EXPERIMENT
history_path = experiment_dir / 'train_history_h1_2.json'
assert history_path.is_file(), f'Thiếu history: {history_path}'
history = json.loads(history_path.read_text(encoding='utf-8'))
print('\nHistory:', json.dumps(history, indent=2), flush=True)


# 7. Bản chính thức chỉ đánh giá validation, không dùng test để chọn lambda.
if not SMOKE:
    checkpoint = experiment_dir / 'checkpoints/model_h1_2_crossattn_epoch_10.pth'
    assert checkpoint.is_file(), f'Thiếu checkpoint: {checkpoint}'
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

print('\nRegion targets:', REGION_TARGETS, REGION_TARGETS.stat().st_size / 1e6, 'MB')
```

Sau smoke, kiểm tra log có dòng `SMOKE MODE: stopping each epoch after 20
batches.` và ba loss hữu hạn. Khi chạy bản chính thức, đổi duy nhất `SMOKE =
False`. Không thêm `--test-after-train` vào thí nghiệm chọn lambda.


