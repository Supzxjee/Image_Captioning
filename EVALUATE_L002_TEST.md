# Đánh giá Region Alignment λ=0.02 trên test 5.000 ảnh

Dùng workflow này sau khi đã chốt λ=0.02 bằng validation. Trước khi chạy, thêm Output của notebook `gated_region_align_l002_dual_gpu` làm Kaggle Input hoặc upload checkpoint epoch 10 thành Dataset. Notebook này không train lại mô hình.

```python
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import torch

REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = '3be0030'

COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
PROMPT_CACHE = Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
VISUAL_CACHE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')

CHECKPOINT_NAME = 'model_h1_2_crossattn_epoch_10.pth'
RESTORED_CHECKPOINT = Path('/kaggle/working') / CHECKPOINT_NAME
EXPERIMENT = 'gated_region_align_l002_dual_gpu_test'


def run(args):
    args = list(map(str, args))
    print('\nRunning:', ' '.join(args), flush=True)
    subprocess.run(args, check=True)


def restore_torch_archive(source, destination):
    """Return a real .pth file whether Kaggle preserved or expanded the archive."""
    if source.is_file():
        return source
    if not source.is_dir():
        raise FileNotFoundError(source)

    print('Kaggle expanded the checkpoint; rebuilding:', source, flush=True)
    with zipfile.ZipFile(destination, mode='w', compression=zipfile.ZIP_STORED) as archive:
        for file in source.rglob('*'):
            if file.is_file():
                relative = file.relative_to(source).as_posix()
                archive.write(file, f'archive/{relative}')
    return destination


# 1. Kiểm tra dữ liệu dùng để đánh giá.
for path in (COCO_JSON, PROMPT_CACHE):
    assert path.is_file(), f'Thiếu Input: {path}'
assert COCO_IMAGES.is_dir(), f'Sai thư mục ảnh: {COCO_IMAGES}'
for part in range(1, 4):
    shard = VISUAL_CACHE / f'visual_part_{part:02d}_of_03.h5'
    assert shard.is_file(), f'Thiếu visual cache: {shard}'


# 2. Tìm đúng checkpoint epoch 10 trong các Kaggle Input đã gắn.
candidates = sorted(Path('/kaggle/input').rglob(CHECKPOINT_NAME))
print('\nCheckpoint candidates:')
for candidate in candidates:
    print(' -', candidate, '| file=', candidate.is_file(), '| dir=', candidate.is_dir())

assert candidates, (
    'Không tìm thấy checkpoint epoch 10. Hãy Add Input từ Output của '
    'notebook gated_region_align_l002_dual_gpu.'
)
assert len(candidates) == 1, (
    'Tìm thấy nhiều checkpoint epoch 10. Chỉ gắn Output λ=0.02 hoặc đặt '
    'CHECKPOINT_SOURCE bằng đúng một đường dẫn được in phía trên.'
)

CHECKPOINT = restore_torch_archive(candidates[0], RESTORED_CHECKPOINT)
assert CHECKPOINT.is_file(), CHECKPOINT

checkpoint_metadata = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
assert checkpoint_metadata['epoch'] == 10, checkpoint_metadata['epoch']
assert abs(float(checkpoint_metadata['alignment_weight']) - 0.02) < 1e-12
assert checkpoint_metadata.get('max_train_batches', 0) == 0, (
    'Đây là checkpoint smoke, không phải bản full.'
)
print('\nCheckpoint verified:')
print(' epoch:', checkpoint_metadata['epoch'])
print(' alignment_weight:', checkpoint_metadata['alignment_weight'])
print(' train loss:', checkpoint_metadata['loss'])
print(' caption loss:', checkpoint_metadata.get('caption_loss'))
print(' alignment loss:', checkpoint_metadata.get('alignment_loss'))
del checkpoint_metadata


# 3. Clone đúng phiên bản code tương thích checkpoint.
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


# 4. Sinh caption và tính metrics trên đúng test split 5.000 ảnh.
run([
    sys.executable, '-u', 'train_h1_2_gated.py',
    '--mode', 'evaluate',
    '--checkpoint', CHECKPOINT,
    '--split', 'test',
    '--limit', '0',
    '--batch-size', '32',
    '--num-workers', '0',
    '--dataset-json-path', COCO_JSON,
    '--base-path', COCO_IMAGES,
    '--prompt-cache-path', PROMPT_CACHE,
    '--visual-cache', VISUAL_CACHE,
    '--visual-cache-id-key', 'coco_id',
    '--visual-preprocessing', 'bilinear',
    '--visual-precision', 'fp32',
    '--experiment-name', EXPERIMENT,
])


# 5. Kiểm tra output để tránh nhầm lại với validation.
evaluation_dir = Path('/kaggle/working') / EXPERIMENT / 'evaluation'
predictions_path = evaluation_dir / 'test_5000_captions_h1_2_gated.json'
ground_truth_path = evaluation_dir / 'test_5000_gt_h1_2_gated.json'
metrics_path = evaluation_dir / 'test_5000_metrics_h1_2_gated.json'

for path in (predictions_path, ground_truth_path, metrics_path):
    assert path.is_file(), f'Thiếu output: {path}'

predictions = json.loads(predictions_path.read_text(encoding='utf-8'))
metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
assert len(predictions) == 5000
assert len({item['image_id'] for item in predictions}) == 5000

print('\nTEST metrics λ=0.02:')
print(json.dumps(metrics, indent=2))
print('\nPredictions:', predictions_path)
print('Ground truth:', ground_truth_path)
print('Metrics:', metrics_path)
```

Kết quả hợp lệ phải có các tên file bắt đầu bằng `test_5000_`, không phải `val_5000_`. Giữ lại cả ba file JSON để đối chiếu caption và tính lại metrics mà không cần chạy inference lần nữa.
