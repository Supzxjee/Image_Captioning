# Đánh giá Q-Former không ITC trên test 5.000 ảnh

Workflow này chỉ chạy inference và COCO metrics; không train lại mô hình. Trước
khi chạy, Add Input từ Output notebook `qformer_32q_2l_gated` hoặc upload
checkpoint epoch 10 thành Kaggle Dataset.

Cell tự kiểm tra checkpoint phải là Q-Former 32 queries, 2 layers, train đủ 10
epoch, không dùng ITC và không dùng region alignment. Vì vậy checkpoint smoke hoặc
checkpoint của thí nghiệm khác sẽ bị từ chối.

```python
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import torch

REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = 'f210ced'

COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
PROMPT_CACHE = Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
VISUAL_CACHE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')

CHECKPOINT_NAME = 'model_h1_2_crossattn_epoch_10.pth'
EXPERIMENT = 'qformer_32q_2l_test'


def run(args):
    args = list(map(str, args))
    print('\nRunning:', ' '.join(args), flush=True)
    subprocess.run(args, check=True)


def restore_torch_archive(source, index):
    """Return a .pth file whether Kaggle kept or expanded the PyTorch archive."""
    if source.is_file():
        return source
    if not source.is_dir():
        raise FileNotFoundError(source)

    destination = Path('/kaggle/working') / f'restored_qformer_candidate_{index}.pth'
    print('Kaggle expanded checkpoint; rebuilding:', source, flush=True)
    data_pickles = list(source.rglob('data.pkl'))
    assert len(data_pickles) == 1, (
        f'Expected exactly one data.pkl below {source}, found {len(data_pickles)}'
    )
    # Kaggle can add a folder named after the original checkpoint. The real
    # PyTorch archive root is the directory that directly contains data.pkl.
    archive_root = data_pickles[0].parent
    with zipfile.ZipFile(destination, mode='w', compression=zipfile.ZIP_STORED) as archive:
        for file in archive_root.rglob('*'):
            if file.is_file():
                relative = file.relative_to(archive_root).as_posix()
                archive.write(file, f'archive/{relative}')
    return destination


def is_full_qformer_checkpoint(metadata):
    return (
        int(metadata.get('epoch', -1)) == 10
        and metadata.get('visual_adapter') == 'qformer'
        and int(metadata.get('num_visual_queries', -1)) == 32
        and int(metadata.get('qformer_layers', -1)) == 2
        and float(metadata.get('alignment_weight', 0.0)) == 0.0
        and float(metadata.get('itc_weight', 0.0)) == 0.0
        and int(metadata.get('max_train_batches', 0)) == 0
    )


# 1. Kiểm tra Input dùng cho test.
for path in (COCO_JSON, PROMPT_CACHE):
    assert path.is_file(), f'Thiếu Input: {path}'
assert COCO_IMAGES.is_dir(), f'Sai thư mục ảnh: {COCO_IMAGES}'
for part in range(1, 4):
    shard = VISUAL_CACHE / f'visual_part_{part:02d}_of_03.h5'
    assert shard.is_file(), f'Thiếu visual cache: {shard}'


# 2. Tìm và xác minh đúng checkpoint Q-Former baseline.
candidates = sorted(Path('/kaggle/input').rglob(CHECKPOINT_NAME))
print('\nCheckpoint candidates:')
for candidate in candidates:
    print(' -', candidate, '| file=', candidate.is_file(), '| dir=', candidate.is_dir())
assert candidates, (
    'Không tìm thấy checkpoint epoch 10. Hãy Add Input từ Output notebook '
    'qformer_32q_2l_gated.'
)

matches = []
for index, candidate in enumerate(candidates, 1):
    restored = restore_torch_archive(candidate, index)
    try:
        metadata = torch.load(restored, map_location='cpu', weights_only=False)
    except Exception as error:
        print('Bỏ qua checkpoint không đọc được:', candidate, '|', repr(error))
        continue
    if is_full_qformer_checkpoint(metadata):
        summary = {
            'epoch': metadata['epoch'],
            'loss': metadata['loss'],
            'visual_adapter': metadata['visual_adapter'],
            'num_visual_queries': metadata['num_visual_queries'],
            'qformer_layers': metadata['qformer_layers'],
            'itc_weight': metadata.get('itc_weight', 0.0),
            'alignment_weight': metadata.get('alignment_weight', 0.0),
        }
        matches.append((restored, candidate, summary))
    else:
        print('Bỏ qua checkpoint sai cấu hình:', candidate)
        print({
            'epoch': metadata.get('epoch'),
            'visual_adapter': metadata.get('visual_adapter'),
            'queries': metadata.get('num_visual_queries'),
            'layers': metadata.get('qformer_layers'),
            'alignment_weight': metadata.get('alignment_weight', 0.0),
            'itc_weight': metadata.get('itc_weight', 0.0),
            'max_train_batches': metadata.get('max_train_batches', 0),
        })
    del metadata

assert len(matches) == 1, (
    f'Cần đúng một checkpoint Q-Former baseline full, nhưng tìm thấy {len(matches)}. '
    'Hãy gỡ các Kaggle Input trùng lặp.'
)
CHECKPOINT, CHECKPOINT_SOURCE, checkpoint_summary = matches[0]

print('\nCheckpoint verified:')
print(' source:', CHECKPOINT_SOURCE)
print(' epoch:', checkpoint_summary['epoch'])
print(' train loss:', checkpoint_summary['loss'])
print(' adapter:', checkpoint_summary['visual_adapter'])
print(' queries/layers:', checkpoint_summary['num_visual_queries'],
      checkpoint_summary['qformer_layers'])
print(' ITC weight:', checkpoint_summary['itc_weight'])
print(' region weight:', checkpoint_summary['alignment_weight'])
del checkpoint_summary, matches


# 3. Clone phiên bản code tương thích.
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


# 4. Sinh caption và tính metrics trên test split 5.000 ảnh.
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
    '--visual-adapter', 'qformer',
    '--num-visual-queries', '32',
    '--qformer-layers', '2',
    '--experiment-name', EXPERIMENT,
])


# 5. Xác nhận đây là output test đầy đủ, không phải validation.
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

print('\nQ-FORMER TEST metrics:')
print(json.dumps(metrics, indent=2))
print('\nPredictions:', predictions_path)
print('Ground truth:', ground_truth_path)
print('Metrics:', metrics_path)
```

Có thể chạy bằng **Save Version / Save & Run All**. Quá trình đánh giá dùng một
GPU vì beam search hiện sinh caption tuần tự theo từng ảnh; chọn T4 x2 không làm
hai GPU cùng chạy ở bước này. Giữ lại ba file `test_5000_*.json` để so caption,
tính lại metrics hoặc thực hiện re-ranking sau này.

Sau khi có kết quả, so sánh trực tiếp với Gate trên cùng test split:

| Mô hình | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---:|---:|---:|---:|---:|
| Cross-Attention có Gate | 0.7660 | 0.3660 | 0.2830 | 0.5710 | 1.1820 |
| Q-Former 32q/2l | chờ chạy | chờ chạy | chờ chạy | chờ chạy | chờ chạy |
