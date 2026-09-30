# Sinh lại test candidates của Q-Former cũ

## Mục đích

Cell này phục hồi thí nghiệm bị mất `test_5000_beam5_candidates.json`. Nó dùng
checkpoint Q-Former 32 query, 2 layer, epoch 10, không ITC, không region loss và
không prompt conditioning. Sau khi sinh năm beam candidates cho 5.000 ảnh test,
cell chấm sẵn CLIPScore để notebook ablation tiếp theo không phải làm lại.

Đây chỉ là inference; **không train lại mô hình**. Chọn một GPU, bật Internet và
gắn các Input sau:

- Save Version hoặc Dataset chứa checkpoint Q-Former cũ epoch 10;
- MS COCO `dataset_coco.json` và thư mục `images`;
- `prompt_clip_tokens_cache.pt`;
- ba shard visual cache.

## Cell Kaggle đầy đủ

```python
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import torch

INPUT_ROOT = Path('/kaggle/input')
REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = 'badda5a'

COCO_JSON = Path(
    '/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json'
)
COCO_IMAGES = Path(
    '/kaggle/input/datasets/vuthetam/mscoco-2014/images'
)
PROMPT_CACHE = Path(
    '/kaggle/input/datasets/ducanh2403/'
    'prompt-cache/prompt_clip_tokens_cache.pt'
)
VISUAL_CACHE = Path(
    '/kaggle/input/datasets/ducanh2403/visual-cache'
)

CHECKPOINT_NAME = 'model_h1_2_crossattn_epoch_10.pth'
EXPERIMENT = 'qformer_test_candidates_rebuilt'


def run(args, cwd=None):
    args = list(map(str, args))
    print('\nRunning:', ' '.join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def restore_torch_archive(source, index):
    """Khôi phục khi Kaggle hiển thị .pth thành một cây thư mục."""
    if source.is_file():
        return source
    data_pickles = list(source.rglob('data.pkl'))
    if len(data_pickles) != 1:
        raise RuntimeError(
            f'Cần một data.pkl dưới {source}, tìm thấy {len(data_pickles)}.'
        )
    archive_root = data_pickles[0].parent
    destination = (
        Path('/kaggle/working') / f'restored_qformer_{index}.pth'
    )
    with zipfile.ZipFile(
        destination, mode='w', compression=zipfile.ZIP_STORED
    ) as archive:
        for file in archive_root.rglob('*'):
            if file.is_file():
                relative = file.relative_to(archive_root).as_posix()
                archive.write(file, f'archive/{relative}')
    return destination


def is_old_qformer(metadata):
    return (
        int(metadata.get('epoch', -1)) == 10
        and metadata.get('visual_adapter') == 'qformer'
        and int(metadata.get('num_visual_queries', -1)) == 32
        and int(metadata.get('qformer_layers', -1)) == 2
        and float(metadata.get('alignment_weight', 0.0)) == 0.0
        and float(metadata.get('itc_weight', 0.0)) == 0.0
        and not bool(metadata.get('prompt_conditioned_qformer', False))
        and int(metadata.get('max_train_batches', 0)) == 0
    )


# 1. Kiểm tra dữ liệu inference.
for path in (COCO_JSON, PROMPT_CACHE):
    assert path.is_file(), f'Thiếu Input: {path}'
assert COCO_IMAGES.is_dir(), f'Thiếu thư mục ảnh: {COCO_IMAGES}'
for part in range(1, 4):
    shard = VISUAL_CACHE / f'visual_part_{part:02d}_of_03.h5'
    assert shard.is_file(), f'Thiếu visual cache: {shard}'


# 2. Tìm checkpoint mà không quét thư mục ảnh/cache lớn.
expected_names = {CHECKPOINT_NAME, Path(CHECKPOINT_NAME).stem}
skip_directories = {
    'images', 'visual-cache', 'visual_cache', 'prompt-cache',
    'prompt_cache', 'Image_Captioning', '__pycache__',
}
checkpoint_sources = set()
for root, directories, files in os.walk(INPUT_ROOT):
    directories[:] = [
        name for name in directories if name not in skip_directories
    ]
    root_path = Path(root)
    if CHECKPOINT_NAME in files:
        checkpoint_sources.add(root_path / CHECKPOINT_NAME)
    if 'data.pkl' in files:
        for parent in (root_path, *list(root_path.parents)[:3]):
            if parent.name in expected_names:
                checkpoint_sources.add(parent)
                break

print('\nCheckpoint candidates:')
matches = []
for index, source in enumerate(sorted(checkpoint_sources, key=str), 1):
    try:
        restored = restore_torch_archive(source, index)
        metadata = torch.load(
            restored, map_location='cpu', weights_only=False
        )
    except Exception as error:
        print('- Bỏ qua:', source, repr(error))
        continue
    profile = {
        'epoch': metadata.get('epoch'),
        'adapter': metadata.get('visual_adapter'),
        'queries': metadata.get('num_visual_queries'),
        'layers': metadata.get('qformer_layers'),
        'region': metadata.get('alignment_weight', 0.0),
        'itc': metadata.get('itc_weight', 0.0),
        'prompt_conditioned': metadata.get(
            'prompt_conditioned_qformer', False
        ),
        'smoke_batches': metadata.get('max_train_batches', 0),
    }
    print('-', source, profile)
    if is_old_qformer(metadata):
        matches.append(restored)
    del metadata

assert len(matches) == 1, (
    'Cần đúng một checkpoint Q-Former 32q/2l epoch 10, không ITC, '
    f'không prompt-conditioned; tìm thấy {len(matches)}: {matches}'
)
CHECKPOINT = matches[0]
print('\nSelected checkpoint:', CHECKPOINT)


# 3. Clone đúng phiên bản code hỗ trợ candidate generation và CLIPScore.
if not REPO.exists():
    run([
        'git', 'clone',
        'https://github.com/Supzxjee/Image_Captioning.git',
        REPO,
    ])
run(['git', 'fetch', 'origin'], cwd=REPO)
run(['git', 'checkout', '--detach', COMMIT], cwd=REPO)
run(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO)
run([
    sys.executable, '-m', 'pip', 'install', '-q',
    '-r', REPO / 'requirements.txt',
])


# 4. Sinh tối đa năm beam candidates cho toàn bộ 5.000 ảnh test.
run([
    sys.executable, '-u', REPO / 'train_h1_2_gated.py',
    '--mode', 'candidates',
    '--candidate-count', '5',
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

evaluation_dir = Path('/kaggle/working') / EXPERIMENT / 'evaluation'
raw_candidates = evaluation_dir / 'test_5000_beam5_candidates.json'
ground_truth = evaluation_dir / 'test_5000_gt_candidates.json'
assert raw_candidates.is_file(), raw_candidates
assert ground_truth.is_file(), ground_truth


# 5. Chấm CLIPScore ngay trong cùng Save Version.
scored_candidates = (
    evaluation_dir / 'test_5000_beam5_clipscore_candidates.json'
)
run([
    sys.executable, '-u', REPO / 'score_clip_candidates.py',
    '--candidates', raw_candidates,
    '--visual-cache', VISUAL_CACHE,
    '--output', scored_candidates,
    '--model', 'openai/clip-vit-base-patch16',
    '--batch-size', '64',
    '--device', 'cuda',
])


# 6. Kiểm tra output trước khi Save Version.
raw = json.loads(raw_candidates.read_text(encoding='utf-8'))
scored = json.loads(scored_candidates.read_text(encoding='utf-8'))
assert raw['metadata']['split'] == 'test'
assert raw['metadata']['count'] == 5000
assert raw['metadata']['visual_adapter'] == 'qformer'
assert raw['metadata']['prompt_conditioned_qformer'] is False
assert len(raw['data']) == len(scored['data']) == 5000
assert all(row['candidates'] for row in raw['data'])
assert all(
    'clipscore' in candidate
    for row in scored['data']
    for candidate in row['candidates']
)

print('\nREBUILD PASS')
print('Raw candidates:', raw_candidates)
print('Scored candidates:', scored_candidates)
print('Ground truth:', ground_truth)
print('Hãy Save Version và gắn output này vào notebook ablation.')
```

## Output bắt buộc

Sau khi cell in `REBUILD PASS`, Save Version phải chứa:

```text
qformer_test_candidates_rebuilt/evaluation/
├── test_5000_beam5_candidates.json
├── test_5000_beam5_clipscore_candidates.json
└── test_5000_gt_candidates.json
```

Gắn Save Version này vào notebook rồi chạy cell trong
`QFORMER_FIXED_RERANKING_ABLATION.md`. Workflow ablation sẽ tự ưu tiên bản đã có
CLIPScore. Không cần tải ba file về máy rồi upload lại.
