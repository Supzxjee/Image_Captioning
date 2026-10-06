# Đánh giá E2 Cascade Alignment trên test 5.000 ảnh

Workflow này chỉ inference và tính COCO metrics, không train lại. Trước khi chạy,
Add Input output của pilot E2 chứa checkpoint epoch 3, object semantic cache,
prompt cache, visual cache và MS COCO 2014.

```python
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import torch

REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = '25b6fa35da2fdd3c0b246e75958272dd557e37a6'

COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
PROMPT_CACHE = Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
VISUAL_CACHE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')

CHECKPOINT_NAME = 'model_h1_2_crossattn_epoch_3.pth'
EXPERIMENT = 'qformer_cascade_alignment_32q_2l_test'


def run(args, cwd=None):
    args = list(map(str, args))
    print('\nRunning:', ' '.join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def restore_torch_archive(source, index):
    """Return a .pth whether Kaggle kept or expanded the PyTorch archive."""
    if source.is_file():
        return source
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination = Path('/kaggle/working') / f'restored_e2_candidate_{index}.pth'
    data_pickles = list(source.rglob('data.pkl'))
    assert len(data_pickles) == 1, (source, data_pickles)
    archive_root = data_pickles[0].parent
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_STORED) as archive:
        for file in archive_root.rglob('*'):
            if file.is_file():
                archive.write(file, f'archive/{file.relative_to(archive_root).as_posix()}')
    return destination


def is_e2_checkpoint(meta):
    return (
        int(meta.get('epoch', -1)) == 3
        and meta.get('visual_adapter') == 'qformer'
        and int(meta.get('num_visual_queries', -1)) == 32
        and int(meta.get('qformer_layers', -1)) == 2
        and bool(meta.get('cascade_semantic_alignment', False))
        and not bool(meta.get('object_semantic_alignment', False))
        and not bool(meta.get('prompt_conditioned_qformer', False))
        and float(meta.get('alignment_weight', 0.0)) == 0.0
        and float(meta.get('itc_weight', 0.0)) == 0.0
        and int(meta.get('max_train_batches', 0)) == 0
    )


# 1. Kiểm tra dữ liệu.
for path in (COCO_JSON, PROMPT_CACHE):
    assert path.is_file(), f'Thiếu Input: {path}'
assert COCO_IMAGES.is_dir(), COCO_IMAGES
for part in range(1, 4):
    shard = VISUAL_CACHE / f'visual_part_{part:02d}_of_03.h5'
    assert shard.is_file(), shard

object_cache_matches = []
for path in Path('/kaggle/input').rglob('object_concepts.pt'):
    try:
        bundle = torch.load(path, map_location='cpu', weights_only=False)
        metadata = bundle.get('metadata', {})
        if (metadata.get('complete') and metadata.get('count') == 123287 and
                metadata.get('max_objects') == 10 and
                float(metadata.get('min_confidence')) == 0.5):
            object_cache_matches.append(path)
    except Exception as error:
        print('Bỏ qua object cache không đọc được:', path, repr(error))
assert len(object_cache_matches) == 1, (
    f'Cần đúng một object cache hoàn chỉnh, tìm thấy: {object_cache_matches}')
OBJECT_CACHE = object_cache_matches[0]
print('Object cache:', OBJECT_CACHE)


# 2. Tìm đúng checkpoint E2; checkpoint control epoch 3 sẽ bị loại.
input_root = Path('/kaggle/input')
expected_names = {CHECKPOINT_NAME, Path(CHECKPOINT_NAME).stem}
candidates = set(input_root.rglob(CHECKPOINT_NAME))
for data_pickle in input_root.rglob('data.pkl'):
    if data_pickle.parent.name in expected_names:
        candidates.add(data_pickle.parent)

matches = []
print('\nCheckpoint candidates:')
for index, candidate in enumerate(sorted(candidates, key=str), 1):
    print(' -', candidate)
    restored = restore_torch_archive(candidate, index)
    try:
        meta = torch.load(restored, map_location='cpu', weights_only=False)
    except Exception as error:
        print('   Bỏ qua:', repr(error))
        continue
    signature = {
        'epoch': meta.get('epoch'),
        'adapter': meta.get('visual_adapter'),
        'queries': meta.get('num_visual_queries'),
        'layers': meta.get('qformer_layers'),
        'cascade': bool(meta.get('cascade_semantic_alignment', False)),
        'object_alignment': bool(meta.get('object_semantic_alignment', False)),
        'max_train_batches': meta.get('max_train_batches', 0),
    }
    print('  ', signature)
    if is_e2_checkpoint(meta):
        matches.append((restored, candidate))
    del meta

assert len(matches) == 1, (
    f'Cần đúng một checkpoint E2 full epoch 3, tìm thấy {len(matches)}: {matches}')
CHECKPOINT, CHECKPOINT_SOURCE = matches[0]
print('\nE2 checkpoint verified:', CHECKPOINT_SOURCE)


# 3. Checkout mã nguồn tương thích.
if REPO.exists() and not (REPO / '.git').is_dir():
    shutil.rmtree(REPO)
if not REPO.exists():
    REPO.mkdir(parents=True)
    run(['git', 'init'], cwd=REPO)
    run(['git', 'remote', 'add', 'origin',
         'https://github.com/Supzxjee/Image_Captioning.git'], cwd=REPO)
for attempt in range(1, 4):
    try:
        run(['git', 'fetch', '--depth', '30', 'origin', 'main'], cwd=REPO)
        break
    except subprocess.CalledProcessError:
        if attempt == 3:
            raise
        print(f'Fetch lần {attempt}/3 thất bại; thử lại sau 10 giây.', flush=True)
        time.sleep(10)
run(['git', 'checkout', '--detach', COMMIT], cwd=REPO)
run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], cwd=REPO)


# 4. Đánh giá test 5.000 ảnh với beam size 5.
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
    '--cascade-semantic-alignment',
    '--object-prompt-cache-path', OBJECT_CACHE,
    '--experiment-name', EXPERIMENT,
], cwd=REPO)


# 5. Xác nhận output test đầy đủ.
evaluation_dir = Path('/kaggle/working') / EXPERIMENT / 'evaluation'
predictions_path = evaluation_dir / 'test_5000_captions_h1_2_gated.json'
ground_truth_path = evaluation_dir / 'test_5000_gt_h1_2_gated.json'
metrics_path = evaluation_dir / 'test_5000_metrics_h1_2_gated.json'
for path in (predictions_path, ground_truth_path, metrics_path):
    assert path.is_file(), path

predictions = json.loads(predictions_path.read_text(encoding='utf-8'))
metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
assert len(predictions) == 5000
assert len({item['image_id'] for item in predictions}) == 5000

print('\nE2 CASCADE TEST METRICS')
print(json.dumps(metrics, indent=2))
print('\nPredictions:', predictions_path)
print('Ground truth:', ground_truth_path)
print('Metrics:', metrics_path)
```

Chạy bằng một GPU T4. Giữ lại toàn bộ thư mục
`/kaggle/working/qformer_cascade_alignment_32q_2l_test/evaluation` để tính CHAIR
mà không phải sinh caption lần nữa.
