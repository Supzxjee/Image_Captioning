# Đánh giá E2 không Soft Selector trên test 5.000 ảnh

Workflow này chỉ inference và tính COCO metrics, không train lại. Add Input output
của `pilot_uniform` chứa checkpoint epoch 3, cùng prompt cache, visual cache,
MS COCO 2014 và object semantic cache. Chạy bằng một GPU T4.

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
COMMIT = 'd2273f4641e0b06d5aaa1a57776b0933d3ddd538'

COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
PROMPT_CACHE = Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
VISUAL_CACHE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')
DETECTIONS = Path(
    '/kaggle/input/datasets/ducanh2403/objectdetectionecache/'
    'objectdetectioncache.json')
BUILT_OBJECT_CACHE = Path('/kaggle/working/object_semantic_cache/object_concepts.pt')

CHECKPOINT_NAME = 'model_h1_2_crossattn_epoch_3.pth'
EXPERIMENT = 'e2_no_soft_selector_32q_2l_test'


def run(args, cwd=None):
    args = list(map(str, args))
    print('\nRunning:', ' '.join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def restore_torch_archive(source, index):
    if source.is_file():
        return source
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination = Path('/kaggle/working') / f'restored_uniform_candidate_{index}.pth'
    data_pickles = list(source.rglob('data.pkl'))
    assert len(data_pickles) == 1, (source, data_pickles)
    archive_root = data_pickles[0].parent
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_STORED) as archive:
        for file in archive_root.rglob('*'):
            if file.is_file():
                archive.write(file, f'archive/{file.relative_to(archive_root).as_posix()}')
    return destination


def is_uniform_checkpoint(meta):
    return (
        int(meta.get('epoch', -1)) == 3
        and meta.get('visual_adapter') == 'qformer'
        and int(meta.get('num_visual_queries', -1)) == 32
        and int(meta.get('qformer_layers', -1)) == 2
        and bool(meta.get('cascade_semantic_alignment', False))
        and meta.get('cascade_selector_mode') == 'uniform'
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
object_cache_matches = sorted(set(object_cache_matches), key=str)
if len(object_cache_matches) > 1:
    print('Tìm thấy nhiều bản sao object cache hoàn chỉnh:')
    for path in object_cache_matches:
        print(' -', path)
    print('Các file đều đã qua kiểm tra metadata; ưu tiên Kaggle Dataset ổn định.')
dataset_caches = [
    path for path in object_cache_matches
    if '/kaggle/input/datasets/' in path.as_posix()
]
OBJECT_CACHE = (dataset_caches[0] if dataset_caches
                else object_cache_matches[0] if object_cache_matches
                else None)
if OBJECT_CACHE is not None:
    print('Dùng object semantic cache:', OBJECT_CACHE)
if OBJECT_CACHE is None:
    assert DETECTIONS.is_file(), (
        'Không có object_concepts.pt và thiếu detection JSON: ' + str(DETECTIONS))


# 2. Tìm đúng checkpoint uniform; checkpoint E2 full và control bị loại.
expected_names = {CHECKPOINT_NAME, Path(CHECKPOINT_NAME).stem}
candidates = set(Path('/kaggle/input').rglob(CHECKPOINT_NAME))
for data_pickle in Path('/kaggle/input').rglob('data.pkl'):
    if data_pickle.parent.name in expected_names:
        candidates.add(data_pickle.parent)

matches = []
print('\nCheckpoint candidates:')
for index, candidate in enumerate(sorted(candidates, key=str), 1):
    restored = restore_torch_archive(candidate, index)
    try:
        meta = torch.load(restored, map_location='cpu', weights_only=False)
    except Exception as error:
        print(' -', candidate, '| bỏ qua:', repr(error))
        continue
    signature = {
        'epoch': meta.get('epoch'),
        'adapter': meta.get('visual_adapter'),
        'queries': meta.get('num_visual_queries'),
        'layers': meta.get('qformer_layers'),
        'cascade': bool(meta.get('cascade_semantic_alignment', False)),
        'selector': meta.get('cascade_selector_mode'),
        'max_train_batches': meta.get('max_train_batches', 0),
    }
    print(' -', candidate, '|', signature)
    if is_uniform_checkpoint(meta):
        matches.append((restored, candidate))
    del meta

assert matches, 'Không tìm thấy checkpoint E2 uniform epoch 3 hợp lệ.'
if len(matches) > 1:
    print('Tìm thấy nhiều bản sao checkpoint uniform hợp lệ:')
    for _, source in matches:
        print(' -', source)
    print('Ưu tiên checkpoint trong Kaggle Dataset ổn định.')
dataset_matches = [
    match for match in matches
    if '/kaggle/input/datasets/' in match[1].as_posix()
]
CHECKPOINT, CHECKPOINT_SOURCE = (
    dataset_matches[0] if dataset_matches else matches[0])
print('\nUniform checkpoint verified:', CHECKPOINT_SOURCE)


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
        time.sleep(10)
run(['git', 'checkout', '--detach', COMMIT], cwd=REPO)
run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], cwd=REPO)


# 4. Nếu cần, xây object semantic cache từ YOLO detection JSON.
if OBJECT_CACHE is None:
    run([
        sys.executable, '-u', 'build_object_concept_cache.py',
        '--source', DETECTIONS,
        '--dataset-json-path', COCO_JSON,
        '--output', BUILT_OBJECT_CACHE,
        '--objects-field', 'objects',
        '--name-key', 'label',
        '--confidence-key', 'conf',
        '--min-confidence', '0.5',
        '--max-objects', '10',
        '--batch-size', '256',
    ], cwd=REPO)
    OBJECT_CACHE = BUILT_OBJECT_CACHE


# 5. Đánh giá test 5.000 ảnh với beam size 5.
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
    '--cascade-selector-mode', 'uniform',
    '--object-prompt-cache-path', OBJECT_CACHE,
    '--experiment-name', EXPERIMENT,
], cwd=REPO)


# 6. Xác nhận output và ghi manifest để notebook CHAIR chọn đúng caption.
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

manifest = {
    'model': 'e2_no_soft_selector',
    'cascade_selector_mode': 'uniform',
    'checkpoint_source': str(CHECKPOINT_SOURCE),
    'predictions': predictions_path.name,
    'ground_truth': ground_truth_path.name,
    'metrics': metrics,
}
manifest_path = evaluation_dir / 'uniform_test_manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

print('\nE2 WITHOUT SOFT SELECTOR TEST METRICS')
print(json.dumps(metrics, indent=2))
print('\nPredictions:', predictions_path)
print('Ground truth:', ground_truth_path)
print('Metrics:', metrics_path)
print('Manifest:', manifest_path)
```

Giữ toàn bộ thư mục
`/kaggle/working/e2_no_soft_selector_32q_2l_test/evaluation`. Sau khi có kết quả,
chạy workflow CHAIR riêng bên dưới; không sinh caption lần thứ hai.
