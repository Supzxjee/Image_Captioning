# Q-Former + OCC: đánh giá test độc lập bằng Save Version

Workflow này dùng `weight=0.1` đã chọn trên validation. Nó không chạy lại
validation và không thử trọng số khác trên test. Tạo notebook Kaggle mới, gắn các
Input cũ rồi chạy toàn bộ cell bằng Save Version.

```python
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import torch

REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = '3f047e0'

COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
PROMPT_CACHE = Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
VISUAL_CACHE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')
DETECTIONS = Path(
    '/kaggle/input/datasets/ducanh2403/'
    'objectdetectionecache/objectdetectioncache.json'
)

CHECKPOINT_NAME = 'model_h1_2_crossattn_epoch_10.pth'
FIXED_WEIGHT = 0.1
EXPERIMENT = 'qformer_occ_test_w010'


def run(args):
    args = list(map(str, args))
    print('\nRunning:', ' '.join(args), flush=True)
    subprocess.run(args, check=True)


def restore_torch_archive(source, index):
    if source.is_file():
        return source
    if not source.is_dir():
        raise FileNotFoundError(source)
    data_pickles = list(source.rglob('data.pkl'))
    assert len(data_pickles) == 1, (
        f'Expected one data.pkl below {source}, found {len(data_pickles)}'
    )
    archive_root = data_pickles[0].parent
    destination = Path('/kaggle/working') / f'restored_candidate_{index}.pth'
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_STORED) as archive:
        for file in archive_root.rglob('*'):
            if file.is_file():
                archive.write(file, f'archive/{file.relative_to(archive_root).as_posix()}')
    return destination


def is_qformer_baseline(metadata):
    return (
        int(metadata.get('epoch', -1)) == 10
        and metadata.get('visual_adapter') == 'qformer'
        and int(metadata.get('num_visual_queries', -1)) == 32
        and int(metadata.get('qformer_layers', -1)) == 2
        and float(metadata.get('alignment_weight', 0.0)) == 0.0
        and float(metadata.get('itc_weight', 0.0)) == 0.0
        and int(metadata.get('max_train_batches', 0)) == 0
    )


# 1. Kiểm tra Input.
for path in (COCO_JSON, PROMPT_CACHE, DETECTIONS):
    assert path.is_file(), f'Thiếu Input: {path}'
assert COCO_IMAGES.is_dir(), COCO_IMAGES
for part in range(1, 4):
    shard = VISUAL_CACHE / f'visual_part_{part:02d}_of_03.h5'
    assert shard.is_file(), f'Thiếu visual cache: {shard}'


# 2. Tìm đúng checkpoint Q-Former không ITC; bỏ qua checkpoint khác.
input_root = Path('/kaggle/input')
expected_names = {CHECKPOINT_NAME, Path(CHECKPOINT_NAME).stem}
candidates = set(input_root.rglob(CHECKPOINT_NAME))
for data_pickle in input_root.rglob('data.pkl'):
    if data_pickle.parent.name in expected_names:
        candidates.add(data_pickle.parent)
candidates = sorted(candidates, key=str)

matches = []
for index, source in enumerate(candidates, 1):
    restored = restore_torch_archive(source, index)
    try:
        metadata = torch.load(restored, map_location='cpu', weights_only=False)
    except Exception as error:
        print('Bỏ qua checkpoint không đọc được:', source, repr(error))
        continue
    print(source, {
        'adapter': metadata.get('visual_adapter'),
        'queries': metadata.get('num_visual_queries'),
        'layers': metadata.get('qformer_layers'),
        'itc': metadata.get('itc_weight', 0.0),
        'region': metadata.get('alignment_weight', 0.0),
        'smoke_batches': metadata.get('max_train_batches', 0),
    })
    if is_qformer_baseline(metadata):
        matches.append(restored)
    del metadata

assert len(matches) == 1, (
    f'Cần đúng một Q-Former baseline checkpoint, tìm thấy {len(matches)}.'
)
CHECKPOINT = matches[0]
print('Selected checkpoint:', CHECKPOINT)


# 3. Clone code có candidate generation và OCC.
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


# 4. Sinh 5 candidates trên test split 5.000 ảnh.
run([
    sys.executable, '-u', 'train_h1_2_gated.py',
    '--mode', 'candidates',
    '--candidate-count', '5',
    '--checkpoint', CHECKPOINT,
    '--split', 'test',
    '--limit', '0',
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


# 5. Áp dụng duy nhất weight=0.1 đã chọn bằng validation.
experiment_dir = Path('/kaggle/working') / EXPERIMENT
test_candidates = experiment_dir / 'evaluation/test_5000_beam5_candidates.json'
test_ground_truth = experiment_dir / 'evaluation/test_5000_gt_candidates.json'
rerank_dir = experiment_dir / 'reranking'
assert test_candidates.is_file(), test_candidates
assert test_ground_truth.is_file(), test_ground_truth

run([
    sys.executable, '-u', 'rerank_candidates.py',
    '--candidates', test_candidates,
    '--ground-truth', test_ground_truth,
    '--detections', DETECTIONS,
    '--output-dir', rerank_dir,
    '--weight', FIXED_WEIGHT,
    '--min-confidence', '0.5',
    '--hallucination-penalty', '1.0',
    '--objects-field', 'objects',
    '--name-key', 'label',
    '--confidence-key', 'conf',
])


# 6. Kiểm tra và in output cuối.
summary_path = rerank_dir / 'test_occ_summary.json'
summary = json.loads(summary_path.read_text(encoding='utf-8'))
assert summary['mode'] == 'apply'
assert len(summary['results']) == 1
assert abs(summary['best_weight'] - FIXED_WEIGHT) < 1e-12
fixed = summary['results'][0]

print('\nFixed OCC test weight:', fixed['weight'])
print('Changed test images:', fixed['changed_images'])
print('Recognized mentions:', fixed['recognized_mentions'])
print('Supported mentions:', fixed['supported_mentions'])
print('Hallucinated mentions:', fixed['hallucinated_mentions'])
print('TEST metrics:')
print(json.dumps(fixed['metrics'], indent=2))
print('\nPredictions:', fixed['predictions'])
print('Details:', fixed['details'])
print('Summary:', summary_path)
```

Output cần giữ trong `/kaggle/working/qformer_occ_test_w010/`:

- `evaluation/test_5000_beam5_candidates.json`;
- `reranking/test_5000_occ_w0p100_predictions.json`;
- `reranking/test_5000_occ_w0p100_details.json`;
- `reranking/test_occ_summary.json`.

So sánh với Q-Former gốc trên test: BLEU-1 `0.7674`, BLEU-4 `0.3707`, METEOR
`0.2850`, ROUGE-L `0.5731`, CIDEr `1.1882`. Không thay đổi weight sau khi xem
kết quả test.
