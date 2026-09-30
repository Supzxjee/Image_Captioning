# Đánh giá test: Prompt-Conditioned Q-Former + CLIPScore + OCC

Trọng số đã khóa bằng validation:

```text
decoder = 0.3
CLIPScore = 0.4
OCC = 0.3
```

Notebook này không train và không tune. Nó sinh năm candidates trên test, tính
CLIPScore, sau đó đánh giá đúng hai cấu hình: rank 0 và trọng số đã khóa.

## Input cần gắn

- Output/Dataset chứa checkpoint epoch 10 của `prompt_conditioned_qformer_32q_2l`;
- MS COCO 2014 và `dataset_coco.json`;
- prompt cache object–relation;
- ba visual-cache shards;
- `objectdetectioncache.json`.

Chọn một GPU và bật Internet.

## Cell Kaggle đầy đủ

```python
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import torch

REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = 'badda5a'

COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
PROMPT_CACHE = Path(
    '/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt'
)
VISUAL_CACHE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')
DETECTIONS = Path(
    '/kaggle/input/datasets/ducanh2403/'
    'objectdetectionecache/objectdetectioncache.json'
)

EXPERIMENT = 'prompt_conditioned_qformer_fixed_test'
CHECKPOINT_NAME = 'model_h1_2_crossattn_epoch_10.pth'
CLIP_WEIGHT = 0.4
OBJECT_WEIGHT = 0.3


def run(args, cwd=None):
    args = list(map(str, args))
    print('\nRunning:', ' '.join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


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
    destination = Path('/kaggle/working') / f'restored_prompt_qformer_{index}.pth'
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_STORED) as archive:
        for file in archive_root.rglob('*'):
            if file.is_file():
                relative = file.relative_to(archive_root).as_posix()
                archive.write(file, f'archive/{relative}')
    return destination


def is_target_checkpoint(metadata):
    return (
        int(metadata.get('epoch', -1)) == 10
        and metadata.get('visual_adapter') == 'qformer'
        and int(metadata.get('num_visual_queries', -1)) == 32
        and int(metadata.get('qformer_layers', -1)) == 2
        and bool(metadata.get('prompt_conditioned_qformer', False))
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
    assert shard.is_file(), shard


# 2. Tìm đúng checkpoint full, hỗ trợ file .pth bị Kaggle bung thành thư mục.
input_root = Path('/kaggle/input')
expected_names = {CHECKPOINT_NAME, Path(CHECKPOINT_NAME).stem}
candidates = set(input_root.rglob(CHECKPOINT_NAME))
for data_pickle in input_root.rglob('data.pkl'):
    if data_pickle.parent.name in expected_names:
        candidates.add(data_pickle.parent)

matches = []
print('\nCheckpoint candidates:')
for index, source in enumerate(sorted(candidates, key=str), 1):
    restored = restore_torch_archive(source, index)
    try:
        metadata = torch.load(restored, map_location='cpu', weights_only=False)
    except Exception as error:
        print('Bỏ qua:', source, repr(error))
        continue
    profile = {
        'epoch': metadata.get('epoch'),
        'adapter': metadata.get('visual_adapter'),
        'queries': metadata.get('num_visual_queries'),
        'layers': metadata.get('qformer_layers'),
        'prompt_conditioned': metadata.get('prompt_conditioned_qformer', False),
        'itc': metadata.get('itc_weight', 0.0),
        'region': metadata.get('alignment_weight', 0.0),
        'smoke_batches': metadata.get('max_train_batches', 0),
    }
    print('-', source, profile)
    if is_target_checkpoint(metadata):
        matches.append(restored)
    del metadata

assert len(matches) == 1, (
    f'Cần đúng một checkpoint Prompt-Conditioned Q-Former full, '
    f'tìm thấy {len(matches)}.'
)
CHECKPOINT = matches[0]
print('Selected checkpoint:', CHECKPOINT)


# 3. Clone code cố định.
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


# 4. Sinh đúng 5 candidates cho test split.
common = [
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
    '--prompt-conditioned-qformer',
    '--experiment-name', EXPERIMENT,
]
run([
    sys.executable, '-u', 'train_h1_2_gated.py',
    '--mode', 'candidates',
    '--candidate-count', '5',
] + common)

experiment_dir = Path('/kaggle/working') / EXPERIMENT
evaluation_dir = experiment_dir / 'evaluation'
raw_candidates = evaluation_dir / 'test_5000_beam5_candidates.json'
ground_truth = evaluation_dir / 'test_5000_gt_candidates.json'
assert raw_candidates.is_file(), raw_candidates
assert ground_truth.is_file(), ground_truth


# 5. Tính CLIPScore từ visual cache.
scored_candidates = evaluation_dir / 'test_5000_beam5_clipscore_candidates.json'
run([
    sys.executable, '-u', 'score_clip_candidates.py',
    '--candidates', raw_candidates,
    '--visual-cache', VISUAL_CACHE,
    '--output', scored_candidates,
    '--model', 'openai/clip-vit-base-patch16',
    '--batch-size', '64',
    '--device', 'cuda',
])


# 6. Đánh giá rank 0 để có baseline test của đúng checkpoint.
baseline_dir = experiment_dir / 'test_baseline_rank0'
run([
    sys.executable, '-u', 'rerank_multiscore.py',
    '--candidates', scored_candidates,
    '--ground-truth', ground_truth,
    '--detections', DETECTIONS,
    '--output-dir', baseline_dir,
    '--clip-weight', '0.0',
    '--object-weight', '0.0',
    '--min-confidence', '0.5',
])


# 7. Áp dụng đúng một bộ weights đã khóa trên validation.
final_dir = experiment_dir / 'test_fixed_d030_c040_o030'
run([
    sys.executable, '-u', 'rerank_multiscore.py',
    '--candidates', scored_candidates,
    '--ground-truth', ground_truth,
    '--detections', DETECTIONS,
    '--output-dir', final_dir,
    '--clip-weight', CLIP_WEIGHT,
    '--object-weight', OBJECT_WEIGHT,
    '--min-confidence', '0.5',
    '--hallucination-penalty', '1.0',
    '--objects-field', 'objects',
    '--name-key', 'label',
    '--confidence-key', 'conf',
])


# 8. Kiểm tra và in kết quả cuối.
baseline_summary = json.loads(
    (baseline_dir / 'test_multiscore_summary.json').read_text(encoding='utf-8')
)
final_summary = json.loads(
    (final_dir / 'test_multiscore_summary.json').read_text(encoding='utf-8')
)
baseline = baseline_summary['results'][0]
final = final_summary['results'][0]
assert baseline['changed_images'] == 0
assert abs(final['decoder_weight'] - 0.3) < 1e-12
assert final['clip_weight'] == 0.4
assert final['object_weight'] == 0.3

print('\nTEST BASELINE METRICS')
print(json.dumps(baseline['metrics'], indent=2))
print('\nTEST FIXED RE-RANKING METRICS')
print(json.dumps(final['metrics'], indent=2))
print('\nChanged images:', final['changed_images'])
print('Recognized mentions:', final['recognized_mentions'])
print('Supported mentions:', final['supported_mentions'])
print('Hallucinated mentions:', final['hallucinated_mentions'])
print('Predictions:', final['predictions'])
print('Details:', final['details'])
print('Hoàn tất:', experiment_dir)
```

## Kết quả test 5.000 ảnh

| Mô hình | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---:|---:|---:|---:|---:|---:|---:|
| Prompt-Conditioned rank 0 | 0.759712 | 0.598945 | 0.463337 | **0.359790** | 0.281008 | 0.566814 | 1.151742 |
| + fixed re-ranking 0.3/0.4/0.3 | **0.762326** | **0.603126** | **0.464612** | 0.356083 | **0.287345** | **0.571062** | **1.188570** |

Re-ranking đổi 2.885/5.000 caption. So với rank 0, CIDEr tăng `0,036827`,
METEOR tăng `0,006337`, ROUGE-L tăng `0,004248`, BLEU-1/2/3 tăng và BLEU-4
giảm `0,003708`. Caption cuối có 7.127 object mentions, gồm 6.598 được YOLO hỗ
trợ và 529 bị nghi ngờ (`7,42%`). Đây là thống kê OCC, chưa phải CHAIR.

So với Q-Former cũ trên test, pipeline mới tăng CIDEr `0,000325` và METEOR
`0,002306`, nhưng giảm BLEU-1/2/3/4 và ROUGE-L. Do đó đóng góp thực nghiệm rõ
nhất nằm ở re-ranking; prompt conditioning đơn lẻ không cải thiện caption model.
