# Q-Former: sinh nhiều ứng viên, Object Consistency Checker và re-ranking

## Mục tiêu

Thí nghiệm này không train lại Q-Former. Beam search hiện tại đã giữ 5 beam nhưng
chỉ trả caption đứng đầu; phiên bản mới lưu tối đa 5 caption riêng biệt cho mỗi ảnh.
Sau đó Object Consistency Checker (OCC) đối chiếu các lớp COCO được nhắc trong mỗi
caption với object cache YOLO.

```text
Q-Former checkpoint
        ↓
5 beam caption candidates + log-probability
        ↓
Nhận diện tên object COCO trong từng caption
        ↓
Đối chiếu object YOLO có confidence ≥ 0.5
        ↓
language score + object consistency score
        ↓
chọn một caption
```

Alias thông dụng được chuẩn hóa, ví dụ `man → person`, `bike → bicycle`,
`sofa → couch`. Caption nhắc một lớp không có trong YOLO bị phạt; lớp được YOLO
hỗ trợ nhận điểm theo confidence. Caption không nhắc lớp COCO nào nhận điểm object
trung tính.

Trọng số re-ranking chỉ được chọn trên validation. `weight=0` là đối chứng và phải
tái tạo đúng metric Q-Former validation trước đây. Không chọn trọng số bằng test.

## Cell Kaggle: validation và chọn trọng số

Add các Input sau trước khi chạy:

- MS COCO 2014 và `dataset_coco.json`.
- Prompt cache và ba visual-cache shards.
- `objectdetectioncache.json`.
- Output/Dataset chứa checkpoint epoch 10 của `qformer_32q_2l_gated`.

Cell hỗ trợ checkpoint `.pth` còn nguyên hoặc checkpoint bị Kaggle bung thành thư
mục.

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
EXPERIMENT = 'qformer_occ_validation'


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


# 2. Tìm checkpoint Q-Former baseline; tự bỏ qua checkpoint ITC/region/smoke.
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


# 3. Clone đúng phiên bản code.
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


# 4. Sinh tối đa 5 beam candidates cho mỗi ảnh validation.
common = [
    '--checkpoint', CHECKPOINT,
    '--split', 'val',
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
]
run([
    sys.executable, '-u', 'train_h1_2_gated.py',
    '--mode', 'candidates',
    '--candidate-count', '5',
] + common)


# 5. Tune OCC weight trên validation; CIDEr là tiêu chí chính.
experiment_dir = Path('/kaggle/working') / EXPERIMENT
candidates_path = experiment_dir / 'evaluation/val_5000_beam5_candidates.json'
ground_truth_path = experiment_dir / 'evaluation/val_5000_gt_candidates.json'
rerank_dir = experiment_dir / 'reranking'
assert candidates_path.is_file(), candidates_path
assert ground_truth_path.is_file(), ground_truth_path

run([
    sys.executable, '-u', 'rerank_candidates.py',
    '--candidates', candidates_path,
    '--ground-truth', ground_truth_path,
    '--detections', DETECTIONS,
    '--output-dir', rerank_dir,
    '--weights', '0,0.05,0.1,0.2,0.3,0.4',
    '--min-confidence', '0.5',
    '--hallucination-penalty', '1.0',
    '--objects-field', 'objects',
    '--name-key', 'label',
    '--confidence-key', 'conf',
])


# 6. Kiểm tra đối chứng và in trọng số được chọn.
summary_path = rerank_dir / 'val_occ_summary.json'
summary = json.loads(summary_path.read_text(encoding='utf-8'))
baseline = next(item for item in summary['results'] if item['weight'] == 0.0)
expected = {
    'Bleu_1': 0.7671533571628831,
    'Bleu_2': 0.608853373137689,
    'Bleu_3': 0.47444502761316537,
    'Bleu_4': 0.3703219250600793,
    'METEOR': 0.28301217655387967,
    'ROUGE_L': 0.571214149949258,
    'CIDEr': 1.1740115041643502,
}
for name, value in expected.items():
    assert abs(baseline['metrics'][name] - value) < 1e-6, (
        name, baseline['metrics'][name], value
    )

print('\nBaseline weight=0 PASS')
print('Best validation weight:', summary['best_weight'])
print('Best validation metrics:')
print(json.dumps(summary['best_metrics'], indent=2))
print('Images changed for each weight:')
for item in summary['results']:
    print(item['weight'], '→', item['changed_images'])
print('Summary:', summary_path)
```

## Cách đọc kết quả

- Nếu `best_weight=0`, OCC không cải thiện CIDEr validation; giữ Q-Former gốc và
  báo cáo OCC như ablation âm.
- Nếu `best_weight>0`, ghi lại chính xác trọng số, metrics, số ảnh bị đổi và tỷ lệ
  object hallucination. Sau đó mới dùng đúng trọng số đó cho test.
- File `*_details.json` lưu toàn bộ candidates, object được hỗ trợ/bị nghi ngờ và
  lý do caption được chọn, nên có thể kiểm tra mẫu mà không chạy inference lại.

OCC chỉ đánh giá 80 lớp COCO. Nó không xác minh thuộc tính hoặc quan hệ, có thể
phạt nhầm khi YOLO bỏ sót object, và từ `orange` có thể là màu thay vì vật thể. Vì
vậy kết quả phải được trình bày như một heuristic re-ranking ablation.
