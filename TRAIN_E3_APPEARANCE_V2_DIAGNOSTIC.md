# E3 Appearance V2: pilot nhỏ và diagnostic

V1 diagnostic: 0/100 caption thay đổi khi tắt nhánh, query gate 0.000210, thay ảnh giữ tọa độ vẫn ưu tiên nhãn heuristic gốc ở 137/138 quan hệ. V2 thay đổi objective để kiểm tra shortcut trước khi train full.

- Scorer riêng chỉ nhận subject/object/union visual features, không ghép vector relative_geometry. Bounding boxes vẫn dùng để pooling, nên đây không phải phép loại bỏ toàn bộ tín hiệu vị trí.
- Text targets là CLIP EOS vectors cố định từ cache; scorer không được học biến đổi text targets để giảm loss. Predicate margin được chặn ở nửa khoảng cách giữa hai normalized text targets để tránh margin không thể đạt.
- Appearance loss chỉ áp dụng holding, wearing, carrying, riding. Spatial relations vẫn giữ trong caption context nhưng không được coi là bằng chứng action grounding.
- Thêm loss correct-image vs wrong-image cùng relation text và boxes. Loại các đối chứng có positive phrase gần trùng (cosine >= 0.9). Đây là weak negative; thiếu annotation không chứng minh quan hệ vắng mặt.
- Giữ gate hiện tại. Không ép mở gate. Kiểm tra sau pilot xem loss mới có giúp nhánh caption hay không.
- Dùng cache hiện tại, không cần sinh cache mới. Khởi tạo lại từ baseline Q-Former epoch 10, không tiếp tục checkpoint E3 V1 đã có gate gần đóng.

## Chạy trên Kaggle

Add Input dataset `qformer-epoch10`, prompt V2.1, output notebook `build-grounded-relation-v1`, visual cache và MS COCO. Dùng một GPU, bật Internet. Chạy cell dưới đây: 1 epoch giới hạn 16 batches (~512 ảnh train), sau đó tự diagnostic 100 ảnh validation. Đây là thử nghiệm nhỏ, chưa dùng để so sánh CIDEr với pilot 3 epoch.

```python
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CHECKPOINT = Path('/kaggle/input/datasets/ducanh2403/qformer-epoch10/qformer_epoch10/model_h1_2_crossattn_epoch_10.pth')
PROMPT_CACHE = Path('/kaggle/input/datasets/ducanh2403/prompt-conditioned-qformer-32q-2l/relation_prompts_v2_1_full/prompt_clip_tokens_relation_v2_1.pt')
GROUNDED_CACHE = Path('/kaggle/input/notebooks/ducanh2403/build-grounded-relation-v1/grounded_relation_v1_full/grounded_relation_cache.pt')
VISUAL_CACHE_SOURCE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')

REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = '8c44631'
VISUAL_CACHE_LOCAL = Path('/tmp/visual-cache-e3-valid')
EXPERIMENT = 'e3_appearance_v2_diagnostic_500'
OUTPUT = Path('/kaggle/working') / EXPERIMENT


def run(args, cwd=None):
    args = list(map(str, args))
    print('\nRunning:', ' '.join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def stage(source):
    destination = VISUAL_CACHE_LOCAL / source.name
    total = source.stat().st_size
    if destination.is_file() and destination.stat().st_size == total:
        print('Already staged:', source.name, flush=True)
        return
    temporary = destination.with_suffix('.h5.partial')
    copied, started, last_log = 0, time.monotonic(), 0
    with source.open('rb') as reader, temporary.open('wb') as writer:
        while True:
            block = reader.read(64 * 1024 * 1024)
            if not block:
                break
            writer.write(block)
            copied += len(block)
            elapsed = time.monotonic() - started
            if elapsed - last_log >= 10 or copied == total:
                print(f'{source.name}: {copied / total:.1%} | {copied / max(elapsed, 0.001) / 1024**2:.1f} MiB/s', flush=True)
                last_log = elapsed
    assert temporary.stat().st_size == total, temporary
    temporary.replace(destination)


print('E3 Appearance V2 bootstrap started.', flush=True)
for path in (CHECKPOINT, PROMPT_CACHE, GROUNDED_CACHE, COCO_JSON):
    assert path.is_file(), f'Missing file: {path}'
assert COCO_IMAGES.is_dir(), COCO_IMAGES
assert VISUAL_CACHE_SOURCE.is_dir(), VISUAL_CACHE_SOURCE

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['PYTHONUNBUFFERED'] = '1'
REPO.parent.mkdir(parents=True, exist_ok=True)
if not REPO.exists():
    run(['git', 'clone', 'https://github.com/Supzxjee/Image_Captioning.git', REPO])
assert (REPO / '.git').is_dir(), REPO
run(['git', 'fetch', '--depth', '30', 'origin', 'main'], cwd=REPO)
run(['git', 'checkout', '--detach', COMMIT], cwd=REPO)
run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], cwd=REPO)

VISUAL_CACHE_LOCAL.mkdir(parents=True, exist_ok=True)
parts = [VISUAL_CACHE_SOURCE / f'visual_part_{i:02d}_of_03.h5' for i in range(1, 4)]
for source in parts:
    assert source.is_file(), source
needed = sum(p.stat().st_size for p in parts if not (VISUAL_CACHE_LOCAL / p.name).is_file()
             or (VISUAL_CACHE_LOCAL / p.name).stat().st_size != p.stat().st_size)
assert shutil.disk_usage('/tmp').free > needed + 5 * 1024**3, 'Not enough space in /tmp.'
for source in parts:
    stage(source)


# Single GPU: 16 batches x 32 images, approximately 512 training images.
common = [
    '--dataset-json-path', COCO_JSON, '--base-path', COCO_IMAGES,
    '--prompt-cache-path', PROMPT_CACHE,
    '--visual-cache', VISUAL_CACHE_LOCAL, '--visual-cache-id-key', 'coco_id',
    '--visual-adapter', 'qformer', '--num-visual-queries', '32', '--qformer-layers', '2',
    '--grounded-relation-alignment', '--grounded-relation-cache-path', GROUNDED_CACHE,
    '--grounded-relation-version', 'appearance_v2', '--relation-alignment-margin', '0.2',
    '--seed', '42', '--num-workers', '0',
]
run([
    sys.executable, '-u', 'train_h1_2_gated.py',
    '--mode', 'train', '--checkpoint', CHECKPOINT,
    '--epochs', '1', '--max-train-batches', '16', '--batch-size', '32',
    '--relation-alignment-weight', '0.1', '--relation-wrong-image-weight', '0.5',
    '--experiment-name', EXPERIMENT,
] + common, cwd=REPO)

V2_CHECKPOINT = OUTPUT / 'checkpoints/model_h1_2_crossattn_epoch_1.pth'
assert V2_CHECKPOINT.is_file(), V2_CHECKPOINT
history = json.loads((OUTPUT / 'train_history_h1_2.json').read_text(encoding='utf-8'))
last = history[-1]
assert last['relation_actions_per_batch'] > 0, 'No action relations trained.'
assert last['relation_wrong_images_per_batch'] > 0, 'No eligible cross-image negatives. Inspect cache/filter.'
print('E3 V2 TRAINING CHECK:')
print(json.dumps(last, indent=2), flush=True)

# Validation-only sensitivity check; not a caption-quality benchmark.
REVIEW = OUTPUT / 'validation_diagnostic_100'
run([
    sys.executable, '-u', 'diagnose_grounded_relations.py',
    '--mode', 'predict', '--split', 'val', '--limit', '0',
    '--checkpoint', V2_CHECKPOINT, '--expected-epoch', '1',
    '--sample-size', '100', '--output', REVIEW,
] + common, cwd=REPO)

summary_path = REVIEW / 'diagnostic_summary.json'
summary = json.loads(summary_path.read_text(encoding='utf-8'))
summary['code_commit'] = COMMIT
summary['warm_start'] = str(CHECKPOINT)
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
archive = shutil.make_archive(str(OUTPUT), 'zip', OUTPUT)
print('E3 APPEARANCE V2 DIAGNOSTIC COMPLETE')
print(json.dumps(summary, indent=2, ensure_ascii=False))
print('Download ZIP:', archive)

```

Output: `/kaggle/working/e3_appearance_v2_diagnostic_500.zip`, chứa checkpoint epoch 1, history và validation diagnostic (summary, scores, review HTML, thumbnails).

Gửi ZIP và Kaggle code version sau khi chạy. Kiểm tra `relation_actions_per_batch`, `relation_wrong_images_per_batch`, `action_ranking`, gate và residual. Chỉ chạy pilot lớn sau khi xác nhận loss có gradient thị giác và đủ negative hợp lệ; không tự coi loss giảm là thành công.

Kiểm tra local: 13 tests relation (bao gồm V1 regression) và 5 tests CLI/workflow đã PASS. Chưa chạy pilot V2 trên dữ liệu Kaggle. Đây là objective prototype; chưa có bằng chứng chất lượng caption tăng hoặc tính mới nghiên cứu được xác nhận.
