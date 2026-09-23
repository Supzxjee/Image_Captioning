# Thí nghiệm căn chỉnh object–region

Thí nghiệm giữ nguyên hai nhánh prompt/ảnh, cross-attention, Gate, decoder và
caption loss. Phần mới lấy patch token CLIP nằm trong từng bounding box YOLO,
mean-pool thành vector vùng rồi phân loại vùng đó bằng các CLIP text prototype
của object label.

```text
L_total = L_caption + lambda * L_region
```

`L_region` là cross-entropy trên cosine similarity giữa region feature và toàn
bộ label prototype. Box nhỏ không chứa tâm patch nào dùng patch gần tâm box nhất.
Confidence YOLO làm trọng số loss; padding region không tham gia loss.

Gate và region loss có vai trò khác nhau: Gate điều tiết lượng thông tin ảnh đưa
vào prompt; region loss trực tiếp ép vùng trong bounding box gần đúng object label.

## Kaggle: cell hoàn chỉnh

Cell dưới đây kiểm tra Input, clone đúng commit, cài dependency, tạo region
target cache, train và tùy chọn full test. Chạy tương tác với `SMOKE = True`
trước; nếu epoch đầu không có NaN/OOM, đổi thành `False` để chạy bản chính thức.

```python
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = '2258b46'

DETECTIONS = Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
PROMPT_CACHE = Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
VISUAL_CACHE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')
REGION_TARGETS = Path('/kaggle/working/region_targets_yolo.pt')

SMOKE = True                  # Sau khi kiểm tra thành công, đổi thành False.
EPOCHS = 1 if SMOKE else 10
EXPERIMENT = 'gated_region_align_smoke' if SMOKE else 'gated_region_align_l01'

def run(args):
    args = list(map(str, args))
    print('\nRunning:', ' '.join(args), flush=True)
    subprocess.run(args, check=True)

# 1. Kiểm tra toàn bộ Input trước khi tải model hoặc train.
for path in (DETECTIONS, COCO_JSON, PROMPT_CACHE):
    assert path.is_file(), f'Thiếu Input: {path}'
assert COCO_IMAGES.is_dir(), f'Sai thư mục ảnh: {COCO_IMAGES}'
for part in range(1, 4):
    shard = VISUAL_CACHE / f'visual_part_{part:02d}_of_03.h5'
    assert shard.is_file(), f'Thiếu visual cache: {shard}'

# 2. Clone và cố định đúng phiên bản code.
if not REPO.exists():
    run(['git', 'clone', 'https://github.com/Supzxjee/Image_Captioning.git', REPO])
os.chdir(REPO)
run(['git', 'fetch', 'origin'])
run(['git', 'checkout', '--detach', COMMIT])
run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'])

# 3. Tạo region target cache một lần trong phiên hiện tại.
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

# 4. Train. SMOKE chỉ chạy một epoch và không full test.
command = [
    sys.executable, '-u', 'train_h1_2_gated.py',
    '--mode', 'train',
    '--epochs', EPOCHS,
    '--seed', '42',
    '--batch-size', '32',
    '--num-workers', '0',
    '--dataset-json-path', COCO_JSON,
    '--base-path', COCO_IMAGES,
    '--prompt-cache-path', PROMPT_CACHE,
    '--visual-cache', VISUAL_CACHE,
    '--visual-cache-id-key', 'coco_id',
    '--visual-preprocessing', 'bilinear',
    '--visual-precision', 'fp32',
    '--region-targets-path', REGION_TARGETS,
    '--alignment-weight', '0.1',
    '--alignment-temperature', '0.07',
    '--max-regions', '10',
    '--experiment-name', EXPERIMENT,
]
if not SMOKE:
    command.append('--test-after-train')
run(command)

# 5. Kiểm tra output quan trọng.
experiment_dir = Path('/kaggle/working') / EXPERIMENT
history = experiment_dir / 'train_history_h1_2.json'
assert history.is_file(), f'Thiếu history: {history}'
print('\nHistory:', history.read_text(encoding='utf-8'), flush=True)
print('Region targets:', REGION_TARGETS, REGION_TARGETS.stat().st_size / 1e6, 'MB')
```

Lưu `region_targets_yolo.pt` thành Kaggle Dataset để những lần train sau không
phải đọc lại kích thước toàn bộ ảnh hay encode label.

## Các tham số của ablation đầu tiên

Cell trên thêm các tham số sau vào pipeline Gate hiện tại:

```text
--region-targets-path /kaggle/working/region_targets_yolo.pt
--alignment-weight 0.1
--alignment-temperature 0.07
--max-regions 10
--experiment-name gated_region_align_l01
```

Giữ seed 42, prompt object+relation, visual cache, 10 epoch và decoding giống
bản Gate đối chứng. Các ngưỡng region supervision được chọn từ audit: bỏ detection
confidence dưới 0.5, box nhỏ hơn 0.1% hoặc lớn hơn 90% diện tích ảnh. Việc lọc
chỉ áp dụng cho `L_region`; prompt object+relation đầu vào vẫn giữ nguyên.
Chạy smoke một epoch trước; log phải hiện riêng `caption` và `align`, cả hai hữu
hạn và giảm hợp lý. So sánh metric với Gate cùng seed, sau đó mới thử lambda 0.05
hoặc 0.2 trên validation; không chọn lambda bằng test.

## Nếu log dừng ở `Train batches per epoch`

Không dùng commit trước `2258b46` cho full train. Các bản cũ gộp patch theo từng
bounding box bằng vòng lặp Python, tạo hàng trăm lần đồng bộ GPU mỗi batch và có
thể làm một epoch kéo dài nhiều giờ. Commit `2258b46` vector hóa mask, pooling và
nearest-patch cho toàn bộ `(batch, regions, 196 patches)` trong một lượt. Dừng
kernel cũ, checkout commit này và chạy lại từ đầu; region target `.pt` đã tạo bằng
ngưỡng giống nhau có thể dùng lại.
