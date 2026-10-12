# E3 Pair-Grounded diagnostic: 100 validation images

Chạy toàn bộ cell dưới đây trên Kaggle với Internet bật và GPU. Các đường dẫn input đã được chốt cho checkpoint E3 epoch 3, prompt V2.1 và grounded relation cache.

Cell dùng script ở commit `8eb21fe`. Nếu visual cache đã có trong `/tmp/visual-cache-e3-valid`, cell dùng lại các shard đó.

Diagnostic chọn ngẫu nhiên 100 ảnh validation với seed 42. Kết quả gồm điểm quan hệ gốc và counterfactual, thử thay đặc trưng bằng ảnh khác nhưng giữ nguyên tọa độ, evidence gate, query gate, mức thay đổi đặc trưng và caption khi bật/tắt nhánh Pair-Grounded. Tắt nhánh E3 vẫn giữ relation prompt V2.1; đây không phải phép thử loại bỏ mọi thông tin quan hệ. Quan hệ gốc là nhãn heuristic, cần kiểm tra bằng ảnh.

```python
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CHECKPOINT = Path('/kaggle/input/notebooks/ducanh2403/kltn-mscoco/pair_grounded_relation_32q_2l_dual_gpu_3ep/checkpoints/model_h1_2_crossattn_epoch_3.pth')
PROMPT_CACHE = Path('/kaggle/input/datasets/ducanh2403/prompt-conditioned-qformer-32q-2l/relation_prompts_v2_1_full/prompt_clip_tokens_relation_v2_1.pt')
GROUNDED_CACHE = Path('/kaggle/input/notebooks/ducanh2403/build-grounded-relation-v1/grounded_relation_v1_full/grounded_relation_cache.pt')
VISUAL_CACHE_SOURCE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')

REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = '8eb21fe'
VISUAL_CACHE_LOCAL = Path('/tmp/visual-cache-e3-valid')
OUTPUT = Path('/kaggle/working/e3_relation_diagnostic_100')


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

run([
    sys.executable, '-u', 'diagnose_grounded_relations.py',
    '--output', OUTPUT, '--sample-size', '100',
    '--mode', 'predict', '--split', 'val', '--limit', '0',
    '--checkpoint', CHECKPOINT,
    '--dataset-json-path', COCO_JSON, '--base-path', COCO_IMAGES,
    '--prompt-cache-path', PROMPT_CACHE,
    '--visual-cache', VISUAL_CACHE_LOCAL, '--visual-cache-id-key', 'coco_id',
    '--visual-adapter', 'qformer', '--num-visual-queries', '32', '--qformer-layers', '2',
    '--grounded-relation-alignment', '--grounded-relation-cache-path', GROUNDED_CACHE,
    '--relation-alignment-margin', '0.2', '--seed', '42', '--num-workers', '0',
], cwd=REPO)

summary_path = OUTPUT / 'diagnostic_summary.json'
summary = json.loads(summary_path.read_text(encoding='utf-8'))
summary['code_commit'] = COMMIT
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
archive = shutil.make_archive(str(OUTPUT), 'zip', OUTPUT)
print('\nE3 DIAGNOSTIC SUMMARY')
print(json.dumps(summary, indent=2, ensure_ascii=False))
print('\nDownload ZIP:', archive)

```

Output: `/kaggle/working/e3_relation_diagnostic_100` và `/kaggle/working/e3_relation_diagnostic_100.zip`.

- `diagnostic_summary.json`: thống kê tổng và theo predicate.
- `diagnostic_records.json`: caption ON/OFF, reference và thông tin từng ảnh.
- `relation_scores.csv`: điểm từng quan hệ.
- `review.html` và ảnh JPEG: xem caption và bounding box (đỏ: subject; xanh: object).

Tải ZIP, giải nén và mở `review.html`. Ghi Kaggle code version cùng kết quả vào Experiment Registry sau khi chạy xong.