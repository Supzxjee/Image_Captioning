# E3 V3: verified relation fusion pilot

V2 đã học scorer nhưng query gate gần đóng sau 1 epoch. V3 đưa điểm xác minh trực tiếp vào caption memory thay vì chỉ vào auxiliary loss.

- Warm-start từ checkpoint Appearance V2 sau 1 epoch đầy đủ, không dùng Q-Former baseline hoặc E3 V1. Scorer và lớp chiếu thị giác đã học ở V2 được giữ cố định. Q-Former, decoder, attention và projection cho fusion vẫn học; các head gate cũ không dùng ở V3.
- Nhánh mới chỉ nhận holding, wearing, carrying, riding. Spatial relations vẫn nằm trong prompt V2.1.
- Chấp nhận relation nếu positive cosine > 0, predicate gap đạt margin khả thi và detection confidence > 0. Đây là heuristic verification, không phải xác suất đúng được calibration.
- Trọng số fusion là confidence x hai sigmoid của score/gap; residual scale cố định 0.1. Đây là thay đổi kiến trúc, không phải sửa khởi tạo learned gate để ép nó mở. Caption attention vẫn có thể học bỏ qua nhánh nếu nhánh không hữu ích.
- Quan hệ không được chấp nhận không làm thay đổi visual queries. Test kiểm tra cả no-relations, zero confidence, counterfactual bị loại và zero scale.

## Chạy

Bật Internet, chọn T4 x2. Add Input output notebook mới E3 Appearance V2 Checkpoint Store (lấy từ source version210), prompt V2.1, grounded cache V1, visual cache, MS COCO. Dùng Copy File Path sửa duy nhất CHECKPOINT nếu mount V2 khác. Cell cũng dùng checkpoint Working nếu bạn vẫn ở session vừa train V2.

Cell diagnostic cùng 100 ảnh trước train; train 128 global batches (~4096 ảnh); diagnostic 100 ảnh sau train. Cả hai diagnostic có caption ON/OFF/counterfactual, accepted relations và đối chứng ảnh khác. Query gate mean là null ở V3 vì kiến trúc không có learned query gate; đọc verification weights và residual.

```python
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CHECKPOINT = Path('/kaggle/input/notebooks/ducanh2403/e3-appearance-v2-checkpoint-store/e3_appearance_v2_checkpoint_store/model_h1_2_crossattn_epoch_1.pth')
PROMPT_CACHE = Path('/kaggle/input/datasets/ducanh2403/prompt-conditioned-qformer-32q-2l/relation_prompts_v2_1_full/prompt_clip_tokens_relation_v2_1.pt')
GROUNDED_CACHE = Path('/kaggle/input/notebooks/ducanh2403/build-grounded-relation-v1/grounded_relation_v1_full/grounded_relation_cache.pt')
VISUAL_CACHE_SOURCE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')

REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = '0dbe6f6'
VISUAL_CACHE_LOCAL = Path('/tmp/visual-cache-e3-valid')
EXPERIMENT = 'e3_verified_v3_pilot_128b'
OUTPUT = Path('/kaggle/working') / EXPERIMENT


def run(args, cwd=None):
    args = list(map(str, args))
    print('\nRunning:', ' '.join(args), flush=True)
    started = time.monotonic()
    with subprocess.Popen(args, cwd=cwd) as process:
        while True:
            try:
                code = process.wait(timeout=30)
                break
            except subprocess.TimeoutExpired:
                print(f'Process still running: {time.monotonic() - started:.0f}s', flush=True)
        if code:
            raise subprocess.CalledProcessError(code, args)


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


print('E3 Verified Fusion V3 bootstrap started.', flush=True)
# Nếu vẫn ở session vừa train V2, dùng đúng checkpoint Working đó.
working_v2 = Path('/kaggle/working/e3_appearance_v2_32q_2l_dual_gpu_1ep/checkpoints/model_h1_2_crossattn_epoch_1.pth')
if not CHECKPOINT.is_file() and working_v2.is_file():
    CHECKPOINT = working_v2
print('V2 checkpoint:', CHECKPOINT, flush=True)
for path in (CHECKPOINT, PROMPT_CACHE, GROUNDED_CACHE, COCO_JSON):
    assert path.is_file(), f'Missing file: {path}'
assert COCO_IMAGES.is_dir(), COCO_IMAGES
assert VISUAL_CACHE_SOURCE.is_dir(), VISUAL_CACHE_SOURCE

os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['TQDM_MININTERVAL'] = '30'
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


import torch
import zipfile

assert torch.cuda.device_count() >= 2, 'Select GPU T4 x2.'
source = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
assert source.get('grounded_relation_version') == 'appearance_v2', 'Use trained appearance V2.'
assert source.get('epoch') == 1 and source.get('max_train_batches') == 0, 'Use full 1-epoch V2, not smoke.'
assert source.get('num_visual_queries') == 32 and source.get('qformer_layers') == 2
del source

common = [
    '--dataset-json-path', COCO_JSON, '--base-path', COCO_IMAGES,
    '--prompt-cache-path', PROMPT_CACHE,
    '--visual-cache', VISUAL_CACHE_LOCAL, '--visual-cache-id-key', 'coco_id',
    '--visual-adapter', 'qformer', '--num-visual-queries', '32', '--qformer-layers', '2',
    '--grounded-relation-alignment', '--grounded-relation-cache-path', GROUNDED_CACHE,
    '--grounded-relation-version', 'verified_v3', '--relation-verified-scale', '0.1',
    '--relation-alignment-weight', '0', '--relation-alignment-margin', '0.2',
    '--seed', '42', '--num-workers', '0',
]

# Verify memory selection before training; missing V3 fusion weights initialized with seed42.
BEFORE = OUTPUT / 'before_training_100'
run([
    sys.executable, '-u', 'diagnose_grounded_relations.py', '--mode', 'predict',
    '--split', 'val', '--limit', '0', '--checkpoint', CHECKPOINT,
    '--expected-epoch', '1', '--sample-size', '100', '--warm-start-verifier',
    '--output', BEFORE,
] + common, cwd=REPO)

V3_CHECKPOINT = OUTPUT / 'checkpoints/model_h1_2_crossattn_epoch_1.pth'
if not V3_CHECKPOINT.is_file():
    run([
        sys.executable, '-u', '-m', 'accelerate.commands.launch',
        '--multi_gpu', '--num_processes', '2', '--num_machines', '1',
        '--mixed_precision', 'no', '--dynamo_backend', 'no',
        '--num_cpu_threads_per_process', '2', '--main_process_port', '29501',
        'train_h1_2_gated.py', '--mode', 'train', '--checkpoint', CHECKPOINT,
        '--epochs', '1', '--max-train-batches', '128', '--batch-size', '32',
        '--experiment-name', EXPERIMENT,
    ] + common, cwd=REPO)

AFTER = OUTPUT / 'after_training_100'
run([
    sys.executable, '-u', 'diagnose_grounded_relations.py', '--mode', 'predict',
    '--split', 'val', '--limit', '0', '--checkpoint', V3_CHECKPOINT,
    '--expected-epoch', '1', '--sample-size', '100', '--output', AFTER,
] + common, cwd=REPO)

manifest = {'code_commit': COMMIT, 'warm_start': str(CHECKPOINT),
            'v3_checkpoint': str(V3_CHECKPOINT), 'train_batches': 128,
            'global_batch_size': 32, 'gpu_processes': 2,
            'verifier_frozen': True, 'shared_vis_projection_frozen': True,
            'verified_scale': 0.1, 'verification_margin': 0.2,
            'notes': 'Action relations only in new branch; spatial relations remain in prompt V2.1. '
                     'Verification scores are not calibrated probabilities. '
                     'Before/after diagnostic compares mechanism, not caption quality.'}
for name, folder in [('before', BEFORE), ('after', AFTER)]:
    path = folder / 'diagnostic_summary.json'
    summary = json.loads(path.read_text(encoding='utf-8'))
    summary['code_commit'] = COMMIT
    path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    manifest[name] = summary
    print(f'\nV3 {name.upper()} SUMMARY')
    print(json.dumps(summary, indent=2), flush=True)
manifest_path = OUTPUT / 'manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
archive = Path('/kaggle/working/e3_verified_v3_pilot_128b_review.zip')
with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as zipped:
    for folder in (BEFORE, AFTER):
        for path in folder.rglob('*'):
            if path.is_file():
                zipped.write(path, path.relative_to(OUTPUT).as_posix())
    zipped.write(manifest_path, 'manifest.json')
    history = OUTPUT / 'train_history_h1_2.json'
    if history.is_file():
        zipped.write(history, history.name)
print('V3 VERIFIED FUSION PILOT COMPLETE')
print('Small review ZIP:', archive)
print('Checkpoint retained:', V3_CHECKPOINT)

```

Gửi `e3_verified_v3_pilot_128b_review.zip` và Kaggle code version. ZIP review không chứa checkpoint. Đừng coi caption đổi là chất lượng tăng: cần đọc ảnh và counterfactual, rồi mới quyết định train full/đánh giá caption metrics.

Kiểm tra local: 19 relation tests và 5 CLI/workflow tests PASS. Chưa chạy dữ liệu Kaggle. Đây là prototype chưa xác nhận tính mới nghiên cứu hoặc hiệu quả caption.
