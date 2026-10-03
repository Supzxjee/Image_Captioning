# E2: Q-Former + Cascade Semantic Prompt Alignment

Thí nghiệm này warm-start từ checkpoint `qformer_32q_2l_gated` tốt nhất. Khác E1, object embeddings không được cộng trực tiếp vào visual queries. Chúng tạo object context để chấm trọng số từng token trong prompt object–relation hiện có; Q-Former queries sau đó cross-attend semantic tokens đã chọn và fusion qua residual gate.

```mermaid
flowchart LR
    I[Ảnh] --> V[CLIP ViT: 197 tokens]
    Q[32 learnable queries] --> QF[Q-Former 2 layers]
    V --> QF
    Y[YOLO object labels] --> O[Per-object CLIP embeddings]
    P[Object-relation prompt: 20 CLIP tokens] --> S[Token selector]
    O --> C[Masked object context]
    C --> S
    S --> R[Selected semantic tokens]
    QF --> A[Query-to-semantic cross-attention]
    R --> A
    A --> G[Residual gate]
    QF --> G
    G --> D[Transformer decoder]
    P --> D
```

## Cell Kaggle đầy đủ

Hai full run bằng DDP đã treo lần lượt sau batch 100 và 200 dù smoke thành công.
Phiên bản này bật `find_unused_parameters`, timeout collective 10 phút và heartbeat
theo từng rank. Chạy 500 batch trên hai GPU trước; nếu hoàn thành mới chạy pilot
3 epoch. Nếu lỗi phân tán tái diễn, timeout sẽ trả traceback thay vì treo nhiều giờ.

```python
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# NCCL báo lỗi bất đồng bộ sớm; mã nguồn còn đặt timeout collective 10 phút.
os.environ['NCCL_P2P_DISABLE'] = '1'
os.environ['NCCL_IB_DISABLE'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '600'

import torch

REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = '7cb84ba96cda67a8163159f23e069867d09ddae1'

COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
PROMPT_CACHE = Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
VISUAL_CACHE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')
DETECTIONS = Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
BUILT_OBJECT_CACHE = Path('/kaggle/working/object_semantic_cache/object_concepts.pt')

# Bắt buộc chạy diagnostic_dual trước khi đổi sang pilot_dual.
RUN_MODE = 'diagnostic_dual'
assert RUN_MODE in {'diagnostic_dual', 'pilot_dual'}
MAX_TRAIN_BATCHES = {'diagnostic_dual': 500, 'pilot_dual': 0}[RUN_MODE]
EPOCHS = 3 if RUN_MODE == 'pilot_dual' else 1
EXPERIMENT = {
    'diagnostic_dual': 'qformer_cascade_alignment_dual_gpu_diagnostic_500',
    'pilot_dual': 'qformer_cascade_alignment_32q_2l_dual_gpu_3ep',
}[RUN_MODE]


def run(args, cwd=None):
    args = list(map(str, args))
    print('\nRunning:', ' '.join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def find_old_qformer_checkpoint():
    matches = []
    for path in Path('/kaggle/input').rglob('model_h1_2_crossattn_epoch_10.pth'):
        if not path.is_file():
            continue
        try:
            meta = torch.load(path, map_location='cpu', weights_only=False)
        except Exception as error:
            print('Bỏ qua checkpoint không đọc được:', path, repr(error))
            continue
        signature = {
            'epoch': meta.get('epoch'),
            'visual_adapter': meta.get('visual_adapter', 'direct'),
            'queries': meta.get('num_visual_queries'),
            'layers': meta.get('qformer_layers'),
            'prompt_conditioned': bool(meta.get('prompt_conditioned_qformer', False)),
            'itc_weight': float(meta.get('itc_weight', 0.0)),
            'object_alignment': bool(meta.get('object_semantic_alignment', False)),
            'cascade_alignment': bool(meta.get('cascade_semantic_alignment', False)),
        }
        print(path, signature)
        if signature == {
            'epoch': 10,
            'visual_adapter': 'qformer',
            'queries': 32,
            'layers': 2,
            'prompt_conditioned': False,
            'itc_weight': 0.0,
            'object_alignment': False,
            'cascade_alignment': False,
        }:
            matches.append(path)
    assert len(matches) == 1, (
        f'Cần đúng một checkpoint Q-Former cũ, tìm thấy {len(matches)}: {matches}')
    return matches[0]


print('PyTorch:', torch.__version__)
print('CUDA devices:', torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(f'GPU {index}:', torch.cuda.get_device_name(index))
assert torch.cuda.device_count() == 2, 'Notebook phải chọn GPU T4 x2.'
for path in (COCO_JSON, PROMPT_CACHE, DETECTIONS):
    assert path.is_file(), path
assert COCO_IMAGES.is_dir(), COCO_IMAGES
assert VISUAL_CACHE.is_dir(), VISUAL_CACHE
OLD_CHECKPOINT = find_old_qformer_checkpoint()
print('Warm-start checkpoint:', OLD_CHECKPOINT)

# Tải đúng một commit thay vì clone toàn bộ repository. Nếu lần trước clone dở,
# thư mục không có .git sẽ được dọn trước khi thử lại.
if REPO.exists() and not (REPO / '.git').is_dir():
    shutil.rmtree(REPO)
if not REPO.exists():
    REPO.mkdir(parents=True)
    run(['git', 'init'], cwd=REPO)
    run(['git', 'remote', 'add', 'origin',
         'https://github.com/Supzxjee/Image_Captioning.git'], cwd=REPO)

last_fetch_error = None
for attempt in range(1, 4):
    try:
        # Fetch an advertised branch. A short commit hash is not a remote ref
        # and makes git fetch exit 128 in a newly initialized repository.
        run(['git', 'fetch', '--depth', '20', 'origin', 'main'], cwd=REPO)
        last_fetch_error = None
        break
    except subprocess.CalledProcessError as error:
        last_fetch_error = error
        print(f'Fetch lần {attempt}/3 thất bại; thử lại sau 10 giây.', flush=True)
        if attempt < 3:
            time.sleep(10)
if last_fetch_error is not None:
    raise RuntimeError(
        'Không tải được mã nguồn sau 3 lần. Kiểm tra Internet của Kaggle.') from last_fetch_error
run(['git', 'checkout', '--detach', COMMIT], cwd=REPO)
run(['git', 'rev-parse', '--short', 'HEAD'], cwd=REPO)
run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt'], cwd=REPO)

# Ưu tiên cache đã tạo ở E1. Hãy Add Input output của notebook E1 để không phải
# nạp lại CLIP text encoder và ghi lại file cache lớn trong mỗi Save Version.
input_object_caches = sorted(Path('/kaggle/input').rglob('object_concepts.pt'))
if input_object_caches:
    OBJECT_CACHE = input_object_caches[0]
    print('Dùng lại object cache từ Input:', OBJECT_CACHE)
    if len(input_object_caches) > 1:
        print('Các object cache khác (không dùng):', input_object_caches[1:])
else:
    OBJECT_CACHE = BUILT_OBJECT_CACHE
    print('Không tìm thấy object cache trong Input; sẽ xây lại:', OBJECT_CACHE)

# Cache chỉ chứa object prompts, không chứa caption reference.
if not OBJECT_CACHE.is_file():
    run([
        sys.executable, '-u', 'build_object_concept_cache.py',
        '--source', DETECTIONS,
        '--dataset-json-path', COCO_JSON,
        '--output', OBJECT_CACHE,
        '--objects-field', 'objects',
        '--name-key', 'label',
        '--confidence-key', 'conf',
        '--min-confidence', '0.5',
        '--max-objects', '10',
        '--batch-size', '256',
    ], cwd=REPO)

object_bundle = torch.load(OBJECT_CACHE, map_location='cpu', weights_only=False)
assert object_bundle['metadata']['count'] == 123287
assert object_bundle['metadata']['complete']
assert object_bundle['metadata']['max_objects'] == 10
print('Object cache:', json.dumps(object_bundle['metadata'], indent=2))
del object_bundle

common = [
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
]

train_args = [
    'train_h1_2_gated.py',
    '--mode', 'train',
    '--checkpoint', OLD_CHECKPOINT,
    '--epochs', str(EPOCHS),
    '--seed', '42',
    '--batch-size', '32',
    '--num-workers', '0',
] + list(map(str, common))
if MAX_TRAIN_BATCHES:
    train_args += ['--max-train-batches', str(MAX_TRAIN_BATCHES)]

run([
    sys.executable, '-m', 'accelerate.commands.launch',
    '--multi_gpu',
    '--num_processes', '2',
    '--num_machines', '1',
    '--mixed_precision', 'no',
    '--dynamo_backend', 'no',
    '--num_cpu_threads_per_process', '1',
] + train_args, cwd=REPO)

experiment_dir = Path('/kaggle/working') / EXPERIMENT
checkpoint = experiment_dir / f'checkpoints/model_h1_2_crossattn_epoch_{EPOCHS}.pth'
assert checkpoint.is_file(), checkpoint
meta = torch.load(checkpoint, map_location='cpu', weights_only=False)
assert meta['visual_adapter'] == 'qformer'
assert meta['num_visual_queries'] == 32
assert meta['qformer_layers'] == 2
assert meta['object_semantic_alignment'] is False
assert meta['cascade_semantic_alignment'] is True
assert meta['object_prompt_metadata']['min_confidence'] == 0.5
assert meta['max_train_batches'] == MAX_TRAIN_BATCHES
print('Checkpoint metadata PASS')
del meta

if RUN_MODE == 'pilot_dual':
    eval_args = [
        'train_h1_2_gated.py',
        '--mode', 'evaluate',
        '--checkpoint', checkpoint,
        '--split', 'val',
        '--limit', '0',
        '--epochs', str(EPOCHS),
        '--batch-size', '32',
        '--num-workers', '0',
    ] + list(map(str, common))
    run([sys.executable, '-u'] + eval_args, cwd=REPO)

    metrics_path = experiment_dir / 'evaluation/val_5000_metrics_h1_2_gated.json'
    assert metrics_path.is_file(), metrics_path
    metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
    print('\nCASCADE ALIGNMENT VALIDATION METRICS')
    print(json.dumps(metrics, indent=2))

print('\nHoàn tất:', experiment_dir)
```

## Quy tắc quyết định

So sánh rank-0 validation với Q-Former cũ: BLEU-4 `0.370322`, CIDEr `1.174012`.
E2 được xem là có tín hiệu nếu CIDEr tăng và BLEU-4 không giảm quá `0.003`.
Chưa chạy test hoặc re-ranking ở giai đoạn chọn kiến trúc. Nếu E2 đạt, bước sau
mới sinh beam candidates và đánh giá CHAIR trên validation/test theo quy trình đã
chốt; nếu không đạt, giữ Q-Former cũ và tập trung cải thiện re-ranking cuối.
