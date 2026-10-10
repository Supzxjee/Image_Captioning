# Train E2 với relation prompt V2.1

Thí nghiệm này đo riêng ảnh hưởng của bộ sinh prompt V2.1. Kiến trúc, checkpoint
khởi tạo, seed, batch size, visual cache, object cache và số epoch giống E2 cũ;
chỉ thay `--prompt-cache-path`.

Trước khi chạy:

- Save Version notebook sinh full cache V2.1;
- Add Input output đó vào notebook train;
- Add Input checkpoint Q-Former baseline epoch 10;
- Add Input `objectdetectionecache` và MS COCO 2014;
- chọn GPU T4 x2.

Workflow này cố ý đọc ảnh gốc và chạy frozen CLIP ViT-B/16 trực tiếp. Không Add
Input `visual-cache`: các shard HDF5 lớn trên Kaggle mount có thể treo DataLoader
giữa epoch. Backbone, preprocessing và precision vẫn giống lúc tạo cache; chỉ thay
cách lấy 197 visual tokens.

```python
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "600"

import torch

REPO = Path("/kaggle/working/Image_Captioning")
COMMIT = "aaf03b8"
COCO_JSON = Path(
    "/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json"
)
COCO_IMAGES = Path(
    "/kaggle/input/datasets/vuthetam/mscoco-2014/images"
)
PROMPT_CACHE = Path(
    "/kaggle/input/datasets/ducanh2403/prompt-conditioned-qformer-32q-2l/"
    "relation_prompts_v2_1_full/prompt_clip_tokens_relation_v2_1.pt"
)
CHECKPOINT_ROOT = Path(
    "/kaggle/input/datasets/ducanh2403/qformer-epoch10"
)
DETECTIONS = Path(
    "/kaggle/input/datasets/ducanh2403/"
    "objectdetectionecache/objectdetectioncache.json"
)
OBJECT_CACHE = Path(
    "/kaggle/working/object_semantic_cache/object_concepts.pt"
)
EXPECTED_PROMPT_SHA = (
    "b816743130a62047b347577d855f5dbf38ee621544050a8f09153e8056a4e479"
)
EXPECTED_TOKEN_SHA = (
    "03a89869a11e137d84204f1ad602f1b80d1fd7b28fb6ed73ea9b3906fcf4c908"
)
EXPERIMENT = "qformer_cascade_alignment_v2_1_raw_images_32q_2l_dual_gpu_3ep"
EPOCHS = 3

print("E2 relation V2.1 bootstrap started.", flush=True)

if CHECKPOINT_ROOT.is_file():
    assert CHECKPOINT_ROOT.name == "model_h1_2_crossattn_epoch_10.pth"
    OLD_CHECKPOINT = CHECKPOINT_ROOT
else:
    assert CHECKPOINT_ROOT.is_dir(), CHECKPOINT_ROOT
    checkpoint_matches = sorted(
        CHECKPOINT_ROOT.rglob("model_h1_2_crossattn_epoch_10.pth")
    )
    assert len(checkpoint_matches) == 1, checkpoint_matches
    OLD_CHECKPOINT = checkpoint_matches[0]
print("Selected epoch-10 checkpoint:", OLD_CHECKPOINT, flush=True)


def run(args, cwd=None):
    args = list(map(str, args))
    print("\nRunning:", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)



def checkpoint_signature(path):
    meta = torch.load(path, map_location="cpu", weights_only=False)
    encoder_state = meta.get("encoder_state_dict", {})
    signature = {
        "epoch": meta.get("epoch"),
        "visual_adapter": meta.get("visual_adapter", "direct"),
        "queries": meta.get("num_visual_queries"),
        "layers": meta.get("qformer_layers"),
        "prompt_conditioned": bool(meta.get("prompt_conditioned_qformer", False)),
        "itc_weight": float(meta.get("itc_weight", 0.0)),
        "object_alignment": bool(meta.get("object_semantic_alignment", False)),
        "cascade_alignment": bool(meta.get("cascade_semantic_alignment", False)),
        "backbone_tensors": sum(
            key.startswith("feature_extractor.") for key in encoder_state
        ),
    }
    del meta
    return signature


print("PyTorch:", torch.__version__)
print("CUDA devices:", torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(f"GPU {index}:", torch.cuda.get_device_name(index))
assert torch.cuda.device_count() == 2, "Notebook phai chon GPU T4 x2."
assert COCO_JSON.is_file(), COCO_JSON
assert COCO_IMAGES.is_dir(), COCO_IMAGES
assert PROMPT_CACHE.is_file(), PROMPT_CACHE
assert OLD_CHECKPOINT.is_file(), OLD_CHECKPOINT
assert DETECTIONS.is_file(), DETECTIONS

prompt_metadata_path = PROMPT_CACHE.with_suffix(".json")
manifest_path = PROMPT_CACHE.parent / "manifest.json"
assert prompt_metadata_path.is_file(), prompt_metadata_path
assert manifest_path.is_file(), manifest_path
prompt_metadata = json.loads(prompt_metadata_path.read_text(encoding="utf-8"))
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
assert prompt_metadata.get("complete") is True
assert prompt_metadata.get("count") == 123287
assert prompt_metadata.get("max_length") == 20
assert prompt_metadata.get("source_sha256") == EXPECTED_PROMPT_SHA
assert prompt_metadata.get("heuristic_version") == "H1_3_semantic_instance_aware_v2_1"
assert manifest.get("prompt_json_sha256") == EXPECTED_PROMPT_SHA
assert manifest.get("token_cache_sha256") == EXPECTED_TOKEN_SHA

expected_checkpoint = {
    "epoch": 10,
    "visual_adapter": "qformer",
    "queries": 32,
    "layers": 2,
    "prompt_conditioned": False,
    "itc_weight": 0.0,
    "object_alignment": False,
    "cascade_alignment": False,
}
signature = checkpoint_signature(OLD_CHECKPOINT)
core_signature = {
    key: value for key, value in signature.items()
    if key != "backbone_tensors"
}
assert core_signature == expected_checkpoint, signature
assert signature["backbone_tensors"] > 0, (
    "Checkpoint epoch 10 khong chua frozen CLIP backbone.", signature
)
print("Explicit input paths and metadata PASS.", flush=True)
print("Checkpoint signature:", signature, flush=True)

if REPO.exists() and not (REPO / ".git").is_dir():
    shutil.rmtree(REPO)
if not REPO.exists():
    REPO.mkdir(parents=True)
    run(["git", "init"], cwd=REPO)
    run([
        "git", "remote", "add", "origin",
        "https://github.com/Supzxjee/Image_Captioning.git",
    ], cwd=REPO)

last_error = None
for attempt in range(1, 4):
    try:
        run(["git", "fetch", "--depth", "30", "origin", "main"], cwd=REPO)
        last_error = None
        break
    except subprocess.CalledProcessError as error:
        last_error = error
        print(f"Fetch attempt {attempt}/3 failed.", flush=True)
        if attempt < 3:
            time.sleep(10)

if last_error is not None:
    raise RuntimeError("Cannot fetch repository. Check Kaggle Internet.") from last_error

run(["git", "checkout", "--detach", COMMIT], cwd=REPO)
run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO)
run([
    sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"
], cwd=REPO)

# The manually uploaded object_concepts.pt was extracted into a directory by
# Kaggle. Rebuild the deterministic cache from the explicit YOLO JSON path.
OBJECT_CACHE.parent.mkdir(parents=True, exist_ok=True)
if OBJECT_CACHE.exists():
    if OBJECT_CACHE.is_dir():
        shutil.rmtree(OBJECT_CACHE)
    else:
        OBJECT_CACHE.unlink()
run([
    sys.executable,
    "-u",
    "build_object_concept_cache.py",
    "--source", DETECTIONS,
    "--dataset-json-path", COCO_JSON,
    "--output", OBJECT_CACHE,
    "--objects-field", "objects",
    "--name-key", "label",
    "--confidence-key", "conf",
    "--min-confidence", "0.5",
    "--max-objects", "10",
    "--batch-size", "256",
], cwd=REPO)
rebuilt = torch.load(OBJECT_CACHE, map_location="cpu", weights_only=False)
rebuilt_data = rebuilt.get("data", {})
rebuilt_metadata = rebuilt.get("metadata", {})
assert len(rebuilt_data) == 123287
assert rebuilt_metadata.get("complete") is True
assert rebuilt_metadata.get("count") == 123287
assert rebuilt_metadata.get("max_objects") == 10
assert float(rebuilt_metadata.get("min_confidence")) == 0.5
print("Rebuilt object cache metadata:")
print(json.dumps(rebuilt_metadata, indent=2))
del rebuilt
del rebuilt_data

common = [
    "--dataset-json-path", COCO_JSON,
    "--base-path", COCO_IMAGES,
    "--prompt-cache-path", PROMPT_CACHE,
    "--visual-preprocessing", "bilinear",
    "--visual-precision", "fp32",
    "--visual-adapter", "qformer",
    "--num-visual-queries", "32",
    "--qformer-layers", "2",
    "--cascade-semantic-alignment",
    "--cascade-selector-mode", "object_context",
    "--object-prompt-cache-path", OBJECT_CACHE,
    "--experiment-name", EXPERIMENT,
]

train_args = [
    "train_h1_2_gated.py",
    "--mode", "train",
    "--checkpoint", OLD_CHECKPOINT,
    "--epochs", str(EPOCHS),
    "--seed", "42",
    "--batch-size", "32",
    "--num-workers", "0",
] + list(map(str, common))

run([
    sys.executable,
    "-m", "accelerate.commands.launch",
    "--multi_gpu",
    "--num_processes", "2",
    "--num_machines", "1",
    "--mixed_precision", "no",
    "--dynamo_backend", "no",
    "--num_cpu_threads_per_process", "1",
] + train_args, cwd=REPO)

experiment_dir = Path("/kaggle/working") / EXPERIMENT
checkpoint = (
    experiment_dir
    / f"checkpoints/model_h1_2_crossattn_epoch_{EPOCHS}.pth"
)
assert checkpoint.is_file(), checkpoint
meta = torch.load(checkpoint, map_location="cpu", weights_only=False)
assert meta["epoch"] == 3
assert meta["visual_adapter"] == "qformer"
assert meta["num_visual_queries"] == 32
assert meta["qformer_layers"] == 2
assert meta["object_semantic_alignment"] is False
assert meta["cascade_semantic_alignment"] is True
assert meta["cascade_selector_mode"] == "object_context"
assert meta["max_train_batches"] == 0
assert meta["visual_cache_files"] == []
assert meta["prompt_metadata"]["complete"] is True
assert meta["prompt_metadata"]["count"] == 123287
assert meta["prompt_metadata"]["source_sha256"] == EXPECTED_PROMPT_SHA
assert (
    meta["prompt_metadata"]["heuristic_version"]
    == "H1_3_semantic_instance_aware_v2_1"
)
print("Checkpoint metadata PASS")
del meta

eval_args = [
    "train_h1_2_gated.py",
    "--mode", "evaluate",
    "--checkpoint", checkpoint,
    "--split", "val",
    "--limit", "0",
    "--epochs", str(EPOCHS),
    "--batch-size", "32",
    "--num-workers", "0",
] + list(map(str, common))
run([sys.executable, "-u"] + eval_args, cwd=REPO)

metrics_path = (
    experiment_dir / "evaluation/val_5000_metrics_h1_2_gated.json"
)
assert metrics_path.is_file(), metrics_path
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
print("\nE2 RELATION V2.1 VALIDATION METRICS")
print(json.dumps(metrics, indent=2))
print("\nCompleted:", experiment_dir)
```

## Tiêu chí quyết định

So sánh validation với E2 cũ:

| Mô hình | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---:|---:|---:|---:|
| E2 prompt V1 | 0.3739 | 0.2869 | 0.5743 | 1.1868 |
| E2 prompt V2.1 | chờ chạy | chờ chạy | chờ chạy | chờ chạy |

V2.1 đạt tín hiệu nếu CIDEr tăng và BLEU-4 không giảm quá `0.003`. Nếu qua cửa
validation, mới chạy test 5.000 ảnh và CHAIR; chưa chạy thêm ablation hoặc E3.
