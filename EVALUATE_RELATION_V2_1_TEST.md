# Đánh giá E2 Relation V2.1 trên test 5.000 ảnh

Workflow này chỉ inference và tính COCO metrics, không train lại. Add Input output
của lần train E2 Relation V2.1 chứa checkpoint epoch 3, cache relation V2.1,
`visual-cache`, `objectdetectionecache` và MS COCO 2014. Chạy bằng một GPU T4.

```python
import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import torch

REPO = Path("/kaggle/working/Image_Captioning")
COMMIT = "1eef0772e994d6805aff6c8bb301e0f9b4d45b14"

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
VISUAL_CACHE = Path(
    "/kaggle/input/datasets/ducanh2403/visual-cache"
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
EXPECTED_HEURISTIC = "H1_3_semantic_instance_aware_v2_1"
CHECKPOINT_NAME = "model_h1_2_crossattn_epoch_3.pth"
EXPERIMENT = "e2_relation_v2_1_32q_2l_test"


def run(args, cwd=None):
    args = list(map(str, args))
    print("\nRunning:", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def restore_torch_archive(source, index):
    """Khôi phục file .pth nếu Kaggle đã bung PyTorch ZIP thành thư mục."""
    if source.is_file():
        return source
    if not source.is_dir():
        raise FileNotFoundError(source)
    destination = Path("/kaggle/working") / f"restored_v2_1_candidate_{index}.pth"
    data_pickles = list(source.rglob("data.pkl"))
    assert len(data_pickles) == 1, (source, data_pickles)
    archive_root = data_pickles[0].parent
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_STORED) as archive:
        for file in archive_root.rglob("*"):
            if file.is_file():
                archive.write(
                    file,
                    f"archive/{file.relative_to(archive_root).as_posix()}",
                )
    return destination


def is_relation_v2_1_checkpoint(meta):
    prompt_metadata = meta.get("prompt_metadata", {})
    return (
        int(meta.get("epoch", -1)) == 3
        and meta.get("visual_adapter") == "qformer"
        and int(meta.get("num_visual_queries", -1)) == 32
        and int(meta.get("qformer_layers", -1)) == 2
        and bool(meta.get("cascade_semantic_alignment", False))
        and meta.get("cascade_selector_mode", "object_context") == "object_context"
        and not bool(meta.get("object_semantic_alignment", False))
        and not bool(meta.get("prompt_conditioned_qformer", False))
        and float(meta.get("alignment_weight", 0.0)) == 0.0
        and float(meta.get("itc_weight", 0.0)) == 0.0
        and int(meta.get("max_train_batches", 0)) == 0
        and prompt_metadata.get("complete") is True
        and int(prompt_metadata.get("count", -1)) == 123287
        and prompt_metadata.get("source_sha256") == EXPECTED_PROMPT_SHA
        and prompt_metadata.get("heuristic_version") == EXPECTED_HEURISTIC
    )


# 1. Kiểm tra các Input cố định.
assert COCO_JSON.is_file(), COCO_JSON
assert COCO_IMAGES.is_dir(), COCO_IMAGES
assert PROMPT_CACHE.is_file(), PROMPT_CACHE
assert DETECTIONS.is_file(), DETECTIONS
for part in range(1, 4):
    shard = VISUAL_CACHE / f"visual_part_{part:02d}_of_03.h5"
    assert shard.is_file(), shard

prompt_metadata_path = PROMPT_CACHE.with_suffix(".json")
assert prompt_metadata_path.is_file(), prompt_metadata_path
prompt_metadata = json.loads(prompt_metadata_path.read_text(encoding="utf-8"))
assert prompt_metadata.get("complete") is True
assert prompt_metadata.get("count") == 123287
assert prompt_metadata.get("source_sha256") == EXPECTED_PROMPT_SHA
assert prompt_metadata.get("heuristic_version") == EXPECTED_HEURISTIC
print("Relation V2.1 prompt cache PASS.", flush=True)


# 2. Chọn checkpoint V2.1 epoch 3 bằng metadata, không dựa vào tên thư mục mount.
expected_names = {CHECKPOINT_NAME, Path(CHECKPOINT_NAME).stem}
candidates = set(Path("/kaggle/input").rglob(CHECKPOINT_NAME))
for data_pickle in Path("/kaggle/input").rglob("data.pkl"):
    if data_pickle.parent.name in expected_names:
        candidates.add(data_pickle.parent)

matches = []
print("\nCheckpoint candidates:")
for index, candidate in enumerate(sorted(candidates, key=str), 1):
    restored = restore_torch_archive(candidate, index)
    try:
        meta = torch.load(restored, map_location="cpu", weights_only=False)
    except Exception as error:
        print(" -", candidate, "| bỏ qua:", repr(error))
        continue
    signature = {
        "epoch": meta.get("epoch"),
        "adapter": meta.get("visual_adapter"),
        "queries": meta.get("num_visual_queries"),
        "layers": meta.get("qformer_layers"),
        "cascade": bool(meta.get("cascade_semantic_alignment", False)),
        "selector": meta.get("cascade_selector_mode", "object_context"),
        "heuristic": meta.get("prompt_metadata", {}).get("heuristic_version"),
        "source_sha256": meta.get("prompt_metadata", {}).get("source_sha256"),
    }
    print(" -", candidate, "|", signature)
    if is_relation_v2_1_checkpoint(meta):
        matches.append((restored, candidate))
    del meta

assert matches, "Không tìm thấy checkpoint E2 Relation V2.1 epoch 3 hợp lệ."
if len(matches) > 1:
    print("Tìm thấy nhiều bản sao checkpoint V2.1 hợp lệ:")
    for _, source in matches:
        print(" -", source)
dataset_matches = [
    match for match in matches
    if "/kaggle/input/datasets/" in match[1].as_posix()
]
CHECKPOINT, CHECKPOINT_SOURCE = (
    dataset_matches[0] if dataset_matches else matches[0]
)
print("Selected V2.1 checkpoint:", CHECKPOINT_SOURCE, flush=True)


# 3. Checkout mã nguồn tương thích.
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
run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=REPO)


# 4. Tạo object semantic cache xác định từ detection JSON.
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

object_bundle = torch.load(OBJECT_CACHE, map_location="cpu", weights_only=False)
object_metadata = object_bundle.get("metadata", {})
assert object_metadata.get("complete") is True
assert object_metadata.get("count") == 123287
assert object_metadata.get("max_objects") == 10
assert float(object_metadata.get("min_confidence")) == 0.5
del object_bundle
print("Object semantic cache PASS.", flush=True)


# 5. Sinh caption và tính metric trên đúng 5.000 ảnh test.
run([
    sys.executable,
    "-u",
    "train_h1_2_gated.py",
    "--mode", "evaluate",
    "--checkpoint", CHECKPOINT,
    "--split", "test",
    "--limit", "0",
    "--batch-size", "32",
    "--num-workers", "0",
    "--dataset-json-path", COCO_JSON,
    "--base-path", COCO_IMAGES,
    "--prompt-cache-path", PROMPT_CACHE,
    "--visual-cache", VISUAL_CACHE,
    "--visual-cache-id-key", "coco_id",
    "--visual-preprocessing", "bilinear",
    "--visual-precision", "fp32",
    "--visual-adapter", "qformer",
    "--num-visual-queries", "32",
    "--qformer-layers", "2",
    "--cascade-semantic-alignment",
    "--cascade-selector-mode", "object_context",
    "--object-prompt-cache-path", OBJECT_CACHE,
    "--experiment-name", EXPERIMENT,
], cwd=REPO)


# 6. Kiểm tra output và ghi manifest để workflow CHAIR chọn đúng captions.
evaluation_dir = Path("/kaggle/working") / EXPERIMENT / "evaluation"
predictions_path = evaluation_dir / "test_5000_captions_h1_2_gated.json"
ground_truth_path = evaluation_dir / "test_5000_gt_h1_2_gated.json"
metrics_path = evaluation_dir / "test_5000_metrics_h1_2_gated.json"
for path in (predictions_path, ground_truth_path, metrics_path):
    assert path.is_file(), path

predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
assert len(predictions) == 5000
assert len({int(item["image_id"]) for item in predictions}) == 5000

manifest = {
    "model": "e2_relation_v2_1",
    "heuristic_version": EXPECTED_HEURISTIC,
    "prompt_source_sha256": EXPECTED_PROMPT_SHA,
    "checkpoint_source": str(CHECKPOINT_SOURCE),
    "predictions": predictions_path.name,
    "ground_truth": ground_truth_path.name,
    "metrics_file": metrics_path.name,
    "metrics": metrics,
}
manifest_path = evaluation_dir / "relation_v2_1_test_manifest.json"
manifest_path.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\nE2 RELATION V2.1 TEST METRICS")
print(json.dumps(metrics, indent=2))
print("\nPredictions:", predictions_path)
print("Ground truth:", ground_truth_path)
print("Metrics:", metrics_path)
print("Manifest:", manifest_path)
```

Giữ toàn bộ thư mục
`/kaggle/working/e2_relation_v2_1_32q_2l_test/evaluation`. Sau khi notebook hoàn
tất, chạy workflow CHAIR riêng; không sinh caption lần thứ hai.

