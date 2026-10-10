# Sinh full relation prompt V2.1 và CLIP token cache

Chạy notebook này sau khi V2.1 đã PASS audit 120 ảnh. Notebook cần:

- Add Input `objectdetectionecache`;
- Add Input MS COCO 2014 của `vuthetam`;
- bật Internet để tải CLIP nếu model chưa có trong cache;
- chọn GPU T4. Bước sinh JSON chạy CPU; bước encode CLIP dùng một GPU và không
  phải là quá trình train.

Chạy nguyên cell Python dưới đây. Không đặt phần mô tả Markdown vào trong cell.

```python
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/kaggle/working/Image_Captioning")
COMMIT = "819992f"
DETECTIONS = Path(
    "/kaggle/input/datasets/ducanh2403/"
    "objectdetectionecache/objectdetectioncache.json"
)
COCO_JSON = Path(
    "/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json"
)
OUTPUT = Path("/kaggle/working/relation_prompts_v2_1_full")
PROMPT_JSON = OUTPUT / "promptcache_relation_v2_1.json"
PROMPT_TOKENS = OUTPUT / "prompt_clip_tokens_relation_v2_1.pt"
MANIFEST = OUTPUT / "manifest.json"


def run(args, cwd=None):
    args = list(map(str, args))
    print("\nRunning:", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


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

assert DETECTIONS.is_file(), DETECTIONS
assert COCO_JSON.is_file(), COCO_JSON

if OUTPUT.exists():
    shutil.rmtree(OUTPUT)
OUTPUT.mkdir(parents=True)

# 1. Build the full JSON prompt cache from saved YOLO detections.
run([
    sys.executable,
    "-u",
    "build_relation_prompts.py",
    "--source", DETECTIONS,
    "--output", PROMPT_JSON,
    "--min-confidence", "0.5",
    "--max-objects", "10",
    "--max-relations", "3",
], cwd=REPO)

PROMPT_SUMMARY = PROMPT_JSON.with_suffix(".summary.json")
assert PROMPT_JSON.is_file(), PROMPT_JSON
assert PROMPT_SUMMARY.is_file(), PROMPT_SUMMARY

prompts = json.loads(PROMPT_JSON.read_text(encoding="utf-8"))
summary = json.loads(PROMPT_SUMMARY.read_text(encoding="utf-8"))
coco = json.loads(COCO_JSON.read_text(encoding="utf-8"))
expected_count = len(coco["images"])

assert len(prompts) == expected_count, (len(prompts), expected_count)
assert summary["images"] == expected_count, summary
assert summary["heuristic_version"] == "H1_3_semantic_instance_aware_v2_1"
assert {entry["heuristic_version"] for entry in prompts.values()} == {
    "H1_3_semantic_instance_aware_v2_1"
}

print("\nFULL V2.1 JSON SUMMARY")
print(json.dumps(summary, indent=2, ensure_ascii=False))

# Release the large parsed JSON before encoding.
del prompts
del coco

# 2. Encode prompts with the same CLIP text encoder used by training.
run([
    sys.executable,
    "-u",
    "embed_relation_prompts.py",
    "--source", PROMPT_JSON,
    "--dataset-json-path", COCO_JSON,
    "--output", PROMPT_TOKENS,
    "--batch-size", "256",
], cwd=REPO)

TOKEN_METADATA = PROMPT_TOKENS.with_suffix(".json")
assert PROMPT_TOKENS.is_file(), PROMPT_TOKENS
assert TOKEN_METADATA.is_file(), TOKEN_METADATA
metadata = json.loads(TOKEN_METADATA.read_text(encoding="utf-8"))

assert metadata["variant"] == "semantic_instance_aware_relation_v2_1"
assert metadata["heuristic_version"] == "H1_3_semantic_instance_aware_v2_1"
assert metadata["count"] == expected_count, metadata
assert metadata["complete"] is True, metadata
assert metadata["source_sha256"] == sha256(PROMPT_JSON)

manifest = {
    "code_commit": COMMIT,
    "heuristic_version": "H1_3_semantic_instance_aware_v2_1",
    "expected_count": expected_count,
    "prompt_json": PROMPT_JSON.name,
    "prompt_json_bytes": PROMPT_JSON.stat().st_size,
    "prompt_json_sha256": metadata["source_sha256"],
    "prompt_summary": PROMPT_SUMMARY.name,
    "token_cache": PROMPT_TOKENS.name,
    "token_cache_bytes": PROMPT_TOKENS.stat().st_size,
    "token_cache_sha256": sha256(PROMPT_TOKENS),
    "token_metadata": TOKEN_METADATA.name,
    "metadata": metadata,
}
MANIFEST.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)

print("\nFULL CACHE COMPLETE")
print(json.dumps(manifest, indent=2, ensure_ascii=False))
print("\nOutput directory:", OUTPUT)
print("Save a Kaggle Notebook Version and keep all output files.")
```

## Output cần giữ

Sau khi cell hoàn tất, thư mục
`/kaggle/working/relation_prompts_v2_1_full` phải có năm tệp:

```text
manifest.json
promptcache_relation_v2_1.json
promptcache_relation_v2_1.summary.json
prompt_clip_tokens_relation_v2_1.pt
prompt_clip_tokens_relation_v2_1.json
```

Chọn **Save Version → Save & Run All** để lưu output. Ở notebook train
E2-V2.1, Add Input output này và trỏ `--prompt-cache-path` đến
`prompt_clip_tokens_relation_v2_1.pt`.
