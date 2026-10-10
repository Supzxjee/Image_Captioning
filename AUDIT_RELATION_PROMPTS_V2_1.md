# Chạy audit 120 ảnh cho relation prompt V2.1

Notebook này không train và không cần GPU. Add Input chứa
`objectdetectioncache.json`, sau đó chạy **nguyên cell Python** bên dưới. Cell
checkout đúng commit `819992f`, sinh prompt cho đúng 120 filename đã audit ở
V1/V2 và đóng gói kết quả thành ZIP.

```python
import json
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
OUTPUT = Path("/kaggle/working/relation_prompts_v2_1_audit_120")
PROMPT_JSON = OUTPUT / "promptcache_relation_v2_1_audit_120.json"


def run(args, cwd=None):
    args = list(map(str, args))
    print("\nRunning:", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


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

FILENAMES = REPO / "audit/relation_prompt_audit_120_filenames.txt"
assert DETECTIONS.is_file(), DETECTIONS
assert FILENAMES.is_file(), FILENAMES

if OUTPUT.exists():
    shutil.rmtree(OUTPUT)
OUTPUT.mkdir(parents=True)

run([
    sys.executable,
    "-u",
    "build_relation_prompts.py",
    "--source", DETECTIONS,
    "--output", PROMPT_JSON,
    "--filenames-file", FILENAMES,
    "--min-confidence", "0.5",
    "--max-objects", "10",
    "--max-relations", "3",
], cwd=REPO)

SUMMARY = OUTPUT / "promptcache_relation_v2_1_audit_120.summary.json"
assert PROMPT_JSON.is_file(), PROMPT_JSON
assert SUMMARY.is_file(), SUMMARY

prompts = json.loads(PROMPT_JSON.read_text(encoding="utf-8"))
summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
assert len(prompts) == 120, len(prompts)
assert {entry["heuristic_version"] for entry in prompts.values()} == {
    "H1_3_semantic_instance_aware_v2_1"
}

print("\nV2.1 AUDIT 120 SUMMARY")
print(json.dumps(summary, indent=2, ensure_ascii=False))

archive = shutil.make_archive(
    "/kaggle/working/relation_prompts_v2_1_audit_120",
    "zip",
    OUTPUT,
)
print("\nPrompt count:", len(prompts))
print("Audit JSON:", PROMPT_JSON)
print("Download ZIP:", archive)
```

Sau khi chạy xong, tải
`/kaggle/working/relation_prompts_v2_1_audit_120.zip`. Chưa sinh cache đầy đủ
và chưa train E2 cho đến khi audit trực quan V2.1 hoàn tất.
