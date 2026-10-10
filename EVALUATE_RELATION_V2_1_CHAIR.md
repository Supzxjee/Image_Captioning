# Đánh giá CHAIR cho E2 Relation V2.1

Workflow này dùng captions test V2.1 đã sinh, không inference và không train lại.
Add Input output của notebook test V2.1 và MS COCO 2014. Chọn accelerator `None`
và bật Internet để tải evaluator CHAIR.

```python
import json
import os
import re
import subprocess
import sys
from pathlib import Path

CHAIR_REPO = Path("/kaggle/working/CHAIR-metric-standalone")
OUTPUT = Path("/kaggle/working/chair_e2_relation_v2_1_test")
COCO_JSON = Path(
    "/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json"
)
CHAIR_COMMIT = "4087a26211aa2339b9a76307cb8f0321ef691d0a"
EXPECTED_HEURISTIC = "H1_3_semantic_instance_aware_v2_1"

# Nếu notebook gắn nhiều output giống nhau, bấm Copy Path tại file captions V2.1
# rồi dán vào đây. Có thể để trống khi manifest chỉ xuất hiện một lần.
PREDICTIONS_PATH = ""


def run(args, cwd=None):
    args = list(map(str, args))
    print("\nRunning:", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)


def coco_id(filename):
    match = re.search(r"_(\d{12})\.jpg$", filename)
    if not match:
        raise ValueError(f"Không đọc được COCO id từ filename: {filename}")
    return int(match.group(1))


# 1. Chọn captions dựa vào manifest V2.1 đi kèm.
prediction_name = "test_5000_captions_h1_2_gated.json"
manifest_name = "relation_v2_1_test_manifest.json"
candidates = []

if PREDICTIONS_PATH:
    direct_path = Path(PREDICTIONS_PATH)
    assert direct_path.is_file(), direct_path
    candidates = [direct_path]
else:
    for search_root in ("/kaggle/input", "/kaggle/working"):
        for root, directories, files in os.walk(search_root):
            directories[:] = [
                name for name in directories
                if name not in {"images", "checkpoints", ".git"}
            ]
            if prediction_name not in files or manifest_name not in files:
                continue
            folder = Path(root)
            try:
                manifest = json.loads(
                    (folder / manifest_name).read_text(encoding="utf-8")
                )
            except Exception as error:
                print("Bỏ qua manifest không đọc được:", folder, repr(error))
                continue
            if (
                manifest.get("model") == "e2_relation_v2_1"
                and manifest.get("heuristic_version") == EXPECTED_HEURISTIC
            ):
                candidates.append(folder / prediction_name)

candidates = sorted(set(candidates), key=str)
print("E2 Relation V2.1 prediction candidates:")
for path in candidates:
    print(" -", path)
assert len(candidates) == 1, (
    f"Cần đúng một caption file V2.1, tìm thấy {len(candidates)}: {candidates}. "
    "Nếu có nhiều bản sao, dán Copy Path vào PREDICTIONS_PATH."
)
PREDICTIONS = candidates[0]
assert COCO_JSON.is_file(), COCO_JSON
print("Selected V2.1 predictions:", PREDICTIONS)


# 2. Đổi eval_id sang COCO image id mà CHAIR yêu cầu.
predictions = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
coco_data = json.loads(COCO_JSON.read_text(encoding="utf-8"))
test_id_map = {
    eval_id: coco_id(image["filename"])
    for eval_id, image in enumerate(coco_data["images"])
    if image["split"] == "test"
}
assert len(predictions) == 5000
assert len(test_id_map) == 5000
prediction_ids = [int(item["image_id"]) for item in predictions]
assert len(set(prediction_ids)) == 5000
assert set(prediction_ids) == set(test_id_map)

chair_input = []
for item in predictions:
    caption = item.get("caption")
    assert isinstance(caption, str) and caption.strip(), item
    chair_input.append({
        "image_id": test_id_map[int(item["image_id"])],
        "caption": caption,
    })
assert len({item["image_id"] for item in chair_input}) == 5000

OUTPUT.mkdir(parents=True, exist_ok=True)
CHAIR_INPUT = OUTPUT / "e2_relation_v2_1_chair_input.json"
CHAIR_RESULT = OUTPUT / "e2_relation_v2_1_chair_output.json"
CHAIR_INPUT.write_text(
    json.dumps(chair_input, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print("CHAIR input:", CHAIR_INPUT)


# 3. Cố định phiên bản evaluator và dependency.
if not CHAIR_REPO.exists():
    run([
        "git", "clone",
        "https://github.com/Maxlinn/CHAIR-metric-standalone.git",
        CHAIR_REPO,
    ])
run(["git", "fetch", "origin"], cwd=CHAIR_REPO)
run(["git", "checkout", "--detach", CHAIR_COMMIT], cwd=CHAIR_REPO)
run([
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    "git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85",
    "nltk",
    "tqdm",
])
run([
    sys.executable,
    "-c",
    "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')",
])


# 4. Chạy CHAIR trên đúng 5.000 captions V2.1.
CHAIR_CACHE = CHAIR_REPO / "chair.pkl"
assert CHAIR_CACHE.is_file(), CHAIR_CACHE
run([
    sys.executable,
    "-u",
    CHAIR_REPO / "chair.py",
    "--cap_file", CHAIR_INPUT,
    "--image_id_key", "image_id",
    "--caption_key", "caption",
    "--cache", CHAIR_CACHE,
    "--save_path", CHAIR_RESULT,
], cwd=CHAIR_REPO)

result = json.loads(CHAIR_RESULT.read_text(encoding="utf-8"))
metrics = result["overall_metrics"]
assert len(result["sentences"]) == 5000
summary = {
    "CHAIRs": float(metrics["CHAIRs"]),
    "CHAIRi": float(metrics["CHAIRi"]),
    "Recall": float(metrics["Recall"]),
}
SUMMARY = OUTPUT / "e2_relation_v2_1_chair_summary.json"
SUMMARY.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\nE2 RELATION V2.1 CHAIR TEST")
print(json.dumps(summary, indent=2))
print("Result:", CHAIR_RESULT)
print("Summary:", SUMMARY)
```

So sánh kết quả với E2 prompt V1 (`CHAIRs=0,0388`, `CHAIRi=0,026636`,
`Recall=0,443922`) và E2 không Soft Selector (`CHAIRs=0,0388`,
`CHAIRi=0,026793`, `Recall=0,443986`).

