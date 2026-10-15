"""Build the complete reproducible CPU notebook, pinned to an existing commit."""
import argparse
import json
from pathlib import Path


def build(commit, destination):
    code = '''import json, os, shutil, subprocess, sys
from pathlib import Path

print("Selective instance evidence routing started; CPU only.", flush=True)
# Local default. On Kaggle, change only this path to the feature ZIP OR its
# extracted root containing manifest.json, features_0.pt, features_1.pt and data/.
FEATURE_SOURCE = Path(r"C:/Users/leduc/Downloads/triplet_detector_training_features_v1_review.zip")
REPO = Path(r"C:/KhoaLuanTotNghiep/Code/github_repo") if os.name == "nt" else Path("/kaggle/working/Image_Captioning")
OUTPUT = Path(r"C:/KhoaLuanTotNghiep/Code/output/selective_evidence_routing_notebook") if os.name == "nt" else Path("/kaggle/working/selective_evidence_routing_v1")
COMMIT = __COMMIT__
EPOCHS, SEED = 30, 42

def run(args, cwd=None):
    args = list(map(str, args))
    print("Running:", " ".join(args), flush=True)
    subprocess.run(args, cwd=cwd, check=True)

assert FEATURE_SOURCE.exists(), f"Set FEATURE_SOURCE to the existing feature ZIP or extracted root: {FEATURE_SOURCE}"
if FEATURE_SOURCE.is_dir():
    for name in ("manifest.json", "features_0.pt", "features_1.pt", "data/train_pairs.json", "data/val_pairs.json", "data/pair_summary.json"):
        assert (FEATURE_SOURCE / name).is_file(), FEATURE_SOURCE / name

if not REPO.exists():
    REPO.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "https://github.com/Supzxjee/Image_Captioning.git", REPO])
assert (REPO / ".git").is_dir(), REPO
# Do not move an existing local checkout. Read-only git archive creates a pinned
# runtime copy without overwriting any uncommitted work.
found = subprocess.run(["git", "cat-file", "-e", COMMIT + "^{commit}"], cwd=REPO, capture_output=True)
if found.returncode:
    run(["git", "fetch", "origin", COMMIT], cwd=REPO)
RUNTIME = OUTPUT / "pinned_source"
assert not OUTPUT.exists() or not any(OUTPUT.iterdir()), "Use a new OUTPUT directory for a separate run."
RUNTIME.mkdir(parents=True, exist_ok=True)
archive_path = OUTPUT / "pinned_source.tar"
run(["git", "archive", "--format=tar", "--output", archive_path, COMMIT], cwd=REPO)
import tarfile
with tarfile.open(archive_path) as archive:
    archive.extractall(RUNTIME, filter="data")
archive_path.unlink()

# Kaggle already provides torch. Local project dependency directory is optional.
environment = os.environ.copy()
local_deps = Path(r"C:/KhoaLuanTotNghiep/Code/.tmp/e3-test-deps")
if os.name == "nt" and local_deps.is_dir():
    environment["PYTHONPATH"] = str(local_deps) + os.pathsep + environment.get("PYTHONPATH", "")
environment["PYTHONUNBUFFERED"] = "1"
subprocess.run([sys.executable, "-u", "run_evidence_routing_cached.py", "--source", str(FEATURE_SOURCE),
                "--output", str(OUTPUT / "result"), "--epochs", str(EPOCHS), "--seed", str(SEED)],
               cwd=RUNTIME, env=environment, check=True)
summary_path = OUTPUT / "result/evidence_routing_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["code_commit"] = COMMIT
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
for name, result in summary["comparisons"].items():
    print(name, json.dumps({"all_accepted": result["evaluation"], "after_cap": result["after_shared_three_claim_cap"]}, indent=2))
print("Verifier trained. Decoder gate is not caption-trained; no CIDEr/CHAIR result.", flush=True)
review_zip = shutil.make_archive(str(OUTPUT) + "_review", "zip", OUTPUT / "result")
print("Review ZIP:", review_zip, flush=True)
'''.replace('__COMMIT__', repr(commit))
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
        'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
        'cells': [
            {'cell_type': 'markdown', 'metadata': {}, 'id': 'scope', 'source': [
                '# Selective instance evidence routing — CPU prototype\n',
                'Runs from cached features. No MSCOCO download, visual-cache copy or GPU.\n',
                'On Kaggle enable Internet for GitHub, choose Accelerator=None and set FEATURE_SOURCE.\n',
                'ZIP and extracted input directory are both supported; no recursive path search.\n',
                'This trains/calibrates the verifier and exports matched prompts/evidence. Decoder adaptation and reranking are not run.']},
            {'cell_type': 'code', 'metadata': {}, 'id': 'run', 'execution_count': None, 'outputs': [],
             'source': code.splitlines(keepends=True)}]}
    Path(destination).write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--commit', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); build(args.commit, args.output)
