import json
import os
import subprocess
import sys
import time
from pathlib import Path

GROUNDED_CACHE = Path('/kaggle/input/notebooks/ducanh2403/build-grounded-relation-v1/grounded_relation_v1_full/grounded_relation_cache.pt')
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
CLIP_MODEL = 'openai/clip-vit-base-patch16'  # Can be an existing local HF CLIP directory.
REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = '2ac5f22'
OUTPUT = Path('/kaggle/working/action_candidate_audit_100')

print('Same-instance action candidate audit bootstrap started.', flush=True)
for path in (GROUNDED_CACHE, COCO_JSON):
    assert path.is_file(), path
assert COCO_IMAGES.is_dir(), COCO_IMAGES

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '30'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'

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
                print(f'Process still running: {time.monotonic()-started:.0f}s', flush=True)
        if code:
            raise subprocess.CalledProcessError(code, args)

REPO.parent.mkdir(parents=True, exist_ok=True)
if not REPO.exists():
    run(['git', 'clone', 'https://github.com/Supzxjee/Image_Captioning.git', REPO])
assert (REPO / '.git').is_dir(), REPO
run(['git', 'fetch', '--depth', '30', 'origin', 'main'], cwd=REPO)
run(['git', 'checkout', '--detach', COMMIT], cwd=REPO)
run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers', 'Pillow'], cwd=REPO)

run([
    sys.executable, '-u', 'audit_action_candidates.py',
    '--grounded-cache', GROUNDED_CACHE,
    '--dataset-json', COCO_JSON, '--images', COCO_IMAGES,
    '--priority-filenames', REPO / 'audit/action_candidate_priority_100.txt',
    '--model', CLIP_MODEL, '--pair-count', '100', '--min-gap', '0.01',
    '--seed', '42', '--output', OUTPUT,
], cwd=REPO)

summary = json.loads((OUTPUT / 'candidate_summary.json').read_text(encoding='utf-8'))
print('\nACTION CANDIDATE AUDIT SUMMARY')
print(json.dumps(summary, indent=2, ensure_ascii=False))
print('Download ZIP:', str(OUTPUT) + '.zip')
