print('UPT FULL COCO CACHE STARTED — no caption training', flush=True)
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
OUTPUT = Path('/kaggle/working/upt_coco_full_cache')
MAX_HOURS = 8  # Save progress and stop before the session time limit.
PREVIOUS_CACHE = None  # Next session: Path('/kaggle/input/.../upt_coco_full_cache')
LOCAL_UPT_CHECKPOINT = None  # Optional official UPT HICO file, not Q-Former weights.

# Keep source checkouts/model files outside Output: completed cache is ~260 files, not thousands.
WORK = Path('/tmp/upt-full-cache-runtime')
REPO = WORK / 'Image_Captioning'
COMMIT = 'e70347809a8e896562f312392a4d4c1df49c9b80'
os.environ.update(PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')


def run(args, cwd=None):
    print('\nRunning:', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024**2):
            h.update(block)
    return h.hexdigest()


def checkout(url, path, revision):
    if not path.exists():
        run(['git', 'clone', url, path])
    if not (path / '.git').exists():
        raise ValueError(f'Not a Git checkout: {path}')
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=path, text=True).strip():
        raise RuntimeError(f'Preserve tracked edits in {path}; choose a fresh WORK directory.')
    run(['git', 'fetch', '--depth', '1', 'origin', revision], cwd=path)
    run(['git', 'checkout', '--detach', revision], cwd=path)


assert COCO_JSON.is_file(), COCO_JSON
assert COCO_IMAGES.is_dir(), COCO_IMAGES
assert not OUTPUT.exists(), 'Use a fresh OUTPUT; preserve the old cache and use PREVIOUS_CACHE.'
assert sha(COCO_JSON) == '2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
if PREVIOUS_CACHE is not None:
    PREVIOUS_CACHE = Path(PREVIOUS_CACHE)
    assert (PREVIOUS_CACHE / 'cache_identity.json').is_file(), PREVIOUS_CACHE
    previous_bytes = sum(p.stat().st_size for p in PREVIOUS_CACHE.iterdir() if p.is_file())
    assert shutil.disk_usage('/kaggle/working').free > previous_bytes + 2 * 1024**3, 'Need space to copy the previous cache and write new shards.'
import torch
assert torch.cuda.device_count() >= 2, 'Select Internet ON and GPU T4 x2.'
for rank in range(2):
    print('GPU', rank, torch.cuda.get_device_name(rank), flush=True)
WORK.mkdir(parents=True, exist_ok=True)
checkout('https://github.com/Supzxjee/Image_Captioning.git', REPO, COMMIT)
run([sys.executable, '-m', 'pip', 'install', '-q', 'numpy', 'Pillow', 'scipy', 'gdown'])
run([sys.executable, '-m', 'unittest', 'tests.test_upt_full_cache', '-v'], cwd=REPO)
UPT, POCKET = WORK / 'upt', WORK / 'pocket'
checkout('https://github.com/fredzzhang/upt.git', UPT, '192743045e4c5f6827299fdc15809f5b97ac1b8e')
run(['git', 'submodule', 'update', '--init', 'detr', 'hicodet'], cwd=UPT)
checkout('https://github.com/fredzzhang/pocket.git', POCKET, 'bff287098bb73427fd8d5525b2ac9867fb56629d')
CHECKPOINT = Path(LOCAL_UPT_CHECKPOINT) if LOCAL_UPT_CHECKPOINT else WORK / 'upt-r50-hicodet.pt'
if not CHECKPOINT.is_file():
    if LOCAL_UPT_CHECKPOINT:
        raise FileNotFoundError(CHECKPOINT)
    import gdown
    print('Downloading the official UPT HICO model (~220 MB).', flush=True)
    gdown.download(id='1lvopTC_JlGHdKOyyEZTUp2VtFfCQ-08z', output=str(CHECKPOINT), quiet=False)
assert sha(CHECKPOINT) == '12f7f84c7c8d5bcc38f53470f57c5cc30ff2f025726e8bcc27addeb0578a258c'
command = [sys.executable, '-u', 'build_upt_full_cache.py', '--dataset-json', COCO_JSON,
           '--images', COCO_IMAGES, '--upt', UPT, '--pocket', POCKET, '--checkpoint', CHECKPOINT,
           '--output', OUTPUT, '--max-hours', str(MAX_HOURS)]
if PREVIOUS_CACHE is not None:
    command += ['--previous-cache', PREVIOUS_CACHE]
run(command, cwd=REPO)
print('\nKEEP THE ENTIRE CACHE FOLDER:', OUTPUT, flush=True)
print('SEND THE SMALL REVIEW ZIP:', str(OUTPUT) + '_review.zip', flush=True)
