print('V-COCO UPT RELATION AUDIT STARTED', flush=True)
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

# Only the existing MS COCO image dataset is required as Kaggle Input.
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
OUTPUT = Path('/kaggle/working/vcoco_upt_full_relation_audit')
WORK = Path('/tmp/vcoco-upt-relation-audit')
REPO = Path('/kaggle/working/Image_Captioning')
COMMIT = '5b397038a24fb17264588c328da9e069ea821507'
# Optional: supply the official UPT HICO checkpoint already attached as a file.
LOCAL_UPT_CHECKPOINT = None

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
        raise RuntimeError(f'Tracked changes in {path}; preserve them and use a new REPO/WORK path.')
    run(['git', 'fetch', '--depth', '1', 'origin', revision], cwd=path)
    run(['git', 'checkout', '--detach', revision], cwd=path)


assert COCO_JSON.is_file(), f'Missing: {COCO_JSON}'
assert COCO_IMAGES.is_dir(), f'Missing: {COCO_IMAGES}'
assert sha(COCO_JSON) == '2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
assert not OUTPUT.exists(), 'Use a new OUTPUT; do not overwrite earlier audit results.'
import torch
assert torch.cuda.device_count() >= 2, 'Select GPU T4 x2.'
for i in range(2):
    print('GPU', i, torch.cuda.get_device_name(i), flush=True)

WORK.mkdir(parents=True, exist_ok=True)
assert shutil.disk_usage(WORK).free > 2 * 1024**3, 'Need at least 2 GiB free in /tmp.'
checkout('https://github.com/Supzxjee/Image_Captioning.git', REPO, COMMIT)
run([sys.executable, '-m', 'pip', 'install', '-q', 'ijson==3.4.0', 'scipy', 'Pillow', 'numpy', 'gdown'])
run([sys.executable, '-m', 'unittest', 'tests.test_vcoco_relation_audit', 'tests.test_vcoco_detector_coverage',
     'tests.test_vcoco_grounded_caption_data', 'tests.test_vcoco_alignment_data', '-v'], cwd=REPO)

SOURCES = WORK / 'sources'
SOURCES.mkdir(exist_ok=True)
if not (SOURCES / 'dataset_coco.json').exists():
    shutil.copyfile(COCO_JSON, SOURCES / 'dataset_coco.json')
for split in ('train', 'val'):
    path = SOURCES / f'vcoco_{split}.json'
    if not path.exists():
        url = f'https://raw.githubusercontent.com/s-gupta/v-coco/489cc4db74f2f10ab4b134f67da3874afbf245ab/data/vcoco/vcoco_{split}.json'
        print('Downloading official relation labels:', split, flush=True)
        with urllib.request.urlopen(url, timeout=90) as response:
            path.write_bytes(response.read())

# Inventory independently verifies the source hashes and regenerates the locked IDs.
INVENTORY = WORK / 'inventory'
if not INVENTORY.exists():
    run([sys.executable, '-u', 'audit_vcoco_alignment_data.py', '--caption-json', SOURCES / 'dataset_coco.json',
         '--vcoco-train', SOURCES / 'vcoco_train.json', '--vcoco-val', SOURCES / 'vcoco_val.json',
         '--output', INVENTORY], cwd=REPO)
assert json.loads((INVENTORY / 'locked_image_groups.json').read_text()) == json.loads(
    (REPO / 'audit/vcoco_grounded_caption_week1_groups.json').read_text()), 'Locked image IDs differ.'
INSTANCES = SOURCES / 'instances_vcoco_trainval2014.json'
if not INSTANCES.exists():
    print('Fetching one official COCO annotation ZIP member; no COCO images downloaded.', flush=True)
    run([sys.executable, '-u', 'fetch_vcoco_coco_instances.py', '--sources', SOURCES, '--output', INSTANCES], cwd=REPO)
GROUNDED = WORK / 'grounded_links_lf_fixed'
if not GROUNDED.exists():
    run([sys.executable, '-u', 'prepare_vcoco_grounded_caption_data.py', '--sources', SOURCES,
         '--instances-json', INSTANCES, '--groups', INVENTORY / 'locked_image_groups.json',
         '--output', GROUNDED], cwd=REPO)
expected = json.loads((REPO / 'audit/vcoco_grounded_caption_box_join_result.json').read_text())
actual = json.loads((GROUNDED / 'box_join_summary.json').read_text())
assert actual == expected, 'Prepared source data differs from the verified box join.'

UPT, POCKET = WORK / 'upt', WORK / 'pocket'
checkout('https://github.com/fredzzhang/upt.git', UPT, '192743045e4c5f6827299fdc15809f5b97ac1b8e')
run(['git', 'submodule', 'update', '--init', 'detr', 'hicodet'], cwd=UPT)
checkout('https://github.com/fredzzhang/pocket.git', POCKET, 'bff287098bb73427fd8d5525b2ac9867fb56629d')
CHECKPOINT = Path(LOCAL_UPT_CHECKPOINT) if LOCAL_UPT_CHECKPOINT else WORK / 'upt-r50-hicodet.pt'
if not CHECKPOINT.is_file():
    if LOCAL_UPT_CHECKPOINT:
        raise FileNotFoundError(CHECKPOINT)
    import gdown
    print('Downloading official UPT checkpoint (~220 MB), not a caption checkpoint.', flush=True)
    gdown.download(id='1lvopTC_JlGHdKOyyEZTUp2VtFfCQ-08z', output=str(CHECKPOINT), quiet=False)
assert sha(CHECKPOINT) == '12f7f84c7c8d5bcc38f53470f57c5cc30ff2f025726e8bcc27addeb0578a258c'

run([sys.executable, '-u', 'run_vcoco_upt_full_audit.py', '--grounded-data', GROUNDED,
     '--instances', INSTANCES, '--upt', UPT, '--pocket', POCKET, '--checkpoint', CHECKPOINT,
     '--base-path', COCO_IMAGES, '--output', OUTPUT], cwd=REPO)
result = json.loads((OUTPUT / 'run_summary.json').read_text())
assert result['status'] == 'FROZEN_RELATION_AUDIT_COMPLETE'
print('RETURN FILE:', str(OUTPUT) + '_review.zip', flush=True)
