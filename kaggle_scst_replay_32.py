print('SCST+CE SAME-GPU REPLAY BOOTSTRAP STARTED', flush=True)
import os
from pathlib import Path
import subprocess
import sys

# Edit these explicit paths if your Input mounts differ.
CHECKPOINT = Path('/kaggle/input/datasets/ducanh2403/qformer-epoch10/qformer_epoch10/model_h1_2_crossattn_epoch_10.pth')
PROMPT_CACHE = Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
VISUAL_CACHE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
OUTPUT = Path('/kaggle/working/qformer_scst_ce_same_gpu_replay_32')
REPO = Path('/kaggle/working/Image_Captioning')
PROJECT_COMMIT = 'ac672acabbd2c79ab21898129f354a651ac65a3d'
os.environ.update(PYTHONUNBUFFERED='1', HDF5_USE_FILE_LOCKING='FALSE',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false')


def run(args, cwd=None):
    print('\nRunning:', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


for path in (CHECKPOINT, PROMPT_CACHE, COCO_JSON):
    assert path.is_file(), f'Missing exact input: {path}'
assert COCO_IMAGES.is_dir(), COCO_IMAGES
for i in range(1, 4):
    assert (VISUAL_CACHE/f'visual_part_{i:02d}_of_03.h5').is_file(), VISUAL_CACHE
assert not OUTPUT.exists(), f'Choose a fresh OUTPUT directory: {OUTPUT}'
if not REPO.exists(): run(['git', 'clone', 'https://github.com/Supzxjee/Image_Captioning.git', REPO])
assert (REPO/'.git').is_dir(), REPO
run(['git', 'fetch', '--depth', '1', 'origin', PROJECT_COMMIT], cwd=REPO)
run(['git', 'checkout', '--detach', PROJECT_COMMIT], cwd=REPO)
run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.57.1',
     'h5py', 'pandas', 'Pillow', 'tqdm', 'nltk==3.9.2', 'pycocoevalcap==1.2'])
# Use the installed matching torch/torchvision pair; no CLIP backbone download needed.
run([sys.executable, '-u', '-c', "import torch,torchvision;from transformers import CLIPModel;assert torch.cuda.is_available(),'Enable GPU in Settings';print('GPU 0:',torch.cuda.get_device_name(0),flush=True)"])
run([sys.executable, '-u', REPO/'scst_preflight.py', '--include-anchor'], cwd=REPO)
run([sys.executable, '-u', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_scst_replay.py', '-v'], cwd=REPO)
run([sys.executable, '-u', REPO/'run_scst_replay.py', '--checkpoint', CHECKPOINT,
     '--prompt-cache-path', PROMPT_CACHE, '--visual-cache', VISUAL_CACHE,
     '--dataset-json-path', COCO_JSON, '--base-path', COCO_IMAGES, '--output', OUTPUT], cwd=REPO)
print('RETURN FILE:', str(OUTPUT)+'_review.zip', flush=True)
