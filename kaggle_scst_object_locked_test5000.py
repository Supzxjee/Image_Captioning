print('QFORMER OBJECT0.5 LOCKED TEST5000 BOOTSTRAP STARTED', flush=True)
import os
import subprocess
import sys
from pathlib import Path

# Locked comparison;no training. Edit exact mounts here if needed.
CHECKPOINT = Path('/kaggle/input/datasets/ducanh2403/qformer-epoch10/qformer_epoch10/model_h1_2_crossattn_epoch_10.pth')
SELECTED_CHECKPOINT = Path('/kaggle/input/datasets/ducanh2403/qformer-scst-object-w05-legacy/qformer_scst_object_w05_legacy.pth')
PROMPT_CACHE = Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
VISUAL_CACHE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
DETECTIONS = Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
OUTPUT = Path('/kaggle/working/qformer_scst_object_w05_test5000_locked')
PROJECT_COMMIT = '21846bc4602f581e9ed2d29424a80f750e5fe3e9'
REPO = Path('/tmp/Image_Captioning-scst-locked-test')
CHAIR = Path('/tmp/scst-reference/CHAIR')
os.environ.update(PYTHONUNBUFFERED='1', HDF5_USE_FILE_LOCKING='FALSE',
                  CUBLAS_WORKSPACE_CONFIG=':4096:8', HF_HUB_DISABLE_XET='1', HF_HUB_DOWNLOAD_TIMEOUT='120', TOKENIZERS_PARALLELISM='false')


def run(args, cwd=None):
    print('\nRunning:', ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def checkout(url, path, revision):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists(): run(['git', 'clone', url, path])
    assert (path/'.git').is_dir(), path
    run(['git', 'fetch', '--depth', '1', 'origin', revision], cwd=path)
    run(['git', 'checkout', '--detach', revision], cwd=path)


for file in (CHECKPOINT, SELECTED_CHECKPOINT, PROMPT_CACHE, COCO_JSON, DETECTIONS):
    assert file.is_file(), f'Missing exact input: {file}. Copy its mounted path from Input.'
assert COCO_IMAGES.is_dir(), COCO_IMAGES
for i in range(1, 4):
    assert (VISUAL_CACHE/f'visual_part_{i:02d}_of_03.h5').is_file(), VISUAL_CACHE
assert not OUTPUT.exists(), f'Choose a fresh OUTPUT directory: {OUTPUT}'
checkout('https://github.com/Supzxjee/Image_Captioning.git', REPO, PROJECT_COMMIT)
run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.57.1', 'pycocotools',
     'pycocoevalcap==1.2', 'h5py', 'pandas', 'Pillow', 'tqdm', 'scipy', 'nltk==3.9.2', 'future'])
run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps',
     'git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'])
run([sys.executable, '-c', "import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"])
run(['java', '-version'])
import torch
assert torch.cuda.device_count() >= 2, 'Select GPU T4 x2 in Notebook Settings.'
for i in range(2): print('GPU', i, torch.cuda.get_device_name(i), flush=True)
print('Evaluation only: original then selected checkpoint;each uses2GPU beam shards. No training.', flush=True)
checkout('https://github.com/Maxlinn/CHAIR-metric-standalone.git', CHAIR, '4087a26211aa2339b9a76307cb8f0321ef691d0a')
assert (CHAIR/'chair.pkl').is_file(), 'Pinned CHAIR cache missing.'
run([sys.executable, '-u', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_scst_locked_test.py', '-v'], cwd=REPO)
# Fail on model downloads and metric dependencies before test inference.
run([sys.executable, '-u', '-c', "from transformers import CLIPModel,AutoTokenizer;m=CLIPModel.from_pretrained('openai/clip-vit-base-patch16',revision='57c216476eefef5ab752ec549e440a49ae4ae5f3');t=AutoTokenizer.from_pretrained('openai/clip-vit-base-patch16',revision='57c216476eefef5ab752ec549e440a49ae4ae5f3');print('Pinned CLIP prefetch PASS',flush=True)"])
smoke = """
import json,tempfile
from pathlib import Path
from captioning.metrics import score_files
with tempfile.TemporaryDirectory() as d:
 p=Path(d);(p/'gt.json').write_text(json.dumps({'info':{},'licenses':[],'type':'captions','images':[{'id':1},{'id':2}],'annotations':[{'id':0,'image_id':1,'caption':'a person holding an umbrella'},{'id':1,'image_id':2,'caption':'a dog running through grass'}]}));(p/'pred.json').write_text(json.dumps([{'image_id':1,'caption':'a person holding an umbrella'},{'image_id':2,'caption':'a dog running through grass'}]));score_files(p/'pred.json',p/'gt.json',p/'metrics.json')
print('Official metric smoke PASS',flush=True)
"""
run([sys.executable, '-u', '-c', smoke], cwd=REPO)
run([sys.executable, '-u', REPO/'run_scst_object_locked_test.py', '--checkpoint', CHECKPOINT,
     '--selected-checkpoint', SELECTED_CHECKPOINT,
     '--prompt-cache-path', PROMPT_CACHE, '--visual-cache', VISUAL_CACHE, '--dataset-json-path', COCO_JSON,
     '--base-path', COCO_IMAGES, '--detections', DETECTIONS, '--chair-directory', CHAIR, '--output', OUTPUT], cwd=REPO)
print('RETURN FILE:', str(OUTPUT)+'_review.zip', flush=True)
