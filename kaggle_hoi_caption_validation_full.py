print('FULL HOI CAPTION VALIDATION BOOTSTRAP STARTED', flush=True)
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# Explicit inputs: original Q-Former epoch10, NOT an E3 checkpoint.
CHECKPOINT = Path('/kaggle/input/datasets/ducanh2403/qformer-epoch10/qformer_epoch10/model_h1_2_crossattn_epoch_10.pth')
PROMPT_CACHE = Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
VISUAL_CACHE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
DETECTIONS = Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
OUTPUT = Path('/kaggle/working/qformer_upt_val5000_fixed')
# Optional: point these two paths at the saved beam+CLIP bundle and protocol.
# Both are in the small qformer_val5000_candidates_reuse.zip recovery package.
REUSE_SCORED_CANDIDATES = None
REUSE_SOURCE_PROTOCOL = None
PREVIOUS_OUTPUT = Path('/kaggle/working/qformer_upt_val5000')
if REUSE_SCORED_CANDIDATES is None and (PREVIOUS_OUTPUT/'val_beam5_clipscore_candidates.json').is_file() and (PREVIOUS_OUTPUT/'protocol.json').is_file():
    REUSE_SCORED_CANDIDATES = PREVIOUS_OUTPUT/'val_beam5_clipscore_candidates.json'
    REUSE_SOURCE_PROTOCOL = PREVIOUS_OUTPUT/'protocol.json'
assert bool(REUSE_SCORED_CANDIDATES)==bool(REUSE_SOURCE_PROTOCOL),'Set both reuse paths or leave both None.'
PROJECT_COMMIT = '8e7a552bf81631fd1480a27ed04db105fd72c277'
REPO = Path('/kaggle/working/Image_Captioning')
REFS = Path('/tmp/hoi-caption-refs')

os.environ.update(PYTHONUNBUFFERED='1', HDF5_USE_FILE_LOCKING='FALSE', HF_HUB_DISABLE_XET='1',
                  HF_HUB_DOWNLOAD_TIMEOUT='120',TOKENIZERS_PARALLELISM='false')


def run(args,cwd=None):
    print('\nRunning:', ' '.join(map(str,args)),flush=True)
    subprocess.run(list(map(str,args)),cwd=cwd,check=True)


def checkout(url,path,revision):
    if not path.exists(): run(['git','clone',url,path])
    assert (path/'.git').exists(),path
    run(['git','fetch','--depth','1','origin',revision],cwd=path)
    run(['git','checkout','--detach',revision],cwd=path)


def restore_exact_archive(path):
    if path.is_file(): return path
    # Bounded recovery under this explicit archive path only; never scan Input.
    root = path if path.is_dir() else path.with_suffix('')
    if not root.is_dir(): raise FileNotFoundError(path)
    pickles=list(root.rglob('data.pkl'))
    if len(pickles)!=1: raise RuntimeError(f'Expected one serialized archive under {root}, found {len(pickles)}')
    destination=Path('/tmp/hoi-caption-inputs')/path.name
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_STORED) as archive:
        for file in sorted(pickles[0].parent.rglob('*')):
            if file.is_file(): archive.write(file,'archive/'+file.relative_to(pickles[0].parent).as_posix())
    print('Restored exact-path Torch archive:',destination,flush=True)
    return destination


for path in (COCO_JSON,DETECTIONS): assert path.is_file(),f'Missing input: {path}'
assert COCO_IMAGES.is_dir(),COCO_IMAGES
for i in range(1,4):
    shard=VISUAL_CACHE/f'visual_part_{i:02d}_of_03.h5'
    assert shard.is_file(),shard
CHECKPOINT=restore_exact_archive(CHECKPOINT)
PROMPT_CACHE=restore_exact_archive(PROMPT_CACHE)

checkout('https://github.com/Supzxjee/Image_Captioning.git',REPO,PROJECT_COMMIT)
run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','pycocotools','pycocoevalcap==1.2',
     'h5py','pandas','Pillow','tqdm','scipy','gdown','nltk','future'])
run([sys.executable,'-m','pip','install','-q','--no-deps',
     'git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'])
run([sys.executable,'-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab');print('CHAIR language dependencies PASS',flush=True)"])
run(['java','-version'])

import torch
print('PyTorch:',torch.__version__,'| CUDA GPUs:',torch.cuda.device_count(),flush=True)
assert torch.cuda.device_count()>=2,'This notebook requires Kaggle GPU T4 x2.'
for i in range(torch.cuda.device_count()):print('GPU',i,torch.cuda.get_device_name(i),flush=True)

REFS.mkdir(parents=True,exist_ok=True)
UPT=REFS/'upt';POCKET=REFS/'pocket';CHAIR=REFS/'chair'
checkout('https://github.com/fredzzhang/upt.git',UPT,'192743045e4c5f6827299fdc15809f5b97ac1b8e')
run(['git','submodule','update','--init','detr','hicodet'],cwd=UPT)
checkout('https://github.com/fredzzhang/pocket.git',POCKET,'bff287098bb73427fd8d5525b2ac9867fb56629d')
checkout('https://github.com/Maxlinn/CHAIR-metric-standalone.git',CHAIR,'4087a26211aa2339b9a76307cb8f0321ef691d0a')
WEIGHTS=REFS/'upt-r50-hicodet.pt'
expected='12f7f84c7c8d5bcc38f53470f57c5cc30ff2f025726e8bcc27addeb0578a258c'


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


if not WEIGHTS.is_file() or digest(WEIGHTS)!=expected:
    import gdown
    print('Downloading official UPT HICO checkpoint (~220 MB).',flush=True)
    gdown.download(id='1lvopTC_JlGHdKOyyEZTUp2VtFfCQ-08z',output=str(WEIGHTS),quiet=False)
assert WEIGHTS.is_file() and digest(WEIGHTS)==expected,'UPT download incomplete or wrong bytes.'
print('UPT checkpoint SHA256 PASS',flush=True)

# Fail on evaluator/model dependencies BEFORE a long beam-generation job.
smoke = """
import json,tempfile
from pathlib import Path
from captioning.metrics import score_files
with tempfile.TemporaryDirectory() as directory:
 p=Path(directory);(p/'gt.json').write_text(json.dumps({'info':{},'licenses':[],'type':'captions','images':[{'id':1},{'id':2}],'annotations':[{'id':0,'image_id':1,'caption':'a person holding an umbrella'},{'id':1,'image_id':2,'caption':'a dog running through grass'}]}));(p/'pred.json').write_text(json.dumps([{'image_id':1,'caption':'a person holding an umbrella'},{'image_id':2,'caption':'a dog running through grass'}]));score_files(p/'pred.json',p/'gt.json',p/'metrics.json')
print('COCO metric evaluator smoke PASS',flush=True)
"""
run([sys.executable,'-u','-c',smoke],cwd=REPO)
run([sys.executable,'-u','-c',
     "from transformers import CLIPModel,AutoTokenizer;print('Prefetching pinned CLIP (~600MB if not cached).',flush=True);m=CLIPModel.from_pretrained('openai/clip-vit-base-patch16',revision='57c216476eefef5ab752ec549e440a49ae4ae5f3');t=AutoTokenizer.from_pretrained('openai/clip-vit-base-patch16',revision='57c216476eefef5ab752ec549e440a49ae4ae5f3');print('Pinned CLIP prefetch PASS',flush=True)"])

print('Launching FULL validation: 5000 images, two GPU shards, no training.',flush=True)
arguments=[sys.executable,'-u',REPO/'run_hoi_caption_validation.py',
     '--checkpoint',CHECKPOINT,'--prompt-cache-path',PROMPT_CACHE,
     '--dataset-json-path',COCO_JSON,'--base-path',COCO_IMAGES,
     '--visual-cache',VISUAL_CACHE,'--detections',DETECTIONS,
     '--upt-directory',UPT,'--pocket-directory',POCKET,'--upt-checkpoint',WEIGHTS,
     '--chair-directory',CHAIR,'--output',OUTPUT,'--workers','2']
if REUSE_SCORED_CANDIDATES is not None:
    arguments += ['--reuse-scored-candidates',REUSE_SCORED_CANDIDATES,'--reuse-protocol',REUSE_SOURCE_PROTOCOL]
run(arguments,cwd=REPO)
summary=json.loads((OUTPUT/'run_summary.json').read_text(encoding='utf-8'))
assert summary['status']=='VALIDATION_COMPLETE' and summary['count']==5000 and not summary['test_run']
print('RETURN FILE:',str(OUTPUT)+'_review.zip',flush=True)
