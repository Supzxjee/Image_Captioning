import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,zipfile,shutil
from pathlib import Path

# Inputs: same full frozen TRAIN/VALID cache, plus the proven TEST/rerank inputs.
CACHE_ROOT=Path('/kaggle/input/datasets/ducanh3/csa-scst-full-karpathy-cache-v1')
COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
PROMPT_TEST_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-prompt-test-cache-v181')
DETECTIONS=Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
CHECKPOINT_ROOT='' # Blank: downloads only exact v199 full checkpoint.bin via API.
RESUME_ROOT='' # Optional whole previous epoch2 output, never epoch1 output here.
SECRET_NAMES=('codex','KAGGLE_API_TOKEN')
RUN_TAG='run1'
WALL_HOURS=5.75 # Invocation stop cap, not ETA or GPU quota billing.
EVALUATION_RESERVE_SECONDS=6000 # 100 minutes after training for VALID/TEST/rerank/CPU scorers.
SOURCE_VERSION='ducanh3/notebook7bc83c4c76/199'
CHECKPOINT_SHA='5b8d33fc0b26f2c77ad1cee4a58d3e4a718f13107a2d2bd9a0884c427faed88c'
OUTPUT=Path(f'/kaggle/working/beam_scst_epoch2_seed43_v199_{RUN_TAG}')
for path in (CACHE_ROOT,COCO_JSON,VISUAL_CACHE,PROMPT_TEST_ROOT,DETECTIONS):
    if not path.exists():raise FileNotFoundError('Attach/correct input before setup: '+str(path))
if OUTPUT.exists():raise RuntimeError('Preserve output; choose a new RUN_TAG or session')
if not 2<=WALL_HOURS<=5.75:raise ValueError('Reviewed cap 2–5.75h, not ETA/quota')
if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<10*1024**3:
    raise RuntimeError('Need10GiB scratch/output for weights, TEST patches and scorers')
OUTPUT.mkdir();WORK=Path(tempfile.mkdtemp(prefix='beam-scst-epoch2-',dir='/tmp'))
SOURCE=WORK/'train_source';SOURCE.mkdir()
TEST_SOURCE=WORK/'test_source';TEST_SOURCE.mkdir()
CLI=WORK/'cli';DEPS=WORK/'deps';CHAIR=WORK/'chair'
CONFIG=WORK/'train_config.json';LOG=OUTPUT/'notebook.log';TOKENS=[]
PARENT_DEADLINE=time.monotonic()+WALL_HOURS*3600
os.environ.update(PYTHONPATH=str(SOURCE),OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
                  USE_TF='0',USE_FLAX='0',HF_HUB_DISABLE_XET='1')
cfg=dict(semantic=True,output=str(OUTPUT),checkpoint_root=CHECKPOINT_ROOT,cache_root=str(CACHE_ROOT),
    dataset_json=str(COCO_JSON),metadata_root=str(WORK/'metadata'),chair=str(CHAIR),
    resume_root=RESUME_ROOT,train_seconds=3600,worker_seconds=900)


def write_config():
    CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    (OUTPUT/'protocol.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')


def redact(line):
    for token in TOKENS:line=line.replace(token,'[REDACTED]')
    return line
