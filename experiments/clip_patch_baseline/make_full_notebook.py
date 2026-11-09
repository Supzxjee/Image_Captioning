"""Author the full10XE notebook; no API submission or model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PREFIX='experiments/clip_patch_baseline/'
sys.path.insert(0,str(ROOT/'experiments/csa_paper'))
from make_smoke_notebook import RUNNER,cell

def make(revision):
    revision=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    files=('full_data.py','full_runtime.py','run_full.py','patch_model.py','patch_beam.py','score_full.py','patch_data.py','test_full_data.py','make_full_notebook.py','SMOKE_SOURCE.json','FULL_XE10_PROTOCOL.md')
    mapping={PREFIX+n:n for n in files};mapping.update({'experiments/csa_paper/'+n:n for n in ('data.py','real_bank_data.py','cuda_runtime_guard.py','saved_quality_data.py')})
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',revision+':'+n],cwd=ROOT)).hexdigest() for n in mapping}
    setup=f'''import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile,shutil,tarfile
from pathlib import Path
REVISION={revision!r}
MAPPING={mapping!r}
HASHES={hashes!r}
'''+'''VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
RESUME_ROOT='' # Optional preserved full-run output from THIS exact recipe. Blank starts fresh.
WALL_HOURS=10.5 # Parent stop cap, not ETA/billed quota. Do not auto-extend.
OUTPUT=Path('/kaggle/working/clip_patch_baseline_full_xe10_v1')
for p in (VISUAL_CACHE,COCO_JSON):
    if not p.exists():raise FileNotFoundError('Attach/correct inputs BEFORE GPU: '+str(p))
if OUTPUT.exists():raise RuntimeError('Preserve previous output; use a fresh runtime/output path')
if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<4*1024**3:raise RuntimeError('Need4GiB free for metadata/checkpoints; no36GB patch copy')
OUTPUT.mkdir(parents=True);SOURCE=Path(tempfile.mkdtemp(prefix='clip-patch-full10-',dir='/tmp'));DEPS=SOURCE/'deps';LOG=OUTPUT/'notebook.log'
PARENT_DEADLINE=time.monotonic()+WALL_HOURS*3600
os.environ['PYTHONPATH']=str(SOURCE);os.environ['OMP_NUM_THREADS']='1'
CONFIG=SOURCE/'config.json';CHAIR=SOURCE/'chair'
cfg=dict(output=str(OUTPUT),stage=str(SOURCE/'stage'),dataset_json=str(COCO_JSON),visual_cache=str(VISUAL_CACHE),worker_seconds=600,resume_root=RESUME_ROOT,chair=str(CHAIR))
def write_config():
    CONFIG.write_text(json.dumps(cfg,indent=2));(OUTPUT/'protocol.json').write_text(json.dumps(cfg,indent=2))
write_config()
def download(url,expected,path):
    remaining=PARENT_DEADLINE-time.monotonic()
    if remaining<=0:raise TimeoutError('Parent cap during download')
    raw=urllib.request.urlopen(url,timeout=min(120,remaining)).read()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Changed pinned asset: '+path.name)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
'''
    runner=RUNNER.replace("print('Running:',", "if time.monotonic()>=PARENT_DEADLINE:raise TimeoutError('Parent cap exceeded')\n    timeout=min(timeout,PARENT_DEADLINE-time.monotonic())\n    print('Running:',").replace('Smoke active;','Full patch XE active;').replace('Smoke wall cap exceeded','Phase/parent cap exceeded')
    execution='''complete=False
try:
    for name,destination in MAPPING.items():download('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+REVISION+'/'+name,HASHES[name],SOURCE/destination)
    (OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=REVISION,sha256=HASHES),indent=2))
    run([sys.executable,'-u',SOURCE/'test_full_data.py'],120)
    run([sys.executable,'-u',SOURCE/'cuda_runtime_guard.py','--output',OUTPUT],60)
    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS,'pycocoevalcap==1.2','nltk==3.9.2','future'],600)
    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS,'git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'],600)
    os.environ['PYTHONPATH']=str(DEPS)+os.pathsep+str(SOURCE)
    run([sys.executable,'-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab');assert nltk.word_tokenize('two dogs')==['two','dogs']"],300)
    base='https://raw.githubusercontent.com/Maxlinn/CHAIR-metric-standalone/4087a26211aa2339b9a76307cb8f0321ef691d0a/'
    for name,digest in {'chair.py':'98728cd2b7fb8b97420b5f3bf14789dcad1d7d980de9c089b82856217387770d','chair.pkl':'73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3'}.items():download(base+name,digest,CHAIR/name)
    java_archive=SOURCE/'java8.tar.gz'
    download('https://github.com/adoptium/temurin8-binaries/releases/download/jdk8u472-b08/OpenJDK8U-jre_x64_linux_hotspot_8u472b08.tar.gz','6f7fb5fd640a0fd00837344b0920cbc4b9b9284b50e66f33789e3b250446a16e',java_archive)
    with tarfile.open(java_archive) as z:z.extractall(SOURCE/'jre',filter='data')
    java=list((SOURCE/'jre').rglob('bin/java'))
    if len(java)!=1:raise ValueError('Missing pinned Java8')
    os.environ['PATH']=str(java[0].parent)+os.pathsep+os.environ['PATH']
    run([java[0],'-version'],60)
    run([sys.executable,'-u',SOURCE/'score_full.py','probe','--config',CONFIG],180)
    run([sys.executable,'-u',SOURCE/'run_full.py','prepare','--config',CONFIG],1800)
    launch=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0']
    cfg['worker_seconds']=min(1200,PARENT_DEADLINE-time.monotonic()-1800);write_config()
    if cfg['worker_seconds']<180:raise TimeoutError('Insufficient budget for real IO preflight')
    run([*launch,SOURCE/'run_full.py','preflight','--config',CONFIG],cfg['worker_seconds']+30)
    cfg['worker_seconds']=PARENT_DEADLINE-time.monotonic()-1200;write_config()
    if cfg['worker_seconds']<1800:raise TimeoutError('Insufficient full-run budget after setup')
    run([*launch,SOURCE/'run_full.py','worker','--config',CONFIG],cfg['worker_seconds']+30)
    run([sys.executable,'-u',SOURCE/'score_full.py','score','--config',CONFIG],1200)
    complete=json.loads((OUTPUT/'result.json').read_text())['status']=='CLIP_PATCH_FULL_XE10_COMPLETE'
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc());raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,epochs_requested=10,test_used=False,SCST=False,reranking=False)))
    review=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUTPUT.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt'):z.write(p,str(p.relative_to(OUTPUT)))
    print('SEND REVIEW:',review,flush=True);print('KEEP ENTIRE OUTPUT:',OUTPUT,flush=True)
    print('LATEST:',OUTPUT/'checkpoint.bin','BEST VALID:',OUTPUT/'best.bin',flush=True)
    print('FULL10 COMPLETE; no TEST or prompt/bank yet.' if complete else 'INCOMPLETE; preserve weights/logs; no automatic restart or partial-model ranking.',flush=True)
'''
    intro='''# Full baseline CLIP patch → standard decoder —10epoch XE

GPU **T4×2**, Internet ON, Save & Run All. Attach only **visual-cache** and **vuthetam/mscoco-2014**. No API token, detector, ROI134, bank or prompt Dataset.

Fresh baseline, not smoke weights or old plain checkpoint. Same architecture validated by smoke192: frozenCLIP-B16 cache→196patch→projection512→standard4layer causal decoder. Direct H5 streaming + bounded CPU LRU +2loader workers/rank; no extra36GB patch copy.

113287TRAIN, all566747captionevents/epoch,10epochs/177110updates/global32; final16/11. XE only, FP32, Adam;1epoch warmup8e-6→1e-4 then cosine→1e-5. No SCST/retrieval/reranking. No110CIDEr promise.

Real full-IO/math/beam preflight before training. Discard preflight updates, verify fresh initialization. Measured budget gates; parent10.5h STOPcap is not ETA/billing guarantee. Latest reported30hGPU was before shortsmoke192; check account quota before launching.

FullVALID5000 beam5/max20 +NLL/CIDEr each epoch; best=maxCIDEr, earlier ties. Best/end saved captions get BLEU1/4, ROUGE-L, CHAIRs/i, recall. METEOR/SPICE/TEST not in this run. No mixing bestCIDEr and another checkpoint's hallucination score.

Atomic checkpoints every500updates/phase transitions, Adam/RNG/cursor/history retained. RESUME_ROOT optional only for exact recipe preserved output; no auto retry. Keep best.bin AND checkpoint.bin. Send `clip_patch_baseline_full_xe10_v1_review.zip`.

Author verified metadata/NumPy/mock/AST only; actual full IO/neural/CUDA/evaluator run by user. Old pipeline results preserved; bank/prompt integration and fair matched comparisons come after this baseline result.
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=True)),cells=[cell('markdown',intro),cell('code',setup),cell('code',runner),cell('code',execution)])
    p=HERE/'CLIP_PATCH_BASELINE_FULL_XE10_2GPU.ipynb';p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8');print(p.name,p.stat().st_size,'bytes',len(mapping),'pins',revision)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);make(p.parse_args().revision)
