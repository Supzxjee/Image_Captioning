"""Generate pinned user-run smoke notebook; never executes models or Kaggle."""
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
    mapping={PREFIX+n:n for n in ('patch_data.py','patch_model.py','run_smoke.py','test_patch_data.py','PROTOCOL.md')}
    mapping.update({'experiments/csa_paper/'+n:n for n in ('data.py','real_bank_data.py','cuda_runtime_guard.py')})
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',revision+':'+n],cwd=ROOT)).hexdigest() for n in mapping}
    setup=f'''import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile,shutil
from pathlib import Path
REVISION={revision!r}
MAPPING={mapping!r}
HASHES={hashes!r}
'''+'''VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
OUTPUT=Path('/kaggle/working/clip_patch_baseline_smoke_v1')
for p in (VISUAL_CACHE,COCO_JSON):
    if not p.exists():raise FileNotFoundError('Attach/correct input before GPU: '+str(p))
if OUTPUT.exists():raise RuntimeError('Preserve prior output; use fresh runtime/output path')
if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<2*1024**3:raise RuntimeError('Need2GiB free for small TRAIN stage')
OUTPUT.mkdir(parents=True);SOURCE=Path(tempfile.mkdtemp(prefix='clip-patch-baseline-',dir='/tmp'));LOG=OUTPUT/'notebook.log'
PARENT_DEADLINE=time.monotonic()+2700
os.environ['PYTHONPATH']=str(SOURCE);os.environ['OMP_NUM_THREADS']='1'
CONFIG=SOURCE/'config.json'
cfg=dict(output=str(OUTPUT),stage=str(SOURCE/'stage'),dataset_json=str(COCO_JSON),visual_cache=str(VISUAL_CACHE),updates=200,prepare_seconds=1200,worker_seconds=900)
CONFIG.write_text(json.dumps(cfg,indent=2));(OUTPUT/'protocol.json').write_text(json.dumps(cfg,indent=2))
'''
    runner=RUNNER.replace("print('Running:',", "if time.monotonic()>=PARENT_DEADLINE:raise TimeoutError('Parent cap exceeded')\n    timeout=min(timeout,PARENT_DEADLINE-time.monotonic())\n    print('Running:',").replace('Smoke active;','Patch baseline active;')
    execution='''complete=False
try:
    for name,destination in MAPPING.items():
        remaining=PARENT_DEADLINE-time.monotonic()
        if remaining<=0:raise TimeoutError('Source download parent cap exceeded')
        raw=urllib.request.urlopen('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+REVISION+'/'+name,timeout=min(90,remaining)).read()
        if hashlib.sha256(raw).hexdigest()!=HASHES[name]:raise ValueError('Changed source: '+name)
        (SOURCE/destination).write_bytes(raw)
    (OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=REVISION,sha256=HASHES),indent=2))
    run([sys.executable,'-u',SOURCE/'test_patch_data.py'],120)
    run([sys.executable,'-u',SOURCE/'cuda_runtime_guard.py','--output',OUTPUT],60)
    run([sys.executable,'-u',SOURCE/'run_smoke.py','prepare','--config',CONFIG],1200)
    cfg['worker_seconds']=min(900,PARENT_DEADLINE-time.monotonic()-120)
    if cfg['worker_seconds']<180:raise TimeoutError('Insufficient budget after staging; preserve inputs/logs')
    CONFIG.write_text(json.dumps(cfg,indent=2));(OUTPUT/'protocol.json').write_text(json.dumps(cfg,indent=2))
    run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0',SOURCE/'run_smoke.py','worker','--config',CONFIG],cfg['worker_seconds']+30)
    complete=json.loads((OUTPUT/'result.json').read_text())['status']=='CLIP_PATCH_BASELINE_TRAIN_SMOKE_COMPLETE'
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc());raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,full_training_complete=False,test_used=False,validation_used=False)))
    review=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUTPUT.rglob('*'):
            if p.is_file():z.write(p,str(p.relative_to(OUTPUT)))
    print('SEND REVIEW:',review,flush=True)
    print('TRAIN SMOKE COMPLETE; not full XE or CIDEr result.' if complete else 'INCOMPLETE; preserve output/logs, no automatic retry.',flush=True)
'''
    intro='''# CLIP patches + standard decoder — TRAIN smoke and benchmark

GPU **T4×2**, Internet ON, Save & Run All. Attach **visual-cache** and **vuthetam/mscoco-2014**. No detector, ROI134, bank, prompt or API token.

FrozenCLIP-B16 raw197tokens dropsCLS;196patch→Linear/LayerNorm512→standard4-layer causal TransformerDecoder→vocab. Fresh initialization; this is NOT old plain/CONTROL checkpoint continuation.

Only256fixedTRAINimages,200XEupdates/global32; no VALID/TEST/CIDEr/SCST/reranking. Check causal mask, image-conditioning, global-token DDP gradient parity and small-pool learning. Measure memory/throughput; small-pool timing does not guarantee full-data speed.

Latest reported budget30hGPU. Parent45min STOPcap is not ETA or billed quota. Full epoch budget/dataIO will be chosen AFTER this result. No target110CIDEr promise.

Send `clip_patch_baseline_smoke_v1_review.zip`. No full baseline trained yet; do not put these TRAIN losses into the quality comparison table.
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=True)),cells=[cell('markdown',intro),cell('code',setup),cell('code',runner),cell('code',execution)])
    p=HERE/'CLIP_PATCH_BASELINE_SMOKE_2GPU.ipynb';p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(p.name,p.stat().st_size,'bytes',len(mapping),'pins',revision)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);make(p.parse_args().revision)
