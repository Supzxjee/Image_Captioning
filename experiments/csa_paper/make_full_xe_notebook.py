"""Write source-pinned full10-epoch notebook; never execute models or Kaggle."""
import argparse
import hashlib
import json
import subprocess
from make_smoke_notebook import HERE,ROOT,PREFIX,cell,RUNNER
from make_sharing_benchmark_notebook import FILES as BENCHMARK_FILES

FILES=list(dict.fromkeys(BENCHMARK_FILES+['run_full_xe.py','full_xe_data.py','full_xe_decode.py','test_full_xe.py','FULL_XE_10_PROTOCOL.md']))

def make(revision):
    commit=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+PREFIX+n],cwd=ROOT)).hexdigest() for n in FILES}
    setup=f'''
import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile,shutil
from pathlib import Path
FULL_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-full-roi-karpathy-v134')
BANK_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-semantic-banks-v130')
# Leave empty for a fresh run. For continuation mount the ENTIRE previous output folder.
RESUME_ROOT=''
OUTPUT=Path('/kaggle/working/csa_full_xe_10')
WALL_HOURS=10.5  # Stop limit, not an ETA or a promise of available quota.
COMMIT={commit!r}
HASHES={hashes!r}
if not FULL_ROOT.is_dir() or not BANK_ROOT.is_dir():raise FileNotFoundError('Add BOTH fullROI134 and semantic-bank130 Datasets.')
if RESUME_ROOT and not Path(RESUME_ROOT).is_dir():raise FileNotFoundError('RESUME_ROOT must be the mounted complete previous output folder.')
if OUTPUT.exists():raise RuntimeError('Preserve existing output; start a fresh session/output path. No automatic overwrite.')
if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<6*1024**3:raise RuntimeError('Need6GiB free for stage/checkpoint/temp files.')
OUTPUT.mkdir(parents=True)
SOURCE=Path(tempfile.mkdtemp(prefix='csa-full-xe10-'));STAGE=SOURCE/'staged-full-cache';LOG=OUTPUT/'notebook.log'
os.environ['OMP_NUM_THREADS']='1'
for n,h in HASHES.items():
    data=urllib.request.urlopen(f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{{COMMIT}}/{PREFIX}{{n}}',timeout=90).read()
    if hashlib.sha256(data).hexdigest()!=h:raise ValueError('Changed source:'+n)
    p=SOURCE/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=COMMIT,sha256=HASHES),indent=2),encoding='utf-8')
print('Full10 XE source:',COMMIT,flush=True)
'''
    execute='''
complete=False;parent_deadline=time.monotonic()+WALL_HOURS*3600
common=['--bank-root',BANK_ROOT,'--stage',STAGE,'--output',OUTPUT]
try:
    run([sys.executable,'-u',SOURCE/'test_full_xe.py'],timeout=120)
    run([sys.executable,'-m','pip','install','-q','pycocoevalcap==1.2'],timeout=300)
    run([sys.executable,'-u',SOURCE/'run_full_xe.py','prepare','--full-root',FULL_ROOT,*common],timeout=900)
    run([sys.executable,'-u',SOURCE/'run_full_xe.py','metric_preflight',*common],timeout=300)
    preflight=OUTPUT/'preflight';preflight.mkdir()
    launch=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0']
    run([*launch,SOURCE/'run_sharing_benchmark.py','worker','--bank-root',BANK_ROOT,'--stage',STAGE,'--output',preflight],timeout=600)
    if json.loads((preflight/'result.json').read_text(encoding='utf-8'))['preflight_cases']!=6:raise RuntimeError('Missing DDP preflight')
    remaining=parent_deadline-time.monotonic()
    if remaining<600:raise TimeoutError('No training budget after preparation')
    run([*launch,SOURCE/'run_full_xe.py','worker',*common,'--resume-root',RESUME_ROOT,'--wall-seconds',remaining-60],timeout=remaining)
    result=OUTPUT/'result.json'
    complete=result.is_file() and json.loads(result.read_text(encoding='utf-8')).get('status')=='FULL_XE_10_COMPLETE'
    if not complete:raise RuntimeError('Stopped before10epochs; preserve ENTIRE output to resume. Do not report full completion.')
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
    raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,requested_epochs=10,test_used=False)),encoding='utf-8')
    review=Path(str(OUTPUT)+'_review.zip')
    with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.txt','.log'):z.write(p,str(p.relative_to(OUTPUT)))
    print('SEND REVIEW:',review,flush=True)
    print('KEEP ENTIRE FOLDER:',OUTPUT,flush=True)
    print('LATEST:',OUTPUT/'checkpoint.bin','BEST VALID:',OUTPUT/'best.bin',flush=True)
    print('COMPLETE10' if complete else 'INCOMPLETE; preserve output and resume, no automatic restart.',flush=True)
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
        cells=[cell('markdown','''# CSA — full Karpathy TRAIN: first10 XE epochs + validation
**GPU T4×2, Internet ON, Save Version → Save & Run All.** No API Secret or old checkpoint.
Add **csa-full-roi-karpathy-v134** and **csa-semantic-banks-v130**; edit only their paths below if needed.

Fresh self-written CSA adaptation, ordinary2GPU FP32/global32captions.113287TRAIN images/566747references per epoch,
17711updates/epoch;10epochs177110updates. Warmup3full epochs8e-6→1e-4, thenflat; no SCST.
After each epoch: all5000val images, beam5/max20/rawscore + CIDEr and NLL; val never joins training, test unused.
Frozen CLIP inputs already cached; learned CSA representations recomputed each update.

Train-only estimate≈5.18h; validation/checkpoints/preparation add time, mature decoding not yet timed.
Wall cap10.5h is a stop limit. Mandatory six DDP guards + batched/per-image beam check must pass before training.
Model execution is performed by you on Kaggle; local agent checks were static/NumPy only.

`checkpoint.bin` stores latest model/Adam/cursor/both-rank RNG every1000updates and before/after validation.
`best.bin` stores greatest valid CIDEr (ties earlier). **Preserve entire output folder**, not just the small review ZIP.
To resume, mount that complete folder and set `RESUME_ROOT`; use a fresh session/output path and this same pinned notebook.
Do not point it at checkpoint131/QFormer/A. Extension beyond10epochs needs a separate approved schedule.
Send `csa_full_xe_10_review.zip`. Hallucination/CHAIR and test metrics are not computed here.
'''),cell('code',setup),cell('code',RUNNER.replace('Smoke active;','Full XE active;').replace('Smoke wall cap exceeded','Full XE wall cap exceeded')),cell('code',execute)])
    path=HERE/'CSA_FULL_XE_10_2GPU.ipynb';path.write_bytes((json.dumps(nb,indent=1,ensure_ascii=False)+'\n').encode('utf-8'))
    assert path.stat().st_size<1_000_000
    print(path,path.stat().st_size,'bytes; unexecuted')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);make(p.parse_args().revision)
