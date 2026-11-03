"""Author continuation notebook only; no local model/API execution."""
import hashlib
import json
import subprocess
from make_smoke_notebook import HERE,ROOT,PREFIX,cell,RUNNER
from make_full_xe_notebook import FILES as BASE_FILES

FILES=list(dict.fromkeys(BASE_FILES+['xe15_resume_data.py','test_xe15_resume.py','XE15_CONTINUATION_PROTOCOL.md',
    'score_xe15_quality.py','full_xe10_review_text/input_identity.json','SAVE_XE139_PACKAGE.json',
    'run_best9_decode.py','saved_quality_data.py','best9_saved_text.json.gz','BEST9_SAVED_TEXT_PACKAGE.json',
    'saved_quality_review_text/comparison.json']))


def make(revision):
    commit=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+PREFIX+n],cwd=ROOT)).hexdigest() for n in FILES}
    setup=f'''
import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile,shutil
from pathlib import Path
FULL_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-full-roi-karpathy-v134')
BANK_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-semantic-banks-v130')
RESUME_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-full-xe10-v139')
OUTPUT=Path('/kaggle/working/csa_full_xe_15')
WALL_HOURS=6.0 # Stop cap for preparation/training, not ETA; CPU quality scoring has a separate cap.
COMMIT={commit!r}
HASHES={hashes!r}
if not all(p.is_dir() for p in (FULL_ROOT,BANK_ROOT,RESUME_ROOT)):raise FileNotFoundError('Add fullROI134, semantic-bank130 and checkpoint139 Datasets.')
if OUTPUT.exists():raise RuntimeError('Preserve existing output; use a fresh session/output folder.')
if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<6*1024**3:raise RuntimeError('Need6GiB free disk.')
OUTPUT.mkdir(parents=True);SOURCE=Path(tempfile.mkdtemp(prefix='csa-xe11-15-'));STAGE=SOURCE/'stage';LOG=OUTPUT/'notebook.log'
os.environ['OMP_NUM_THREADS']='1'
for n,h in HASHES.items():
    raw=urllib.request.urlopen(f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{{COMMIT}}/{PREFIX}{{n}}',timeout=90).read()
    if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Changed source:'+n)
    p=SOURCE/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=COMMIT,sha256=HASHES),indent=2),encoding='utf-8')
'''
    nb=json.loads((HERE/'CSA_BEST9_GREEDY_BEAM5.ipynb').read_text(encoding='utf-8'))
    old=next(c['source'] for c in nb['cells'] if c['cell_type']=='code' and 'complete=False' in c['source'])
    dependencies=old[old.index("    run([sys.executable,'-m','pip'"):old.index("    run([sys.executable,'-u','-m','torch.distributed.run'")]
    execute='''
training_complete=False;quality_complete=False;parent_deadline=time.monotonic()+WALL_HOURS*3600
common=['--bank-root',BANK_ROOT,'--stage',STAGE,'--output',OUTPUT,'--target-epochs','15']
try:
    # Fail missing/changed checkpoints before staging/training. Recursive directory lookup accepts nested mount folders.
    receipts=list(RESUME_ROOT.rglob('checkpoint_receipt.json'))
    if len(receipts)!=1:raise RuntimeError('Expected one checkpoint_receipt.json under RESUME_ROOT.')
    RESUME_ROOT=receipts[0].parent
    digest=lambda p:hashlib.file_digest(Path(p).open('rb'),'sha256').hexdigest()
    receipt=json.loads(receipts[0].read_text(encoding='utf-8'))
    best=json.loads((RESUME_ROOT/'best_receipt.json').read_text(encoding='utf-8'))
    if digest(RESUME_ROOT/'checkpoint.bin')!=receipt['sha256'] or digest(RESUME_ROOT/'best.bin')!=best['sha256']:raise ValueError('Checkpoint/best bytes differ from receipts')
    if not 177110<=receipt['updates']<=265665:raise ValueError('Expected latest epoch10 or interrupted11–15 output')
    run([sys.executable,'-u',SOURCE/'test_full_xe.py'],timeout=120)
    run([sys.executable,'-u',SOURCE/'test_xe15_resume.py'],timeout=120)
'''+dependencies+'''
    run([sys.executable,'-u',SOURCE/'run_full_xe.py','prepare','--full-root',FULL_ROOT,*common],timeout=900)
    run([sys.executable,'-u',SOURCE/'run_full_xe.py','metric_preflight',*common],timeout=300)
    sys.path.insert(0,str(SOURCE))
    from run_best9_decode import saved_text
    raw,_=saved_text()
    for n in ('val_epoch09_rank0.json','val_epoch09_rank1.json'):(OUTPUT/n).write_bytes(raw[n])
    # Carry previous extension validation text when resuming an interrupted11–15 run.
    for p in RESUME_ROOT.glob('val_epoch*_rank*.json'):
        if p.name not in ('val_epoch09_rank0.json','val_epoch09_rank1.json'):shutil.copy2(p,OUTPUT/p.name)
    preflight=OUTPUT/'preflight';preflight.mkdir()
    launch=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0']
    run([*launch,SOURCE/'run_sharing_benchmark.py','worker','--bank-root',BANK_ROOT,'--stage',STAGE,'--output',preflight],timeout=600)
    if json.loads((preflight/'result.json').read_text(encoding='utf-8'))['preflight_cases']!=6:raise ValueError('Missing six DDP guards')
    remaining=parent_deadline-time.monotonic()
    if remaining<600:raise TimeoutError('Insufficient worker time after preparation')
    run([*launch,SOURCE/'run_full_xe.py','worker',*common,'--resume-root',RESUME_ROOT,'--wall-seconds',remaining-60],timeout=remaining)
    result=OUTPUT/'result.json'
    training_complete=result.is_file() and json.loads(result.read_text(encoding='utf-8'))['status']=='FULL_XE_15_COMPLETE'
    if not training_complete:raise RuntimeError('Training interrupted; preserve ENTIRE folder, resume same pinned notebook with updated RESUME_ROOT')
    run([sys.executable,'-u',SOURCE/'score_xe15_quality.py','--output',OUTPUT,'--chair',CHAIR],timeout=2400)
    quality_complete=json.loads((OUTPUT/'quality_result.json').read_text(encoding='utf-8'))['status']=='XE15_SAVED_QUALITY_COMPLETE'
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=training_complete and quality_complete,training_complete=training_complete,
        quality_complete=quality_complete,target_total_epochs=15,requested_added_epochs=5,test_used=False)),encoding='utf-8')
    archive=Path(str(OUTPUT)+'_review.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.log','.txt'):z.write(p,str(p.relative_to(OUTPUT)))
    print('SEND REVIEW:',archive,flush=True)
    print('KEEP ENTIRE FOLDER:',OUTPUT,flush=True)
    for n in ('checkpoint.bin','best.bin'):
        if (OUTPUT/n).is_file():print('KEEP:',OUTPUT/n,flush=True)
    print('COMPLETE15 + QUALITY' if training_complete and quality_complete else 'INCOMPLETE: preserve output/logs; do not restart training automatically.',flush=True)
'''
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),kaggle=dict(isGpuEnabled=True,isInternetEnabled=True)),
        cells=[cell('markdown','''# CSA XE continuation — epochs11–15, two GPUs
**T4×2, Internet ON, Save Version → Save & Run All.** Add Input:
- `csa-full-roi-karpathy-v134`
- `csa-semantic-banks-v130`
- `csa-full-xe10-v139` (latest `checkpoint.bin`, `best.bin` and their receipts)

Paths prefilled. No API token or mounted review ZIP. Mandatory resume from latest epoch10/177110updates; never from best9/fresh weights.
Five additional fullTRAIN epochs:88555updates; absolute endpoint15=265665updates. Same model/Adam/RNG/data/global32/FP32; LR remains1e-4, no warmup restart/SCST/architecture change. TwoT4DDP, ordinary features; six current-environment gradient/loss checks before actual training.
Full5000val NLL/CIDEr and originalbeam5/max20 after every epoch. Keep oldbest9 unless a new validationCIDEr exceeds it (earlier ties). No TEST.
Then CPUscore savedbest9 and11–15 CIDEr/BLEU4/ROUGE-L/CHAIRs/i/recall/counts/length, with oldbest9 metric replay. Selection is stillCIDEr, not post-hoc CHAIR. Training and CPUscore completion reported separately; scorer failure does not erase training checkpoint.
Measured old10epoch worker≈6h;5extra may take roughly3h plussetup/scoring, not a guarantee.6h parent training/preparation cap +40minCPUscorecap.23h reported quota; elapsedtime and billing are different.
Preserve ENTIRE `csa_full_xe_15` folder (latest,best,receipts,validation captions). For interruption, save it asDataset and change RESUME_ROOT to that entire folder in the SAME pinned notebook/fresh output path. Never substitute a new source revision for an interrupted15run without audit.
Send `csa_full_xe_15_review.zip`. Local agent checks are metadata/NumPy/static only; CUDA/real resume/metrics are user-run pending.
'''),cell('code',setup),cell('code',RUNNER.replace('Smoke active;','CSA continuation active;').replace('Smoke wall cap exceeded','Continuation phase cap exceeded')),cell('code',execute)])
    p=HERE/'CSA_FULL_XE_11_TO_15_2GPU.ipynb';p.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
    print(p.name,p.stat().st_size,'bytes; unexecuted')


if __name__=='__main__':
    import sys
    make(sys.argv[1] if len(sys.argv)>1 else 'HEAD')
