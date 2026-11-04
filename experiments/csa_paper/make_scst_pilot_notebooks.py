"""Author notebooks only; no API calls, neural imports or notebook execution."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from make_smoke_notebook import HERE,ROOT,PREFIX,cell,RUNNER

FILES=['scst_pilot_data.py','scst_pilot_model.py','run_scst_pilot.py','test_scst_pilot_data.py',
       'test_scst_pilot_model.py','model.py','real_bank_data.py','full_cache_data.py','full_xe_data.py',
       'sharing_data.py','package_full_roi_dataset.py','saved_quality_data.py','SAVE_FULL_ROI_V134_PACKAGE.json',
       'SCST_PILOT_PROTOCOL.md','make_scst_pilot_notebooks.py','make_smoke_notebook.py']
EXTERNAL=['captioning/__init__.py','captioning/scst.py','captioning/object_reward.py','captioning/object_consistency.py']


def save147():
    old=json.loads((HERE/'SAVE_CSA_XE10_V139_DATASET.ipynb').read_text(encoding='utf-8'))
    setup=next(c['source'] for c in old['cells'] if c['cell_type']=='code' and 'def run_cli(' in c['source']).replace('139','147')
    upload=next(c['source'] for c in old['cells'] if c['cell_type']=='code' and 'CREATE_DATASET_SCRIPT=' in c['source'])
    start=upload.index("description='");end=upload.index("')",start)
    upload=upload[:start]+"description='Private byte-preserved CSA epoch15 weights from exact SaveVersion147. Upstream terms retained; no training.'"+upload[end+1:]
    config="""
from pathlib import Path
KERNEL_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/147'
OWNER='ducanh2403'
SLUG='csa-full-xe15-v147'
TITLE='CSA XE15 v147 preserved weights dataset'
SECRET_NAME='KAGGLE_API_TOKEN' # Set to codex if that is your actual enabled Secret name.
DESTINATION=Path('/kaggle/working/csa_full_xe15_v147_dataset')
"""
    transport=r"""
import hashlib,json,shutil,tempfile
EXPECTED='e02593ea079bdb563a1afe0c5898abdbf81297aa5e688b3b0df839cb820e5b25'
if DESTINATION.exists():raise RuntimeError('Preserve existing output; fresh destination required.')
DOWNLOAD=Path(tempfile.mkdtemp(prefix='csa147-',dir='/tmp'))
run_cli([sys.executable,'-m','kaggle','kernels','output',KERNEL_VERSION,'-p',str(DOWNLOAD),'--file-pattern',r'(^|/)best\.bin$'])
matches=[p for p in DOWNLOAD.rglob('best.bin') if hashlib.file_digest(p.open('rb'),'sha256').hexdigest()==EXPECTED]
if len(matches)!=1:raise RuntimeError('Expected one exact epoch15 best.bin from147; no latest/version fallback.')
DESTINATION.mkdir(parents=True)
shutil.copy2(matches[0],DESTINATION/'best.bin')
if hashlib.file_digest((DESTINATION/'best.bin').open('rb'),'sha256').hexdigest()!=EXPECTED:raise RuntimeError('Copy SHA differs')
(DESTINATION/'checkpoint_receipt.json').write_text(json.dumps(dict(source=KERNEL_VERSION,sha256=EXPECTED,validated_epoch=15,updates=265665,original_bytes=True),indent=2),encoding='utf-8')
print('Exact147 weights copied unchanged; no model loaded.',flush=True)
"""
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        kaggle=dict(isInternetEnabled=True,isGpuEnabled=False)),cells=[
        cell('markdown','# Save CSA epoch15 — exact147\nAccelerator **None**, Internet ON. Enable Secret KAGGLE_API_TOKEN (or change SECRET_NAME to codex). Downloads only best.bin from exact147, verifies known SHA and creates a new Private Dataset. best/latest15 are identical; one binary is sufficient. Keep original147 output for logs/optimizer. No model execution.'),
        cell('code',config),cell('code',setup),cell('code',transport),cell('code',upload),
        cell('markdown','Expected mount: `/kaggle/input/datasets/ducanh2403/csa-full-xe15-v147`. Attach this Dataset to BOTH GPU notebooks; share private Dataset with the second account if required.')])
    write('SAVE_CSA_XE15_V147_DATASET.ipynb',nb)


def write(name,nb):
    p=HERE/name;p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8');print(name,p.stat().st_size)


def make(revision):
    commit=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    mapping={PREFIX+n:n for n in FILES};mapping.update({n:n for n in EXTERNAL})
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT)).hexdigest() for n in mapping}
    for arm in ('SCST_CE','SCST_CE_OBJECT'):
        setup=f'''
import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile,shutil
from pathlib import Path
ARM={arm!r}
WALL_HOURS=4 # Stop cap, NOT an ETA or billing estimate. User reports unchanged20h/23h.
FULL_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-full-roi-karpathy-v134')
CHECKPOINT_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-full-xe15-v147')
COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
DETECTIONS=Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
RESUME_ROOT='' # Same-arm prior output ONLY. Empty for first run; never automatically retries.
OUTPUT=Path('/kaggle/working/csa15_'+ARM.lower()+'_pilot500')
for p in (FULL_ROOT,CHECKPOINT_ROOT,COCO_JSON,DETECTIONS):
    if not p.exists():raise FileNotFoundError('Attach all4inputs before GPU: '+str(p))
probe="import json,torch;print(json.dumps(dict(version=torch.__version__,built=torch.backends.cuda.is_built(),available=torch.cuda.is_available(),count=torch.cuda.device_count())))"
GPU_ENV=json.loads(subprocess.check_output([sys.executable,'-c',probe],text=True,timeout=60))
print('GPU environment',GPU_ENV,flush=True)
if not GPU_ENV['built'] or not GPU_ENV['available'] or GPU_ENV['count']!=2:raise RuntimeError('Select GPU T4x2 and restart session; do not reinstall Torch blindly.')
if OUTPUT.exists():raise RuntimeError('Preserve old output; use a fresh output path/session.')
if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<8*1024**3:raise RuntimeError('Need8GiB free for stage/checkpoints.')
OUTPUT.mkdir(parents=True);SOURCE=Path(tempfile.mkdtemp(prefix='csa15-scst-'));STAGE=SOURCE/'stage';LOG=OUTPUT/'notebook.log'
COMMIT={commit!r}
HASHES={hashes!r}
MAPPING={mapping!r}
for name,expected in HASHES.items():
    raw=urllib.request.urlopen(f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{{COMMIT}}/{{name}}',timeout=90).read()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Changed pinned source:'+name)
    path=SOURCE/MAPPING[name];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
os.environ['OMP_NUM_THREADS']='1'
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=COMMIT,sha256=HASHES),indent=2),encoding='utf-8')
(OUTPUT/'gpu_environment.json').write_text(json.dumps(GPU_ENV),encoding='utf-8')
CHAIR=SOURCE/'chair';CONFIG=SOURCE/'config.json'
cfg=dict(arm=ARM,full_root=str(FULL_ROOT),checkpoint_root=str(CHECKPOINT_ROOT),dataset_json=str(COCO_JSON),detections=str(DETECTIONS),
    output=str(OUTPUT),stage=str(STAGE),chair=str(CHAIR),resume_root=RESUME_ROOT,worker_seconds=9000)
CONFIG.write_text(json.dumps(cfg),encoding='utf-8')
'''
        execute='''
complete=False;deadline=time.monotonic()+WALL_HOURS*3600
try:
    run([sys.executable,'-u',SOURCE/'test_scst_pilot_data.py'],timeout=120)
    run([sys.executable,'-m','pip','install','-q','pycocoevalcap==1.2','nltk==3.9.2','future'],timeout=300)
    run([sys.executable,'-m','pip','install','-q','--no-deps','git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'],timeout=300)
    run([sys.executable,'-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"],timeout=300)
    run(['java','-version'],timeout=30)
    run(['git','init',CHAIR],timeout=60)
    run(['git','-C',CHAIR,'remote','add','origin','https://github.com/Maxlinn/CHAIR-metric-standalone.git'],timeout=60)
    run(['git','-C',CHAIR,'fetch','--depth','1','origin','4087a26211aa2339b9a76307cb8f0321ef691d0a'],timeout=300)
    run(['git','-C',CHAIR,'checkout','--detach','FETCH_HEAD'],timeout=60)
    run([sys.executable,'-u',SOURCE/'test_scst_pilot_model.py'],timeout=180)
    launch=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0']
    run([*launch,SOURCE/'test_scst_pilot_model.py','--ddp'],timeout=300)
    run([sys.executable,'-u',SOURCE/'run_scst_pilot.py','prepare','--config',CONFIG],timeout=min(2400,deadline-time.monotonic()-1800))
    run([sys.executable,'-u',SOURCE/'run_scst_pilot.py','metric_preflight','--config',CONFIG],timeout=300)
    remaining=deadline-time.monotonic()
    if remaining<2400:raise TimeoutError('Not enough worker/scoring margin; do not train automatically.')
    cfg['worker_seconds']=remaining-900;CONFIG.write_text(json.dumps(cfg),encoding='utf-8')
    run([*launch,SOURCE/'run_scst_pilot.py','worker','--config',CONFIG],timeout=cfg['worker_seconds']+60)
    run([sys.executable,'-u',SOURCE/'run_scst_pilot.py','score','--config',CONFIG],timeout=min(900,deadline-time.monotonic()))
    complete=json.loads((OUTPUT/'result.json').read_text(encoding='utf-8'))['status']=='CSA_SCST_PILOT_COMPLETE'
    if not complete:raise RuntimeError('Incomplete endpoint/VALID')
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,arm=ARM,updates_requested=500,test_used=False)),encoding='utf-8')
    archive=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt') and p.name not in ('training_references.json','object_targets.json'):
                z.write(p,str(p.relative_to(OUTPUT)))
    print('SEND REVIEW:',archive,flush=True)
    print('KEEP ENTIRE OUTPUT:',OUTPUT,flush=True)
    print('KEEP CHECKPOINT:',OUTPUT/'checkpoint.bin' if (OUTPUT/'checkpoint.bin').is_file() else 'No checkpoint yet',flush=True)
    print('COMPLETE fixed500 + VALID5000' if complete else 'INCOMPLETE; preserve logs/weights; no automatic restart.',flush=True)
'''
        intro=f'''# CSA epoch15 → {arm}, two GPUs
        Run **SAVE_CSA_XE15_V147_DATASET** on CPU first. Then select **T4x2**, Internet ON,
        Save Version → Save & Run All. One arm per account; both notebooks use2GPUs.
        Attach4inputs: csa-full-roi-karpathy-v134; csa-full-xe15-v147;
        mscoco-2014 (dataset_coco.json only); objectdetectionecache. Edit mount paths in cell1.
        No API secrets, raw images, bank130 or new feature extraction in these GPU notebooks.

        {'CE +0.05SCST(CIDEr-D)' if arm=='SCST_CE' else 'CE +0.05SCST(CIDEr-D +0.5Robj)'}.
        Same CSA15 weights, freshAdam5e-6, FP32, dropoutOFF, global8=4/GPU.
        Fixed500updates=4000unique TRAIN images drawn from a locked10k pool. This is a pilot,
        not a full RL epoch. FullTRAIN reference IDF; held-out labels excluded from reward.
        Original CSA alignment retained; no extra word-region attention loss.

        Final VALID5000 beam5/max20 with CIDEr, BLEU4, ROUGE-L, CHAIRs/i, recall, NLL.
        No TEST or mid-checkpoint selection. Four-hour stop cap is not an ETA/billing promise.
        Atomic checkpoints every100updates; measured throughput gate at25updates preserves
        partial weights if remaining budget is insufficient. Do not compare partial arms.
        Local author checks are NumPy/AST only; real policy/DDP/CUDA tests run here before training.
        '''
        nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
            kaggle=dict(isInternetEnabled=True,isGpuEnabled=True)),cells=[cell('markdown',intro),cell('code',setup),
            cell('code',RUNNER.replace('Smoke active;','CSA SCST active;')),cell('code',execute)])
        write('CSA15_'+arm+'_PILOT500_2GPU.ipynb',nb)
    save147()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',default='HEAD');a=p.parse_args();make(a.revision)
