"""Author CPU cache transport and two isolated GPU arms; never execute notebooks."""
import argparse
import hashlib
import json
import subprocess
from make_smoke_notebook import HERE,ROOT,PREFIX,cell,RUNNER

FILES=['grounding_pilot_data.py','grounding_model.py','run_grounding_pilot.py','test_grounding_model.py',
       'test_grounding_pilot_data.py','model.py','real_bank_data.py','full_cache_data.py','package_full_roi_dataset.py',
       'full_xe_data.py','sharing_data.py','saved_quality_data.py','SAVE_FULL_ROI_V134_PACKAGE.json',
       'grounding_audit_v2_review_text/summary.json','GROUNDING_PILOT_PROTOCOL.md','package_grounding150.py']


def pins(commit,names):
    return {n:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+PREFIX+n],cwd=ROOT)).hexdigest() for n in names}


def source_setup(commit,hashes):
    return f'''
COMMIT={commit!r}
HASHES={hashes!r}
for name,expected in HASHES.items():
    raw=urllib.request.urlopen(f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{{COMMIT}}/{PREFIX}{{name}}',timeout=90).read()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Changed pinned source:'+name)
    p=SOURCE/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
'''


def save_notebook(commit):
    names=['package_grounding150.py','real_bank_data.py','grounding_audit_v2_review_text/summary.json']
    hashes=pins(commit,names)
    old=json.loads((HERE/'SAVE_CSA_XE10_V139_DATASET.ipynb').read_text(encoding='utf-8'))
    setup=next(c['source'] for c in old['cells'] if c['cell_type']=='code' and 'def run_cli(' in c['source']).replace('139','150')
    upload=next(c['source'] for c in old['cells'] if c['cell_type']=='code' and 'CREATE_DATASET_SCRIPT=' in c['source'])
    upload=upload.replace('Private checkpoints from exact notebook139: CSA XE10 latest and validation-best epoch9, original binary bytes, receipts and complete small review ZIP. Retains upstream model/data terms. No model/training rerun.',
        'Private weak category-word/region labels from exact CSA audit output150. TRAIN10k only; original labels and provenance preserved. No model/checkpoint/features. Retains COCO upstream terms. Human label review pending.')
    config='''
import hashlib,json,os,re,subprocess,sys,tempfile,urllib.request
from pathlib import Path
KERNEL_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/150'
OWNER='ducanh2403'
SLUG='csa-grounding-audit10k-v150'
TITLE='CSA grounding audit10k v150 dataset'
SECRET_NAME='KAGGLE_API_TOKEN' # Change to codex if that is the secret's name.
DESTINATION=Path('/kaggle/working/csa_grounding_audit10k_v150_dataset')
SOURCE=Path(tempfile.mkdtemp(prefix='save-grounding150-'))
'''
    execute='''
sys.path.insert(0,str(SOURCE))
from package_grounding150 import package,NAMES
DOWNLOAD=Path(tempfile.mkdtemp(prefix='audit150-download-'))
pattern=r'(^|/)(?:'+'|'.join(re.escape(n) for n in NAMES)+r')$'
run_cli([sys.executable,'-m','kaggle','kernels','output',KERNEL_VERSION,'-p',str(DOWNLOAD),'--file-pattern',pattern])
package(DOWNLOAD,DESTINATION)
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
        cells=[cell('markdown','''# Save complete grounding labels — exact Output150
        Accelerator **None**, Internet ON. No Add Input. Enable secret `KAGGLE_API_TOKEN`
        belonging to the account that can read private source150; if named `codex`, change SECRET_NAME.
        Downloads full TRAIN label output, verifies reviewed summary/cohort/lexicon and copies
        unchanged into a NEW private Dataset. No training. Keep source150 until Dataset is ready.
        If creation/status fails, preserve payload; do not repeatedly create/overwrite.
        '''),cell('code',config),cell('code',source_setup(commit,hashes)),cell('code',setup),cell('code',execute),cell('code',upload),
        cell('markdown','''Expected mount: `/kaggle/input/datasets/ducanh2403/csa-grounding-audit10k-v150`.
        Share this private Dataset and the CSA checkpoint/ROI inputs with the second account
        before running its GPU arm. Send actual mount if different.''')])
    write('SAVE_CSA_GROUNDING_AUDIT_V150_DATASET.ipynb',nb)


def write(name,nb):
    target=HERE/name;target.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    assert target.stat().st_size<1_000_000
    print(name,target.stat().st_size,'bytes; unexecuted')


def make(revision):
    commit=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip();hashes=pins(commit,FILES)
    for arm,budget in [('XE',20),('GROUND',23)]:
        setup=f'''
import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile,shutil
from pathlib import Path
ARM={arm!r}
AVAILABLE_GPU_HOURS={budget} # Last user report; cap below is not a billing estimate.
WALL_HOURS=4
FULL_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-full-roi-karpathy-v134')
AUDIT_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-grounding-audit10k-v150')
CHECKPOINT_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-full-xe10-v139')
RESUME_ROOT='' # Exact prior SAME-ARM output folder containing checkpoint.bin + receipt; otherwise fresh continuation.
OUTPUT=Path('/kaggle/working/csa_grounding_pilot_'+ARM)
for p in (FULL_ROOT,AUDIT_ROOT,CHECKPOINT_ROOT):
    if not p.is_dir():raise FileNotFoundError('Attach all three input Datasets before GPU: '+str(p))
if OUTPUT.exists():raise RuntimeError('Preserve prior output; use fresh runtime/output path for resume.')
if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<8*1024**3:raise RuntimeError('Need8GiB free for readonly stage and atomic checkpoints.')
OUTPUT.mkdir(parents=True)
SOURCE=Path(tempfile.mkdtemp(prefix='csa-grounding-'+ARM+'-'));STAGE=SOURCE/'stage';LOG=OUTPUT/'notebook.log'
os.environ['OMP_NUM_THREADS']='1'
'''+source_setup(commit,hashes)+'''
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=COMMIT,sha256=HASHES),indent=2),encoding='utf-8')
CONFIG=SOURCE/'config.json';CHAIR=SOURCE/'chair'
cfg=dict(arm=ARM,full_root=str(FULL_ROOT),audit_root=str(AUDIT_ROOT),checkpoint_root=str(CHECKPOINT_ROOT),output=str(OUTPUT),stage=str(STAGE),chair=str(CHAIR),resume_root=RESUME_ROOT,worker_seconds=9000)
CONFIG.write_text(json.dumps(cfg),encoding='utf-8')
print('Prepared',ARM,'source',COMMIT,'fixed2000updates; VALID only.',flush=True)
'''
        execute='''
complete=False;parent_deadline=time.monotonic()+WALL_HOURS*3600
try:
    run([sys.executable,'-u',SOURCE/'test_grounding_pilot_data.py'],timeout=120)
    run([sys.executable,'-m','pip','install','-q','pycocoevalcap==1.2','nltk==3.9.2','future'],timeout=300)
    run([sys.executable,'-m','pip','install','-q','--no-deps','git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'],timeout=300)
    run([sys.executable,'-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"],timeout=300)
    run(['java','-version'],timeout=30)
    run(['git','init',CHAIR],timeout=60)
    run(['git','-C',CHAIR,'remote','add','origin','https://github.com/Maxlinn/CHAIR-metric-standalone.git'],timeout=60)
    run(['git','-C',CHAIR,'fetch','--depth','1','origin','4087a26211aa2339b9a76307cb8f0321ef691d0a'],timeout=300)
    run(['git','-C',CHAIR,'checkout','--detach','FETCH_HEAD'],timeout=60)
    run([sys.executable,'-u',SOURCE/'run_grounding_pilot.py','prepare','--config',CONFIG],timeout=2400)
    run([sys.executable,'-u',SOURCE/'run_grounding_pilot.py','metric_preflight','--config',CONFIG],timeout=300)
    run([sys.executable,'-u',SOURCE/'test_grounding_model.py'],timeout=180)
    launch=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0']
    run([*launch,SOURCE/'test_grounding_model.py','--ddp'],timeout=300)
    remaining=parent_deadline-time.monotonic()
    if remaining<1800:raise TimeoutError('Insufficient worker + scoring margin; no automatic training')
    cfg['worker_seconds']=min(10800,remaining-1200);CONFIG.write_text(json.dumps(cfg),encoding='utf-8')
    run([*launch,SOURCE/'run_grounding_pilot.py','worker','--config',CONFIG],timeout=cfg['worker_seconds']+60)
    run([sys.executable,'-u',SOURCE/'run_grounding_pilot.py','score','--config',CONFIG],timeout=min(1200,parent_deadline-time.monotonic()))
    complete=json.loads((OUTPUT/'result.json').read_text(encoding='utf-8'))['status']=='CSA_GROUNDING_PILOT_COMPLETE'
    if not complete:raise RuntimeError('Pilot incomplete')
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,arm=ARM,updates_requested=2000,test_used=False,labels_human_verified=False)),encoding='utf-8')
    archive=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.log','.txt'):z.write(p,str(p.relative_to(OUTPUT)))
    print('SEND REVIEW:',archive,flush=True)
    print('KEEP ENTIRE OUTPUT:',OUTPUT,flush=True)
    print('KEEP CHECKPOINT:',OUTPUT/'checkpoint.bin' if (OUTPUT/'checkpoint.bin').is_file() else 'No checkpoint yet',flush=True)
    print('COMPLETE endpoint2000 + VALID5000' if complete else 'INCOMPLETE: preserve output/logs; no automatic restart.',flush=True)
'''
        nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
            cells=[cell('markdown',f'''# CSA10 continuation — {arm}, two GPUs
            **T4 ×2**, Internet ON, Save Version → Save & Run All. One notebook per account;
            both run two GPUs. No API secrets needed for these training notebooks.

            Add Input: `csa-full-roi-karpathy-v134`, `csa-full-xe10-v139`,
            and saved `csa-grounding-audit10k-v150`. Edit mount paths below if necessary.
            No raw images/detector/bank130/new CLIP extraction required; banks are in checkpoint139.
            Exact epoch10 `checkpoint.bin` required, NOT best9 or epoch15.

            {arm}: {'only XE' if arm=='XE' else 'XE + 0.1 × word–region grounding'}.
            Fresh Adam LR1e-5, global32, FP32, same 64,000 events/2,000 updates (~1.28 subset epochs).
            Attention instrumentation identical across arms; original inference attention/beam5/max20.
            Fixed endpoint then VALID5000 CIDEr/BLEU4/ROUGE-L/CHAIR/recall/NLL; no TEST or best selection.
            Weak labels are NOT human-verified; this is an exploratory pilot.

            Parent4h stop cap, not ETA/billing guarantee. Checkpoints every500updates plus endpoint;
            preserve whole output for same-arm resume. First new session still initializes weights
            from CSA10, not fresh random weights. Both CPU cache staging and user-run neural/DDP
            preflights must pass; assistant has not run models locally.
            '''),cell('code',setup),cell('code',RUNNER.replace('Smoke active;','Grounding pilot active;')),cell('code',execute)])
        write(f'CSA_GROUNDING_{arm}_2000_2GPU.ipynb',nb)
    save_notebook(commit)
    (HERE/'GROUNDING_PILOT_PACKAGE.json').write_text(json.dumps(dict(core_commit=commit,sha256=hashes,user_run_pending=True,source_audit_version=150),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('revision');make(p.parse_args().revision)
