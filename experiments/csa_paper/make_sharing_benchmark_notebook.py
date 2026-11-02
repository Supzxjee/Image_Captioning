"""Author matched two-GPU benchmark; no model/Kaggle execution by agent."""
import argparse
import hashlib
import json
import subprocess
from make_smoke_notebook import HERE,ROOT,PREFIX,cell,RUNNER

FILES=['run_sharing_benchmark.py','sharing_data.py','full_cache_data.py','model.py','real_bank_data.py','pilot_data.py',
       'package_full_roi_dataset.py','SAVE_FULL_ROI_V134_PACKAGE.json','SAVE_BANK_V130_PACKAGE.json','REAL_BANK_INPUT_PACKAGE.json',
       'lemma_correction_review_text/bank_words_corrected.json','DDP_SHARING_BENCHMARK_PROTOCOL.md']

def make(revision):
    commit=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+PREFIX+n],cwd=ROOT)).hexdigest() for n in FILES}
    setup=f'''
import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile,shutil
from pathlib import Path
FULL_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-full-roi-karpathy-v134')
BANK_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-semantic-banks-v130')
OUTPUT=Path('/kaggle/working/csa_ddp_sharing_benchmark')
COMMIT={commit!r}
HASHES={hashes!r}
if not FULL_ROOT.is_dir() or not BANK_ROOT.is_dir():raise FileNotFoundError('Add BOTH fullROI134 and bank130 Datasets or correct the paths.')
if OUTPUT.exists():raise RuntimeError('Preserve existing review; use a fresh session.')
if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<6*1024**3:raise RuntimeError('Need6GiB free for temporary stage.')
OUTPUT.mkdir(parents=True)
SOURCE=Path(tempfile.mkdtemp(prefix='csa-ddp-sharing-'))
STAGE=SOURCE/'staged-full-cache'
LOG=OUTPUT/'notebook.log'
os.environ['OMP_NUM_THREADS']='1'
for n,h in HASHES.items():
    data=urllib.request.urlopen(f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{{COMMIT}}/{PREFIX}{{n}}',timeout=90).read()
    if hashlib.sha256(data).hexdigest()!=h:raise ValueError('Changed source: '+n)
    path=SOURCE/n;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=COMMIT,sha256=HASHES),indent=2),encoding='utf-8')
print('Two-GPU benchmark source:',COMMIT,flush=True)
'''
    execute='''
complete=False
deadline=time.monotonic()+1800
try:
    run([sys.executable,'-u',SOURCE/'run_sharing_benchmark.py','prepare','--full-root',FULL_ROOT,
         '--bank-root',BANK_ROOT,'--stage',STAGE,'--output',OUTPUT],timeout=900)
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('Notebook benchmark budget exhausted after staging')
    run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0',
         SOURCE/'run_sharing_benchmark.py','worker','--bank-root',BANK_ROOT,'--stage',STAGE,'--output',OUTPUT],timeout=remaining)
    complete=True
except Exception:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
    raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,full_training_started=False,checkpoint_saved=False)),encoding='utf-8')
    review=Path(str(OUTPUT)+'_review.zip')
    with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.iterdir()):
            if p.is_file() and p.suffix in ('.json','.txt','.log'):z.write(p,p.name)
    print('SEND REVIEW:',review,flush=True)
    print('COMPLETE' if complete else 'INCOMPLETE; do not compare partial modes as completed.',flush=True)
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
        cells=[cell('markdown','''# CSA — two GPUs: ordinary vs feature sharing
Select **GPU T4×2 + Internet ON**. Add `csa-full-roi-karpathy-v134` and `csa-semantic-banks-v130`. No Secret or checkpoint.
Stages fullvalidatedcache once. Then six dropout0 loss/gradientparitycases before timing ordinary/shared modes on samegroupedcaptionorder.
Eachmode:10warmup+30timedupdates, FP32,16caption/GPU,globalbatch32; sharedVo/Vr autogradkept. DropoutONsharingchangesnoise correlations; noqualityequivalenceclaim.
No full epoch/RL/checkpoint/newdecodingscore. Notebook30mincap is notETA. Send `csa_ddp_sharing_benchmark_review.zip`.
'''),cell('code',setup),cell('code',RUNNER.replace('Smoke active;','DDP benchmark active;').replace('Smoke wall cap exceeded','DDP benchmark wall cap exceeded')),cell('code',execute)])
    path=HERE/'CSA_DDP_SHARING_BENCHMARK.ipynb';path.write_bytes((json.dumps(nb,indent=1,ensure_ascii=False)+'\n').encode('utf-8'));print(path,path.stat().st_size)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);make(p.parse_args().revision)
