"""Generate source-pinned user-run notebook; no neural/API execution."""
import argparse
import hashlib
import json
import subprocess
from make_smoke_notebook import HERE,ROOT,PREFIX,cell,RUNNER

FILES=['run_xe_pilot.py','pilot_data.py','real_bank_data.py','model.py','REAL_BANK_INPUT_PACKAGE.json',
       'SAVE_BANK_V130_PACKAGE.json','REAL_XE_PILOT_PROTOCOL.md']

def make(revision):
    commit=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+PREFIX+n],cwd=ROOT)).hexdigest() for n in FILES}
    setup=f'''
import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile,shutil
from pathlib import Path
PREPARED_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-real-v126-lemma-fixed')
BANK_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-semantic-banks-v130')
REPORTED_GPU_HOURS=19
OUTPUT=Path('/kaggle/working/csa_real_xe_pilot_256')
COMMIT={commit!r}
HASHES={hashes!r}
if OUTPUT.exists():raise RuntimeError('Preserve existing output/checkpoint; use a fresh session.')
if not PREPARED_ROOT.is_dir() or not BANK_ROOT.is_dir():raise FileNotFoundError('Add both Dataset inputs first.')
if shutil.disk_usage('/kaggle/working').free<3*1024**3:raise RuntimeError('Need3GiB free for atomic checkpoint writes.')
OUTPUT.mkdir(parents=True)
SOURCE=Path(tempfile.mkdtemp(prefix='csa-real-xe-'))
LOG=OUTPUT/'notebook.log'
print('Source:',COMMIT,'Pilot:15small-subsetepochs/600updates/GPU0/hardcap90min',flush=True)
for name,h in HASHES.items():
    payload=urllib.request.urlopen(f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{{COMMIT}}/{PREFIX}{{name}}',timeout=90).read()
    if hashlib.sha256(payload).hexdigest()!=h:raise ValueError('Source SHA differs: '+name)
    (SOURCE/name).write_bytes(payload)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=COMMIT,sha256=HASHES,reported_gpu_hours=REPORTED_GPU_HOURS),indent=2),encoding='utf-8')
'''
    execution='''
complete=False
try:
    # Existing Torch/NumPy only: no encoder/model downloads or pip installs.
    run([sys.executable,'-u',SOURCE/'run_xe_pilot.py','--prepared-root',PREPARED_ROOT,
         '--bank-root',BANK_ROOT,'--output',OUTPUT],timeout=5400)
    complete=True
except Exception:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
    raise
finally:
    review=Path(str(OUTPUT)+'_review.zip')
    with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUTPUT.iterdir()):
            if path.is_file() and path.suffix in ('.json','.txt','.log','.md'):z.write(path,path.name)
    print('SEND REVIEW:',review,flush=True)
    print('KEEP CHECKPOINT:',OUTPUT/'checkpoint.bin',flush=True)
    print('COMPLETE' if complete else 'INCOMPLETE; report completed epochs/updates only.',flush=True)
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
        cells=[cell('markdown','''# CSA — real-data XE learning pilot
Add BOTH Datasets: `csa-real-v126-lemma-fixed` and `csa-semantic-banks-v130` (owner ducanh2403). Enable Internet + GPU; GPU0 only, no API Secret.
Fresh self-written CSA;256TRAIN×5captions,64val;15small-subsetepochs=600updates;effectivebatch32(micro4×8),FP32;3epochwarmup8e-6→1e-4.
No CLIP/ROI recache, full COCO, RL or official CIDEr/CHAIR. This pilot tests learning, not novelty/grounding accuracy.
Hard worker cap90minutes (not ETA). Save & Run All. Send reviewZIP; preserve checkpoint.bin and both input Datasets.
'''),cell('code',setup),cell('code',RUNNER.replace('Smoke active;','Pilot active;').replace('Smoke wall cap exceeded','Pilot wall cap exceeded')),cell('code',execution)])
    path=HERE/'CSA_REAL_XE_PILOT_256.ipynb';path.write_bytes((json.dumps(nb,indent=1,ensure_ascii=False)+'\n').encode('utf-8'))
    print(path,path.stat().st_size,'bytes')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--revision',required=True);make(parser.parse_args().revision)
