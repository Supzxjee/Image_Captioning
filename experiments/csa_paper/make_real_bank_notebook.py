"""Author pinned notebook; metadata only, never submits or executes it."""
import argparse
import hashlib
import json
import subprocess
from make_smoke_notebook import HERE, ROOT, PREFIX, cell, RUNNER

FILES=['real_bank_data.py','run_real_bank.py','REAL_BANK_INPUT_PACKAGE.json','REAL_BANK_CACHE_PROTOCOL.md','model.py']

def make(revision):
    commit=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+PREFIX+n],cwd=ROOT)).hexdigest() for n in FILES}
    setup=f'''
import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile
from pathlib import Path
PREPARED_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-real-v126-lemma-fixed')
OUTPUT=Path('/kaggle/working/csa_real_bank_cache_smoke')
COMMIT={commit!r}
HASHES={hashes!r}
if OUTPUT.exists(): raise RuntimeError('Preserve existing output; use a fresh session.')
OUTPUT.mkdir(parents=True)
SOURCE=Path(tempfile.mkdtemp(prefix='csa-real-bank-'))
LOG=OUTPUT/'notebook.log'
os.environ['HF_HOME']=str(SOURCE/'hf-cache')
os.environ['HF_HUB_DISABLE_XET']='1'
print('Pinned source:',COMMIT,flush=True)
for name,h in HASHES.items():
    payload=urllib.request.urlopen(f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{{COMMIT}}/{PREFIX}{{name}}',timeout=90).read()
    if hashlib.sha256(payload).hexdigest()!=h: raise ValueError('Changed source: '+name)
    (SOURCE/name).write_bytes(payload)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=COMMIT,sha256=HASHES),indent=2),encoding='utf-8')
'''
    execution='''
complete=False
try:
    # No global dependency changes; preserve system Torch/NumPy.
    env=SOURCE/'text-env'
    run([sys.executable,'-m','venv','--without-pip','--system-site-packages',env],timeout=120)
    PY=env/'bin/python'
    run([sys.executable,'-m','pip','--python',PY,'install','--disable-pip-version-check',
         'transformers==4.57.1','huggingface-hub==0.36.2','tokenizers==0.22.1','ftfy==6.3.1'],timeout=900)
    shared=['--prepared-root',PREPARED_ROOT,'--output',OUTPUT,'--hf-cache',SOURCE/'hf-cache']
    run([PY,'-u',SOURCE/'run_real_bank.py','cache',*shared],timeout=1800)
    # Separate process frees CLIP weights before CSA allocation; successful cache persists.
    run([PY,'-u',SOURCE/'run_real_bank.py','smoke',*shared],timeout=600)
    complete=True
except Exception:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
    raise
finally:
    review=Path(str(OUTPUT)+'_review.zip')
    with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(OUTPUT.iterdir()):
            if p.is_file() and p.suffix in ('.json','.txt','.log','.md'): archive.write(p,p.name)
    print('SEND REVIEW:',review,flush=True)
    print('KEEP ENTIRE CACHE FOLDER:',OUTPUT,flush=True)
    print('COMPLETE' if complete else 'INCOMPLETE; successful cache receipt remains valid if present.',flush=True)
'''
    cells=[cell('markdown','''# CSA — frozen text bank cache + real ROI smoke
Add Dataset `ducanh2403/csa-real-v126-lemma-fixed`; enable Internet and GPU (GPU0 only). No API secret.
Encodes 5,269 nouns / 1,594 verbs with pinned frozen CLIP. Then 2 XE smoke updates on 4 TRAIN images and beam generation on 2 val images.
This is not long training or quality evaluation. Randomly initialized captions are not research results.
Preserve the entire output folder, including `semantic_banks.npz`; send only the small review ZIP.
'''),cell('code',setup),cell('code',RUNNER.replace('Smoke active;','Worker active;')),cell('code',execution)]
    value=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),nbformat=4,nbformat_minor=5)
    path=HERE/'CSA_REAL_BANK_CACHE_SMOKE.ipynb'; path.write_bytes((json.dumps(value,indent=1,ensure_ascii=False)+'\n').encode('utf-8'))
    print(path)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--revision',required=True); make(parser.parse_args().revision)
