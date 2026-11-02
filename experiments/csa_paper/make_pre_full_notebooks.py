"""Build user-run decode and CPU full-ROI notebooks. No APIs/models executed."""
import argparse
import hashlib
import json
import subprocess
from make_smoke_notebook import HERE,ROOT,PREFIX,cell,RUNNER

COMMON=['real_bank_data.py','data.py','REAL_BANK_INPUT_PACKAGE.json','PRE_FULL_CHECK_PROTOCOL.md']

def notebook(revision,kind):
    commit=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    extra=(['run_decode_check.py','model.py','pilot_data.py','SAVE_BANK_V130_PACKAGE.json',
        'xe_pilot_review_text/input_receipt.json','xe_pilot_review_text/captions_final.json'] if kind=='decode' else
        ['prepare_full_regions.py','prepare_real_assets.py','audit_assets.py'])
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+PREFIX+n],cwd=ROOT)).hexdigest() for n in COMMON+extra}
    name='csa_decode_check_v131' if kind=='decode' else 'csa_full_roi_karpathy'
    setup=f'''
import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile,shutil
from pathlib import Path
COMMIT={commit!r}
HASHES={hashes!r}
OUTPUT=Path('/kaggle/working/{name}')
PREPARED_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-real-v126-lemma-fixed')
BANK_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-semantic-banks-v130')
if OUTPUT.exists():raise RuntimeError('Preserve existing output; use a fresh session.')
if shutil.disk_usage('/kaggle/working').free<{2 if kind=='decode' else 8}*1024**3:raise RuntimeError('Insufficient free disk; preserve existing files.')
OUTPUT.mkdir(parents=True)
SOURCE=Path(tempfile.mkdtemp(prefix='{name}-'))
LOG=OUTPUT/'notebook.log'
for n,h in HASHES.items():
    data=urllib.request.urlopen(f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{{COMMIT}}/{PREFIX}{{n}}',timeout=90).read()
    if hashlib.sha256(data).hexdigest()!=h:raise ValueError('Changed source: '+n)
    target=SOURCE/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=COMMIT,sha256=HASHES),indent=2),encoding='utf-8')
print('Source pinned:',COMMIT,flush=True)
'''
    if kind=='decode':
        old=json.loads((HERE/'SAVE_CSA_REAL_PREPARATION_V126_DATASET.ipynb').read_text(encoding='utf-8'))
        api_setup=next(c['source'] for c in old['cells'] if c['cell_type']=='code' and 'def run_cli(' in c['source'])
        api_setup=api_setup.replace('/126','/131').replace('v126','v131').replace('pins126','pins131')
        body='API_SETUP='+repr(api_setup)+'\n'+r'''
# Validate both cached inputs before API installation, authentication or checkpoint download.
sys.path.insert(0,str(SOURCE))
from real_bank_data import load_prepared
from pilot_data import load_banks
_,prepared_inputs,regions,images,vocab,words=load_prepared(PREPARED_ROOT)
load_banks(BANK_ROOT,words,prepared_inputs)
del regions,images,vocab,words
print('Both ROI and semantic-bank inputs verified. Now fetching exact checkpoint131.',flush=True)
KERNEL_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/131'
OWNER='ducanh2403'
SLUG='csa-decode-check'
TITLE='CSA decode check'
SECRET_NAME='KAGGLE_API_TOKEN'  # Change to codex if that is your enabled secret name.
exec(compile(API_SETUP,'pinned_cli_setup.py','exec'))
DOWNLOAD=SOURCE/'checkpoint-download';DOWNLOAD.mkdir()
run_cli([sys.executable,'-m','kaggle','kernels','output',KERNEL_VERSION,'-p',str(DOWNLOAD),
         '--file-pattern',r'(^|/)checkpoint\.bin$'])
found=list(DOWNLOAD.rglob('checkpoint.bin'))
if len(found)!=1:raise RuntimeError('Expected exactly one Output131 checkpoint; no latest fallback.')
h=hashlib.sha256()
with found[0].open('rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
if h.hexdigest()!='99b1f9936b864e549182f2307c369a221e4ca4f17b0f0cad9c60f3fb1cb7017e':raise ValueError('Output131 checkpoint SHA differs')
(OUTPUT/'download_receipt.json').write_text(json.dumps(dict(kernel_version=KERNEL_VERSION,checkpoint_sha256=h.hexdigest()),indent=2),encoding='utf-8')
run([sys.executable,'-u',SOURCE/'run_decode_check.py','--checkpoint',found[0],
     '--prepared-root',PREPARED_ROOT,'--bank-root',BANK_ROOT,'--output',OUTPUT],timeout=900)
'''
        title='CSA — same checkpoint131: greedy / beam5 / EOS'
        instructions='''GPU + Internet ON. Add `csa-real-v126-lemma-fixed` and `csa-semantic-banks-v130`.
Enable Kaggle Secret `KAGGLE_API_TOKEN` (or edit SECRET_NAME to `codex`). Downloads ONLY checkpoint.bin from exact Output131, verifies full SHA.
No training, recache, minimumlength, lengthpenalty or changed weights. Same64val, original beam5replay plus greedy/EOS trace. GPU0 worker cap15min is notETA.
'''
        filename='CSA_DECODE_CHECK_V131.ipynb'
    else:
        body='''
os.environ['CUDA_VISIBLE_DEVICES']=''
CONFIG=SOURCE/'full_roi_config.json'
config=dict(dataset_json='/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json',
    detections='/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json',
    visual_cache='/kaggle/input/datasets/ducanh2403/visual-cache',
    image_root='/kaggle/input/datasets/vuthetam/mscoco-2014/images',prepared_root=str(PREPARED_ROOT),output=str(OUTPUT))
# Edit paths here only if your mounted Dataset paths differ.
CONFIG.write_text(json.dumps(config,indent=2),encoding='utf-8')
run([sys.executable,'-u',SOURCE/'prepare_full_regions.py','--config',CONFIG],timeout=36000)
receipt=json.loads((OUTPUT/'receipt.json').read_text(encoding='utf-8'))
if receipt['status']!='FULL_ROI_PREPARATION_COMPLETE':raise RuntimeError('Full ROI incomplete')
'''
        title='CSA — full Karpathy TRAIN/val ROI preparation (CPU)'
        instructions='''Accelerator NONE + Internet ON. Add FOUR Datasets:
`mscoco-2014` (vuthetam), `objectdetectionecache`, `visual-cache`, `csa-real-v126-lemma-fixed` (ducanh2403).
No Secret, checkpoint, bank130, spaCy, CLIP encoding or training. Uses existing rawCLIPpatches+predictedboxes.
Prepares113287TRAIN+5000val, excludes test. Writes119pairedNPZ/JSONshards. CPUcap10h/internal9h50 is notETA; stops early if first-shard projection cannot fit the new cap.
Keep entire output folder; review ZIP excludes shards. Partial completion is not training-ready; no automatic resume.
'''
        filename='CSA_FULL_ROI_KARPATHY_CPU.ipynb'
    execute='''
complete=False
try:
'''+''.join('    '+line+'\n' for line in body.splitlines())+'''
    complete=True
except Exception:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
    raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,training_executed=False)),encoding='utf-8')
    review=Path(str(OUTPUT)+'_review.zip')
    globals_only={'source_receipt.json','input_receipt.json','progress.json','manifest.json','receipt.json',
                  'throughput.json','failure.txt','notebook.log','run_status.json'}
    with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.iterdir()):
            if p.is_file() and p.suffix in ('.json','.log','.txt') and ('''+repr(kind)+'''=='decode' or p.name in globals_only):z.write(p,p.name)
    print('SEND REVIEW:',review,flush=True)
    print('KEEP ENTIRE OUTPUT:',OUTPUT,flush=True)
    print('COMPLETE' if complete else 'INCOMPLETE; preserve completed artifacts/logs.',flush=True)
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
        cells=[cell('markdown','# '+title+'\n\n'+instructions),cell('code',setup),
               cell('code',RUNNER.replace('Smoke active;','Worker active;').replace('Smoke wall cap exceeded','Worker wall cap exceeded')),cell('code',execute)])
    path=HERE/filename;path.write_bytes((json.dumps(nb,ensure_ascii=False,indent=1)+'\n').encode('utf-8'));print(path,path.stat().st_size)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);p.add_argument('--kind',choices=['decode','roi','both'],default='both');args=p.parse_args()
    for kind in (('decode','roi') if args.kind=='both' else (args.kind,)):notebook(args.revision,kind)
