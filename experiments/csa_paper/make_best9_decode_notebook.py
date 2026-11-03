"""Author only; never executes Kaggle or neural code."""
import hashlib
import json
import subprocess
from make_smoke_notebook import HERE,ROOT,PREFIX,cell,RUNNER

FILES=['run_best9_decode.py','model.py','real_bank_data.py','saved_quality_data.py','SAVE_XE139_PACKAGE.json',
       'SAVE_FULL_ROI_V134_PACKAGE.json','comparison_baseline4000.json.gz','saved_quality_review_text/comparison.json',
       'DECODE_BEST9_PROTOCOL.md','best9_saved_text.json.gz','BEST9_SAVED_TEXT_PACKAGE.json']


def make(revision):
    commit=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+PREFIX+n],cwd=ROOT)).hexdigest() for n in FILES}
    setup=f'''
import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile
from pathlib import Path
FULL_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-full-roi-karpathy-v134')
# User-confirmed Dataset; lookup also accepts a nested output folder.
CHECKPOINT_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-full-xe10-v139')
OUTPUT=Path('/kaggle/working/csa_best9_greedy_beam5')
COMMIT={commit!r}
HASHES={hashes!r}
if not FULL_ROOT.is_dir() or not CHECKPOINT_ROOT.is_dir():raise FileNotFoundError('Add fullROI134 and saved checkpoint139 Datasets.')
if OUTPUT.exists():raise RuntimeError('Preserve existing output; use a fresh session/path.')
OUTPUT.mkdir(parents=True);SOURCE=Path(tempfile.mkdtemp(prefix='csa-best9-decode-'));LOG=OUTPUT/'notebook.log'
for n,h in HASHES.items():
    raw=urllib.request.urlopen(f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{{COMMIT}}/{PREFIX}{{n}}',timeout=90).read()
    if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Changed source:'+n)
    target=SOURCE/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=COMMIT,sha256=HASHES),indent=2),encoding='utf-8')
'''
    old=json.loads((HERE/'CSA_SAVED_QUALITY_V139_CPU.ipynb').read_text(encoding='utf-8'))
    scoring=next(c['source'] for c in old['cells'] if c['cell_type']=='code' and 'complete=False' in c['source'])
    dependencies=scoring[scoring.index("    run([sys.executable,'-m','pip'"):scoring.index("    run([sys.executable,'-u',SOURCE/'score_saved_quality.py'")]
    execute='''
complete=False
try:
    run([sys.executable,'-u',SOURCE/'run_best9_decode.py','prepare','--full-root',FULL_ROOT,
         '--checkpoint-root',CHECKPOINT_ROOT,'--output',OUTPUT],timeout=1200)
'''+dependencies+'''
    run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0',
         SOURCE/'run_best9_decode.py','worker','--output',OUTPUT,'--wall-seconds','5400'],timeout=5400)
    run([sys.executable,'-u',SOURCE/'run_best9_decode.py','score','--output',OUTPUT,'--chair',CHAIR],timeout=2400)
    result=json.loads((OUTPUT/'result.json').read_text(encoding='utf-8'))
    complete=result['status']=='BEST9_DECODE_COMPLETE'
    if not complete:raise RuntimeError('Incomplete diagnostic')
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,training=False,test_used=False)),encoding='utf-8')
    archive=Path(str(OUTPUT)+'_review.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.txt','.log'):z.write(p,str(p.relative_to(OUTPUT)))
    print('SEND REVIEW:',archive,flush=True)
    print('COMPLETE' if complete else 'INCOMPLETE; no partial-cohort ranking.',flush=True)
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        kaggle=dict(isGpuEnabled=True,isInternetEnabled=True)),cells=[cell('markdown','''# CSA best epoch9 — greedy vs saved beam5
Select **GPU T4 x2**, Internet ON. Add Input:
1. `csa-full-roi-karpathy-v134`.
2. Your saved checkpoint139 Dataset containing `best.bin`.

No API token, original images, review ZIP or separate semantic-bank Dataset required. The exact five saved text files from Output139 are bundled with pinned source; byte hashes and original review identity are checked. Exact SHA lookup under CHECKPOINT_ROOT; set that path to your Dataset folder if several mounted copies exist.
Frozen epoch9; no training, no resume, no TEST. Validation5000 only; max20 tokens for both decoders. Existing beam_search(width1) generates greedy; beam5 saved captions are reused. Two independent GPU workers, no DDP collectives/backward. First8 original images per GPU replay beam5 as a guard.
Reports same full5000/common4000 CIDEr/BLEU4/ROUGE-L/CHAIR/recall/counts/length. Saved beam metrics must replay before COMPLETE. A search diagnostic does not establish a new alignment method. GPU phase capped90min, not an ETA. CPU metrics afterward may take up to40min. No automatic15epoch/RL training.
Send `csa_best9_greedy_beam5_review.zip`; checkpoint/ROI arrays excluded from review.
'''),cell('code',setup),cell('code',RUNNER.replace('Smoke active;','Decode/metrics active;').replace('Smoke wall cap exceeded','Phase wall cap exceeded')),cell('code',execute)])
    out=HERE/'CSA_BEST9_GREEDY_BEAM5.ipynb';out.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
    print(out.name,out.stat().st_size,'bytes; unexecuted')
    recovery=f'''
# Paste ONE new cell in the CURRENT failed session, then rerun the last execution cell.
import hashlib,json,urllib.request
from pathlib import Path
if not all(n in globals() for n in ('SOURCE','OUTPUT','run')):raise RuntimeError('Use the corrected full notebook in a fresh session; current globals missing.')
SOURCE=Path(SOURCE);OUTPUT=Path(OUTPUT)
if (OUTPUT/'prepared.json').exists() or list(OUTPUT.glob('greedy_rank*.json')):raise RuntimeError('This recovery is only for the missing-review prepare failure; preserve later progress.')
COMMIT={commit!r}
HASHES={hashes!r}
for n,h in HASHES.items():
    raw=urllib.request.urlopen(f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{{COMMIT}}/{PREFIX}{{n}}',timeout=90).read()
    if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Changed source:'+n)
    p=SOURCE/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
for name in ('failure.txt','run_status.json','source_receipt.json'):
    p=OUTPUT/name;previous=OUTPUT/('before_prepare_fix_'+name)
    if p.exists() and not previous.exists():p.rename(previous)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=COMMIT,sha256=HASHES),indent=2),encoding='utf-8')
print('Fixed source and exact saved text ready. Rerun only the last execution cell; do not rerun setup.')
'''
    cell('code',recovery)  # AST validation only.
    (HERE/'RECOVER_BEST9_PREPARE_CELL.py').write_text(recovery.strip()+'\n',encoding='utf-8')


if __name__=='__main__':
    import sys
    make(sys.argv[1] if len(sys.argv)>1 else 'HEAD')
