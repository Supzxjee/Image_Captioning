"""Build CPU comparison notebook. Does not call API/scorers/models locally."""
import argparse
import hashlib
import json
import subprocess
from make_smoke_notebook import HERE,ROOT,PREFIX,cell,RUNNER

FILES=['score_saved_quality.py','saved_quality_data.py','comparison_baseline4000.json.gz','COMPARISON_INPUTS.json','QUALITY_COMPARISON_PROTOCOL.md']

def make(revision):
    commit=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+PREFIX+n],cwd=ROOT)).hexdigest() for n in FILES}
    old=json.loads((HERE/'SAVE_CSA_XE10_V139_DATASET.ipynb').read_text(encoding='utf-8'))
    auth=next(c['source'] for c in old['cells'] if c['cell_type']=='code' and 'def run_cli(' in c['source'])
    # Read-only download needs no Dataset owner/slug/title validation.
    auth='\n'.join(line for line in auth.splitlines() if not (line.startswith('if ') and any(n in line for n in ('OWNER','SLUG','TITLE'))))+'\n'
    setup=f'''
import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile,re,shutil
from pathlib import Path
KERNEL_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/139'
SECRET_NAME='KAGGLE_API_TOKEN' # Change to codex if that is your actual Secret name.
OUTPUT=Path('/kaggle/working/csa_saved_quality_v139_v2')
COMMIT={commit!r}
HASHES={hashes!r}
if OUTPUT.exists():raise RuntimeError('Preserve existing output and use a fresh session/path.')
OUTPUT.mkdir(parents=True);SOURCE=Path(tempfile.mkdtemp(prefix='csa-saved-quality-'));LOG=OUTPUT/'notebook.log'
for n,h in HASHES.items():
    raw=urllib.request.urlopen(f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{{COMMIT}}/{PREFIX}{{n}}',timeout=90).read()
    if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Changed source:'+n)
    (SOURCE/n).write_bytes(raw)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=COMMIT,sha256=HASHES),indent=2),encoding='utf-8')
'''
    execute=r'''
complete=False
try:
    DOWNLOAD=SOURCE/'download';DOWNLOAD.mkdir()
    run_cli([sys.executable,'-m','kaggle','kernels','output',KERNEL_VERSION,'-p',str(DOWNLOAD),
             '--file-pattern',r'(^|/)csa_full_xe_10_v2_review\.zip$'])
    manifest=json.loads((SOURCE/'COMPARISON_INPUTS.json').read_text(encoding='utf-8'))
    matches=[p for p in DOWNLOAD.rglob('csa_full_xe_10_v2_review.zip') if hashlib.sha256(p.read_bytes()).hexdigest()==manifest['new_review_zip_sha256']]
    if len(matches)!=1:raise RuntimeError('Exact139 review missing or changed; no latest fallback.')
    review=matches[0]
    run([sys.executable,'-m','pip','install','-q','pycocoevalcap==1.2','nltk==3.9.2','future'],timeout=300)
    run([sys.executable,'-m','pip','install','-q','--no-deps','git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'],timeout=300)
    run([sys.executable,'-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"],timeout=300)
    run(['java','-version'],timeout=30)
    CHAIR=SOURCE/'chair'
    run(['git','init',CHAIR],timeout=60)
    run(['git','-C',CHAIR,'remote','add','origin','https://github.com/Maxlinn/CHAIR-metric-standalone.git'],timeout=60)
    run(['git','-C',CHAIR,'fetch','--depth','1','origin','4087a26211aa2339b9a76307cb8f0321ef691d0a'],timeout=300)
    run(['git','-C',CHAIR,'checkout','--detach','FETCH_HEAD'],timeout=60)
    if hashlib.sha256((CHAIR/'chair.pkl').read_bytes()).hexdigest()!='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':raise ValueError('Changed CHAIR cache')
    run([sys.executable,'-u',SOURCE/'score_saved_quality.py','--review',review,'--baseline',SOURCE/'comparison_baseline4000.json.gz',
         '--manifest',SOURCE/'COMPARISON_INPUTS.json','--chair',CHAIR,'--output',OUTPUT],timeout=2400)
    result=json.loads((OUTPUT/'comparison.json').read_text(encoding='utf-8'))
    complete=result['status']=='SAVED_CAPTION_COMPARISON_COMPLETE'
    if not complete:raise RuntimeError('Incomplete comparison')
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,model_run=False,test_used=False)),encoding='utf-8')
    archive=Path(str(OUTPUT)+'_review.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(OUTPUT)))
    print('SEND REVIEW:',archive,flush=True)
    print('COMPLETE' if complete else 'INCOMPLETE; no partial-model ranking.',flush=True)
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),kaggle=dict(isGpuEnabled=False,isInternetEnabled=True)),
        cells=[cell('markdown','''# CSA vs old A/C0/C — saved captions, CPU only
Accelerator **NONE**, Internet **ON**, enable Secret `KAGGLE_API_TOKEN` (or change SECRET_NAME to `codex`). **No Add Input or checkpoint.**
Downloads only the small review ZIP from exact notebook139 and historical caption text. Does not train/generate captions/load models.
Common4000: re-score A,C0,C,CSA9,CSA10 with identical first5references/image and identical PTB/CIDEr/CHAIR cache.
Full5000: score CSA9/10 separately with original25010references. Output CIDEr/BLEU4/ROUGE-L/CHAIRs/i/recall/false counts/length.
Historical A/C0/C metrics and stored CSA5000 CIDEr must replay; otherwise stop. Best9 fixed before CHAIR.
Loss: includes CSA NLL history; old comparable NLL absent, vocabularies differ. No fabricated cross-model loss ranking.
Oldgreedy vs CSAbeam5 and historically reused validation remain limitations; this is not an independent test.
V2 restores exact old image-ID/input order and original CSA5000rank-concat order. Saves actual/expected metric diagnostics.
All CHAIR outputs are preserved even if CIDEr replay fails; any replay mismatch still makes overall comparison INCOMPLETE.
Send `csa_saved_quality_v139_v2_review.zip`. CPU dependency/scorer execution remains user-run pending.
'''),cell('code',setup),cell('code',auth),cell('code',RUNNER.replace('Smoke active;','CPU metrics active;').replace('Smoke wall cap exceeded','CPU metric cap exceeded')),cell('code',execute)])
    p=HERE/'CSA_SAVED_QUALITY_V139_CPU.ipynb';p.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
    print(p,p.stat().st_size,'bytes; unexecuted')
    recovery=f'''
# USER-RUN ONLY: paste into ONE new cell in the CURRENT failed Kaggle session.
# Reuses downloaded139review, installed dependencies and CHAIR; no new API/download of weights.
import hashlib,json,sys,tempfile,traceback,urllib.request,zipfile
from pathlib import Path
if not all(n in globals() for n in ('review','CHAIR','run')):raise RuntimeError('Current session globals missing; use the full corrected CPU notebook in a fresh session.')
REVIEW_PATH=Path(review);CHAIR_PATH=Path(CHAIR)
if not REVIEW_PATH.is_file() or not (CHAIR_PATH/'chair.pkl').is_file():raise FileNotFoundError('Preserve current downloaded review and CHAIR; otherwise use fresh full notebook.')
COMMIT={commit!r}
HASHES={hashes!r}
OUTPUT=Path('/kaggle/working/csa_saved_quality_v139_v2')
if OUTPUT.exists():raise RuntimeError('Preserve existing V2 output; do not overwrite it.')
OUTPUT.mkdir(parents=True);SOURCE=Path(tempfile.mkdtemp(prefix='csa-quality-recovery-'));LOG=OUTPUT/'notebook.log'
for n,h in HASHES.items():
    raw=urllib.request.urlopen(f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{{COMMIT}}/{PREFIX}{{n}}',timeout=90).read()
    if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Changed source:'+n)
    (SOURCE/n).write_bytes(raw)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=COMMIT,sha256=HASHES,recovery=True),indent=2),encoding='utf-8')
complete=False
try:
    run([sys.executable,'-u',SOURCE/'score_saved_quality.py','--review',REVIEW_PATH,'--baseline',SOURCE/'comparison_baseline4000.json.gz',
         '--manifest',SOURCE/'COMPARISON_INPUTS.json','--chair',CHAIR_PATH,'--output',OUTPUT],timeout=2400)
    complete=json.loads((OUTPUT/'comparison.json').read_text(encoding='utf-8'))['status']=='SAVED_CAPTION_COMPARISON_COMPLETE'
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,model_run=False,test_used=False)),encoding='utf-8')
    archive=Path(str(OUTPUT)+'_review.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(OUTPUT)))
    print('SEND REVIEW:',archive,flush=True)
    print('COMPLETE' if complete else 'INCOMPLETE; actual/expected diagnostics and CHAIR outputs preserved.',flush=True)
'''
    recovery_path=HERE/'RECOVER_SAVED_QUALITY_V139_CELL.py'
    recovery_path.write_text(recovery.lstrip(),encoding='utf-8')
    import ast
    ast.parse(recovery)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);make(p.parse_args().revision)
