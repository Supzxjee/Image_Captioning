# USER-RUN ONLY: paste into ONE new cell in the CURRENT failed Kaggle session.
# Reuses downloaded139review, installed dependencies and CHAIR; no new API/download of weights.
import hashlib,json,sys,tempfile,traceback,urllib.request,zipfile
from pathlib import Path
if not all(n in globals() for n in ('review','CHAIR','run')):raise RuntimeError('Current session globals missing; use the full corrected CPU notebook in a fresh session.')
REVIEW_PATH=Path(review);CHAIR_PATH=Path(CHAIR)
if not REVIEW_PATH.is_file() or not (CHAIR_PATH/'chair.pkl').is_file():raise FileNotFoundError('Preserve current downloaded review and CHAIR; otherwise use fresh full notebook.')
COMMIT='3a6532941a98384a8501e7dcfb5be3453ec407ab'
HASHES={'score_saved_quality.py': '19bced5082a59d51c933413a40662ecd0fbf5e79031f408ea4b8fad39e043bf5', 'saved_quality_data.py': 'f8a092506e3adf96292203190ea2f83de397d3028a3ddc6293d58fd360199a1e', 'comparison_baseline4000.json.gz': '8360e82244995afb12fc4eb0ede17d4a092d7f7b846105c7fcd59cb15cc7e046', 'COMPARISON_INPUTS.json': '68ad3fdf4af3ac4f38e0125ceb8d87e2085bdf02e7d3e60d9ca41bcc321dce69', 'QUALITY_COMPARISON_PROTOCOL.md': '7ef4fe5b3f000822ce89b46bee774e937a30533d7863dddd28de5a823c7e6ceb'}
OUTPUT=Path('/kaggle/working/csa_saved_quality_v139_v2')
if OUTPUT.exists():raise RuntimeError('Preserve existing V2 output; do not overwrite it.')
OUTPUT.mkdir(parents=True);SOURCE=Path(tempfile.mkdtemp(prefix='csa-quality-recovery-'));LOG=OUTPUT/'notebook.log'
for n,h in HASHES.items():
    raw=urllib.request.urlopen(f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{COMMIT}/experiments/csa_paper/{n}',timeout=90).read()
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
