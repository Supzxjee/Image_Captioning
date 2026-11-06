"""Small CPU notebook, SHA-pinned saved captions; no Kaggle/API execution."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def cell(kind,text):
    c=dict(cell_type=kind,metadata={},source=text)
    if kind=='code':c.update(execution_count=None,outputs=[])
    return c


def make(revision):
    revision=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    files=['score_prompt_test_extra.py','score_plain_csa10_extra.py','test_plain_csa10_extra.py','PLAIN_CSA10_EXTRA_INPUTS.json','plain_csa10_test_snapshot.json.gz']
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',revision+':experiments/csa_paper/'+n],cwd=ROOT)).hexdigest() for n in files}
    old=json.loads((HERE/'CSA_PROMPT_TEST_EXTRA_CONTROL_CPU.ipynb').read_text(encoding='utf-8'))
    runner=old['cells'][1]['source'];runner=runner[runner.index('def run('):]
    setup='import os,sys,json,hashlib,urllib.request,subprocess,tempfile,time,threading,queue,signal,zipfile,traceback,shutil\nfrom pathlib import Path\n'
    setup+=f'REVISION={revision!r}\nHASHES={hashes!r}\n'
    setup+='''OUTPUT=Path('/kaggle/working/plain_csa10_TEST_extra_v1')
if OUTPUT.exists():raise RuntimeError('Preserve existing output; use fresh runtime/output path.')
if shutil.disk_usage('/kaggle/working').free<3*1024**3:raise RuntimeError('Need3GiB free for Java/CoreNLP/cache')
OUTPUT.mkdir(parents=True)
SOURCE=Path(tempfile.mkdtemp(prefix='plain-csa10-extra-',dir='/tmp'))
DEPS=SOURCE/'deps';LOG=OUTPUT/'notebook.log';DEADLINE=time.monotonic()+4*3600
for name,digest in HASHES.items():
    raw=urllib.request.urlopen('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+REVISION+'/experiments/csa_paper/'+name,timeout=90).read()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Changed pinned source/snapshot:'+name)
    (SOURCE/name).write_bytes(raw)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=REVISION,sha256=HASHES),indent=2),encoding='utf-8')
ENV=os.environ.copy()
for key in ('KAGGLE_API_TOKEN','KAGGLE_USERNAME','KAGGLE_KEY'):ENV.pop(key,None)
ENV.update(PYTHONPATH=str(DEPS)+os.pathsep+str(SOURCE),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
'''+runner
    execution='''complete=False
try:
    run([sys.executable,'-u',SOURCE/'test_plain_csa10_extra.py'],120)
    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS,'pycocoevalcap==1.2'],600)
    for mode,cap in [('environment',2400),('smoke',900),('tokenize',600),('basic',1200),('meteor',2400),('spice',9000),('combine',120)]:
        print('CPU PHASE:',mode,flush=True)
        run([sys.executable,'-u',SOURCE/'score_plain_csa10_extra.py',mode,'--output',OUTPUT],cap)
    result=json.loads((OUTPUT/'result.json').read_text(encoding='utf-8'))
    complete=result['status']=='PLAIN_CSA10_TEST_EIGHT_METRICS_COMPLETE'
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,training=False,decoding=False,reranking=False)),encoding='utf-8')
    REVIEW=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(REVIEW,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUTPUT.rglob('*'):
            if p.is_file() and 'assets' not in p.relative_to(OUTPUT).parts and p.suffix in ('.json','.csv','.txt','.log'):z.write(p,str(p.relative_to(OUTPUT)))
    print('SEND REVIEW:',REVIEW,'\\nKEEP WHOLE OUTPUT:',OUTPUT,flush=True)
    print('COMPLETE: both baselines/8metrics/TEST5000' if complete else 'INCOMPLETE: preserve inputs/cache/logs; no inference or automatic retry.',flush=True)
'''
    intro='''# NoCSA10 vs CSA10 — eight TEST5000 metrics
**CPU / Accelerator None; Internet ON.** No Secret, API token, Dataset, JPEG or checkpoint needed. Run once on either account.
Loads SHA-pinned512KB archived caption/reference snapshot from GitHub; same5000TEST IDs and25010references as latest PROMPT/CONTROL. No training, decoding or reranking.

Both exactXE10/beam5/max20. Compute BLEU1/METEOR/SPICE and replayBLEU4/ROUGE/CIDEr within1e-7; carry auditedCHAIRs/i only after snapshot SHA matches. Uses same COCO1.2/Java8/CoreNLP scorers as completed185/62. Output raw +×100 andCSV. NoCSA's lowerCHAIRs alone is not bettercaptionquality (many short sentences); encodercapacity differs.

4hCPU STOPcap, notETA; heartbeat30s and preservedoutput on any failure. Author checks only stdlib/metadata/mock/static, no actual Java/scorers. Send `plain_csa10_TEST_extra_v1_review.zip`.
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=False)),cells=[cell('markdown',intro),cell('code',setup),cell('code',execution)])
    for c in nb['cells']:
        if c['cell_type']=='code':compile(c['source'],'baseline-notebook','exec')
    path=HERE/'PLAIN_CSA10_TEST_EXTRA_CPU.ipynb';path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(path.name,path.stat().st_size,'bytes; pinned',revision)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);make(p.parse_args().revision)
