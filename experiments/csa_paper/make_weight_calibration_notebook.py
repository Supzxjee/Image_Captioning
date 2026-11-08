"""CPU notebook for one predeclared saved-VALID weight calibration."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def make(revision):
    revision=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    files=['experiments/csa_paper/run_weight_calibration.py',
           'experiments/csa_paper/weight_calibration_data.py',
           'experiments/csa_paper/test_weight_calibration_data.py',
           'experiments/csa_paper/make_weight_calibration_notebook.py',
           'experiments/csa_paper/valid_rerank_diagnostic/CALIBRATION_PARITY.json.gz',
           'experiments/csa_paper/valid_rerank_diagnostic/CALIBRATION_SPLIT.json',
           'experiments/csa_paper/valid_rerank_diagnostic/VALID_CANDIDATES.json.gz']
    digests={n:hashlib.sha256(subprocess.check_output(['git','show',revision+':'+n],cwd=ROOT)).hexdigest() for n in files}
    setup='import os,sys,json,time,hashlib,urllib.request,subprocess,threading,queue,signal,tempfile,zipfile,traceback,tarfile\nfrom pathlib import Path\n'
    setup+=f'REVISION={revision!r}\nFILES={digests!r}\n'
    setup+='''OUTPUT=Path('/kaggle/working/valid_weight_calibration_v1')
if OUTPUT.exists():raise RuntimeError('Preserve old output; use a fresh output/runtime.')
OUTPUT.mkdir(parents=True)
WORK=Path(tempfile.mkdtemp(prefix='valid-weight-',dir='/tmp'));DEPS=WORK/'deps';LOG=OUTPUT/'notebook.log'
DEADLINE=time.monotonic()+3600
ENV=os.environ.copy()
for k in ('KAGGLE_API_TOKEN','KAGGLE_USERNAME','KAGGLE_KEY'):ENV.pop(k,None)
ENV.update(PYTHONPATH=str(DEPS),PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2')
def run(args,timeout=900):
    start=time.monotonic();beat=start;q=queue.Queue()
    process=subprocess.Popen(list(map(str,args)),env=ENV,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
    def reader():
        for line in process.stdout:q.put(line)
        q.put(None)
    threading.Thread(target=reader,daemon=True).start()
    try:
        with LOG.open('a',encoding='utf-8') as log:
            while True:
                try:
                    line=q.get(timeout=1)
                    if line is None:break
                    print(line,end='',flush=True);log.write(line);log.flush()
                except queue.Empty:pass
                now=time.monotonic()
                if now>DEADLINE or now-start>timeout:raise TimeoutError('CPU phase/parent cap exceeded')
                if now-beat>=30:print('VALID CPU metric active;',round(now-start),'s; no model.',flush=True);beat=now
        if process.wait():raise subprocess.CalledProcessError(process.returncode,args)
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
'''
    execution='''complete=False
try:
    for name,digest in FILES.items():
        print('Downloading saved source/data:',Path(name).name,flush=True)
        raw=urllib.request.urlopen('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+REVISION+'/'+name,timeout=90).read()
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Changed pinned file: '+name)
        (WORK/Path(name).name).write_bytes(raw)
    (OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=REVISION,sha256=FILES),indent=2))
    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS,'pycocoevalcap==1.2','nltk==3.9.2','future'],600)
    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS,'git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'],600)
    run([sys.executable,'-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"],300)
    for name,digest in {'chair.py':'98728cd2b7fb8b97420b5f3bf14789dcad1d7d980de9c089b82856217387770d','chair.pkl':'73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3'}.items():
        print('Downloading official CHAIR asset:',name,flush=True)
        raw=urllib.request.urlopen('https://raw.githubusercontent.com/Maxlinn/CHAIR-metric-standalone/4087a26211aa2339b9a76307cb8f0321ef691d0a/'+name,timeout=120).read()
        if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Changed official asset: '+name)
        (WORK/name).write_bytes(raw)
    run([sys.executable,'-u',WORK/'test_weight_calibration_data.py'],120)
    url='https://github.com/adoptium/temurin8-binaries/releases/download/jdk8u472-b08/OpenJDK8U-jre_x64_linux_hotspot_8u472b08.tar.gz'
    print('Downloading pinned Java8 for PTB tokenizer; no CoreNLP/SPICE models.',flush=True)
    raw=urllib.request.urlopen(url,timeout=120).read()
    if hashlib.sha256(raw).hexdigest()!='6f7fb5fd640a0fd00837344b0920cbc4b9b9284b50e66f33789e3b250446a16e':raise ValueError('Changed Java asset')
    archive=WORK/'java8.tar.gz';archive.write_bytes(raw)
    with tarfile.open(archive) as z:z.extractall(WORK/'jre',filter='data')
    java=list((WORK/'jre').rglob('bin/java'))
    if len(java)!=1:raise ValueError('Missing Java executable')
    ENV['PATH']=str(java[0].parent)+os.pathsep+ENV['PATH']
    run([java[0],'-version'],60)
    bundle_name='experiments/csa_paper/valid_rerank_diagnostic/VALID_CANDIDATES.json.gz'
    run([sys.executable,'-u',WORK/'run_weight_calibration.py','--bundle',WORK/'VALID_CANDIDATES.json.gz','--parity',WORK/'CALIBRATION_PARITY.json.gz','--chair-cache',WORK/'chair.pkl','--output',OUTPUT],2400)
    complete=json.loads((OUTPUT/'result.json').read_text())['status']=='VALID_WEIGHT_CALIBRATION_COMPLETE'
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc());raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,training=False,decoding=False,test_used=False)))
    review=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUTPUT.iterdir():
            if p.is_file():z.write(p,p.name)
    print('SEND REVIEW:',review,flush=True)
    print('COMPLETE' if complete else 'INCOMPLETE; preserve outputs; no retraining.',flush=True)
'''
    cell=lambda kind,text:dict(cell_type=kind,metadata={},source=text,**(dict(execution_count=None,outputs=[]) if kind=='code' else {}))
    header='''# VALID reranking: tune weights, then check once

Accelerator **None**, Internet ON. No input Dataset, checkpoint, API token or GPU.
Saved CONTROL180/PROMPT60 five-caption candidates; no training, decoding or TEST.
Fixed shared 2000 TUNE / 3000 CHECK split; 66 shared D/CLIP/object weight rules.
Tune with CIDEr + nonworsening CHAIRs/i and recall guards. Lock choice before CHECK.
If CHECK fails guards/improvement, retain original .3D+.4C+.3O; no second search.
Historical CIDEr/CHAIR must replay. Subset CIDEr uses separate reference IDF.
CHECK is internal reused-VALID verification, not a fresh independent test.
One-hour CPU STOP cap, not ETA. No SPICE/CoreNLP parser model download.
Send `valid_weight_calibration_v1_review.zip`; preserve entire output on failure.
'''

    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=False)),cells=[cell('markdown',header),cell('code',setup),cell('code',execution)])
    for c in nb['cells']:
        if c['cell_type']=='code':compile(c['source'],'valid_oracle_cell','exec')
    p=HERE/'VALID_WEIGHT_CALIBRATION_CPU.ipynb';p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(p.name,p.stat().st_size,'bytes; pinned',revision)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);make(p.parse_args().revision)
