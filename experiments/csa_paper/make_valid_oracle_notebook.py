"""Small CPU-only notebook. Downloads saved VALID text, never a checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def make(revision):
    revision=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    files=['experiments/csa_paper/score_valid_candidate_oracle.py',
           'experiments/csa_paper/valid_rerank_diagnostic/VALID_CANDIDATES.json.gz']
    digests={n:hashlib.sha256(subprocess.check_output(['git','show',revision+':'+n],cwd=ROOT)).hexdigest() for n in files}
    setup='import os,sys,json,time,hashlib,urllib.request,subprocess,threading,queue,signal,tempfile,zipfile,traceback,tarfile\nfrom pathlib import Path\n'
    setup+=f'REVISION={revision!r}\nFILES={digests!r}\n'
    setup+='''OUTPUT=Path('/kaggle/working/valid_candidate_oracle_v1')
if OUTPUT.exists():raise RuntimeError('Preserve old output; use a fresh output/runtime.')
OUTPUT.mkdir(parents=True)
WORK=Path(tempfile.mkdtemp(prefix='valid-oracle-',dir='/tmp'));DEPS=WORK/'deps';LOG=OUTPUT/'notebook.log'
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
    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS,'pycocoevalcap==1.2'],600)
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
    run([sys.executable,'-u',WORK/'score_valid_candidate_oracle.py','--bundle',WORK/'VALID_CANDIDATES.json.gz','--sha256',FILES[bundle_name],'--output',OUTPUT],2400)
    complete=json.loads((OUTPUT/'result.json').read_text())['status']=='VALID_CANDIDATE_ORACLE_COMPLETE'
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc());raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,training=False,decoding=False,test_used=False,selector_changed=False)))
    review=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUTPUT.iterdir():
            if p.is_file():z.write(p,p.name)
    print('SEND REVIEW:',review,flush=True)
    print('COMPLETE' if complete else 'INCOMPLETE; preserve outputs; no retraining.',flush=True)
'''
    cell=lambda kind,text:dict(cell_type=kind,metadata={},source=text,**(dict(execution_count=None,outputs=[]) if kind=='code' else {}))
    header='''# VALID: how much CIDEr is missed by reranking?

Accelerator **None**, Internet ON. No input Dataset, checkpoint, API token or GPU needed.
One notebook covers CONTROL180 and PROMPT60 END candidates: same5000VALID images; no TEST.
Reuses all saved D/C/O scores and captions; computes CIDEr of each of5ranks with same reference corpus.
Historical rank0 and selected per-image CIDEr must replay before reporting oracle.
Oracle uses references: diagnostic upper bound, not an inference method or visual truth.
One-hour CPU STOPcap, not ETA; Java8/PTB only, no SPICE/CoreNLP parser downloads.
Send `valid_candidate_oracle_v1_review.zip`. No training, decoding or selector weight changes.
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=False)),cells=[cell('markdown',header),cell('code',setup),cell('code',execution)])
    for c in nb['cells']:
        if c['cell_type']=='code':compile(c['source'],'valid_oracle_cell','exec')
    p=HERE/'VALID_CANDIDATE_ORACLE_CPU.ipynb';p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(p.name,p.stat().st_size,'bytes; pinned',revision)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);make(p.parse_args().revision)
