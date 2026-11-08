"""Generate two small own-account CPU notebooks; do not execute Kaggle."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def cell(kind, code):
    c = dict(cell_type=kind, metadata={}, source=code)
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    return c


def make(revision):
    revision = subprocess.check_output(['git', 'rev-parse', revision], cwd=ROOT, text=True).strip()
    files = ['score_prompt_test_extra.py', 'score_spice_subsets.py', 'test_spice_subsets_data.py', 'compare_spice_subsets.py', 'PROMPT_TEST_EXTRA_INPUTS.json', 'PROMPT_TEST_EXTRA_RESULT.json']
    payload = {n: subprocess.check_output(['git', 'show', revision+':experiments/csa_paper/'+n], cwd=ROOT) for n in files}
    hashes = {n: hashlib.sha256(raw).hexdigest() for n, raw in payload.items()}
    manifest = json.loads(payload['PROMPT_TEST_EXTRA_INPUTS.json'])
    for arm in ('CONTROL', 'PROMPT'):
        header = f'''# TEST5000: SPICE Objects / Relations / Attributes — {arm}
**CPU / Accelerator None**, Internet ON. No input Dataset, checkpoint or JPEG required.
Enable your own Kaggle API secret named `codex`. Downloads only caption/reference/metadata files from **{manifest['arms'][arm]['version']}**, never latest or the other account.
If exact output is already attached, set `SOURCE_ROOT` to that folder and no API secret is needed.

Scores only SPICE on normal + fixed_rerank saved captions. Replays All and preserves per-image Objects/Relations/Attributes. Undefined values excluded with denominators; no new model quality claim. SPICE All must match; no training/decoding/reranking. Fixed Java8 and COCO1.2; CoreNLP3.6 models downloaded with progress and recorded SHA. All CPU scorers need to pass the smoke check first.
Four-hour CPU STOPcap (not ETA); heartbeat30s, logs and review preserved on failure. Actual Java/scorers not run by author. Send the review ZIP; do not rerun GPU inference if a metric fails.
'''
        setup = 'import os,sys,json,hashlib,urllib.request,subprocess,tempfile,time,threading,queue,signal,zipfile,traceback,shutil\nfrom pathlib import Path\n'
        setup += f'ARM={arm!r}\nKERNEL_VERSION={manifest["arms"][arm]["version"]!r}\nREVISION={revision!r}\nHASHES={hashes!r}\n'
        setup += '''SECRET_NAME='codex'
SOURCE_ROOT='' # Optional mounted exact output; blank downloads own version.
OUTPUT=Path('/kaggle/working/csa_prompt_TEST_spice_subsets_'+ARM+'_v1')
if OUTPUT.exists():raise RuntimeError('Preserve previous output; use a fresh session or output path.')
if shutil.disk_usage('/kaggle/working').free < 3*1024**3:raise RuntimeError('Need 3GiB free for Java/CoreNLP and metric caches')
OUTPUT.mkdir(parents=True)
SOURCE=Path(tempfile.mkdtemp(prefix='prompt-test-extra-',dir='/tmp'))
DEPS=SOURCE/'deps';LOG=OUTPUT/'notebook.log';DEADLINE=time.monotonic()+4*3600
for name,digest in HASHES.items():
    raw=urllib.request.urlopen('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+REVISION+'/experiments/csa_paper/'+name,timeout=90).read()
    if hashlib.sha256(raw).hexdigest()!=digest:raise ValueError('Changed source:'+name)
    (SOURCE/name).write_bytes(raw)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=REVISION,sha256=HASHES),indent=2),encoding='utf-8')
ENV=os.environ.copy()
for key in ('KAGGLE_API_TOKEN','KAGGLE_USERNAME','KAGGLE_KEY'):ENV.pop(key,None)
ENV.update(PYTHONPATH=str(DEPS)+os.pathsep+str(SOURCE),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')

def run(args,timeout=600,environment=None):
    remaining=DEADLINE-time.monotonic()
    if remaining<=0:raise TimeoutError('Parent CPU cap exceeded')
    timeout=min(timeout,remaining);start=time.monotonic();heartbeat=start;q=queue.Queue()
    args=list(map(str,args));print('Running:',' '.join(args),flush=True)
    process=subprocess.Popen(args,env=ENV if environment is None else environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
    def reader():
        for line in process.stdout:q.put(line)
        q.put(None)
    thread=threading.Thread(target=reader,daemon=True);thread.start()
    try:
        with LOG.open('a',encoding='utf-8') as log:
            ended=False
            while not ended:
                try:
                    line=q.get(timeout=1)
                    if line is None:ended=True
                    else:print(line,end='',flush=True);log.write(line);log.flush()
                except queue.Empty:pass
                now=time.monotonic()
                if now-start>timeout:raise TimeoutError('CPU metric/phase cap exceeded; saved logs remain.')
                if now-heartbeat>=30:
                    print('CPU metrics active;',round(now-start),'s; no model inference.',flush=True);heartbeat=now
            if process.wait():raise subprocess.CalledProcessError(process.returncode,args)
    finally:
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
'''
        execution = '''complete=False
try:
    run([sys.executable,'-u',SOURCE/'test_spice_subsets_data.py'],120)
    if not SOURCE_ROOT:
        run([sys.executable,'-m','pip','install','-q','--target',DEPS,'kaggle==2.2.5'],600)
        from kaggle_secrets import UserSecretsClient
        try:token=UserSecretsClient().get_secret(SECRET_NAME).strip()
        except Exception:raise RuntimeError('Enable your own API secret named '+SECRET_NAME) from None
        if not token:raise RuntimeError('Empty Kaggle API secret')
        api_env={**ENV,'KAGGLE_API_TOKEN':token}
        try:
            route="from types import SimpleNamespace;from unittest.mock import patch;from kaggle.api.kaggle_api_extended import KaggleApi;import tempfile;api=KaggleApi();source="+repr(KERNEL_VERSION)+";owner,slug,version=source.split('/');ctx=tempfile.TemporaryDirectory();root=ctx.__enter__();p=patch.object(api,'build_kaggle_client');f=p.__enter__();ep=f.return_value.__enter__.return_value.kernels.kernels_api_client.list_kernel_session_output;ep.return_value=SimpleNamespace(files=[],next_page_token=None,log=None);api.kernels_output(source,path=root);r=ep.call_args.args[0];assert (r.user_name,r.kernel_slug,r.version_label)==(owner,slug,'v'+version);p.__exit__(None,None,None);ctx.__exit__(None,None,None);print('Exact own-version API route PASS')"
            run([sys.executable,'-c',route],60,api_env)
            downloaded=OUTPUT/'saved_inputs';downloaded.mkdir()
            pattern=r'(^|/)(test_references[.]json|test_lock[.]json|run_status[.]json|result[.]json|normal_captions[.]json|fixed_rerank_captions[.]json)$'
            run([sys.executable,'-c','from kaggle.cli import main;main()','kernels','output',KERNEL_VERSION,'-p',downloaded,'--file-pattern',pattern],900,api_env)
            SOURCE_ROOT=str(downloaded)
        finally:
            api_env.clear();del token
    sys.path.insert(0,str(SOURCE))
    from score_prompt_test_extra import verify,read
    verify(SOURCE_ROOT,ARM,read(SOURCE/'PROMPT_TEST_EXTRA_INPUTS.json'))
    print('Exact caption/reference/status SHA and full TEST5000 PASS',flush=True)
    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS,'pycocoevalcap==1.2'],600)
    launch=[sys.executable,'-u',SOURCE/'score_spice_subsets.py']
    flags=['--arm',ARM,'--input',SOURCE_ROOT,'--output',OUTPUT]
    for mode,cap in [('environment',2400),('smoke',900),('tokenize',600),('spice',9000)]:
        print('PHASE:',mode,flush=True)
        run([*launch,mode,*flags],cap)
    complete=read(OUTPUT/'result.json')['status']=='SPICE_SUBSETS_COMPLETE'
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,arm=ARM,training=False,decoding=False,reranking=False)),encoding='utf-8')
    REVIEW=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(REVIEW,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUTPUT.rglob('*'):
            if p.is_file() and 'assets' not in p.relative_to(OUTPUT).parts and p.suffix in ('.json','.csv','.txt','.log'):
                z.write(p,str(p.relative_to(OUTPUT)))
    print('SEND REVIEW:',REVIEW,'\\nKEEP ENTIRE OUTPUT:',OUTPUT,flush=True)
    print('SPICE SUBSETS COMPLETE' if complete else 'INCOMPLETE; preserve inputs/cache/log; no automatic retries or model runs.',flush=True)
'''
        nb = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
            kaggle=dict(isInternetEnabled=True,isGpuEnabled=False)), cells=[cell('markdown',header),cell('code',setup),cell('code',execution)])
        path = HERE/f'CSA_PROMPT_SPICE_SUBSETS_{arm}_CPU.ipynb'
        path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
        for c in nb['cells']:
            if c['cell_type']=='code':
                compile(c['source'], str(path), 'exec')
        print(path.name, path.stat().st_size, 'bytes; CPU; pinned', revision)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--revision',required=True)
    make(p.parse_args().revision)
