"""Author CPU cache+fixture notebook pinned to exact mapping output167."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from make_smoke_notebook import cell,RUNNER
from make_prompt_adapter_notebook import FILES as AUDIT_FILES,EXTERNAL

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PREFIX='experiments/csa_paper/'


def make(revision):
    revision=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    files=AUDIT_FILES+['prompt_text_cache.py','test_prompt_text_cache_data.py','make_prompt_text_cache_notebook.py','make_smoke_notebook.py','RERANK_TEST_INPUTS.json']
    mapping={PREFIX+n:n for n in files};mapping.update({n:n for n in EXTERNAL})
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',revision+':'+n],cwd=ROOT)).hexdigest() for n in mapping}
    setup='''
import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile,shutil
from pathlib import Path
LOCAL_SOURCE_ROOT='' # Optional folder/mount containing the full exact mapping167 output.
SECRET_NAME='codex' # Enabled Kaggle API Secret, used only if LOCAL_SOURCE_ROOT is empty.
OUTPUT=Path('/kaggle/working/csa_prompt_text_cache_v1')
if OUTPUT.exists():raise RuntimeError('Preserve existing output; use a fresh runtime/output path.')
SOURCE=Path(tempfile.mkdtemp(prefix='csa-prompt-text-',dir='/tmp'))
LOG=SOURCE/'notebook.log'
'''
    setup+='\nREVISION='+repr(revision)+'\nMAPPING='+repr(mapping)+'\nHASHES='+repr(hashes)+'\n'
    setup+='''
print('CPU-only frozenCLIP text cache + neural fixture smoke. No caption training or TEST.',flush=True)
for original,destination in MAPPING.items():
    blob=urllib.request.urlopen('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+REVISION+'/'+original,timeout=90).read()
    if hashlib.sha256(blob).hexdigest()!=HASHES[original]:raise RuntimeError('Changed source:'+original)
    target=SOURCE/destination;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(blob)
(SOURCE/'source_receipt.json').write_text(json.dumps(dict(commit=REVISION,sha256=HASHES),indent=2),encoding='utf-8')
os.environ['PYTHONPATH']=str(SOURCE)
'''
    runner=RUNNER.replace('def run(args, timeout=1800):','def run(args, timeout=1800, environment=None):').replace('cwd=SOURCE, stdout=','cwd=SOURCE, env=environment, stdout=')
    runner=runner.replace('Smoke active;','Prompt text CPU active;').replace('Smoke wall cap exceeded','Prompt text CPU phase cap exceeded')
    final=r'''
complete=False
try:
    run([sys.executable,'-u',SOURCE/'test_prompt_adapter_data.py'],timeout=120)
    run([sys.executable,'-u',SOURCE/'test_prompt_text_cache_data.py'],timeout=120)
    if LOCAL_SOURCE_ROOT:
        SOURCE_ROOT=Path(LOCAL_SOURCE_ROOT)
    else:
        from kaggle_secrets import UserSecretsClient
        try:token=UserSecretsClient().get_secret(SECRET_NAME).strip()
        except Exception:raise RuntimeError('Enable the named API Secret for your own account to read exact version167, or set LOCAL_SOURCE_ROOT to the full saved167 folder.') from None
        if not token:raise RuntimeError('Empty API Secret')
        run([sys.executable,'-m','pip','install','-q','kaggle==2.2.5'],timeout=300)
        env=os.environ.copy()
        for k in ('KAGGLE_API_TOKEN','KAGGLE_USERNAME','KAGGLE_KEY'):env.pop(k,None)
        env['KAGGLE_API_TOKEN']=token
        SOURCE_ROOT=SOURCE/'download167';SOURCE_ROOT.mkdir()
        cli=[sys.executable,'-c','from kaggle.cli import main; main()','kernels','output','ducanh2403/2-vi-cai-kia-qua-nhieu/167',
             '-p',str(SOURCE_ROOT),'--file-pattern',r'(^|/)(receipt.json|summary.json|cohort.json|prompt_links.jsonl|prompt_texts.json)$']
        run(cli,timeout=900,environment=env)
        del token,env
    sys.path.insert(0,str(SOURCE))
    from prompt_text_cache import checked_source,validate_links
    VERIFIED_ROOT,_=checked_source(SOURCE_ROOT)
    _,validation=validate_links(VERIFIED_ROOT)
    print('Exact167 payload SHA and full15000links/318texts PASS',validation,flush=True)
    run([sys.executable,'-u',SOURCE/'test_prompt_adapter_model.py'],timeout=180)
    DEPS=SOURCE/'hf-deps'
    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS,
         'transformers==4.57.1','tokenizers==0.22.1','huggingface-hub==0.36.2','safetensors==0.6.2','ftfy==6.3.1'],timeout=300)
    os.environ['PYTHONPATH']=str(DEPS)+os.pathsep+str(SOURCE)
    run([sys.executable,'-c',"from importlib.metadata import version;expected={'transformers':'4.57.1','tokenizers':'0.22.1','huggingface-hub':'0.36.2','safetensors':'0.6.2'};assert all(version(k)==v for k,v in expected.items());print('Isolated HF dependency versions PASS')"],timeout=60)
    CONFIG=SOURCE/'cache_config.json'
    CONFIG.write_text(json.dumps(dict(source_root=str(VERIFIED_ROOT),output=str(OUTPUT),hf_cache=str(SOURCE/'hf-cache'))),encoding='utf-8')
    run([sys.executable,'-u',SOURCE/'prompt_text_cache.py','--config',CONFIG],timeout=1800)
    complete=json.loads((OUTPUT/'cache_receipt.json').read_text())['status']=='PROMPT_TEXT_CACHE_COMPLETE'
finally:
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if LOG.exists():shutil.copy2(LOG,OUTPUT/'notebook.log')
    shutil.copy2(SOURCE/'source_receipt.json',OUTPUT/'source_receipt.json')
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,training=False,test_used=False,
        neural_fixture_passed=complete,actual_checkpoint161_smoke=False)),encoding='utf-8')
    REVIEW=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(REVIEW,'w',zipfile.ZIP_DEFLATED) as z:
        for n in ('cache_manifest.json','cache_receipt.json','run_status.json','source_receipt.json','notebook.log'):
            if (OUTPUT/n).is_file():z.write(OUTPUT/n,n)
    print('SEND REVIEW:',REVIEW,'\nKEEP ENTIRE OUTPUT:',OUTPUT,flush=True)
    print('CPU prompt cache complete; no caption model trained.' if complete else 'INCOMPLETE; preserve log/output; do not use partial cache.',flush=True)
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        kaggle=dict(isInternetEnabled=True,isGpuEnabled=False)),cells=[
        cell('markdown','# Cache318prompt CLIP — exact mapping167\n**Accelerator None, Internet ON.** CPU frozenCLIP inference and tiny neural fixture tests are user-run; no caption training/TEST. No AddInput needed if using your enabled API Secret `codex` to fetch exact167. Alternative: set LOCAL_SOURCE_ROOT to full mapping167 output; no API needed then. Verifies all5payload hashes,15000links/318texts before modeldownload. Uses pinned CLIP bytes/revision, FP32 normalized512vectors. No Torch reinstall. Preserve whole output for matched training later; send small reviewZIP. Tiny fixtures do not validate actualcheckpoint161 or CUDA/DDP.'),
        cell('code',setup),cell('code',runner),cell('code',final)])
    p=HERE/'CSA_PROMPT_TEXT_CACHE_CPU.ipynb';p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(p.name,p.stat().st_size,'bytes; source',revision)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);make(p.parse_args().revision)
