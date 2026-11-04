"""Author source-pinned baselineTEST with exact-version API downloads; no API run."""
import hashlib
import json
import subprocess
from make_smoke_notebook import HERE,ROOT,PREFIX,cell,RUNNER
from make_comparison_notebooks import FILES as BASE

FILES=list(dict.fromkeys(BASE+['plain_test_data.py','compare_plain_test.py','test_plain_test_data.py','make_plain_test_notebook.py',
    'PLAIN10_TEST_PROTOCOL.md','plain10_review_text/input_identity.json','plain10_review_text/checkpoint_receipt.json',
    'csa10_test_review_text/test_receipt.json','csa10_test_review_text/result.json','csa10_test_saved_text.json.gz','CSA10_TEST_TEXT_PACKAGE.json']))


def make(revision):
    commit=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+PREFIX+n],cwd=ROOT)).hexdigest() for n in FILES}
    setup=f'''
import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile
from pathlib import Path
SECRET_PLAIN='KAGGLE_API_TOKEN' # Enable this Secret; change to codex if that is your Secret name.
SECRET_TEST='KAGGLE_API_TOKEN' # May be a different Secret/token for private ducanh2403 output.
PLAIN_VERSION='ducanh3/notebook7bc83c4c76/45'
TEST_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/143'
OUTPUT=Path('/kaggle/working/plain_epoch10_karpathy_test')
COMMIT={commit!r}
HASHES={hashes!r}
if OUTPUT.exists():raise RuntimeError('Preserve existing output; use a fresh session/output path.')
OUTPUT.mkdir(parents=True);SOURCE=Path(tempfile.mkdtemp(prefix='plain10-test-'));LOG=OUTPUT/'notebook.log';TOKENS={{}}
def redact(line):
    for token in TOKENS.values():line=line.replace(token,'[REDACTED]')
    return line
for n,h in HASHES.items():
    raw=urllib.request.urlopen(f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{{COMMIT}}/{PREFIX}{{n}}',timeout=90).read()
    if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Changed source:'+n)
    p=SOURCE/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=COMMIT,sha256=HASHES),indent=2),encoding='utf-8')
'''
    runner=RUNNER.replace('def run(args, timeout=1800):','def run(args, timeout=1800, environment=None):')
    runner=runner.replace('cwd=SOURCE, stdout=', 'cwd=SOURCE, env=environment, stdout=')
    runner=runner.replace("else: print(line, end='', flush=True);", "else: line=redact(line); print(line, end='', flush=True);")
    runner=runner.replace('Smoke active;','PlainTEST active;').replace('Smoke wall cap exceeded','Phase wall cap exceeded')
    auth='''
from kaggle_secrets import UserSecretsClient
def run_cli(args,secret):
    if secret not in TOKENS:
        try:
            token=UserSecretsClient().get_secret(secret).strip()
            if not token:raise ValueError('Empty')
        except Exception:raise RuntimeError('Enable the configured API Secret in Add-ons > Secrets; do not paste token in code/chat.') from None
        TOKENS[secret]=token
    env=os.environ.copy()
    for key in ('KAGGLE_API_TOKEN','KAGGLE_USERNAME','KAGGLE_KEY'):env.pop(key,None)
    env['KAGGLE_API_TOKEN']=TOKENS[secret];env['PYTHONUNBUFFERED']='1'
    run(args,timeout=900,environment=env)
'''
    old=json.loads((HERE/'CSA_BEST9_GREEDY_BEAM5.ipynb').read_text(encoding='utf-8'))
    oldcode=next(c['source'] for c in old['cells'] if c['cell_type']=='code' and 'complete=False' in c['source'])
    deps=oldcode[oldcode.index("    run([sys.executable,'-m','pip'"):oldcode.index("    run([sys.executable,'-u','-m','torch.distributed.run'")]
    routing='''
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi()
for source in ('ducanh3/notebook7bc83c4c76/45','ducanh2403/2-vi-cai-kia-qua-nhieu/143'):
    owner,slug,version=source.split('/')
    with tempfile.TemporaryDirectory() as directory,patch.object(api,'build_kaggle_client') as factory:
        endpoint=factory.return_value.__enter__.return_value.kernels.kernels_api_client.list_kernel_session_output
        endpoint.return_value=SimpleNamespace(files=[],next_page_token=None,log=None)
        api.kernels_output(source,path=directory)
        request=endpoint.call_args.args[0]
        assert request.version_label=='v'+version and request.user_name==owner and request.kernel_slug==slug,'CLI ignored exact version'
print('Exact45/143 API routing mock preflight PASS; no network.',flush=True)
'''
    execute=r'''
complete=False;deadline=time.monotonic()+5400
try:
    if PLAIN_VERSION!='ducanh3/notebook7bc83c4c76/45' or TEST_VERSION!='ducanh2403/2-vi-cai-kia-qua-nhieu/143':raise ValueError('Do not change exact source versions')
    run([sys.executable,'-u',SOURCE/'test_comparison_setup.py'],timeout=120)
    run([sys.executable,'-u',SOURCE/'test_plain_test_data.py'],timeout=120)
    run([sys.executable,'-m','pip','install','--quiet','--upgrade','--force-reinstall','https://github.com/Kaggle/kaggle-cli/archive/6894864ff4095cf98e9e36d3a694ff5fc9ba3818.zip'],timeout=300)
    run([sys.executable,'-u','-c',ROUTING_PREFLIGHT],timeout=120)
    sys.path.insert(0,str(SOURCE))
    from plain_test_data import stage_inputs,exact_file,PLAIN_SHA
    plain=OUTPUT/'inputs_plain45';plain.mkdir(exist_ok=True);test=SOURCE/'inputs_test143';test.mkdir(exist_ok=True)
    try:exact_file(plain,'checkpoint.bin',PLAIN_SHA);plain_ready=True
    except ValueError:plain_ready=False
    if not plain_ready:
        run_cli([sys.executable,'-m','kaggle','kernels','output',PLAIN_VERSION,'-p',str(plain),'--file-pattern',r'(^|/)checkpoint[.]bin$'],SECRET_PLAIN)
    else:print('Reusing SHA-verified baseline checkpoint; no repeat download.',flush=True)
    expected=json.loads((SOURCE/'csa10_test_review_text/test_receipt.json').read_text(encoding='utf-8'))
    try:
        for n,h in expected['files'].items():exact_file(test,n,h)
        test_ready=True
    except ValueError:test_ready=False
    if not test_ready:
        run_cli([sys.executable,'-m','kaggle','kernels','output',TEST_VERSION,'-p',str(test),'--file-pattern',r'(^|/)(test_regions[.]npz|test_metadata[.]json|vocabulary[.]json)$'],SECRET_TEST)
    checkpoint=stage_inputs(plain,test,OUTPUT)
    print('Exact baseline45 and SAME TESTcache143 SHA-verified.',flush=True)
'''+deps+'''
    launch=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0']
    run([*launch,SOURCE/'run_csa10_test.py','worker','--arm','plain','--output',OUTPUT,'--checkpoint-root',checkpoint.parent],timeout=min(2460,deadline-time.monotonic()))
    run([sys.executable,'-u',SOURCE/'run_csa10_test.py','score','--arm','plain','--output',OUTPUT,'--chair',CHAIR],timeout=min(1200,deadline-time.monotonic()))
    run([sys.executable,'-u',SOURCE/'compare_plain_test.py','--output',OUTPUT],timeout=min(600,deadline-time.monotonic()))
    complete=json.loads((OUTPUT/'comparison.json').read_text(encoding='utf-8'))['status']=='PLAIN10_VS_CSA10_TEST_COMPLETE'
except BaseException:
    (OUTPUT/'failure.txt').write_text(redact(traceback.format_exc()),encoding='utf-8');raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,training=False,test_used=True,plain_version=PLAIN_VERSION,test_version=TEST_VERSION)),encoding='utf-8')
    archive=Path(str(OUTPUT)+'_review.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.txt','.log'):z.write(p,str(p.relative_to(OUTPUT)))
    print('SEND REVIEW:',archive,flush=True)
    print('KEEP ENTIRE OUTPUT:',OUTPUT,flush=True)
    print('COMPLETE' if complete else 'INCOMPLETE; preserve logs/cache; no automatic retraining.',flush=True)
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),kaggle=dict(isGpuEnabled=True,isInternetEnabled=True)),cells=[
        cell('markdown','''# Plain epoch10 TEST5000 — same cache as CSA10
T4x2, InternetON, noAddInput. Enable APISecret(s) underAdd-ons>Secrets. SECRET_PLAIN needsreadaccess to ducanh3/notebook7bc83c4c76/45; SECRET_TEST to ducanh2403/2-vi-cai-kia-qua-nhieu/143. Defaultboth KAGGLE_API_TOKEN; change to codex/yourSecretname or use twoSecrets forprivateaccounts. Tokensneverbelongincode/chat. If403, grantlegitimateaccess/share source orusecorrespondingaccounttoken; no versionfallback.
Downloads exactbaselinecheckpoint45 and3TESTcache143payloads. SHAverified, noDatasetcreation/upload required. No train/newfeatureextraction or originalimages. Exactplain10/177110updates/beam5/max20, SAME5000arrays/refs/vocab asCSA. TwoindependentGPUworkers, separateteacherNLL; nooptimizer.
ScoresCIDEr/BLEU4/ROUGE-L/CHAIRs/i/recall/counts/length/NLL andcompareswithlockedCSA10TEST. SavedCSAtextmetricsmustreplay beforeCOMPLETE. Differentparamcapacity/one seedlimits remain. This comparison doesnotestablishnoveltyofCSA.
90minelapsedstopcap, notETA/billing. Send plain_epoch10_karpathy_test_review.zip; preservewholeoutputcache/checkpoint. Localagentchecksaremetadata/NumPy/static only; userCUDA/API/scorerspending.
'''),cell('code',setup),cell('code',runner),cell('code',auth+'\nROUTING_PREFLIGHT='+repr(routing)),cell('code',execute)])
    out=HERE/'PLAIN_EPOCH10_KARPATHY_TEST_2GPU.ipynb';out.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8');print(out.name,out.stat().st_size,'bytes; unexecuted')
    recovery=f'''
# First enable an APISecret that can read ducanh2403 output143 and set SECRET_TEST to its name.
# CURRENT failed API-download session only. This cell continues the corrected execution automatically.
import hashlib,json,sys,urllib.request
from pathlib import Path
if not all(n in globals() for n in ('SOURCE','OUTPUT','run','run_cli','SECRET_TEST','TOKENS')):raise RuntimeError('Use the corrected full notebook; current-session globals missing.')
SOURCE=Path(SOURCE);OUTPUT=Path(OUTPUT)
if (OUTPUT/'result.json').exists() or list(OUTPUT.glob('test_worker*.json')):raise RuntimeError('Inference progress exists; preserve it rather than applying first-download recovery.')
COMMIT={commit!r}
HASHES={hashes!r}
for n,h in HASHES.items():
    raw=urllib.request.urlopen(f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{{COMMIT}}/{PREFIX}{{n}}',timeout=90).read()
    if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Changed source:'+n)
    p=SOURCE/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
sys.path.insert(0,str(SOURCE));sys.modules.pop('plain_test_data',None)
TOKENS.pop(SECRET_TEST,None) # Refresh a replaced Secret value; never print it.
for n in ('failure.txt','run_status.json','source_receipt.json'):
    p=OUTPUT/n;previous=OUTPUT/('before_download_fix_'+n)
    if p.exists() and not previous.exists():p.rename(previous)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=COMMIT,sha256=HASHES),indent=2),encoding='utf-8')
EXECUTION_CODE={execute!r}
exec(compile(EXECUTION_CODE,'plain_test_download_recovery','exec'))
'''
    cell('code',recovery)
    (HERE/'RECOVER_PLAIN_TEST_DOWNLOAD_CELL.py').write_text(recovery.strip()+'\n',encoding='utf-8')


if __name__=='__main__':
    import sys
    make(sys.argv[1] if len(sys.argv)>1 else 'HEAD')
