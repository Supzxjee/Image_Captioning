"""Author baseline's own TEST workflow; no foreign-notebook cache download."""
import hashlib
import json
import subprocess
from make_smoke_notebook import HERE,ROOT,PREFIX,cell,RUNNER
from make_plain_test_notebook import FILES as BASE

FILES=list(dict.fromkeys(BASE+['make_plain_self_test_notebook.py','PLAIN_SELF_TEST_PROTOCOL.md']))


def make(revision):
    commit=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+PREFIX+n],cwd=ROOT)).hexdigest() for n in FILES}
    setup=f'''
import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile
from pathlib import Path
COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
DETECTIONS=Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
IMAGE_ROOT=COCO_JSON.parent/'images'
SECRET_PLAIN='KAGGLE_API_TOKEN' # Only needed if own checkpoint45 is not already under /kaggle/working.
PLAIN_VERSION='ducanh3/notebook7bc83c4c76/45'
OUTPUT=Path('/kaggle/working/plain_epoch10_karpathy_test_self')
COMMIT={commit!r}
HASHES={hashes!r}
if OUTPUT.exists():raise RuntimeError('Preserve existing output; use a fresh session/path.')
if not COCO_JSON.is_file() or not DETECTIONS.is_file() or not VISUAL_CACHE.is_dir():raise FileNotFoundError('Add mscoco-2014, visual-cache and objectdetectionecache Datasets first.')
OUTPUT.mkdir(parents=True);SOURCE=Path(tempfile.mkdtemp(prefix='plain-self-test-'));LOG=OUTPUT/'notebook.log';TOKENS={{}}
def redact(line):
    for token in TOKENS.values():line=line.replace(token,'[REDACTED]')
    return line
for n,h in HASHES.items():
    raw=urllib.request.urlopen(f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{{COMMIT}}/{PREFIX}{{n}}',timeout=90).read()
    if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Changed source:'+n)
    p=SOURCE/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=COMMIT,sha256=HASHES),indent=2),encoding='utf-8')
'''
    old=json.loads((HERE/'PLAIN_EPOCH10_KARPATHY_TEST_2GPU.ipynb').read_text(encoding='utf-8'))
    auth=old['cells'][3]['source'].split('\nROUTING_PREFLIGHT=')[0]
    runner=RUNNER.replace('def run(args, timeout=1800):','def run(args, timeout=1800, environment=None):').replace('cwd=SOURCE, stdout=','cwd=SOURCE, env=environment, stdout=')
    runner=runner.replace("else: print(line, end='', flush=True);","else: line=redact(line); print(line, end='', flush=True);").replace('Smoke active;','Plain selfTEST active;').replace('Smoke wall cap exceeded','Phase cap exceeded')
    raw_old=json.loads((HERE/'CSA_BEST9_GREEDY_BEAM5.ipynb').read_text(encoding='utf-8'))
    oldcode=next(c['source'] for c in raw_old['cells'] if c['cell_type']=='code' and 'complete=False' in c['source'])
    deps=oldcode[oldcode.index("    run([sys.executable,'-m','pip'"):oldcode.index("    run([sys.executable,'-u','-m','torch.distributed.run'")]
    route='''
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi()
with tempfile.TemporaryDirectory() as directory,patch.object(api,'build_kaggle_client') as factory:
    endpoint=factory.return_value.__enter__.return_value.kernels.kernels_api_client.list_kernel_session_output
    endpoint.return_value=SimpleNamespace(files=[],next_page_token=None,log=None)
    api.kernels_output('ducanh3/notebook7bc83c4c76/45',path=directory)
    request=endpoint.call_args.args[0]
    assert request.version_label=='v45' and request.user_name=='ducanh3' and request.kernel_slug=='notebook7bc83c4c76'
print('Own45 API routing mock PASS; no foreign notebook API.',flush=True)
'''
    execute='''
complete=False;deadline=time.monotonic()+7200
try:
    sys.path.insert(0,str(SOURCE))
    from plain_test_data import exact_file,PLAIN_SHA,check_test_equivalence
    run([sys.executable,'-u',SOURCE/'test_comparison_setup.py'],timeout=120)
    run([sys.executable,'-u',SOURCE/'test_plain_test_data.py'],timeout=120)
    try:checkpoint=exact_file('/kaggle/working','checkpoint.bin',PLAIN_SHA)
    except ValueError:checkpoint=None
    if checkpoint is None:
        if PLAIN_VERSION!='ducanh3/notebook7bc83c4c76/45':raise ValueError('Own checkpoint must be exact45')
        run([sys.executable,'-m','pip','install','--quiet','--upgrade','--force-reinstall','https://github.com/Kaggle/kaggle-cli/archive/6894864ff4095cf98e9e36d3a694ff5fc9ba3818.zip'],timeout=300)
        run([sys.executable,'-u','-c',OWN_ROUTING_PREFLIGHT],timeout=120)
        own=OUTPUT/'inputs_plain45';own.mkdir()
        run_cli([sys.executable,'-m','kaggle','kernels','output',PLAIN_VERSION,'-p',str(own),'--file-pattern',r'(^|/)checkpoint[.]bin$'],SECRET_PLAIN)
        checkpoint=exact_file(own,'checkpoint.bin',PLAIN_SHA)
    print('Using exact own baseline checkpoint:',checkpoint,flush=True)
    (OUTPUT/'test_lock.json').write_text(json.dumps(dict(epoch=10,updates=177110,checkpoint_sha256=PLAIN_SHA,checkpoint=str(checkpoint),beam=5,max_new_tokens=20,selection='exact10 prelocked',foreign_cache_api=False)),encoding='utf-8')
'''+deps+'''
    CONFIG=SOURCE/'test_config.json'
    CONFIG.write_text(json.dumps(dict(dataset_json=str(COCO_JSON),detections=str(DETECTIONS),visual_cache=str(VISUAL_CACHE),image_root=str(IMAGE_ROOT),output=str(OUTPUT))),encoding='utf-8')
    run([sys.executable,'-u',SOURCE/'prepare_test_regions.py','--config',CONFIG],timeout=min(3720,deadline-time.monotonic()))
    check_test_equivalence(json.loads((OUTPUT/'test_receipt.json').read_text(encoding='utf-8')))
    print('Own TEST cache matches CSA TEST NPZ/metadata/vocab bytes and profile exactly.',flush=True)
    launch=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0']
    run([*launch,SOURCE/'run_csa10_test.py','worker','--arm','plain','--output',OUTPUT,'--checkpoint-root',checkpoint.parent],timeout=min(2460,deadline-time.monotonic()))
    run([sys.executable,'-u',SOURCE/'run_csa10_test.py','score','--arm','plain','--output',OUTPUT,'--chair',CHAIR],timeout=min(1200,deadline-time.monotonic()))
    run([sys.executable,'-u',SOURCE/'compare_plain_test.py','--output',OUTPUT],timeout=min(600,deadline-time.monotonic()))
    complete=json.loads((OUTPUT/'comparison.json').read_text(encoding='utf-8'))['status']=='PLAIN10_VS_CSA10_TEST_COMPLETE'
except BaseException:
    (OUTPUT/'failure.txt').write_text(redact(traceback.format_exc()),encoding='utf-8');raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,training=False,test_used=True,foreign_cache_api=False)),encoding='utf-8')
    archive=Path(str(OUTPUT)+'_review.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.txt','.log'):z.write(p,str(p.relative_to(OUTPUT)))
    print('SEND REVIEW:',archive,flush=True)
    print('KEEP ENTIRE OUTPUT:',OUTPUT,flush=True)
    print('COMPLETE' if complete else 'INCOMPLETE; preserve logs/progress, no automatic retraining.',flush=True)
'''
    auth+='\nOWN_ROUTING_PREFLIGHT='+repr(route)
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),kaggle=dict(isGpuEnabled=True,isInternetEnabled=True)),cells=[
        cell('markdown','''# Baseline notebook evaluates its own epoch10 TEST
T4x2, InternetON. Add3rawDatasets: mscoco-2014(images+datasetJSON), visual-cache(originalCLIPH5), objectdetectionecache. Paths are the same as CSA's rawpreparation. No API toCSA notebook/cache143 andnoSECRET_TEST.
Reuses own exactbaseline45checkpoint a75d7bad alreadyunder/kaggle/working (includingthepreviousAPI download). Ifnotpresent, downloadONLYown ducanh3/notebook7bc83c4c76/45 usingSECRET_PLAIN; enablethematchingownAPISecret then. Nevertrainagain.
Regenerate5000TESTROI usingSAMErawcache/detector/pooling/vocab asCSA; requireNPZ/metadata/vocabbyteSHA+profile identities match beforeinference. Elapsedpreparationseconds maydiffer. Stoponanydata mismatch, notsilentlysubstitute.
Fixedplain10/beam5/max20; independentT4workers andsameofficialscorers, savedCSAtext metricreplay beforecomparison.2hstopcap notETA. No train/checkpointtuning. SaveVersion/RunAll, sendplain_epoch10_karpathy_test_self_review.zip andpreservewholeoutput.
Localagentmodels/scorers/API notexecuted; userruntimepending. RawTESTpoolingpreviously≈23min, plusinference/metrics; notguaranteedruntime.
'''),cell('code',setup),cell('code',runner),cell('code',auth),cell('code',execute)])
    p=HERE/'PLAIN_EPOCH10_SELF_TEST_2GPU.ipynb';p.write_text(json.dumps(nb,indent=1,ensure_ascii=False)+'\n',encoding='utf-8');print(p.name,p.stat().st_size,'bytes, unexecuted')
    recovery='''# Add3rawDatasets first, then execute thiscell in thecurrentfailedAPI session.
# Uses a NEW outputfolder and finds the alreadydownloadedbaselinecheckpoint under /kaggle/working.
'''+setup+'\n'+runner+'\n'+auth+'\n'+execute
    cell('code',recovery)
    (HERE/'RECOVER_PLAIN_SELF_TEST_CELL.py').write_text(recovery,encoding='utf-8')


if __name__=='__main__':
    import sys
    make(sys.argv[1] if len(sys.argv)>1 else 'HEAD')
