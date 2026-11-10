"""Author fixed199 TEST notebook; never submit/execute inference."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PREFIX='experiments/clip_patch_baseline/'
sys.path.insert(0,str(ROOT/'experiments/csa_paper'))
from make_smoke_notebook import RUNNER,cell

def make(revision,semantic=False):
    revision=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    mapping={PREFIX+n:n for n in ('TEST199_LOCK.json','TEST199_PROTOCOL.md','TEST199_BUILD_V2_PROTOCOL.md','SMOKE_SOURCE.json','test199_data.py','run_test199.py','test_test199.py','patch_model.py','patch_beam.py','patch_data.py','full_data.py','make_test199_notebook.py')}
    mapping.update({'experiments/csa_paper/'+n:n for n in ('data.py','real_bank_data.py','cuda_runtime_guard.py','saved_quality_data.py','score_prompt_test_extra.py')})
    if semantic:
        mapping.update({PREFIX+n:n for n in ('semantic_test_data.py','semantic_data.py','semantic_patch_model.py','test_semantic_test.py','BANK_PROMPT_TEST_PROTOCOL.md')})
        mapping.update({'experiments/csa_paper/model.py':'bank_layers.py','experiments/csa_paper/PROMPT_TEST_PREPARATION_RESULT.json':'PROMPT_TEST_PREPARATION_RESULT.json'})
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',revision+':'+n],cwd=ROOT)).hexdigest() for n in mapping}
    setup=f"""import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile,shutil
from pathlib import Path
REVISION={revision!r}
MAPPING={mapping!r}
HASHES={hashes!r}
"""+"""VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
CHECKPOINT_ROOT='' # Optional exact199 mounted file/root. Blank downloads ONLY own199/best.bin.
SECRET_NAMES=('codex','KAGGLE_API_TOKEN') # Enable the matching Kaggle Secret; token is never logged.
WALL_HOURS=2 # STOP cap including CPU metrics, not ETA or billing.
OUTPUT=Path('/kaggle/working/clip_patch_199_test5000_v2')
for p in (VISUAL_CACHE,COCO_JSON):
    if not p.exists():raise FileNotFoundError('Attach/correct original inputs: '+str(p))
if OUTPUT.exists():raise RuntimeError('Preserve existing captions/output; use fresh session/output path')
if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<6*1024**3:raise RuntimeError('Need6GiB free for TEST patches/checkpoint/CPU scorer assets')
OUTPUT.mkdir(parents=True);SOURCE=Path(tempfile.mkdtemp(prefix='clip-patch199-test-',dir='/tmp'));DEPS=SOURCE/'deps';CLI=SOURCE/'cli';CHAIR=SOURCE/'chair';LOG=OUTPUT/'notebook.log'
PARENT_DEADLINE=time.monotonic()+WALL_HOURS*3600;TOKENS=[]
os.environ['PYTHONPATH']=str(SOURCE);os.environ['OMP_NUM_THREADS']='1'
CONFIG=SOURCE/'config.json'
cfg=dict(output=str(OUTPUT),stage=str(SOURCE/'test_patches'),dataset_json=str(COCO_JSON),visual_cache=str(VISUAL_CACHE),checkpoint_root=CHECKPOINT_ROOT,chair=str(CHAIR),worker_seconds=1800)
def write_config():
    CONFIG.write_text(json.dumps(cfg,indent=2));(OUTPUT/'protocol.json').write_text(json.dumps(cfg,indent=2))
write_config()
def redact(line):
    for token in TOKENS:line=line.replace(token,'[REDACTED]')
    return line
def download(url,expected,path):
    remaining=PARENT_DEADLINE-time.monotonic()
    if remaining<=0:raise TimeoutError('Parent deadline before download')
    raw=urllib.request.urlopen(url,timeout=min(120,remaining)).read()
    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Changed pinned input: '+path.name)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
"""
    runner=RUNNER.replace('def run(args, timeout=1800):','def run(args, timeout=1800, environment=None):').replace('cwd=SOURCE, stdout=','cwd=SOURCE, env=environment, stdout=')
    runner=runner.replace("print('Running:',", "if time.monotonic()>=PARENT_DEADLINE:raise TimeoutError('TEST parent cap')\n    timeout=min(timeout,PARENT_DEADLINE-time.monotonic())\n    print('Running:',").replace("else: print(line, end='', flush=True);", "else: line=redact(line); print(line, end='', flush=True);").replace('Smoke active;','TEST/CPU metrics active;').replace('Smoke wall cap exceeded','TEST parent/phase cap exceeded')
    routing="""import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi()
with tempfile.TemporaryDirectory() as directory,patch.object(api,'build_kaggle_client') as factory:
    endpoint=factory.return_value.__enter__.return_value.kernels.kernels_api_client.list_kernel_session_output
    endpoint.return_value=SimpleNamespace(files=[],next_page_token=None,log=None)
    api.kernels_output('ducanh2403/2-vi-cai-kia-qua-nhieu/199',path=directory)
    request=endpoint.call_args.args[0]
    assert (request.user_name,request.kernel_slug,request.version_label)==('ducanh2403','2-vi-cai-kia-qua-nhieu','v199')
print('Exact own199 API routing mock PASS; no network.',flush=True)
"""
    if semantic:
        routing=routing.replace('ducanh2403/2-vi-cai-kia-qua-nhieu/199','ducanh2403/kltnnnnnff/9').replace("('ducanh2403','2-vi-cai-kia-qua-nhieu','v199')","('ducanh2403','kltnnnnnff','v9')").replace('Exact own199','Exact own9')
    execution='ROUTING_PREFLIGHT='+repr(routing)+'\n'+"""complete=False
try:
    for path,name in MAPPING.items():download('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+REVISION+'/'+path,HASHES[path],SOURCE/name)
    (OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=REVISION,sha256=HASHES),indent=2))
    run([sys.executable,'-u',SOURCE/'test_test199.py'],120)
    run([sys.executable,'-u',SOURCE/'cuda_runtime_guard.py','--output',OUTPUT],60)
    sys.path.insert(0,str(SOURCE))
    from test199_data import checkpoint,VERSION,CHECKPOINT_SHA
    existing=Path('/kaggle/working/clip_patch_199_test5000_v1/inputs199/clip_patch_baseline_full_xe10_v3/best.bin')
    available=CHECKPOINT_ROOT or (str(existing) if existing.is_file() else '')
    if available:
        weights=checkpoint(available);origin='mounted or existing exact199 bytes'
    else:
        run([sys.executable,'-m','pip','install','-q','--target',CLI,'https://github.com/Kaggle/kaggle-cli/archive/6894864ff4095cf98e9e36d3a694ff5fc9ba3818.zip'],600)
        env=os.environ.copy();env['PYTHONPATH']=str(CLI)
        run([sys.executable,'-u','-c',ROUTING_PREFLIGHT],120,env)
        from kaggle_secrets import UserSecretsClient
        token=None
        for name in SECRET_NAMES:
            try:value=UserSecretsClient().get_secret(name).strip()
            except Exception:continue
            if value:token=value;break
        if not token:raise RuntimeError('Enable own Kaggle API Secret codex or KAGGLE_API_TOKEN; no token is printed.')
        TOKENS.append(token)
        for key in ('KAGGLE_API_TOKEN','KAGGLE_USERNAME','KAGGLE_KEY'):env.pop(key,None)
        env['KAGGLE_API_TOKEN']=token;env['PYTHONUNBUFFERED']='1'
        target=OUTPUT/'inputs199';target.mkdir()
        # CLI main entry works even on builds without kaggle.__main__.
        run([sys.executable,'-u','-c','from kaggle.cli import main; main()','kernels','output',VERSION,'-p',str(target),'--file-pattern',r'(^|/)best[.]bin$'],900,env)
        weights=checkpoint(target);origin='exact own SaveVersion199 API'
    cfg['checkpoint_root']=str(weights);write_config()
    (OUTPUT/'checkpoint_source.json').write_text(json.dumps(dict(version=VERSION,sha256=CHECKPOINT_SHA,path=str(weights),origin=origin),indent=2))
    run([sys.executable,'-u',SOURCE/'run_test199.py','checkpoint_preflight','--config',CONFIG],120)
    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS,'pycocoevalcap==1.2','nltk==3.9.2','future'],600)
    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS,'git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'],600)
    os.environ['PYTHONPATH']=str(DEPS)+os.pathsep+str(SOURCE)
    run([sys.executable,'-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"],300)
    base='https://raw.githubusercontent.com/Maxlinn/CHAIR-metric-standalone/4087a26211aa2339b9a76307cb8f0321ef691d0a/'
    for name,digest in {'chair.py':'98728cd2b7fb8b97420b5f3bf14789dcad1d7d980de9c089b82856217387770d','chair.pkl':'73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3'}.items():download(base+name,digest,CHAIR/name)
    run([sys.executable,'-u',SOURCE/'run_test199.py','environment','--config',CONFIG],1200)
    run([sys.executable,'-u',SOURCE/'run_test199.py','smoke','--config',CONFIG],600)
    run([sys.executable,'-u',SOURCE/'run_test199.py','prepare','--config',CONFIG],1800)
    cfg['worker_seconds']=min(1800,PARENT_DEADLINE-time.monotonic()-1800);write_config()
    if cfg['worker_seconds']<300:raise TimeoutError('Insufficient budget before TEST inference; no automatic extension')
    launch=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0']
    run([*launch,SOURCE/'run_test199.py','worker','--config',CONFIG],cfg['worker_seconds']+30)
    run([sys.executable,'-u',SOURCE/'run_test199.py','tokenize','--config',CONFIG],180)
    for mode,cap in (('basic',180),('meteor',600),('spice',1800),('chair',600),('combine',120)):
        run([sys.executable,'-u',SOURCE/'run_test199.py',mode,'--config',CONFIG],cap)
    complete=json.loads((OUTPUT/'result.json').read_text())['status']=='CLIP_PATCH_199_TEST5000_COMPLETE'
except BaseException:
    (OUTPUT/'failure.txt').write_text(redact(traceback.format_exc()));raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,training=False,test_tuning=False,source_version='199')))
    review=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUTPUT.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt'):z.write(p,str(p.relative_to(OUTPUT)))
    print('SEND REVIEW:',review,flush=True);print('KEEP OUTPUT:',OUTPUT,flush=True)
    print('TEST COMPLETE; no training/tuning.' if complete else 'INCOMPLETE; preserve captions/logs; no automatic re-decode or retraining.',flush=True)
"""
    intro="""# Baseline199 — TEST5000 V2, sửa tương thích CUDA

**T4×2**, InternetON, Save & Run All. Chỉ gắn `ducanh2403/visual-cache` và `vuthetam/mscoco-2014`. Không cần detector/ROI/bank/prompt/testcachecũ. Chỉ stage5000TESTpatch~1,4GiB, không chép full33GiB và không chạyCLIP lại.

Checkpointbestepoch10 cốđịnh từ **ducanh2403/2-vi-cai-kia-qua-nhieu/199**, SHA5061e305...; CHECKPOINT_ROOT tùychọn nếuđãcófile/DS, đểtrống thì tảiONLYbest.bin từđúng199. Bật Secret **codex** hoặc **KAGGLE_API_TOKEN** của tàikhoảnđó. Khônglatest/phiênbảnkhác/foreignnotebook API; khônglộtoken.

Beam5/max20/rawlogprob/rank0 giữnhưVALID, khôngrerank/lengthnormalization/training/tuning. CaptionthamchiếuTEST chỉdùng chấmđiểm vàNLL saukhisinhcaption. Haiworker2500ảnh/rank, lưucaptiontrướcCPUmetrics. ChấmBLEU-1/4,METEOR,ROUGE-L,CIDEr,SPICE,CHAIRs/i (+Recall,NLL); thiếuphase thìkhôngđánhcomplete. Giữảnh/split/model/sốcaptioncốđịnh; priorprojectTESTexposure vẫnđượcghi.

V2 cho phép Torch2.11.0 CUDAcu128/cu130 đã quan sát, ghi rõ khác biệtbuild và không cam kết caption giốngbitwise. Kiểm checkpoint/runtime trước khi stage; strictweights và GPUbeam-pathcheck vẫn bắt buộc. Không cài lại Torch. Nếu filecheckpoint199 cũ còn trongWorking, tự dùng lại sau kiểmSHA.

Cap2giờgồmcảdownload/scorerCPU, khôngphảiETA/quota. StanfordCoreNLP/JavaMeteorspicecóthểcầnthờigian. MỗiCPUphase lưukếtquảriêng; nếulỗi, giữoutputcaption đểphụchồiCPU, khôngtựchạymôhìnhlại. Gửi **clip_patch_199_test5000_v2_review.zip**. Tácgiảchỉkiểmmetadata/NumPy/mock/AST; chưa chạyCUDA/scorerslocal.
"""
    if semantic:
        setup=setup.replace("CHECKPOINT_ROOT=''", "PROMPT_TEST_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-prompt-test-cache-v181')\nCHECKPOINT_ROOT=''")
        setup=setup.replace('exact199','exact bank+prompt best9').replace('own199','own kltnnnnnff/9').replace('clip_patch_199_test5000_v2','clip_patch_bank_prompt_test5000_v1')
        setup=setup.replace('for p in (VISUAL_CACHE,COCO_JSON):','for p in (VISUAL_CACHE,COCO_JSON,PROMPT_TEST_ROOT):')
        setup=setup.replace('cfg=dict(output=',"cfg=dict(semantic=True,prompt_test_root=str(PROMPT_TEST_ROOT),output=")
        execution=execution.replace('ducanh2403/2-vi-cai-kia-qua-nhieu/199','ducanh2403/kltnnnnnff/9')
        execution=execution.replace("'2-vi-cai-kia-qua-nhieu', 'v199'", "'kltnnnnnff', 'v9'")
        execution=execution.replace('from test199_data import checkpoint,VERSION,CHECKPOINT_SHA','from semantic_test_data import checkpoint,VERSION,CHECKPOINT_SHA,prompt_inputs')
        execution=execution.replace("Path('/kaggle/working/clip_patch_199_test5000_v1/inputs199/clip_patch_baseline_full_xe10_v3/best.bin')", "Path('/kaggle/working/clip_patch_bank_prompt_xe10_v1/best.bin')")
        execution=execution.replace("SOURCE/'test_test199.py'],120)","SOURCE/'test_test199.py'],120)\n    run([sys.executable,'-u',SOURCE/'test_semantic_test.py'],120)")
        execution=execution.replace("cfg['checkpoint_root']=str(weights);write_config()", "cfg['checkpoint_root']=str(weights);write_config()\n    prompt_inputs(cfg) # exact TEST181 seal/scope check before expensive setup/stage")
        execution=execution.replace("TOKENS.append(token)","if token.startswith('sk-'):raise RuntimeError('Use a Kaggle API Secret, not an OpenAI key.')\n        TOKENS.append(token)")
        execution=execution.replace('exact199 bytes','exact best9 bytes').replace('exact own SaveVersion199 API','exact own kltnnnnnff/9 API')
        execution=execution.replace('CLIP_PATCH_199_TEST5000_COMPLETE','CLIP_PATCH_BANK_PROMPT_TEST5000_COMPLETE').replace("source_version='199'","source_version='ducanh2403/kltnnnnnff/9'")
        intro='''# Bank + prompt, best epoch9 — TEST5000

**T4 ×2**, Internet ON, Save & Run All. Attach THREE Datasets:
- `ducanh2403/visual-cache`
- `vuthetam/mscoco-2014`
- `ducanh2403/csa-prompt-test-cache-v181`

Enable your **Kaggle API** Secret `codex` or `KAGGLE_API_TOKEN` (not an OpenAI key). Blank CHECKPOINT_ROOT downloads only best.bin from **ducanh2403/kltnnnnnff/9**, exact SHA7c11be52. User confirmed notebook slug/version; default owner is own ducanh2403. Missing access fails early; no latest/other checkpoint fallback. Alternatively mount the exact knownSHA file/root and set CHECKPOINT_ROOT. No API token printed.

Best epoch9 is159399updates; the full experiment ran10epochs but epoch10 has different weights. Keep epoch9 chosen on VALID before TEST. Model frozen, no training/selection/reranking. Frozen TRAIN noun/verb banks are inside the checkpoint; no bank130 or prompt175 dataset needed. TEST181 only supplies258 fixed prompt embeddings +instance links. Uses exact same ROI-box→14×14patch pooling/masks as training; old ROI feature arrays and old181 vocabulary are NOT used. No CLIP/UPT inference.

Only5000TEST patches staged (~1.4GiB), no full33GiB copy or TRAIN/VALID feature reads. Same5000IDs/rawreferences, TRAIN vocabulary and beam5/max20/rawlogprob/rank0 as baseline199. Captions saved before teacherNLL and CPUscoring. Metrics: BLEU1/4,METEOR,ROUGE-L,CIDEr,SPICE,CHAIRs/i (+Recall,NLL). Complete only when all eight metrics finished. Previous project TEST exposure disclosed; no TEST tuning.

Parent2h STOP cap inherited from baselineTEST, not ETA/billing. Latest answer confirmed cache path but did not supply remaining GPU hours; check account quota before launch. GPUweights/runtime/firstbeam-path guards before full decode. No Torch reinstall. Preserve caption/output on errors; no automatic model rerun. Send **clip_patch_bank_prompt_test5000_v1_review.zip**. Author checks NumPy/mock/AST only; models/CUDA/scorers run only by user.
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=True)),cells=[cell('markdown',intro),cell('code',setup),cell('code',runner),cell('code',execution)])
    p=HERE/('CLIP_PATCH_BANK_PROMPT_TEST5000_2GPU.ipynb' if semantic else 'CLIP_PATCH_199_TEST5000_V2_2GPU.ipynb');p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8');print(p.name,p.stat().st_size,'bytes',len(mapping),'pins',revision)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);p.add_argument('--semantic',action='store_true');a=p.parse_args();make(a.revision,a.semantic)
