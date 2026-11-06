"""Author two pinned Kaggle notebooks; no model, API or notebook execution."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from make_smoke_notebook import cell,RUNNER

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PREFIX='experiments/csa_paper/'
ENTRY=['run_prompt_scst.py','prompt_scst_rerank.py','test_prompt_scst_data.py','test_scst_full_model.py',
       'test_prompt_adapter_model.py','make_prompt_scst_notebooks.py','cuda_runtime_guard.py']
ASSETS=['PROMPT_SCST_PROTOCOL.md','SAVE_FULL_ROI_V134_PACKAGE.json','FULL_PROMPT_FRESH_V175_RESULT.json','RERANK_TEST_INPUTS.json']


def source_bundle(revision):
    """Include local imports, even function-local imports, from the exact Git tree."""
    tracked=set(subprocess.check_output(['git','ls-tree','-r','--name-only',revision],cwd=ROOT,text=True).splitlines())
    pending=[PREFIX+n for n in ENTRY+ASSETS];mapping={};payloads={}
    while pending:
        source=pending.pop()
        if source in mapping:continue
        if source not in tracked:raise ValueError('Missing pinned source:'+source)
        dest=source[len(PREFIX):] if source.startswith(PREFIX) else source
        raw=subprocess.check_output(['git','show',revision+':'+source],cwd=ROOT);mapping[source]=dest;payloads[source]=raw
        if not source.endswith('.py'):continue
        tree=ast.parse(raw.decode('utf-8-sig'));modules=[]
        for n in ast.walk(tree):
            if isinstance(n,ast.Import):modules.extend(a.name for a in n.names)
            if isinstance(n,ast.ImportFrom):
                name=n.module or ''
                if n.level:
                    package=Path(dest).parent.parts
                    name='.'.join((*package[:len(package)-n.level+1],name)).strip('.')
                modules.append(name)
        for module in modules:
            rel=module.replace('.','/')
            for candidate in (PREFIX+rel+'.py',rel+'.py',rel+'/__init__.py'):
                if candidate in tracked and (candidate.startswith(PREFIX) or candidate.startswith('captioning/')):pending.append(candidate)
    # External packages need their package marker as well as imported submodules.
    if any(n.startswith('captioning/') for n in mapping):
        marker='captioning/__init__.py';mapping[marker]=marker;payloads[marker]=subprocess.check_output(['git','show',revision+':'+marker],cwd=ROOT)
    if len(set(mapping.values()))!=len(mapping):raise ValueError('Source destination collision')
    return mapping,{n:hashlib.sha256(raw).hexdigest() for n,raw in payloads.items()}


def make(revision):
    revision=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip();mapping,hashes=source_bundle(revision)
    for arm,version in [('CONTROL','ducanh2403/2-vi-cai-kia-qua-nhieu/178'),('PROMPT','ducanh3/notebook7bc83c4c76/58')]:
        setup=f'''import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile,shutil
from pathlib import Path
ARM={arm!r}
KERNEL_VERSION={version!r} # This account's own output only; never replace with latest.
'''+'''
SECRET_NAME='codex'
WEIGHTS_ROOT='' # Optional local mount containing the exact selected best.bin; blank downloads own version.
RESUME_ROOT='' # Only a complete preserved output from THIS arm/source/recipe. Blank starts the approved continuation.
WALL_HOURS=4 # Parent STOPcap, not a promised runtime or billed GPU quota.
FULL_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-full-roi-karpathy-v134')
PROMPT_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-prompt-full-karpathy-v175')
COCO_ROOT=Path('/kaggle/input/datasets/vuthetam/mscoco-2014')
COCO_JSON=COCO_ROOT/'dataset_coco.json'
DETECTIONS=Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
OUTPUT=Path('/kaggle/working/csa_prompt_scst_'+ARM+'_full1_v2')
PARENT_DEADLINE=time.monotonic()+WALL_HOURS*3600
for p in (FULL_ROOT,PROMPT_ROOT,COCO_ROOT,COCO_JSON,DETECTIONS):
    if not p.exists():raise FileNotFoundError('Attach inputs BEFORE GPU; correct this path: '+str(p))
if OUTPUT.exists():raise RuntimeError('Preserve prior output. Use a fresh session/output path; no automatic restart.')
if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<10*1024**3:raise RuntimeError('Need10GiB free for ROI stage, weights and atomic checkpoints')
OUTPUT.mkdir(parents=True);SOURCE=Path(tempfile.mkdtemp(prefix='csa-prompt-scst-'+ARM.lower()+'-',dir='/tmp'));LOG=OUTPUT/'notebook.log'
'''
        setup+='REVISION='+repr(revision)+'\nMAPPING='+repr(mapping)+'\nHASHES='+repr(hashes)+'\n'
        setup+='''
for original,destination in MAPPING.items():
    remaining=PARENT_DEADLINE-time.monotonic()
    if remaining<=0:raise TimeoutError('Parent deadline during source download')
    raw=urllib.request.urlopen('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+REVISION+'/'+original,timeout=min(90,remaining)).read()
    if hashlib.sha256(raw).hexdigest()!=HASHES[original]:raise RuntimeError('Changed pinned source:'+original)
    target=SOURCE/destination;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=REVISION,sha256=HASHES),indent=2),encoding='utf-8')
os.environ['OMP_NUM_THREADS']='1';os.environ['PYTHONPATH']=str(SOURCE)
CHAIR=SOURCE/'chair';CONFIG=SOURCE/'config.json';DEPS=SOURCE/'hf-deps'
cfg=dict(arm=ARM,updates=14161,full_root=str(FULL_ROOT),prompt_root=str(PROMPT_ROOT),checkpoint_root=WEIGHTS_ROOT,
    dataset_json=str(COCO_JSON),coco_root=str(COCO_ROOT),detections=str(DETECTIONS),output=str(OUTPUT),stage=str(SOURCE/'stage'),
    chair=str(CHAIR),model_cache=str(SOURCE/'hf-cache'),worker_seconds=10000,rerank_seconds=1200,resume_root=RESUME_ROOT)
def write_config():
    CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    (OUTPUT/'protocol.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
write_config()
'''
        runner=RUNNER.replace("print('Running:',", "if time.monotonic()>=PARENT_DEADLINE:raise TimeoutError('Parent deadline exceeded')\n    timeout=min(timeout,max(.01,PARENT_DEADLINE-time.monotonic()))\n    print('Running:',").replace('Smoke active;','Prompt SCST active;').replace('Smoke wall cap exceeded','Parent/phase deadline exceeded')
        execution='''
complete=False
try:
    run([sys.executable,'-u',SOURCE/'test_prompt_scst_data.py'],timeout=120)
    run([sys.executable,'-u',SOURCE/'cuda_runtime_guard.py','--output',OUTPUT],timeout=60)
    if not WEIGHTS_ROOT:
        run([sys.executable,'-m','pip','install','-q','kaggle==2.2.5'],timeout=300)
        # Local mock checks exact version routing before the authenticated download.
        route="from types import SimpleNamespace;from unittest.mock import patch;from kaggle.api.kaggle_api_extended import KaggleApi;import tempfile;api=KaggleApi();owner,slug,version="+repr(KERNEL_VERSION)+".split('/');ctx=tempfile.TemporaryDirectory();d=ctx.__enter__();factory=patch.object(api,'build_kaggle_client');f=factory.__enter__();ep=f.return_value.__enter__.return_value.kernels.kernels_api_client.list_kernel_session_output;ep.return_value=SimpleNamespace(files=[],next_page_token=None,log=None);api.kernels_output("+repr(KERNEL_VERSION)+",path=d);r=ep.call_args.args[0];assert (r.user_name,r.kernel_slug,r.version_label)==(owner,slug,'v'+version);factory.__exit__(None,None,None);ctx.__exit__(None,None,None);print('Exact version route mock PASS')"
        from kaggle_secrets import UserSecretsClient
        token=UserSecretsClient().get_secret(SECRET_NAME).strip()
        if not token:raise RuntimeError('Enable configured Kaggle API Secret for THIS account')
        previous={k:os.environ.pop(k,None) for k in ('KAGGLE_API_TOKEN','KAGGLE_USERNAME','KAGGLE_KEY')}
        try:
            os.environ['KAGGLE_API_TOKEN']=token
            run([sys.executable,'-c',route],timeout=60)
            weight_dir=OUTPUT/'source_xe';weight_dir.mkdir()
            run([sys.executable,'-c','from kaggle.cli import main;main()','kernels','output',KERNEL_VERSION,
                 '-p',weight_dir,'--file-pattern',r'(^|/)best\\.bin$'],timeout=900)
        finally:
            for k in previous:os.environ.pop(k,None)
            for k,v in previous.items():
                if v is not None:os.environ[k]=v
            del token
        cfg['checkpoint_root']=str(weight_dir)
    sys.path.insert(0,str(SOURCE))
    from prompt_scst_data import source_checkpoint
    print('Exact selected checkpoint:',source_checkpoint(cfg['checkpoint_root'],ARM),flush=True);write_config()
    run([sys.executable,'-m','pip','install','-q','pycocoevalcap==1.2','nltk==3.9.2','future'],timeout=300)
    run([sys.executable,'-m','pip','install','-q','--no-deps','git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'],timeout=300)
    run([sys.executable,'-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"],timeout=300)
    run(['java','-version'],timeout=30)
    run(['git','init',CHAIR],timeout=60)
    run(['git','-C',CHAIR,'remote','add','origin','https://github.com/Maxlinn/CHAIR-metric-standalone.git'],timeout=60)
    run(['git','-C',CHAIR,'fetch','--depth','1','origin','4087a26211aa2339b9a76307cb8f0321ef691d0a'],timeout=300)
    run(['git','-C',CHAIR,'checkout','--detach','FETCH_HEAD'],timeout=60)
    # Fail inference dependencies/assets before expensive training; isolate HF packages from host apps.
    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS,'transformers==4.57.1','tokenizers==0.22.1',
         'huggingface-hub==0.36.2','safetensors==0.6.2'],timeout=300)
    os.environ['PYTHONPATH']=str(DEPS)+os.pathsep+str(SOURCE)
    run([sys.executable,'-c',"from importlib.metadata import version;from transformers import CLIPModel,CLIPProcessor;assert version('transformers')=='4.57.1';assert version('huggingface-hub')=='0.36.2';print('Isolated HF imports PASS')"],timeout=60)
    run([sys.executable,'-u',SOURCE/'prompt_scst_rerank.py','prefetch','--config',CONFIG],timeout=1200)
    run([sys.executable,'-u',SOURCE/'run_prompt_scst.py','prepare','--config',CONFIG],timeout=1800)
    run([sys.executable,'-u',SOURCE/'prompt_scst_rerank.py','prepare','--config',CONFIG],timeout=1800)
    run([sys.executable,'-u',SOURCE/'run_prompt_scst.py','metric_preflight','--config',CONFIG],timeout=240)
    run([sys.executable,'-u',SOURCE/'test_prompt_adapter_model.py'],timeout=180)
    launch=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0']
    run([*launch,SOURCE/'test_scst_full_model.py'],timeout=240)
    run([*launch,SOURCE/'run_prompt_scst.py','preflight','--config',CONFIG],timeout=600)
    cfg['worker_seconds']=PARENT_DEADLINE-time.monotonic()-1800
    if cfg['worker_seconds']<2400:raise TimeoutError('Insufficient parent budget after preflight; preserve inputs/logs')
    write_config()
    run([*launch,SOURCE/'run_prompt_scst.py','worker','--config',CONFIG],timeout=cfg['worker_seconds']+30)
    run([sys.executable,'-u',SOURCE/'run_prompt_scst.py','score','--config',CONFIG],timeout=600)
    cfg['rerank_seconds']=PARENT_DEADLINE-time.monotonic()-600
    if cfg['rerank_seconds']<180:raise TimeoutError('Training complete, insufficient rerank reserve; preserve5beams, do not retrain')
    write_config()
    run([*launch,SOURCE/'prompt_scst_rerank.py','worker','--config',CONFIG],timeout=cfg['rerank_seconds']+30)
    run([sys.executable,'-u',SOURCE/'prompt_scst_rerank.py','score','--config',CONFIG],timeout=600)
    complete=json.loads((OUTPUT/'result.json').read_text(encoding='utf-8'))['status']=='PROMPT_SCST_RERANK_VALID_COMPLETE'
    if not complete:raise RuntimeError('Incomplete full result')
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,arm=ARM,updates_requested=14161,test_used=False)),encoding='utf-8')
    REVIEW=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(REVIEW,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt') and p.name!='training_references.json':z.write(p,str(p.relative_to(OUTPUT)))
    checkpoint=OUTPUT/'checkpoint.bin'
    print('SEND REVIEW:',REVIEW,'\\nKEEP ENTIRE OUTPUT:',OUTPUT,'\\nCHECKPOINT:',checkpoint if checkpoint.is_file() else 'No SCST checkpoint saved yet',flush=True)
    print('COMPLETE — ORIGINAL/END, rank0/fixed rerank, full VALID5000.' if complete else 'INCOMPLETE — preserve output. No automatic restart or partial-model ranking.',flush=True)
'''
        intro=f'''# CSA + prompt → SCST + CE + fixed rerank — {arm} V2

Một notebook/một tài khoản, **GPU T4 ×2**, Internet ON, Save Version → Save & Run All. Nạp **selected {arm} từ {version}**, không train từ đầu, không reset adapter.

Gắn4Dataset: `csa-full-roi-karpathy-v134`, `csa-prompt-full-karpathy-v175`, `vuthetam/mscoco-2014` (**cả JPEG gốc và dataset_coco.json**), `objectdetectionecache`.
Không cần bank130 riêng: bank nằm trong checkpoint. Dataset175 phải được chia sẻ quyền cho tài khoản chạy.

Nếu chưa lưu checkpoint thành Dataset, bật Secret tên `codex` (API token của **chính tài khoản đang chạy**). Notebook chỉ tải best.bin đúng version của tài khoản đó. Nếu gắn checkpoint sẵn, điền WEIGHTS_ROOT, không cần API.

Giữ nguyên kiến trúc. CONTROL tắt adapter; PROMPT dùng adapter đã học. Học **1 lượt toàn bộ113287ảnh TRAIN =14161update/global8**, cuối4/3; freshAdam5e-6, CE+.05SCST, không thưởng đối tượng.
Trước/sau học: fullVALID5000, beam5/max20; cùng fixed rerank .3decoder+.4CLIP+.3object. KhôngBLIP/TEST/tuning/oracle. C10/P9 có số epoch XE ban đầu khác nhau; không gọi là bằng nhau tổng ngân sách.

Giới hạn4giờ theo đồng hồ notebook, không phải cam kết thời gian hay quota tính phí. Lưu checkpoint mỗi500update và khi dừng; giữ cả output nếu incomplete. Hai tài khoản có quota10/15giờ theo báo cáo mới; không tự tăngcap/epoch.
V2 sửa chuẩn bị JPEG: quét thư mục một lần, chỉ index detectorVALID,8luồng đọc/kiểmảnh,log từng bước/mỗi100ảnh; cap chuẩn bị30phút nằm trong cap tổng4giờ. Output mới `_full1_v2`; không xóa outputV1. Không đổi mô hình/loss/nguồn XE.
Tác giả chỉ kiểm NumPy/metadata/AST/mock, không chạy model/DDP/notebook. Gửi ZIPreview và giữ checkpoint.bin/nguồn XE/candidates.
'''
        nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
            kaggle=dict(isInternetEnabled=True,isGpuEnabled=True)),cells=[cell('markdown',intro),cell('code',setup),cell('code',runner),cell('code',execution)])
        path=HERE/f'CSA_PROMPT_SCST_{arm}_FULL1_2GPU.ipynb';path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
        print(path.name,path.stat().st_size,'bytes;',len(mapping),'pins;',revision)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);make(p.parse_args().revision)
