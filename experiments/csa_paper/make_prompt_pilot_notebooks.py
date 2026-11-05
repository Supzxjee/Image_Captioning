"""Author source-pinned matched GPU notebooks; no neural imports or notebook execution."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from make_smoke_notebook import cell,RUNNER

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PREFIX='experiments/csa_paper/'
FILES=['prompt_pilot_data.py','prompt_pilot_model.py','run_prompt_pilot.py','test_prompt_pilot_data.py','make_prompt_pilot_notebooks.py',
    'prompt_adapter_data.py','prompt_adapter_model.py','test_prompt_adapter_model.py','prompt_text_cache.py',
    'scst_pilot_data.py','scst_pilot_model.py','run_scst_pilot.py','test_scst_pilot_model.py',
    'model.py','real_bank_data.py','full_cache_data.py','full_xe_data.py','sharing_data.py','package_full_roi_dataset.py',
    'saved_quality_data.py','SAVE_FULL_ROI_V134_PACKAGE.json','PROMPT_PILOT_PROTOCOL.md','make_smoke_notebook.py']
EXTERNAL=['captioning/__init__.py','captioning/scst.py']


def make(revision):
    revision=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    mapping={PREFIX+n:n for n in FILES};mapping.update({n:n for n in EXTERNAL})
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',revision+':'+n],cwd=ROOT)).hexdigest() for n in mapping}
    for arm in ('CONTROL','PROMPT'):
        setup='''
import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile,shutil
from pathlib import Path
'''+f'ARM={arm!r}\n'+'''
WALL_HOURS=3 # Parent STOPcap, not ETA or Kaggle quota/billing promise.
INPUT_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-prompt-pilot-inputs-v169')
FULL_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-full-roi-karpathy-v134')
COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
OUTPUT=Path('/kaggle/working/csa_prompt_pilot_'+ARM+'_500_v1')
PARENT_DEADLINE=time.monotonic()+WALL_HOURS*3600
for p in (INPUT_ROOT,FULL_ROOT,COCO_JSON):
    if not p.exists():raise FileNotFoundError('Attach3inputDatasets BEFORE GPU: '+str(p))
if OUTPUT.exists():raise RuntimeError('Preserve existing output; use fresh runtime/output path. No automatic restart.')
if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<8*1024**3:raise RuntimeError('Need8GiB free for fullROI stage and atomic checkpoints')
OUTPUT.mkdir(parents=True);SOURCE=Path(tempfile.mkdtemp(prefix='csa-prompt-'+ARM.lower()+'-',dir='/tmp'));LOG=OUTPUT/'notebook.log'
'''
        setup+='REVISION='+repr(revision)+'\nMAPPING='+repr(mapping)+'\nHASHES='+repr(hashes)+'\n'
        setup+='''
for original,destination in MAPPING.items():
    remaining=PARENT_DEADLINE-time.monotonic()
    if remaining<=0:raise TimeoutError('Parent cap during source download')
    raw=urllib.request.urlopen('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+REVISION+'/'+original,timeout=min(90,remaining)).read()
    if hashlib.sha256(raw).hexdigest()!=HASHES[original]:raise RuntimeError('Changed pinned source:'+original)
    target=SOURCE/destination;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=REVISION,sha256=HASHES),indent=2),encoding='utf-8')
os.environ['OMP_NUM_THREADS']='1';os.environ['PYTHONPATH']=str(SOURCE)
CHAIR=SOURCE/'chair';CONFIG=SOURCE/'config.json'
cfg=dict(arm=ARM,updates=500,input_root=str(INPUT_ROOT),full_root=str(FULL_ROOT),dataset_json=str(COCO_JSON),
    output=str(OUTPUT),stage=str(SOURCE/'stage'),chair=str(CHAIR),worker_seconds=6000)
CONFIG.write_text(json.dumps(cfg),encoding='utf-8')
'''
        runner=RUNNER.replace("print('Running:',", "if time.monotonic()>=PARENT_DEADLINE:raise TimeoutError('Parent cap exceeded')\n    timeout=min(timeout,max(.01,PARENT_DEADLINE-time.monotonic()))\n    print('Running:',")
        runner=runner.replace('Smoke active;','Prompt pilot active;').replace('Smoke wall cap exceeded','Prompt pilot phase/parent cap exceeded')
        final='''
complete=False
try:
    run([sys.executable,'-u',SOURCE/'test_prompt_pilot_data.py'],timeout=120)
    # Fail weight/cache identities before slow fullROI staging or GPU smoke.
    sys.path.insert(0,str(SOURCE))
    from prompt_pilot_data import checkpoint,cache_root
    print('Exact161 checkpoint:',checkpoint(INPUT_ROOT),flush=True)
    print('Exact169 cache:',cache_root(INPUT_ROOT)[3],flush=True)
    probe="import json,torch;print(json.dumps(dict(version=torch.__version__,available=torch.cuda.is_available(),count=torch.cuda.device_count())));assert torch.cuda.is_available() and torch.cuda.device_count()==2,'Select GPU T4x2 and restart; do not reinstall Torch blindly'"
    run([sys.executable,'-c',probe],timeout=60)
    run([sys.executable,'-m','pip','install','-q','pycocoevalcap==1.2','nltk==3.9.2','future'],timeout=300)
    run([sys.executable,'-m','pip','install','-q','--no-deps','git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'],timeout=300)
    run([sys.executable,'-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"],timeout=300)
    run(['java','-version'],timeout=30)
    run(['git','init',CHAIR],timeout=60)
    run(['git','-C',CHAIR,'remote','add','origin','https://github.com/Maxlinn/CHAIR-metric-standalone.git'],timeout=60)
    run(['git','-C',CHAIR,'fetch','--depth','1','origin','4087a26211aa2339b9a76307cb8f0321ef691d0a'],timeout=300)
    run(['git','-C',CHAIR,'checkout','--detach','FETCH_HEAD'],timeout=60)
    run([sys.executable,'-u',SOURCE/'run_prompt_pilot.py','prepare','--config',CONFIG],timeout=1800)
    run([sys.executable,'-u',SOURCE/'run_prompt_pilot.py','metric_preflight','--config',CONFIG],timeout=240)
    run([sys.executable,'-u',SOURCE/'test_prompt_adapter_model.py'],timeout=180)
    launch=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0']
    run([*launch,SOURCE/'test_scst_pilot_model.py','--ddp'],timeout=240)
    run([*launch,SOURCE/'run_prompt_pilot.py','preflight','--config',CONFIG],timeout=600)
    remaining=PARENT_DEADLINE-time.monotonic()
    if remaining<2400:raise RuntimeError('Too little parent budget after staging/smoke; preserve stage/log, do not train automatically')
    cfg['worker_seconds']=remaining-900;CONFIG.write_text(json.dumps(cfg),encoding='utf-8')
    (OUTPUT/'protocol.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    run([*launch,SOURCE/'run_prompt_pilot.py','worker','--config',CONFIG],timeout=cfg['worker_seconds']+30)
    run([sys.executable,'-u',SOURCE/'run_prompt_pilot.py','score','--config',CONFIG],timeout=850)
    complete=json.loads((OUTPUT/'result.json').read_text(encoding='utf-8'))['status']=='PROMPT_PILOT500_COMPLETE'
    if not complete:raise RuntimeError('Incomplete endpoint/VALID, no partial ranking')
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,arm=ARM,updates_requested=500,test_used=False,rerank_measured=False)),encoding='utf-8')
    REVIEW=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(REVIEW,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt') and p.name!='training_references.json':z.write(p,str(p.relative_to(OUTPUT)))
    print('SEND REVIEW:',REVIEW,'\\nKEEP ENTIRE OUTPUT:',OUTPUT,flush=True)
    print('KEEP CHECKPOINT:',OUTPUT/'checkpoint.bin' if (OUTPUT/'checkpoint.bin').is_file() else 'No checkpoint yet',flush=True)
    print('COMPLETE500 + original/END VALID5000.5beams saved; rerank is the next scoring step.' if complete else 'INCOMPLETE; preserve output; no automatic restart.',flush=True)
'''
        intro=f'''# CSA prompt pilot — {arm},500updates,T4×2
**Một notebook/một tài khoản, mỗi notebook2GPU.** Select **GPU T4×2**, Internet ON, Save Version → Save & Run All. Không cần Secret/API.
Gắn3inputDatasets: `csa-prompt-pilot-inputs-v169` (cache+161weights, cần share quyền cho Ngân), `csa-full-roi-karpathy-v134`, `vuthetam/mscoco-2014` (dataset_coco.json).
Chỉnh path ở cell đầu nếu Kaggle mount khác; không đổi SHA/nguồn. Không cần dataset detector/UPT/JPEG/bank130 riêng; banks nằm trong161.

Giữ noun/verb banks. {'CONTROL tắt prompt adapter và chỉ học CSA.' if arm=='CONTROL' else 'PROMPT học CSA + object/relation residual adapters từ zero-output initialization.'}
Cùng exact161/freshAdam5e-6/CE+.05SCST/FP32/dropoutOFF/clip1. TRAIN pool10k,500×8=4000uniqueimageevents, không phải full epoch.
Actual161/CUDA/DDP guards trước train;2smokeupdates bị bỏ,worker nạp161 lại. Parent3h STOPcap,không hứa ETA/billing; checkpoint100updates, không tự restart.
ORIGINAL+END fullVALID5000 beam5/max20 và fullrefNLL/CIDEr/B4/ROUGE/CHAIR/recall; PROMPT thêm64image sensitivity probe. KhôngTEST,không chọn checkpoint theoVALID.
Lưu5beams/ảnh cho **bước rerank cố định tiếp theo**, notebook này chỉ đo rank0; chưa có điểm sau rerank. Giữ checkpoint.bin/wholeoutput và gửi ZIPreview.
Tác giả chỉ kiểm metadata/NumPy/AST/sourcebundle; thực thi model/DDP/API trên máy tác giả không được thực hiện.
'''
        nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
            kaggle=dict(isInternetEnabled=True,isGpuEnabled=True)),cells=[cell('markdown',intro),cell('code',setup),cell('code',runner),cell('code',final)])
        path=HERE/f'CSA_PROMPT_{arm}_PILOT500_2GPU.ipynb';path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
        print(path.name,path.stat().st_size,'bytes; source',revision)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);make(p.parse_args().revision)
