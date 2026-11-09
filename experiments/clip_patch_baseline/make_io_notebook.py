"""Generate source-pinned CPU IO notebook; never submit it."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/csa_paper'))
from make_smoke_notebook import RUNNER,cell

def make(revision):
    revision=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    mapping={'experiments/clip_patch_baseline/'+n:n for n in ('io_probe.py','test_io_probe.py','full_data.py','patch_data.py','SMOKE_SOURCE.json','IO_PROBE_PROTOCOL.md')}
    mapping.update({'experiments/csa_paper/'+n:n for n in ('data.py','real_bank_data.py')})
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',revision+':'+n],cwd=ROOT)).hexdigest() for n in mapping}
    intro="""# Kiểm tra luồng đọc cache — CPU ONLY

**Accelerator: None**, Internet ON, Save & Run All. Không GPU, không API token, không huấn luyện hay sinh caption.

Gắn hai Dataset giống full baseline: `ducanh2403/visual-cache` và `vuthetam/mscoco-2014`. Giữ cùng2.048captionevents TRAIN của64globalbatches trước;4tiến trình đọc/CPU LRU256. Hai lượt đọc mountedH5 → chép riêng mẫu theo thứ tự vật lý sang localfloat16NPY → hai lượt localmemmap. Kiểm tra hash toàn luồng đặc trưng khớp; không mất thông tin.

Chỉ chép mẫu khoảng0,6GB, không phải full35GB. Ghi disk/RAM/staging/startup/read+stack/walltime; CPUdelivery không phải thời gian GPUupdate. Không xóa OS cache: không gọi là phép đo cold-cache, cachelocal nhỏ có thể nằmRAM; chưa suy ra tốc độfull hay ETA10epoch.

Giới hạn30phút là capdừng; không cam kết runtime. Gửi `clip_patch_cache_io_probe_v1_review.zip`; chưa chạy lại fullGPU. Tác giả chỉ kiểm NumPy/mock/AST; phép đo Kaggle thực do bạn chạy.
"""
    setup=f"""import os,sys,json,hashlib,shutil,time,tempfile,urllib.request,zipfile,traceback,queue,signal,subprocess,threading
from pathlib import Path
REVISION={revision!r}
MAPPING={mapping!r}
HASHES={hashes!r}
"""+"""VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
OUTPUT=Path('/kaggle/working/clip_patch_cache_io_probe_v1')
for p in (VISUAL_CACHE,COCO_JSON):
    if not p.exists():raise FileNotFoundError('Attach/correct input path: '+str(p))
if OUTPUT.exists():raise RuntimeError('Preserve old output; use fresh session or output path')
if shutil.disk_usage('/tmp').free<4*1024**3:raise RuntimeError('Need4GiB scratch free for metadata/sample; do not copy fullcache')
OUTPUT.mkdir(parents=True);SOURCE=Path(tempfile.mkdtemp(prefix='patch-cache-io-',dir='/tmp'));LOG=OUTPUT/'notebook.log'
PARENT_DEADLINE=time.monotonic()+1800
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONPATH']=str(SOURCE)
CONFIG=SOURCE/'config.json'
cfg=dict(output=str(OUTPUT),stage=str(SOURCE/'metadata'),local=str(SOURCE/'local_sample'),visual_cache=str(VISUAL_CACHE),dataset_json=str(COCO_JSON),wall_seconds=1740)
CONFIG.write_text(json.dumps(cfg,indent=2));(OUTPUT/'protocol.json').write_text(json.dumps(cfg,indent=2))
"""
    runner=RUNNER.replace("print('Running:',", "if time.monotonic()>=PARENT_DEADLINE:raise TimeoutError('CPU parent cap')\n    timeout=min(timeout,PARENT_DEADLINE-time.monotonic())\n    print('Running:',").replace('Smoke active;','CPU cache IO active;').replace('Smoke wall cap exceeded','CPU IO parent/phase cap exceeded')
    execution="""complete=False
try:
    for path,name in MAPPING.items():
        raw=urllib.request.urlopen('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+REVISION+'/'+path,timeout=90).read()
        if hashlib.sha256(raw).hexdigest()!=HASHES[path]:raise ValueError('Changed source: '+name)
        (SOURCE/name).write_bytes(raw)
    (OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=REVISION,sha256=HASHES),indent=2))
    run([sys.executable,'-u',SOURCE/'test_io_probe.py'],120)
    cfg['wall_seconds']=PARENT_DEADLINE-time.monotonic()-30;CONFIG.write_text(json.dumps(cfg,indent=2))
    run([sys.executable,'-u',SOURCE/'io_probe.py','--config',CONFIG],cfg['wall_seconds'])
    complete=json.loads((OUTPUT/'result.json').read_text())['status']=='CPU_PATCH_IO_PROBE_COMPLETE'
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc());raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,models_executed=False,test_features_read=False)))
    review=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUTPUT.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt'):z.write(p,str(p.relative_to(OUTPUT)))
    print('SEND REVIEW:',review,flush=True)
    print('CPU IO COMPLETE; do not automatically start fullGPU.' if complete else 'INCOMPLETE; preserve log and send review, no automatic retry.',flush=True)
"""
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=False)),cells=[cell('markdown',intro),cell('code',setup),cell('code',runner),cell('code',execution)])
    path=HERE/'CLIP_PATCH_CACHE_IO_PROBE_CPU.ipynb';path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8');print(path.name,path.stat().st_size,'bytes',revision)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);make(p.parse_args().revision)
