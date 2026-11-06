"""Author own-account frozen TEST notebooks; never execute them locally."""
import argparse
import json
from pathlib import Path
import subprocess
import make_prompt_scst_notebooks as bundle
from make_smoke_notebook import cell
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def make(revision):
    revision=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    bundle.ENTRY=['run_prompt_test.py','test_prompt_test_data.py','make_prompt_test_notebooks.py','cuda_runtime_guard.py']
    bundle.ASSETS=['PROMPT_TEST_PROTOCOL.md','PROMPT_TEST_PREPARATION_RESULT.json','PROMPT_SCST_RESULT.json','RERANK_TEST_INPUTS.json']
    mapping,hashes=bundle.source_bundle(revision)
    for arm,version in [('CONTROL','ducanh2403/2-vi-cai-kia-qua-nhieu/180'),('PROMPT','ducanh3/notebook7bc83c4c76/60')]:
        old=json.loads((HERE/f'CSA_PROMPT_SCST_{arm}_FULL1_2GPU.ipynb').read_text(encoding='utf-8'));old_code=[c['source'] for c in old['cells'] if c['cell_type']=='code']
        setup='''import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile,shutil
from pathlib import Path
'''+f'ARM={arm!r}\nKERNEL_VERSION={version!r}\n'+'''
WEIGHTS_ROOT='' # Optional mounted exact checkpoint.bin; otherwise download own-account output.
SECRET_NAME='codex'
CACHE_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-prompt-test-cache-v181')
COCO_ROOT=Path('/kaggle/input/datasets/vuthetam/mscoco-2014')
COCO_JSON=COCO_ROOT/'dataset_coco.json'
DETECTIONS=Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
OUTPUT=Path('/kaggle/working/csa_prompt_TEST_'+ARM+'_v1')
WALL_HOURS=3 # STOPcap, not an ETA or billed GPU quota guarantee.
PARENT_DEADLINE=time.monotonic()+WALL_HOURS*3600
for p in (CACHE_ROOT,COCO_JSON,DETECTIONS):
    if not p.exists():raise FileNotFoundError('Attach3inputDatasets before GPU: '+str(p))
if OUTPUT.exists():raise RuntimeError('Preserve existing output; use a fresh runtime/output path')
if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<5*1024**3:raise RuntimeError('Need5GiB free for weights/models/output')
OUTPUT.mkdir(parents=True);SOURCE=Path(tempfile.mkdtemp(prefix='csa-prompt-test-'+ARM.lower()+'-',dir='/tmp'));LOG=OUTPUT/'notebook.log'
'''
        setup+='REVISION='+repr(revision)+'\nMAPPING='+repr(mapping)+'\nHASHES='+repr(hashes)+'\n'
        setup+='''
for original,destination in MAPPING.items():
    if time.monotonic()>=PARENT_DEADLINE:raise TimeoutError('Source download deadline')
    raw=urllib.request.urlopen('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+REVISION+'/'+original,timeout=90).read()
    if hashlib.sha256(raw).hexdigest()!=HASHES[original]:raise ValueError('Changed pinned source:'+original)
    p=SOURCE/destination;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=REVISION,sha256=HASHES),indent=2),encoding='utf-8')
os.environ['OMP_NUM_THREADS']='1';os.environ['PYTHONPATH']=str(SOURCE)
CHAIR=SOURCE/'chair';DEPS=SOURCE/'deps';CONFIG=SOURCE/'config.json'
cfg=dict(arm=ARM,cache_root=str(CACHE_ROOT),checkpoint_root=WEIGHTS_ROOT,dataset_json=str(COCO_JSON),coco_root=str(COCO_ROOT),detections=str(DETECTIONS),
    output=str(OUTPUT),chair=str(CHAIR),model_cache=str(SOURCE/'hf-cache'),decode_seconds=5400,rerank_seconds=1800)
def write_config():
    CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8');(OUTPUT/'protocol.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
write_config()
'''
        # Retain the proven own-version CLI routing, secret cleanup, metric and isolated-HF setup.
        execution=old_code[2][:old_code[2].index("    run([sys.executable,'-u',SOURCE/'run_prompt_scst.py','prepare'")]
        execution=execution.replace("SOURCE/'test_prompt_scst_data.py'","SOURCE/'test_prompt_test_data.py'")
        execution=execution.replace('from prompt_scst_data import source_checkpoint','from prompt_test_data import source_checkpoint').replace('best\\.bin','checkpoint\\.bin')
        execution+='''
    run([sys.executable,'-u',SOURCE/'run_prompt_test.py','prepare','--config',CONFIG],timeout=1800)
    run([sys.executable,'-u',SOURCE/'run_prompt_test.py','metric_preflight','--config',CONFIG],timeout=240)
    launch=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0']
    cfg['decode_seconds']=PARENT_DEADLINE-time.monotonic()-2400
    if cfg['decode_seconds']<1200:raise TimeoutError('Insufficient decode/score reserve after preparation')
    write_config();run([*launch,SOURCE/'run_prompt_test.py','decode','--config',CONFIG],timeout=cfg['decode_seconds']+30)
    cfg['rerank_seconds']=PARENT_DEADLINE-time.monotonic()-1200
    if cfg['rerank_seconds']<180:raise TimeoutError('Captions saved; insufficient rerank reserve, do not regenerate')
    write_config();run([*launch,SOURCE/'run_prompt_test.py','rerank','--config',CONFIG],timeout=cfg['rerank_seconds']+30)
    run([sys.executable,'-u',SOURCE/'run_prompt_test.py','score','--config',CONFIG],timeout=1200)
    complete=json.loads((OUTPUT/'result.json').read_text(encoding='utf-8'))['status']=='PROMPT_TEST5000_COMPLETE'
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,arm=ARM,training=False,test_tuning=False)),encoding='utf-8')
    REVIEW=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(REVIEW,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUTPUT.iterdir():
            if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt'):z.write(p,p.name)
    print('SEND REVIEW:',REVIEW,'\\nKEEP WHOLE OUTPUT:',OUTPUT,flush=True)
    print('TEST COMPLETE; all locked conditions reported.' if complete else 'INCOMPLETE; preserve captions/logs, no automatic retry or policy change.',flush=True)
'''
        intro=f'''# Frozen TEST5000 — {arm}
**T4 ×2**, Internet ON, Save&RunAll. Một notebook/một tài khoản. Secret `codex` của tài khoản chạy để tải checkpoint.bin đúng `{version}`, không tải output tài khoản kia. Nếu có mount exact checkpoint, điền WEIGHTS_ROOT, không cần API.
Gắn3Dataset: `csa-prompt-test-cache-v181` (share quyền cho Ngân), `vuthetam/mscoco-2014` (JPEG+JSON), `objectdetectionecache`. Không cần ROI134/175/visual-cache/UPT riêng, không train/cache lại.
CONTROL:normal raw+rerank; PROMPT:normal raw+rerank vàzero/shuffle(raw). Cùng checkpoint, giữROI/pairs/geometry/masks, donor ngoại ảnh khóahash42 theo từng loại slot. Beam5/max20, fixed .3decoder+.4CLIP+.3object, khôngBLIP/oracle/tuning hoặc fallback theo điểmTEST.
3h parent STOPcap, không phải ETA/quota; không tự chạy lại. Giữ output/5beams nếu scorer lỗi. Không optimizer/cập nhật trọng số. PriorTEST exposure và C10/P9 lịch XE ban đầu khác nhau được ghi trong kết quả. Zero/shuffle là độ nhạy khi suy luận, chưa phải đối chứng huấn luyện.
Tác giả chỉ kiểm static/NumPy/mock; model/CUDA/CLIP/scorer chưa được tác giả chạy. Gửi reviewZIP sau khi xong.
'''
        nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=True)),cells=[cell('markdown',intro),cell('code',setup),cell('code',old_code[1]),cell('code',execution)])
        p=HERE/f'CSA_PROMPT_TEST_{arm}_2GPU.ipynb';p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8');print(p.name,len(mapping),'pins',revision)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);make(p.parse_args().revision)
