"""Author source-pinned paired full10 XE notebooks; no model/API execution."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from make_smoke_notebook import cell,RUNNER
from make_full_xe_notebook import FILES as XE_FILES
from make_full_prompt_cache_notebook import FILES as PROMPT_FILES
from make_prompt_adapter_notebook import EXTERNAL

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PREFIX='experiments/csa_paper/'
FILES=list(dict.fromkeys(XE_FILES+PROMPT_FILES+['prompt_scratch_data.py','prompt_scratch_model.py','prompt_scratch_preflight.py',
    'test_prompt_scratch_data.py','test_prompt_scratch_model.py','run_prompt_scratch.py','make_prompt_scratch_notebooks.py','FULL_PROMPT_FRESH_V175_RESULT.json']))


def make(revision):
    revision=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    mapping={PREFIX+n:n for n in FILES};mapping.update({n:n for n in EXTERNAL})
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',revision+':'+n],cwd=ROOT)).hexdigest() for n in mapping}
    for arm in ('CONTROL','PROMPT'):
        setup=f'''import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile,shutil
from pathlib import Path
ARM={arm!r}
FULL_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-full-roi-karpathy-v134')
BANK_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-semantic-banks-v130')
PROMPT_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-prompt-full-karpathy-v175')
RESUME_ROOT='' # Fresh run: empty. Same-arm continuation: entire previous output folder.
OUTPUT=Path('/kaggle/working/csa_prompt_scratch_'+ARM+'_xe10')
WALL_HOURS=10.5 # STOPcap within reported15h; not ETA or guaranteed session/quota.
PARENT_DEADLINE=time.monotonic()+WALL_HOURS*3600
for p in (FULL_ROOT,BANK_ROOT,PROMPT_ROOT):
    if not p.is_dir():raise FileNotFoundError('Attach all3Datasets or correct mount: '+str(p))
if RESUME_ROOT and not Path(RESUME_ROOT).is_dir():raise FileNotFoundError('Mount complete same-arm resume folder')
if OUTPUT.exists():raise RuntimeError('Preserve existing output; choose fresh runtime/output path')
if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<8*1024**3:raise RuntimeError('Need8GiB free for stage/checkpoints')
OUTPUT.mkdir(parents=True);SOURCE=Path(tempfile.mkdtemp(prefix='csa-prompt-xe10-',dir='/tmp'));STAGE=SOURCE/'stage';LOG=OUTPUT/'notebook.log'
REVISION={revision!r}
MAPPING={mapping!r}
HASHES={hashes!r}
for original,destination in MAPPING.items():
    remaining=PARENT_DEADLINE-time.monotonic()
    if remaining<=0:raise TimeoutError('Parent cap during source download')
    raw=urllib.request.urlopen('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+REVISION+'/'+original,timeout=min(90,remaining)).read()
    if hashlib.sha256(raw).hexdigest()!=HASHES[original]:raise ValueError('Source SHA differs:'+original)
    p=SOURCE/destination;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=REVISION,sha256=HASHES),indent=2),encoding='utf-8')
os.environ['OMP_NUM_THREADS']='1';os.environ['PYTHONPATH']=str(SOURCE)
'''
        runner=RUNNER.replace("print('Running:',","if time.monotonic()>=PARENT_DEADLINE:raise TimeoutError('Parent wall cap')\n    timeout=min(timeout,max(.01,PARENT_DEADLINE-time.monotonic()))\n    print('Running:',").replace('Smoke active;','Prompt XE active;').replace('Smoke wall cap exceeded','Prompt XE wall cap exceeded')
        final='''complete=False
common=['--bank-root',BANK_ROOT,'--prompt-root',PROMPT_ROOT,'--stage',STAGE,'--output',OUTPUT,'--arm',ARM]
launch=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0']
try:
    run([sys.executable,'-u',SOURCE/'test_prompt_adapter_data.py'],timeout=120)
    run([sys.executable,'-u',SOURCE/'test_prompt_scratch_data.py'],timeout=120)
    # Neural fixtures run here on Kaggle only; no trained weights survive these processes.
    run([sys.executable,'-u',SOURCE/'test_prompt_adapter_model.py'],timeout=180)
    run([sys.executable,'-u',SOURCE/'test_prompt_scratch_model.py'],timeout=180)
    run([sys.executable,'-m','pip','install','-q','pycocoevalcap==1.2'],timeout=300)
    run([sys.executable,'-u',SOURCE/'run_prompt_scratch.py','prepare','--full-root',FULL_ROOT,*common],timeout=1800)
    run([sys.executable,'-u',SOURCE/'run_prompt_scratch.py','metric_preflight',*common],timeout=300)
    run([*launch,SOURCE/'prompt_scratch_preflight.py',*common],timeout=1200)
    guard=json.loads((OUTPUT/'prompt_preflight_result.json').read_text(encoding='utf-8'))
    if not guard['passed'] or guard['cases']!=4 or guard['arm']!=ARM:raise RuntimeError('Missing exact-arm DDP guards')
    remaining=PARENT_DEADLINE-time.monotonic()
    if remaining<600:raise TimeoutError('No training budget after preparation')
    run([*launch,SOURCE/'run_prompt_scratch.py','worker',*common,'--resume-root',RESUME_ROOT,'--wall-seconds',remaining-60],timeout=remaining)
    result=OUTPUT/'result.json'
    complete=result.is_file() and json.loads(result.read_text(encoding='utf-8')).get('status')=='PROMPT_SCRATCH_XE10_COMPLETE'
    if not complete:raise RuntimeError('Stopped before10epochs; preserve ENTIRE output for same-arm resume')
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,arm=ARM,requested_epochs=10,test_used=False)),encoding='utf-8')
    REVIEW=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(REVIEW,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.log','.txt'):z.write(p,str(p.relative_to(OUTPUT)))
    print('SEND REVIEW:',REVIEW,'\\nKEEP ENTIRE OUTPUT:',OUTPUT,flush=True)
    for n in ('checkpoint.bin','best.bin'):
        if (OUTPUT/n).is_file():print('KEEP CHECKPOINT:',OUTPUT/n,flush=True)
    print('COMPLETE10' if complete else 'INCOMPLETE; preserve output/logs, no automatic restart.',flush=True)
'''
        intro=f'''# CSA {arm} — XE từ đầu, full Karpathy10epoch
**T4×2, Internet ON; Save Version → Save & Run All.** Gắn3Datasets: ROI134, semantic-banks130 và prompt-full175 (path ở cell đầu). Không API Secret/ảnhJPEG/COCOJSON/checkpoint caption cũ.
Hai notebook độc lập, chạy song song mỗi notebook2GPU. Cùng fresh commonseed42, bank danh từ/động từ CLIP giữ nguyên, global32caption, groupedimage-sharing, Adam(.9,.99), clip1, FP32. Warmup3epoch8e-6→1e-4;10epoch177110updates/5667470captionevents. NhánhCONTROL khóa/bypass adapter; PROMPT dùng object–relation prompt đúng instance/ROI, zero-output ởstep0 rồi học từupdate1. Ảnh thiếu mappedprompt có residual0; không coi là nhãn âm.
Mỗi epoch đánh giá đủ5000VALID với originalbeam5/max20/CIDEr/NLL và lưu captions. CHAIR/BLEU/ROUGE chấm từ captions đã lưu sautrain; chưa có TEST/SCST/rerank ở bước này. Chỉso với nhánhCONTROL cùng lượt; CSA10cũ là tham khảo do encoderdropout-sharing khác.
4DDP guards cho loss/gradient sharing ởclosed/opened +batch32/tail27 và neuralfixtures phải PASS trướctrain. Guard weights bị bỏ; mainworker khởi tạo mới. So `common_init_sha256` của hai output khi review. Không cam kết tăngCIDEr hoặc hội tụ10epochs.
Parentcap10.5h là giới hạn dừng, không phảiETA/quota. Checkpoint latest chứamodel/Adam/cursor/RNG mỗi1000updates vàtrước/sauVALID; best theoVALIDCIDEr/tieearlier. **Giữ ENTIRE output**, không chỉZIP. Nếu dừng, gửireview và mount toàn bộ output vàoRESUME_ROOT cùngnhánh, cùngnotebook/source trên phiên mới; khôngkhởiđộng lại từ đầu tựđộng. Dữ liệu cache không cần tạo lại. No author-localmodelrun.
'''
        nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=True)),cells=[cell('markdown',intro),cell('code',setup),cell('code',runner),cell('code',final)])
        for c in nb['cells']:
            if c['cell_type']=='code':ast.parse(c['source'])
        path=HERE/f'CSA_PROMPT_SCRATCH_{arm}_XE10_2GPU.ipynb';path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
        print(path.name,path.stat().st_size,'bytes;',len(hashes),'source pins;',revision)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);a=p.parse_args();make(a.revision)
