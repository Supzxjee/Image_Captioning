"""Author pinned candidate notebook; never executes notebook/models/API."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from make_smoke_notebook import HERE,ROOT,PREFIX,cell,RUNNER

FILES=['candidate_data.py','run_candidates.py','test_candidate_data.py','make_candidate_notebook.py',
       'CANDIDATE_BASELINE161.json.gz','CANDIDATE_DIAGNOSTIC_PROTOCOL.md','SAVE_FULL_ROI_V134_PACKAGE.json',
       'model.py','real_bank_data.py','saved_quality_data.py','make_smoke_notebook.py']


def make(revision):
    commit=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+PREFIX+n],cwd=ROOT)).hexdigest() for n in FILES}
    setup=f'''
import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile,shutil
from pathlib import Path
FULL_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-full-roi-karpathy-v134')
CHECKPOINT_ROOT='' # Optional local mount/file. Empty downloads exact SaveVersion161 from your notebook.
KERNEL_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/161'
SECRET_NAME='codex' # Set to the actual ENABLED Kaggle Secret name, e.g. KAGGLE_API_TOKEN.
WALL_HOURS=2 # STOP cap only, not expected runtime/billing. Latest user quota16h.
OUTPUT=Path('/kaggle/working/csa_ce161_candidates_valid5000')
if not FULL_ROOT.is_dir():raise FileNotFoundError('Attach fullROI134 Dataset before GPU: '+str(FULL_ROOT))
if OUTPUT.exists():raise RuntimeError('Preserve prior output; fresh output/session required.')
if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<2*1024**3:raise RuntimeError('Need2GiB free.')
GPU=json.loads(subprocess.check_output([sys.executable,'-c',"import json,torch;print(json.dumps(dict(built=torch.backends.cuda.is_built(),available=torch.cuda.is_available(),count=torch.cuda.device_count())))"],text=True,timeout=60))
if not GPU['built'] or not GPU['available'] or GPU['count']!=2:raise RuntimeError('Select T4x2 and restart runtime.')
OUTPUT.mkdir();SOURCE=Path(tempfile.mkdtemp(prefix='csa-candidates-'));LOG=OUTPUT/'notebook.log'
START=time.monotonic();DEADLINE=START+WALL_HOURS*3600
COMMIT={commit!r}
HASHES={hashes!r}
for n,h in HASHES.items():
    raw=urllib.request.urlopen(f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{{COMMIT}}/{PREFIX}{{n}}',timeout=90).read()
    if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Source differs: '+n)
    (SOURCE/n).write_bytes(raw)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=COMMIT,sha256=HASHES,gpu=GPU),indent=2),encoding='utf-8')
os.environ['OMP_NUM_THREADS']='1'
CONFIG=SOURCE/'config.json';CHAIR=SOURCE/'chair'
'''
    execute=r'''
complete=False
try:
    run([sys.executable,'-u',SOURCE/'test_candidate_data.py'],timeout=120)
    # Metadata fixture only; actual model is loaded exclusively by worker subprocesses.
    sys.path.insert(0,str(SOURCE))
    from candidate_data import CHECKPOINT_SHA
    from real_bank_data import sha
    if CHECKPOINT_ROOT:
        root=Path(CHECKPOINT_ROOT)
        found=[root] if root.is_file() else list(root.rglob('best.bin'))+list(root.rglob('checkpoint.bin'))
    else:
        if KERNEL_VERSION!='ducanh2403/2-vi-cai-kia-qua-nhieu/161':raise ValueError('No version/latest fallback.')
        from kaggle_secrets import UserSecretsClient
        token=UserSecretsClient().get_secret(SECRET_NAME).strip()
        if not token:raise RuntimeError('Enabled API Secret is empty.')
        environment=os.environ.copy()
        for key in ('KAGGLE_API_TOKEN','KAGGLE_USERNAME','KAGGLE_KEY'):environment.pop(key,None)
        environment['KAGGLE_API_TOKEN']=token;environment['PYTHONUNBUFFERED']='1'
        download=OUTPUT/'checkpoint161';download.mkdir()
        print('Downloading ONLY best.bin from exact161; no Dataset creation.',flush=True)
        command=[sys.executable,'-m','kaggle','kernels','output',KERNEL_VERSION,'-p',str(download),'--file-pattern',r'(^|/)best\.bin$']
        subprocess.run(command,env=environment,check=True,timeout=900)
        environment.pop('KAGGLE_API_TOKEN',None);del token,environment
        found=list(download.rglob('best.bin'))
    matches=[p for p in found if sha(p)==CHECKPOINT_SHA]
    if not matches:raise ValueError('Missing exact SCST+CE END161 weights; preserve output, no substitute.')
    checkpoint=sorted(matches)[0]
    (OUTPUT/'checkpoint_source.json').write_text(json.dumps(dict(source=KERNEL_VERSION if not CHECKPOINT_ROOT else CHECKPOINT_ROOT,sha256=CHECKPOINT_SHA)),encoding='utf-8')
    run([sys.executable,'-m','pip','install','-q','pycocoevalcap==1.2','nltk==3.9.2','future'],timeout=300)
    run([sys.executable,'-m','pip','install','-q','--no-deps','git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'],timeout=300)
    run([sys.executable,'-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"],timeout=300)
    run(['java','-version'],timeout=30)
    run(['git','init',CHAIR],timeout=60)
    run(['git','-C',CHAIR,'remote','add','origin','https://github.com/Maxlinn/CHAIR-metric-standalone.git'],timeout=60)
    run(['git','-C',CHAIR,'fetch','--depth','1','origin','4087a26211aa2339b9a76307cb8f0321ef691d0a'],timeout=300)
    run(['git','-C',CHAIR,'checkout','--detach','FETCH_HEAD'],timeout=60)
    cfg=dict(full_root=str(FULL_ROOT),checkpoint=str(checkpoint),output=str(OUTPUT),stage=str(SOURCE/'stage'),chair=str(CHAIR))
    CONFIG.write_text(json.dumps(cfg),encoding='utf-8')
    run([sys.executable,'-u',SOURCE/'run_candidates.py','prepare','--config',CONFIG],timeout=min(900,DEADLINE-time.monotonic()))
    run([sys.executable,'-u',SOURCE/'run_candidates.py','metric_preflight','--config',CONFIG],timeout=min(600,DEADLINE-time.monotonic()))
    cfg['worker_seconds']=DEADLINE-time.monotonic()-1200
    if cfg['worker_seconds']<600:raise TimeoutError('Insufficient decode/metrics reserve; preserve output.')
    CONFIG.write_text(json.dumps(cfg),encoding='utf-8')
    launch=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0']
    run([*launch,SOURCE/'run_candidates.py','worker','--config',CONFIG],timeout=cfg['worker_seconds']+60)
    run([sys.executable,'-u',SOURCE/'run_candidates.py','score','--config',CONFIG],timeout=min(1200,DEADLINE-time.monotonic()))
    result=json.loads((OUTPUT/'result.json').read_text(encoding='utf-8'))
    complete=result['status']=='CSA_CANDIDATE_DIAGNOSTIC_COMPLETE'
    if not complete:raise RuntimeError('Incomplete diagnostic')
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,training=False,test_used=False,actual_reranking=False)),encoding='utf-8')
    archive=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt'):z.write(p,str(p.relative_to(OUTPUT)))
    print('SEND REVIEW:',archive,flush=True)
    print('KEEP ENTIRE OUTPUT:',OUTPUT,flush=True)
    print('COMPLETE — oracle is diagnostic only; no trained/reranked result.' if complete else 'INCOMPLETE — preserve logs; no automatic retry.',flush=True)
'''
    intro='''# CSA SCST+CE161 — kiểm tra năm caption trước reranking
    Chạy trên notebook của bạn, GPU **T4x2**, Internet ON. Attach **csa-full-roi-karpathy-v134**.
    Checkpoint chưa lưu Dataset: notebook tải **best.bin từ đúng version161**; bật Secret API
    và sửa SECRET_NAME trong cell đầu nếu tên không phải `codex`. Không cần Dataset checkpoint mới.

    Giữ nguyên mô hình và beam5/max20. Lấy đủ25.000 ứng viên trên VALID5000; kiểm câu đầu
    khớp kết quả161, chấm từng beam và phép chọn lý tưởng theo CIDEr tham chiếu.
    **Oracle dùng đáp án nên chỉ là chẩn đoán dư địa, không phải kết quả reranker triển khai.**
    Báo cả CHAIR/recall, số câu quá ngắn, độ đa dạng. Giữ output ứng viên để dùng cho reranker sau.
    Không train/TEST/ITM, không tải ảnh hay chạy lại CLIP, không stage TRAIN.2h là giới hạn dừng,
    không phải dự báo thời gian hoặc quota. User tự chạy; author chỉ kiểm metadata/NumPy/static.
    Save Version → Save & Run All. Gửi ZIP review sau khi chạy.
    '''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        kaggle=dict(isInternetEnabled=True,isGpuEnabled=True)),cells=[cell('markdown',intro),cell('code',setup),
        cell('code',RUNNER.replace('Smoke active;','Candidate diagnostic active;')),cell('code',execute)])
    path=HERE/'CSA_CE161_CANDIDATE_DIAGNOSTIC_2GPU.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8');print(path.name,path.stat().st_size)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',default='HEAD');make(p.parse_args().revision)
