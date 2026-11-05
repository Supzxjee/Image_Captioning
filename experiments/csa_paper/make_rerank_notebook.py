"""Author pinned reranking notebook only; no API/model execution."""
import argparse
import hashlib
import json
import subprocess
from make_smoke_notebook import HERE,ROOT,PREFIX,cell,RUNNER

FILES=['rerank_data.py','run_rerank.py','test_rerank_data.py','make_rerank_notebook.py','make_smoke_notebook.py',
    'candidate_data.py','real_bank_data.py','saved_quality_data.py','CANDIDATE_BASELINE161.json.gz','RERANK_VALID_PROTOCOL.md']
EXTERNAL=['captioning/__init__.py','captioning/multiscore_reranking.py','captioning/object_consistency.py']


def make(revision):
    commit=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    mapping={PREFIX+n:n for n in FILES};mapping.update({n:n for n in EXTERNAL})
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT)).hexdigest() for n in mapping}
    setup=f'''
import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile,shutil
from pathlib import Path
COCO_ROOT=Path('/kaggle/input/datasets/vuthetam/mscoco-2014') # Must contain original JPEGs, not only features.
DATASET_JSON=COCO_ROOT/'dataset_coco.json'
DETECTIONS=Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
CANDIDATE_ROOT='' # Optional mounted version163 output; otherwise download the two small candidate files via API.
KERNEL_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/163'
SECRET_NAME='codex' # Actual ENABLED API Secret name, e.g. KAGGLE_API_TOKEN.
WALL_HOURS=2 # STOP cap only. Last user quota16h; no automatic retry.
OUTPUT=Path('/kaggle/working/csa_ce161_rerank_valid5000')
for p in (COCO_ROOT,DATASET_JSON,DETECTIONS):
    if not p.exists():raise FileNotFoundError('Attach rawCOCO + datasetJSON + detector Dataset before GPU: '+str(p))
if OUTPUT.exists():raise RuntimeError('Preserve previous scores/logs; fresh output/session required.')
if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<6*1024**3:raise RuntimeError('Need6GiB free for scoring-model snapshots/dependencies.')
GPU=json.loads(subprocess.check_output([sys.executable,'-c',"import json,torch;print(json.dumps(dict(built=torch.backends.cuda.is_built(),available=torch.cuda.is_available(),count=torch.cuda.device_count())))"],text=True,timeout=60))
if not GPU['built'] or not GPU['available'] or GPU['count']!=2:raise RuntimeError('Select T4x2 and restart runtime.')
OUTPUT.mkdir();SOURCE=Path(tempfile.mkdtemp(prefix='csa-rerank-'));LOG=OUTPUT/'notebook.log'
START=time.monotonic();DEADLINE=START+WALL_HOURS*3600
COMMIT={commit!r}
HASHES={hashes!r}
MAPPING={mapping!r}
for n,h in HASHES.items():
    raw=urllib.request.urlopen(f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{{COMMIT}}/{{n}}',timeout=90).read()
    if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('Pinned source differs: '+n)
    p=SOURCE/MAPPING[n];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=COMMIT,sha256=HASHES,gpu=GPU),indent=2),encoding='utf-8')
os.environ.update(OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',USE_TF='0',USE_FLAX='0',HF_HUB_DISABLE_XET='1')
CONFIG=SOURCE/'config.json';CHAIR=SOURCE/'chair';DEPS=SOURCE/'hf_deps'
'''
    execute=r'''
complete=False
try:
    run([sys.executable,'-u',SOURCE/'test_rerank_data.py'],timeout=120)
    if CANDIDATE_ROOT:
        cache=Path(CANDIDATE_ROOT)
    else:
        if KERNEL_VERSION!='ducanh2403/2-vi-cai-kia-qua-nhieu/163':raise ValueError('Exact163 only; no latest fallback.')
        run([sys.executable,'-m','pip','install','-q','kaggle==2.2.5'],timeout=300)
        routing="""
from types import SimpleNamespace
from unittest.mock import patch
from kaggle.api.kaggle_api_extended import KaggleApi
import tempfile
api=KaggleApi()
with tempfile.TemporaryDirectory() as directory,patch.object(api,'build_kaggle_client') as factory:
    endpoint=factory.return_value.__enter__.return_value.kernels.kernels_api_client.list_kernel_session_output
    endpoint.return_value=SimpleNamespace(files=[],next_page_token=None,log=None)
    api.kernels_output('ducanh2403/2-vi-cai-kia-qua-nhieu/163',path=directory)
    request=endpoint.call_args.args[0]
    assert (request.user_name,request.kernel_slug,request.version_label)==('ducanh2403','2-vi-cai-kia-qua-nhieu','v163')
print('Exact163 API routing mock PASS; no network.',flush=True)
"""
        run([sys.executable,'-c',routing],timeout=60)
        from kaggle_secrets import UserSecretsClient
        token=UserSecretsClient().get_secret(SECRET_NAME).strip()
        if not token:raise RuntimeError('Enabled API Secret empty.')
        env=os.environ.copy()
        for key in ('KAGGLE_API_TOKEN','KAGGLE_USERNAME','KAGGLE_KEY'):env.pop(key,None)
        env['KAGGLE_API_TOKEN']=token;env['PYTHONUNBUFFERED']='1'
        cache=OUTPUT/'candidates163';cache.mkdir()
        command=[sys.executable,'-c','from kaggle.cli import main; main()','kernels','output',KERNEL_VERSION,'-p',str(cache),
            '--file-pattern',r'(^|/)candidates_rank[01]\.jsonl$']
        print('Download only two small beam files from exact163; no captioner checkpoint.',flush=True)
        subprocess.run(command,env=env,check=True,timeout=900)
        env.pop('KAGGLE_API_TOKEN',None);del env,token
    cfg=dict(candidate_root=str(cache),coco_root=str(COCO_ROOT),dataset_json=str(DATASET_JSON),detections=str(DETECTIONS),
        output=str(OUTPUT),chair=str(CHAIR),model_cache=str(SOURCE/'model_cache'))
    CONFIG.write_text(json.dumps(cfg),encoding='utf-8')
    run([sys.executable,'-u',SOURCE/'run_rerank.py','prepare','--config',CONFIG],timeout=min(900,DEADLINE-time.monotonic()))
    run([sys.executable,'-m','pip','install','-q','pycocoevalcap==1.2','nltk==3.9.2','future'],timeout=300)
    run([sys.executable,'-m','pip','install','-q','--no-deps','git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'],timeout=300)
    run([sys.executable,'-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"],timeout=300)
    run(['java','-version'],timeout=30)
    run(['git','init',CHAIR],timeout=60)
    run(['git','-C',CHAIR,'remote','add','origin','https://github.com/Maxlinn/CHAIR-metric-standalone.git'],timeout=60)
    run(['git','-C',CHAIR,'fetch','--depth','1','origin','4087a26211aa2339b9a76307cb8f0321ef691d0a'],timeout=300)
    run(['git','-C',CHAIR,'checkout','--detach','FETCH_HEAD'],timeout=60)
    run([sys.executable,'-u',SOURCE/'run_rerank.py','metric_preflight','--config',CONFIG],timeout=min(600,DEADLINE-time.monotonic()))
    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS,'transformers==4.57.1','tokenizers==0.22.1',
        'huggingface-hub==0.36.2','safetensors==0.6.2'],timeout=300)
    os.environ['PYTHONPATH']=str(DEPS)+os.pathsep+str(SOURCE)
    run([sys.executable,'-c',"from importlib.metadata import version;expected={'transformers':'4.57.1','tokenizers':'0.22.1','huggingface-hub':'0.36.2','safetensors':'0.6.2'};assert all(version(k)==v for k,v in expected.items());print('Isolated pinned HF dependencies PASS')"],timeout=60)
    if DEADLINE-time.monotonic()<2400:raise TimeoutError('Insufficient model-download/inference/metric margin; preserve cache.')
    run([sys.executable,'-u',SOURCE/'run_rerank.py','prefetch','--config',CONFIG],timeout=min(1200,DEADLINE-time.monotonic()-1800))
    cfg['worker_seconds']=DEADLINE-time.monotonic()-1200
    if cfg['worker_seconds']<600:raise TimeoutError('Insufficient inference/metric reserve; preserve cache.')
    CONFIG.write_text(json.dumps(cfg),encoding='utf-8')
    launch=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0']
    run([*launch,SOURCE/'run_rerank.py','worker','--config',CONFIG],timeout=cfg['worker_seconds']+60)
    run([sys.executable,'-u',SOURCE/'run_rerank.py','score','--config',CONFIG],timeout=min(1200,DEADLINE-time.monotonic()))
    complete=json.loads((OUTPUT/'result.json').read_text(encoding='utf-8'))['status']=='CSA_FIXED_RERANK_VALID_COMPLETE'
    if not complete:raise RuntimeError('Incomplete fixed-policy VALID comparison')
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,training=False,test_used=False,oracle_used_for_selection=False,actual_reranking=True)),encoding='utf-8')
    archive=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt'):z.write(p,str(p.relative_to(OUTPUT)))
    print('SEND REVIEW:',archive,flush=True)
    print('KEEP WHOLE SCORE OUTPUT:',OUTPUT,flush=True)
    print('COMPLETE fixed three-policy VALID comparison' if complete else 'INCOMPLETE; preserve scores/logs, no automatic retry.',flush=True)
'''
    intro='''# CSA161 — chọn lại caption bằng CLIP/đối tượng và BLIP ITM
    **T4x2**, Internet ON, Save Version → Save & Run All. Chạy trên notebook của bạn.
    Attach **mscoco-2014 có ảnh JPEG + dataset_coco.json**, và **objectdetectionecache**.
    Không cần ROI134, bank hay checkpointCSA. Sửa đường dẫn cell đầu nếu mount khác.
    Bật Secret API (mặc định `codex`): tải đúng hai file ứng viên từ **version163**, không lấy latest.
    Nếu đã mount output163, điền CANDIDATE_ROOT để bỏ qua API.

    Cùng5000ảnh/25000câu: decoder rank0; công thức cũ .3decoder+.4CLIP+.3đối tượng;
    BLIP ITM chọn margin đúng-cặp lớn nhất. Không dùng caption tham chiếu hay oracle khi chọn.
    Chấm CIDEr/BLEU4/ROUGE/CHAIR/recall/độ dài sau khi chọn. Kiểm lại baseline trước tải mô hình.
    Không train hoặc sinh lại caption. CLIP/BLIP vẫn cần chạy inference trên ảnh gốc.
    Mô hình BLIP là checkpoint ngoài đã fine-tune COCO, chưa audit độc lập split huấn luyện.
    Chỉ tái dùng công thức CLIP cũ: lượt này dùng processor chính thức trên JPEG, khác nguồn CLS cũ.

    HF dependencies cài vào thư mục tạm riêng, không thay Torch hay HF toàn cục.
   2h là giới hạn dừng, không phải dự đoán quota. Dừng sớm nếu thiếu ảnh/model/lệch baseline hoặc
    tốc độ đo không đủ. Giữ điểm từng câu để tái sử dụng; không tự chạy lại khi lỗi.
    Agent chỉ kiểm metadata/NumPy/static; GPU/model/scorer do người dùng chạy.
    '''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        kaggle=dict(isInternetEnabled=True,isGpuEnabled=True)),cells=[cell('markdown',intro),cell('code',setup),
        cell('code',RUNNER.replace('Smoke active;','Reranking active;')),cell('code',execute)])
    path=HERE/'CSA_CE161_RERANK_VALID5000_2GPU.ipynb';path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(path.name,path.stat().st_size)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',default='HEAD');make(p.parse_args().revision)
