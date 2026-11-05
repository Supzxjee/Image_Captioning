"""Author pinned user-run TEST notebook; no execution/submission/model imports."""
import argparse
import hashlib
import json
import subprocess
from make_smoke_notebook import HERE,ROOT,PREFIX,cell,RUNNER

FILES=['rerank_test_data.py','run_rerank_test.py','test_rerank_test_data.py','make_rerank_test_notebook.py',
    'RERANK_TEST_PROTOCOL.md','RERANK_TEST_INPUTS.json','make_smoke_notebook.py','real_bank_data.py',
    'candidate_data.py','rerank_data.py','plain_test_data.py','full_xe_decode.py','saved_quality_data.py','model.py',
    'prepare_test_regions.py','prepare_real_assets.py','audit_assets.py','data.py','run_best9_decode.py',
    'best9_saved_text.json.gz','BEST9_SAVED_TEXT_PACKAGE.json','SAVE_XE139_PACKAGE.json','full_roi_review_text/input_receipt.json']
EXTERNAL=['captioning/__init__.py','captioning/multiscore_reranking.py','captioning/object_consistency.py']


def make(revision):
    commit=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    mapping={PREFIX+n:n for n in FILES};mapping.update({n:n for n in EXTERNAL})
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT)).hexdigest() for n in mapping}
    setup=f'''
import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile,shutil
from pathlib import Path
COCO_ROOT=Path('/kaggle/input/datasets/vuthetam/mscoco-2014')
DATASET_JSON=COCO_ROOT/'dataset_coco.json'
DETECTIONS=Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
CHECKPOINT_ROOT='' # Optional mounted root containing exact161 best.bin; otherwise download your own exact161.
TEST_ROOT='' # Optional mounted exact TEST cache; otherwise CPU restore from original raw H5, exact byte hashes required.
VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
SECRET_NAME='codex' # Enabled API Secret name. Change if yours is KAGGLE_API_TOKEN.
WALL_HOURS=2 # STOP cap, not ETA or billing. Last reported quota16h before VALID164.
OUTPUT=Path('/kaggle/working/csa_ce161_fixed_rerank_test5000_v2')
for p in (COCO_ROOT,DATASET_JSON,DETECTIONS):
    if not p.exists():raise FileNotFoundError('Attach the same2rawDatasets as164 before GPU: '+str(p))
if not TEST_ROOT and not VISUAL_CACHE.is_dir():raise FileNotFoundError('Add THIRD Dataset visual-cache for CPU TEST restoration: '+str(VISUAL_CACHE))
if OUTPUT.exists():raise RuntimeError('Preserve previous output/scores; fresh runtime/path required.')
if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<6*1024**3:raise RuntimeError('Need6GiB free for immutable inputs/CLIP assets.')
GPU=json.loads(subprocess.check_output([sys.executable,'-c',"import json,torch;print(json.dumps(dict(built=torch.backends.cuda.is_built(),available=torch.cuda.is_available(),count=torch.cuda.device_count())))"],text=True,timeout=60))
if not GPU['built'] or not GPU['available'] or GPU['count']!=2:raise RuntimeError('Select T4x2 and restart runtime.')
OUTPUT.mkdir();SOURCE=Path(tempfile.mkdtemp(prefix='csa-fixed-test-'));LOG=OUTPUT/'notebook.log'
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
    run([sys.executable,'-u',SOURCE/'test_rerank_test_data.py'],timeout=120)
    sys.path.insert(0,str(SOURCE))
    locked=json.loads((SOURCE/'RERANK_TEST_INPUTS.json').read_text(encoding='utf-8'))
    downloads=[]
    if not CHECKPOINT_ROOT:
        from plain_test_data import exact_file
        prior=Path('/kaggle/working/csa_ce161_fixed_rerank_test5000/inputs161')
        if prior.is_dir():
            try:
                reused=exact_file(prior,'best.bin',locked['checkpoint_sha256']);CHECKPOINT_ROOT=str(reused.parent)
                print('Reusing verified161weights already downloaded:',reused,flush=True)
            except ValueError:pass
    if not CHECKPOINT_ROOT:
        CHECKPOINT_ROOT=str(OUTPUT/'inputs161');Path(CHECKPOINT_ROOT).mkdir()
        downloads.append((locked['checkpoint_version'],CHECKPOINT_ROOT,r'(^|/)best\.bin$'))
    restore_cache=not bool(TEST_ROOT)
    if restore_cache:TEST_ROOT=str(OUTPUT/'restored_test_cache')
    if downloads:
        run([sys.executable,'-m','pip','install','-q','kaggle==2.2.5'],timeout=300)
        routing="""
from types import SimpleNamespace
from unittest.mock import patch
from kaggle.api.kaggle_api_extended import KaggleApi
import tempfile
api=KaggleApi()
for version in ('161',):
    with tempfile.TemporaryDirectory() as directory,patch.object(api,'build_kaggle_client') as factory:
        endpoint=factory.return_value.__enter__.return_value.kernels.kernels_api_client.list_kernel_session_output
        endpoint.return_value=SimpleNamespace(files=[],next_page_token=None,log=None)
        api.kernels_output('ducanh2403/2-vi-cai-kia-qua-nhieu/'+version,path=directory)
        request=endpoint.call_args.args[0]
        assert (request.user_name,request.kernel_slug,request.version_label)==('ducanh2403','2-vi-cai-kia-qua-nhieu','v'+version)
print('Exact own161 API routing mock PASS; no143cache download.',flush=True)
"""
        run([sys.executable,'-c',routing],timeout=60)
        from kaggle_secrets import UserSecretsClient
        token=UserSecretsClient().get_secret(SECRET_NAME).strip()
        if not token:raise RuntimeError('Enabled API Secret empty.')
        env=os.environ.copy()
        for key in ('KAGGLE_API_TOKEN','KAGGLE_USERNAME','KAGGLE_KEY'):env.pop(key,None)
        env['KAGGLE_API_TOKEN']=token;env['PYTHONUNBUFFERED']='1'
        for version,directory,pattern in downloads:
            if version!='ducanh2403/2-vi-cai-kia-qua-nhieu/161':raise ValueError('No latest/version fallback.')
            print('Downloading frozen own input',version,flush=True)
            command=[sys.executable,'-c','from kaggle.cli import main; main()','kernels','output',version,'-p',directory,'--file-pattern',pattern]
            subprocess.run(command,env=env,check=True,timeout=900)
        env.pop('KAGGLE_API_TOKEN',None);del env,token
    cfg=dict(checkpoint_root=CHECKPOINT_ROOT,test_root=TEST_ROOT,coco_root=str(COCO_ROOT),dataset_json=str(DATASET_JSON),detections=str(DETECTIONS),
        output=str(OUTPUT),chair=str(CHAIR),model_cache=str(SOURCE/'model_cache'),visual_cache=str(VISUAL_CACHE),
        test_cache_mode='CPU_original_H5_exact_byte_match' if restore_cache else 'verified_mounted')
    CONFIG.write_text(json.dumps(cfg),encoding='utf-8')
    if restore_cache:
        remaining=DEADLINE-time.monotonic()-1800
        if remaining<1800:raise TimeoutError('Insufficient CPU restoration/inference reserve; preserve weights.')
        run([sys.executable,'-u',SOURCE/'run_rerank_test.py','rebuild_cache','--config',CONFIG],timeout=min(3720,remaining))
    run([sys.executable,'-u',SOURCE/'run_rerank_test.py','prepare','--config',CONFIG],timeout=min(900,DEADLINE-time.monotonic()))
    run([sys.executable,'-m','pip','install','-q','pycocoevalcap==1.2','nltk==3.9.2','future'],timeout=300)
    run([sys.executable,'-m','pip','install','-q','--no-deps','git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'],timeout=300)
    run([sys.executable,'-c',"import nltk;from pattern.en import singularize;from pycocoevalcap.cider.cider import Cider;from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"],timeout=300)
    run(['java','-version'],timeout=30)
    run(['git','init',CHAIR],timeout=60)
    run(['git','-C',CHAIR,'remote','add','origin','https://github.com/Maxlinn/CHAIR-metric-standalone.git'],timeout=60)
    run(['git','-C',CHAIR,'fetch','--depth','1','origin','4087a26211aa2339b9a76307cb8f0321ef691d0a'],timeout=300)
    run(['git','-C',CHAIR,'checkout','--detach','FETCH_HEAD'],timeout=60)
    assert hashlib.sha256((CHAIR/'chair.pkl').read_bytes()).hexdigest()=='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3'
    (OUTPUT/'metric_preflight.json').write_text(json.dumps(dict(passed=True,dependencies_only=True,no_test_metrics_yet=True)),encoding='utf-8')
    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS,'transformers==4.57.1','tokenizers==0.22.1','huggingface-hub==0.36.2','safetensors==0.6.2'],timeout=300)
    os.environ['PYTHONPATH']=str(DEPS)+os.pathsep+str(SOURCE)
    run([sys.executable,'-c',"from importlib.metadata import version;expected={'transformers':'4.57.1','tokenizers':'0.22.1','huggingface-hub':'0.36.2','safetensors':'0.6.2'};assert all(version(k)==v for k,v in expected.items());print('Isolated pinned HF dependencies PASS')"],timeout=60)
    if DEADLINE-time.monotonic()<2400:raise TimeoutError('Insufficient model/decode/scoring reserve; preserve inputs.')
    run([sys.executable,'-u',SOURCE/'run_rerank_test.py','prefetch','--config',CONFIG],timeout=min(1200,DEADLINE-time.monotonic()-1800))
    launch=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0']
    cfg['phase_seconds']=DEADLINE-time.monotonic()-1800
    if cfg['phase_seconds']<600:raise TimeoutError('Insufficient TEST decode and scoring reserve.')
    CONFIG.write_text(json.dumps(cfg),encoding='utf-8')
    run([*launch,SOURCE/'run_rerank_test.py','decode','--config',CONFIG],timeout=cfg['phase_seconds']+60)
    run([sys.executable,'-u',SOURCE/'run_rerank_test.py','text_preflight','--config',CONFIG],timeout=180)
    cfg['phase_seconds']=DEADLINE-time.monotonic()-900
    if cfg['phase_seconds']<300:raise TimeoutError('Insufficient CLIP/metric reserve; beams preserved.')
    CONFIG.write_text(json.dumps(cfg),encoding='utf-8')
    run([*launch,SOURCE/'run_rerank_test.py','worker','--config',CONFIG],timeout=cfg['phase_seconds']+60)
    run([sys.executable,'-u',SOURCE/'run_rerank_test.py','score','--config',CONFIG],timeout=min(900,DEADLINE-time.monotonic()))
    complete=json.loads((OUTPUT/'result.json').read_text(encoding='utf-8'))['status']=='CSA161_FIXED_RERANK_TEST5000_COMPLETE'
    if not complete:raise RuntimeError('Incomplete locked TEST evaluation')
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,training=False,test_used=True,tuned_on_test=False)),encoding='utf-8')
    archive=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file() and p.suffix in ('.json','.jsonl','.log','.txt'):z.write(p,str(p.relative_to(OUTPUT)))
    print('SEND REVIEW:',archive,flush=True)
    print('KEEP WHOLE OUTPUT:',OUTPUT,flush=True)
    print('COMPLETE frozen same-checkpoint TEST5000 comparison' if complete else 'INCOMPLETE; preserve partial beams/scores/logs; no automatic retry.',flush=True)
'''
    intro='''# CSA161 — TEST5000 có và không reranking, cấu hình đã khóa
    Chạy trên **notebook của bạn**, **T4x2**, Internet ON, Save Version → Save & Run All.
    Attach3Datasets: **mscoco-2014 có JPEG + dataset_coco.json**, **objectdetectionecache**, **visual-cache gốc**.
    Bật Secret API (mặc định `codex`, sửa cell đầu nếu tên khác).
    Tái dùng best.bin161 đã tải trong phiên lỗi nếu còn, hoặc tải đúng161. Giữ output cũ, ghi folder **_v2**.
    Version143 không trả file cache: tạo lại TESTROI bằng CPU từ visual-cache H5 gốc, dùng đúng mã cũ;
    bắt buộc NPZ/metadata/vocab khớp cả ba SHA đã chốt. Không chạy lại CLIP hay train để tạo cache.
    Nếu đã có cache TEST đúng SHA được mount, điền TEST_ROOT để bỏ qua CPU restore/visual-cache.
    Không cần Dataset checkpoint mới, ROI134, bank hay BLIP. CPU pooling cũ từng khoảng23phút, không đảm bảo thời gian.

    Bộ ứng viên163 là VALID nên không dùng làm TEST. Lượt này sinh5câu cho5000TESTảnh một lần,
    rồi so cùng checkpoint: decoder rank0 và công thức cũ .3decoder+.4CLIP+.3đối tượng.
    Giữ nguyên beam5/max20, CLIP/processor/hệ số như VALID164; không train/tune/thay checkpoint.
    Tham chiếu chỉ dùng chấm điểm sau chọn. Báo cả hai cách dù kết quả tăng hay giảm.
    CIDEr/BLEU4/ROUGE/CHAIR/recall/độ dài; không dùng oracle hay đo loss mới cho reranking.
    TEST đã có các lượt đánh giá trước trong dự án, không gọi là tập kiểm tra chưa từng xem.

    Giới hạn dừng2h, không phải dự báo quota; đo tốc độ từng phase, giữ kết quả dở khi không đủ.
    HF dependencies trong thư mục tạm riêng, không thay Torch. Agent chỉ kiểm dữ liệu/NumPy/static;
    việc tải model, chạy GPU và scorer do người dùng thực hiện. Gửi ZIP review sau chạy.
    '''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        kaggle=dict(isInternetEnabled=True,isGpuEnabled=True)),cells=[cell('markdown',intro),cell('code',setup),
        cell('code',RUNNER.replace('Smoke active;','Frozen TEST active;')),cell('code',execute)])
    path=HERE/'CSA_CE161_FIXED_RERANK_TEST5000_2GPU.ipynb';path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8');print(path.name,path.stat().st_size)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',default='HEAD');make(p.parse_args().revision)
