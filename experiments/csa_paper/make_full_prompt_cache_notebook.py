"""Author CPU-only full Karpathy prompt preparation; no model, API or notebook execution."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
from make_smoke_notebook import cell,RUNNER
from make_prompt_adapter_notebook import FILES as AUDIT_FILES,EXTERNAL

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PREFIX='experiments/csa_paper/'
FILES=AUDIT_FILES+['prompt_text_cache.py','test_prompt_text_cache_data.py','make_prompt_text_cache_notebook.py',
    'make_smoke_notebook.py','RERANK_TEST_INPUTS.json','RECOVER_PROMPT_TEXT_167_CELL.py',
    'full_prompt_cache_data.py','run_full_prompt_cache.py','test_full_prompt_cache_data.py',
    'make_full_prompt_cache_notebook.py','PROMPT_SCRATCH_XE_PLAN.md','recover_full_prompt_mapping.py']


def make(revision,recover=False,fresh=False):
    if recover and fresh:raise ValueError('Choose recovery OR fresh rebuild')
    revision=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    mapping={PREFIX+n:n for n in FILES};mapping.update({n:n for n in EXTERNAL})
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',revision+':'+n],cwd=ROOT)).hexdigest() for n in mapping}
    setup='''
import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile,shutil
from pathlib import Path
FULL_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-full-roi-karpathy-v134')
PROMPTS=Path('/kaggle/input/datasets/ducanh2403/upt-prompts-full-all-actions-v2/prompt_cache.json')
DETECTIONS=Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
BASE_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-prompt-pilot-inputs-v169')
OUTPUT=Path('/kaggle/working/csa_prompt_full_karpathy_cache_v1')
WALL_HOURS=4 # CPU STOPcap, not ETA or a claim about Kaggle session limits.
PARENT_DEADLINE=time.monotonic()+WALL_HOURS*3600
for p in (FULL_ROOT,PROMPTS,DETECTIONS,BASE_ROOT):
    if not p.exists():raise FileNotFoundError('Attach all4inputDatasets before starting: '+str(p))
if OUTPUT.exists():raise RuntimeError('Preserve existing output; fresh runtime/output path required')
if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<6*1024**3:raise RuntimeError('Need6GiB free for readonly metadata/download/copies; no files removed')
OUTPUT.mkdir(parents=True);SOURCE=Path(tempfile.mkdtemp(prefix='csa-full-prompt-',dir='/tmp'));LOG=OUTPUT/'notebook.log'
'''
    setup+='REVISION='+repr(revision)+'\nMAPPING='+repr(mapping)+'\nHASHES='+repr(hashes)+'\n'
    setup+='''
for original,destination in MAPPING.items():
    remaining=PARENT_DEADLINE-time.monotonic()
    if remaining<=0:raise TimeoutError('CPU parent cap during source download')
    raw=urllib.request.urlopen('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+REVISION+'/'+original,timeout=min(90,remaining)).read()
    if hashlib.sha256(raw).hexdigest()!=HASHES[original]:raise ValueError('Changed pinned source:'+original)
    target=SOURCE/destination;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=REVISION,sha256=HASHES),indent=2),encoding='utf-8')
os.environ['PYTHONPATH']=str(SOURCE);os.environ['OMP_NUM_THREADS']='1'
CONFIG=SOURCE/'config.json';MAPPING_ROOT=OUTPUT/'mapping'
cfg=dict(full_root=str(FULL_ROOT),prompts=str(PROMPTS),detections=str(DETECTIONS),base_root=str(BASE_ROOT),
    output=str(OUTPUT),mapping_root=str(MAPPING_ROOT),hf_cache=str(SOURCE/'hf-cache'))
CONFIG.write_text(json.dumps(cfg),encoding='utf-8')
MAP_CONFIG=SOURCE/'mapping_config.json'
MAP_CONFIG.write_text(json.dumps(dict(full_root=str(FULL_ROOT),prompts=str(PROMPTS),detections=str(DETECTIONS),
    output=str(MAPPING_ROOT),scope='full',wall_seconds=10800)),encoding='utf-8')
'''
    runner=RUNNER.replace("print('Running:',", "if time.monotonic()>=PARENT_DEADLINE:raise TimeoutError('CPU parent cap exceeded')\n    timeout=min(timeout,max(.01,PARENT_DEADLINE-time.monotonic()))\n    print('Running:',")
    runner=runner.replace('Smoke active;','Full prompt CPU active;').replace('Smoke wall cap exceeded','Full prompt CPU phase/parent cap exceeded')
    final='''
complete=False
try:
    run([sys.executable,'-u',SOURCE/'test_prompt_adapter_data.py'],timeout=120)
    run([sys.executable,'-u',SOURCE/'test_prompt_text_cache_data.py'],timeout=120)
    run([sys.executable,'-u',SOURCE/'test_full_prompt_cache_data.py'],timeout=120)
    # Verify reusable169 bytes and embedding space before processing full Karpathy.
    run([sys.executable,'-c',"from full_prompt_cache_data import checked_base;from pathlib import Path;import sys;r,t,a,m=checked_base(Path(sys.argv[1]));print('Exact169 base verified:',len(t),a.shape,'No weights/model loaded.',flush=True)",BASE_ROOT],timeout=180)
    run([sys.executable,'-u',SOURCE/'run_prompt_mapping_audit.py','--config',MAP_CONFIG],timeout=10800)
    run([sys.executable,'-u',SOURCE/'run_full_prompt_cache.py','inspect','--config',CONFIG],timeout=900)
    plan=json.loads((OUTPUT/'cache_plan.json').read_text(encoding='utf-8'))
    print('NEW text vectors to encode:',plan['encode_new_count'],'; reuse318 unchanged.',flush=True)
    if plan['encode_new_count']:
        DEPS=SOURCE/'hf-deps'
        run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS,
            'transformers==4.57.1','tokenizers==0.22.1','huggingface-hub==0.36.2','safetensors==0.6.2','ftfy==6.3.1'],timeout=300)
        os.environ['PYTHONPATH']=str(DEPS)+os.pathsep+str(SOURCE)
        run([sys.executable,'-c',"from importlib.metadata import version;expected={'transformers':'4.57.1','tokenizers':'0.22.1','huggingface-hub':'0.36.2','safetensors':'0.6.2'};assert all(version(k)==v for k,v in expected.items());print('Isolated HF dependency versions PASS')"],timeout=60)
    run([sys.executable,'-u',SOURCE/'run_full_prompt_cache.py','cache','--config',CONFIG],timeout=3600)
    receipt=json.loads((OUTPUT/'cache_receipt.json').read_text(encoding='utf-8'))
    complete=receipt['status']=='FULL_PROMPT_TEXT_CACHE_COMPLETE'
    if not complete:raise RuntimeError('Incomplete full cache')
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,training=False,test_used=False,caption_weights_loaded=False)),encoding='utf-8')
    REVIEW=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(REVIEW,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ('cache_plan.json','cache_manifest.json','cache_receipt.json','mapping_receipt.json','reuse_report.json',
            'summary.json','cohort.json','review_examples.json','source_receipt.json','failure.txt','notebook.log','run_status.json',
            'comparison_diagnostic.json','recovery_receipt.json','mapping_source_receipt.json'):
            if (OUTPUT/name).is_file():z.write(OUTPUT/name,name)
        # Preserve mapping diagnostics even when the later cache phase did not start.
        for name in ('summary.json','receipt.json','cohort.json','review_examples.json','failure.json'):
            if (MAPPING_ROOT/name).is_file():z.write(MAPPING_ROOT/name,'mapping/'+name)
    print('SEND REVIEW:',REVIEW,'\\nKEEP ENTIRE OUTPUT:',OUTPUT,flush=True)
    print('FULL prompt cache ready for review; no caption training.' if complete else 'INCOMPLETE; preserve output/logs; do not use a partial cache.',flush=True)
'''
    intro='''# Full Karpathy prompt cache trước XE từ đầu
**Accelerator None, Internet ON.** CPU map liên kết và frozenCLIP chỉ cho text mới; không UPT inference, caption training, TEST, API Secret hoặc checkpointload.

Gắn4Datasets: `csa-full-roi-karpathy-v134`, `upt-prompts-full-all-actions-v2`, `objectdetectionecache`, `csa-prompt-pilot-inputs-v169`. Sửa4path ở cell đầu nếu mount khác.
Kiểm toàn bộ113287TRAIN+5000VALID, cùngIoU/instance/cap8object/3relation. Giữ318vector169 byte-identical; nếu cótext mới chỉ encode phần mới cùngCLIP/revision/assetSHA/frozenFP32CPU. Không lấy prompt từ captionthamchiếu.
Fullmapping phải giữ nguyên các trường ngữ nghĩa và float32input của15000record169. Sai số FP64 chỉ ở IoU/roi_bbox/geometry ≤1e-12 và castfloat32 giốnghệt mới được nhận; diagnostic ghi rõ trường lệch. ID/text/boxnguồn/S/O/ROI/confidence/threshold luônso chínhxác. Dừng nếu thiếudữliệu hoặc đổi semantic/input; không coi ảnhthiếuprompt lànegative/no_objects.
Chạy Save Version → Save & Run All vớiCPU. Cap4h là giới hạn dừng, không phảiETA/Kagglequota. Giữwholeoutput, gửireviewZIP. Sau review sẽ lưuDataset trước hai nhánhXEfromscratch; notebook này chưa huấn luyện và không thay kết quả172/56.
'''
    if recover:
        setup=setup.replace("OUTPUT=Path('/kaggle/working/csa_prompt_full_karpathy_cache_v1')","OUTPUT=Path('/kaggle/working/csa_prompt_full_karpathy_cache_v2')\nSECRET_NAME='codex'")
        setup=setup.replace('for p in (FULL_ROOT,PROMPTS,DETECTIONS,BASE_ROOT):','for p in (BASE_ROOT,):')
        original=json.loads(subprocess.check_output(['git','show','1caf4fa:'+PREFIX+'CSA_PROMPT_FULL_KARPATHY_CACHE_CPU.ipynb'],cwd=ROOT))
        original_setup=next(c['source'] for c in original['cells'] if c['cell_type']=='code' and 'HASHES=' in c['source'])
        old={}
        for n in ast.parse(original_setup).body:
            if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('REVISION','HASHES'):old[n.targets[0].id]=ast.literal_eval(n.value)
        setup+='\nORIGINAL_SOURCE='+repr(dict(commit=old['REVISION'],sha256=old['HASHES']))+'\n'
        runner=runner.replace('def run(args, timeout=1800):','def run(args, timeout=1800, environment=None):').replace('cwd=SOURCE, stdout=','cwd=SOURCE, env=environment, stdout=')
        recovery='''
    from kaggle_secrets import UserSecretsClient
    try:token=UserSecretsClient().get_secret(SECRET_NAME).strip()
    except Exception:raise RuntimeError('Enable Secret codex for the account owning output173') from None
    if not token:raise RuntimeError('Empty API Secret')
    run([sys.executable,'-m','pip','install','-q','kaggle==2.2.5'],timeout=300)
    preflight="import tempfile;from types import SimpleNamespace;from unittest.mock import patch;from kaggle.api.kaggle_api_extended import KaggleApi;api=KaggleApi()\\nwith tempfile.TemporaryDirectory() as d,patch.object(api,'build_kaggle_client') as f:\\n e=f.return_value.__enter__.return_value.kernels.kernels_api_client.list_kernel_session_output;e.return_value=SimpleNamespace(files=[],next_page_token=None,log=None);api.kernels_output('ducanh2403/2-vi-cai-kia-qua-nhieu/173',path=d);r=e.call_args.args[0];assert r.version_label=='v173' and r.user_name=='ducanh2403' and r.kernel_slug=='2-vi-cai-kia-qua-nhieu'\\nprint('Exact173 request routing PASS; mock only')"
    run([sys.executable,'-c',preflight],timeout=60)
    import re
    from full_prompt_cache_data import MAPPING_FILES
    from recover_full_prompt_mapping import preserve
    env=os.environ.copy()
    for k in ('KAGGLE_API_TOKEN','KAGGLE_USERNAME','KAGGLE_KEY'):env.pop(k,None)
    env['KAGGLE_API_TOKEN']=token
    DOWNLOAD=SOURCE/'download173';DOWNLOAD.mkdir()
    selector=r'(^|/)('+'|'.join(re.escape(n) for n in MAPPING_FILES+('receipt.json','source_receipt.json'))+r')$'
    run([sys.executable,'-c','from kaggle.cli import main; main()','kernels','output','ducanh2403/2-vi-cai-kia-qua-nhieu/173',
        '-p',str(DOWNLOAD),'--file-pattern',selector],timeout=1200,environment=env)
    del token,env
    preserve(DOWNLOAD,MAPPING_ROOT,ORIGINAL_SOURCE)
    print('Original173 mapping restored with all payloadSHA; no mapping rerun.',flush=True)
'''
        final=final.replace("    run([sys.executable,'-u',SOURCE/'run_prompt_mapping_audit.py','--config',MAP_CONFIG],timeout=10800)",recovery)
        intro='''# Tiếp tục cache full từ Output173 — giữ mapping đã hoàn tất
**Accelerator None, Internet ON**, bật Secret `codex` của tài khoản sở hữu Output173. Chỉ gắn Dataset `csa-prompt-pilot-inputs-v169`; không cần ROI/UPT/detector lúc phục hồi.
Tải đúng173 gồm5mappingpayload+receipt+source receipt; kiểm toàn bộSHA và27sourcepins gốc. Không dùnglatest, không maplại118287ảnh, khôngxóaOutput173.
So toàn bộ15krecord169; ghi `comparison_diagnostic.json`. Chỉcho phép sai số FP64≤1e-12 ở trường tính toán IoU/roi_bbox/geometry nếu float32 hữu hạn giốngbit. Các trường ngữ nghĩa/ID/ROI/text/boxnguồn/S/O/confidence/threshold giữ chính xác; bất kỳ lệch thực thìdừng, khôngencode/COMPLETE.
Nếu đạtguard: reuse318vector, encodechỉtextmới frozenCLIPCPU. Giữwholeoutputv2 và gửireviewZIP kể cả khi dừng. Chưa biết trường lệchthực trước diagnostic; không khẳng định nguyênnhân từlog173.
'''
    if fresh:
        setup=setup.replace("BASE_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-prompt-pilot-inputs-v169')\n",'')
        setup=setup.replace('for p in (FULL_ROOT,PROMPTS,DETECTIONS,BASE_ROOT):','for p in (FULL_ROOT,PROMPTS,DETECTIONS):')
        setup=setup.replace('Attach all4inputDatasets','Attach all3inputDatasets')
        setup=setup.replace('base_root=str(BASE_ROOT),','rebuild_all=True,')
        setup=setup.replace('csa_prompt_full_karpathy_cache_v1','csa_prompt_full_karpathy_cache_fresh_v3')
        final='\n'.join(line for line in final.split('\n') if '# Verify reusable169' not in line and "checked_base;from pathlib" not in line)
        final=final.replace("'; reuse318 unchanged.'","'; fresh encoding of ALL unique texts.'")
        intro='''# Tạo lại toàn bộ cache prompt Karpathy từ đầu
**Accelerator None, Internet ON. Không cần API Secret hoặc cache169/output173.**

Gắn3Datasets: `csa-full-roi-karpathy-v134`, `upt-prompts-full-all-actions-v2`, `objectdetectionecache`. Kiểm3path trong cell đầu.
Tạo mới mapping113287TRAIN+5000VALID từ ROI134/UPT/detector, sau đó encode TOÀN BỘ text duy nhất bằng frozenCLIP FP32 CPU cùng revision và SHA assets đã khóa. Không so record với169, không tái dùng vectors169, không dùng caption tham chiếu để tạo prompt, không TEST/train mô hình caption/UPT inference.
Vẫn kiểm SHA đầu vào, đủsplit/ROI/text/tensorpacking và toàn bộ payload trước receipt COMPLETE. Cache mới không được khẳng định giống bit với169; cần review trước XE từ đầu.
Save Version → Save & Run All, CPU. Cap4h là giới hạn dừng, không phải ETA. Giữ toàn bộ `/kaggle/working/csa_prompt_full_karpathy_cache_fresh_v3`, gửi ZIP review cả khi lỗi.
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        kaggle=dict(isInternetEnabled=True,isGpuEnabled=False)),cells=[cell('markdown',intro),cell('code',setup),cell('code',runner),cell('code',final)])
    path=HERE/('CSA_PROMPT_FULL_CACHE_FRESH_CPU.ipynb' if fresh else 'CSA_PROMPT_FULL_CACHE_RECOVER173_CPU.ipynb' if recover else 'CSA_PROMPT_FULL_KARPATHY_CACHE_CPU.ipynb')
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8');print(path.name,path.stat().st_size,'bytes;source',revision)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);p.add_argument('--recover173',action='store_true');p.add_argument('--fresh',action='store_true');a=p.parse_args();make(a.revision,a.recover173,a.fresh)
