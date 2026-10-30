"""Generate pinned D/E notebooks; never execute or submit their cells."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap
from make_kv_prepare_notebook import cell

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREFIX = 'experiments/semantic_grounding_method/'
FILES = [PREFIX+n for n in ('kv_pilot_data.py','kv_pilot_model.py','kv_pilot_preflight.py',
    'kv_pilot_training.py','kv_pilot_evaluate.py','run_kv_pilot.py','test_kv_pilot_data.py',
    'test_relation_kv_math.py','kv_data.py','KV_SAVE_V115_PACKAGE.json','KV_NATIVE_TEXT_AUDIT.json',
    'data/gain_metadata.json.gz')]
FILES += ['experiments/relation_frames_inventory/inventory.py','experiments/shared_pair_readout/metric_preflight.py']
FILES += ['captioning/'+n+'.py' for n in ('__init__','models','config','grounded_relations',
    'decoder_relation_memory','scst','scst_runtime','tokenizer','object_reward','object_consistency','semantic_prompts')]

RUNNER = '''
def run(args,timeout=900,cwd=None):
    args=list(map(str,args));print('Running:',' '.join(args),flush=True)
    process=subprocess.Popen(args,cwd=cwd or REPO,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        text=True,errors='replace',bufsize=1,start_new_session=True)
    messages=queue.Queue()
    def reader():
        try:
            for line in process.stdout:messages.put(line)
        finally:messages.put(None)
    thread=threading.Thread(target=reader,daemon=True);thread.start()
    start=heartbeat=time.monotonic();ended=False
    with LOG.open('a',encoding='utf-8') as log:
        try:
            while not ended or process.poll() is None or not messages.empty():
                try:
                    line=messages.get(timeout=1)
                    if line is None:ended=True
                    else:print(line,end='',flush=True);log.write(line);log.flush()
                except queue.Empty:pass
                now=time.monotonic()
                if now-heartbeat>=30:
                    print('Worker active;',round(now-start),'seconds; see phase/step logs.',flush=True);heartbeat=now
                if now-start>timeout:raise TimeoutError('Locked phase cap exceeded; INCOMPLETE')
            if process.wait():raise subprocess.CalledProcessError(process.returncode,args)
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            thread.join(timeout=5)
'''


def make(revision):
    core = subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    assets = {name:hashlib.sha256(subprocess.check_output(['git','show',core+':'+name],cwd=ROOT)).hexdigest() for name in FILES}
    package = dict(core_commit=core,assets=assets,model_executed_locally=False,kaggle_submitted=False,notebooks={})
    for arm in ('D','E'):
        cells = [cell('markdown',f'''
            # Relation K/V — {arm} · pilot500 · T4×2
            **Internet ON · T4×2 · Save & Run All. Mỗi notebook một arm, dùng cả2GPU.**
            D: directional keys/values. E: giữ keys, chia sẻ value giữa hai equivalent views cùng fact.
            Cả hai cùng C0 init, SCST+own-C0 KL+BCE+observed MP, endpoint500; chưa có kết quả mô hình.

            Add Input: relation-kv-preparation-v115; concept-mass-c-c0-v111-v112;
            prompt-cache; objectdetectionecache; mscoco-2014 của vuthetam.
            Không cần raw visual-cache hoặc UPT dataset. Prompt-cache chỉ bổ sung đúng98nativeheldprompts
            cho supporting decoder diagnostic; không sửa H5 nguồn hoặc tạo prompt từ nhãn.
            Kiểm5path ở cell đầu. Dataset Private phải chia sẻ với tài khoản Ngân nếu chạy E ở đó.

            FullSHA/preflight/20stepbenchmark phải qua trước training; forecast không vừa trần thì STOP,
            không tự rút steps. GPUphase cap7200s, CPU scorer1800s; caps không ETA hoặc cách tính quota.
            CPU bootstrap vẫn có thể tiêu quota khi accelerator bật. User báo khoảng20h/account.
            Giữ checkpoint.bin và gửi review ZIP. Không tự promote/checkpoint selection/sweep/full run.
            '''),cell('code',f'''
            from pathlib import Path
            ARM={arm!r}
            PREPARED_ROOT=Path('/kaggle/input/datasets/ducanh2403/relation-kv-preparation-v115')
            CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/concept-mass-c-c0-v111-v112/C0.bin')
            COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
            DETECTIONS=Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
            LEGACY_PROMPT=Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
            OUTPUT=Path('/kaggle/working/relation_kv_'+ARM+'_pilot')
            REPORTED_GPU_HOURS_PER_ACCOUNT=20
            '''),cell('code',textwrap.dedent(f'''
            CORE={core!r}
            ASSETS={assets!r}
            import hashlib,json,os,signal,subprocess,sys,tempfile,threading,time,zipfile,queue,traceback,urllib.request
            os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                PYTHONUNBUFFERED='1',PYTHONHASHSEED='42',TOKENIZERS_PARALLELISM='false',HDF5_USE_FILE_LOCKING='FALSE')
            LOG=OUTPUT.with_name(OUTPUT.name+'_notebook.log');REVIEW=OUTPUT.with_name(OUTPUT.name+'_review.zip')
            for p in (OUTPUT,LOG,REVIEW):
                if p.exists():raise FileExistsError('Preserve old output; use a fresh session/path: '+str(p))
            for p in (CHECKPOINT,COCO_JSON,DETECTIONS,LEGACY_PROMPT,PREPARED_ROOT/'cpu_preparation.json',PREPARED_ROOT/'inputs.h5'):
                if not p.is_file():raise FileNotFoundError('Add Input or edit path: '+str(p))
            WORK=Path(tempfile.mkdtemp(prefix='relation-kv-'+ARM+'-'))
            REPO=WORK/'repo';REPO.mkdir()
            HERE=REPO/{PREFIX!r};CONFIG=WORK/'config.json'
            print('Bootstrap starting; arm',ARM,'source',CORE,flush=True)
            ''')+RUNNER),cell('code','''
            try:
                for i,(relative,expected) in enumerate(ASSETS.items(),1):
                    with urllib.request.urlopen('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+CORE+'/'+relative,timeout=60) as response:raw=response.read()
                    if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Changed pinned source '+relative)
                    p=REPO/relative;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
                    if i%5==0 or i==len(ASSETS):print('Verified source files',i,'/',len(ASSETS),flush=True)
                os.environ['PYTHONPATH']=str(REPO)
                run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','pycocoevalcap==1.2','h5py','numpy','Pillow','nltk==3.9.2','future'])
                run([sys.executable,'-m','pip','install','-q','--no-deps','git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'])
                run([sys.executable,'-u','-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"])
                run(['java','-version'])
                CHAIR=WORK/'chair'
                run(['git','init',CHAIR]);run(['git','remote','add','origin','https://github.com/Maxlinn/CHAIR-metric-standalone.git'],cwd=CHAIR)
                run(['git','fetch','--depth','1','origin','4087a26211aa2339b9a76307cb8f0321ef691d0a'],cwd=CHAIR)
                run(['git','checkout','--detach','FETCH_HEAD'],cwd=CHAIR)
                sys.path.insert(0,str(REPO));sys.path.insert(0,str(HERE))
                from kv_pilot_data import POLICY
                cfg=dict(arm=ARM,prepared_root=str(PREPARED_ROOT),checkpoint=str(CHECKPOINT),dataset_json=str(COCO_JSON),
                    detections=str(DETECTIONS),legacy_prompt=str(LEGACY_PROMPT),chair=str(CHAIR),output=str(OUTPUT),
                    policy=POLICY,source_commit=CORE,source_assets=ASSETS,reported_gpu_hours_per_account=REPORTED_GPU_HOURS_PER_ACCOUNT)
                CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
            except BaseException:
                with zipfile.ZipFile(REVIEW,'x',zipfile.ZIP_DEFLATED) as archive:
                    if LOG.exists():archive.write(LOG,'notebook.log')
                    archive.writestr('bootstrap_error.txt',traceback.format_exc())
                print('BOOTSTRAP FAILED; SEND',REVIEW,flush=True)
                raise
            '''),cell('code','''
            success=False
            try:
                run([sys.executable,'-u',HERE/'test_kv_pilot_data.py'],120)
                run([sys.executable,'-u',HERE/'test_relation_kv_math.py'],120)
                run([sys.executable,'-u',REPO/'experiments/shared_pair_readout/metric_preflight.py'],180)
                run([sys.executable,'-u',HERE/'run_kv_pilot.py','prepare','--config',CONFIG],1800)
                run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1',
                    '--nproc_per_node=2','--max_restarts=0',HERE/'run_kv_pilot.py','worker','--config',CONFIG],7200)
                run([sys.executable,'-u',HERE/'run_kv_pilot.py','evaluate','--config',CONFIG],1800)
                result=json.loads((OUTPUT/'summary.json').read_bytes())
                if result['status']!='ENDPOINT_500_EVALUATED_WAIT_MATCHED_DE':raise RuntimeError('Incomplete endpoint')
                success=True
            finally:
                with zipfile.ZipFile(REVIEW,'x',zipfile.ZIP_DEFLATED) as archive:
                    if LOG.exists():archive.write(LOG,'notebook.log')
                    if CONFIG.exists():archive.write(CONFIG,'notebook_config.json')
                    for p in sorted(OUTPUT.glob('*')):
                        if p.suffix in ('.json','.jsonl','.log'):archive.write(p,p.name)
                    for relative in ASSETS:
                        if relative.endswith('.py'):archive.write(REPO/relative,'source/'+relative)
                    archive.writestr('execution_status.json',json.dumps(dict(success=success,arm=ARM,source_commit=CORE,assistant_launched_job=False)))
                print('SEND REVIEW ZIP:',REVIEW,flush=True)
                if (OUTPUT/'checkpoint.bin').exists():print('KEEP CHECKPOINT:',OUTPUT/'checkpoint.bin',flush=True)
                if not success:print('INCOMPLETE; do not compare a partial run with endpoint500.',flush=True)
            '''),cell('markdown',f'''
            Gửi `relation_kv_{arm}_pilot_review.zip`. Giữ `checkpoint.bin`; không cần tải rawcache xuống máy.
            Chờ đủ D/E để so cùng C0, nguồn/init/token identities và paired CIDEr/CHAIR/semantic scores.
            Model/CUDA/gradient/parity chưa được Codex chạy; chỉ user-run preflight mới xác nhận.
            ''')]
        notebook = dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
            kernelspec=dict(display_name='Python 3',language='python',name='python3'),
            language_info=dict(name='python'),kaggle=dict(isGpuEnabled=True,isInternetEnabled=True,accelerator='nvidiaTeslaT4')))
        target = HERE/f'RELATION_KV_{arm}_2GPU.ipynb'
        target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
        for c in cells:
            if c['cell_type']=='code':compile(c['source'],target.name,'exec')
        if target.stat().st_size >= 1_000_000:raise ValueError('Notebook exceeds Kaggle source limit')
        package['notebooks'][arm] = dict(file=target.name,bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        print(target.name,target.stat().st_size,'bytes; cells compiled, NOT executed')
    (HERE/'KV_DE_NOTEBOOK_PACKAGE.json').write_text(json.dumps(package,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--core',default='HEAD')
    make(p.parse_args().core)
