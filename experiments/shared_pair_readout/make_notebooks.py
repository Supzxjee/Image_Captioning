"""Generate two pinned Kaggle notebooks, never launch them."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def cell(kind,text):
    source=textwrap.dedent(text).strip()+'\n'
    c=dict(cell_type=kind,source=source,metadata={},id=hashlib.sha256((kind+source).encode()).hexdigest()[:12])
    if kind=='code':c.update(execution_count=None,outputs=[])
    return c


def make(core):
    commit=subprocess.check_output(['git','rev-parse',core],cwd=ROOT,text=True).strip()
    files=['experiments/shared_pair_readout/'+n for n in ('readout_data.py','prepare_readout.py','readout_model.py','run_readout.py',
        'evaluate_readout.py','metric_preflight.py','test_readout_data.py','DEV_DIMENSIONS.json')]
    files+=['captioning/'+n+'.py' for n in ('__init__','models','config','grounded_relations','decoder_relation_memory',
        'scst','scst_runtime','tokenizer','object_reward','object_consistency','visual_cache','semantic_prompts')]
    files+=['experiments/relation_frame_pretrain/frame_data.py','experiments/relation_frames_inventory/V2_MANIFEST.json.gz',
        'experiments/semantic_grounding_method/data/gain_metadata.json.gz']
    assets={p:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+p],cwd=ROOT)).hexdigest() for p in files}
    package=[]
    for arm in ('I','J'):
        cells=[cell('markdown',f'''
            # Shared-weight pair readout — {arm} · T4×2 · endpoint500
            **Internet ON · T4×2 · Save & Run All. Một tài khoản chạy I, tài khoản còn lại chạy J.**

            Hai nhánh dùng cùng attention cặp, đặc trưng, tham số, loss và khởi tạo.
            I đọc thực thể bằng priors riêng; J dùng marginals của attention cặp.
            Đây là cải tiến tích hợp kế thừa, chưa có kết quả hoặc chứng nhận tính mới.

            Add Input sáu dataset: `relation-frames-v2-cache-v105`, `pair-source-fullframe-ce`,
            `mscoco-2014` của vuthetam, `prompt-cache`, `visual-cache`, `objectdetectionecache`.
            Không cần UPT, checkpoint frame200, API token hoặc ảnhRGB. Chỉ COCO JSON được đọc từ mscoco.
            Đường dẫn điền sẵn ở cell tiếp theo; sửa mount nếu khác, giữ nguồn/SHA.

            Pipeline: kiểm phụ thuộc metric → sao chép cache612+dev1000 trên CPU → actual-model2GPU preflight
            → baseline A hiện tại →500updates → caption A/end/off và heldout → metrics/ZIP.
            CPU prepare cap30phút, pha2GPU cap60phút/nhánh, CPU metrics cap15phút; đều là giới hạn, không ETA.
            Dừng khi lỗi/cap, không tự kéo dài hoặc đổi cấu hình. Giữ checkpoint endpoint.bin nếu đã có.
            User báo còn25giờ, không suy thành50giờ hoặc khẳng định quota riêng của tài khoản khác.
            '''),cell('code',f'''
            from pathlib import Path
            ARM={arm!r}
            CACHE=Path('/kaggle/input/datasets/ducanh2403/relation-frames-v2-cache-v105')
            CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin')
            COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
            LEGACY_PROMPT=Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
            VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
            DETECTIONS=Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
            OUTPUT=Path('/kaggle/working/shared_pair_readout_'+ARM)
            REPORTED_REMAINING_HOURS=25  # User report; account distribution and Kaggle accounting not inferred.
            '''),cell('code',f'''
            import hashlib,json,os,signal,subprocess,sys,tempfile,threading,time,traceback,zipfile
            os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                PYTHONUNBUFFERED='1',PYTHONHASHSEED='42',TOKENIZERS_PARALLELISM='false',HDF5_USE_FILE_LOCKING='FALSE')
            SOURCE_COMMIT={commit!r}
            SOURCE_ASSETS={assets!r}
            REPO=Path(tempfile.mkdtemp(prefix='shared-pair-'+ARM+'-'))/'repo'
            LOG=OUTPUT.with_name(OUTPUT.name+'_notebook.log')
            REVIEW=OUTPUT.with_name(OUTPUT.name+'_review.zip')
            if OUTPUT.exists():raise FileExistsError('Giữ output cũ; dùng phiên mới trước khi chạy lại.')
            if LOG.exists() or REVIEW.exists():raise FileExistsError('Giữ log/review cũ; đổi OUTPUT hoặc dùng phiên mới.')
            for p in (CHECKPOINT,COCO_JSON,LEGACY_PROMPT,DETECTIONS):
                if not p.is_file():raise FileNotFoundError('Add Input dataset chứa '+str(p))
            for p in (CACHE,VISUAL_CACHE):
                if not p.is_dir():raise FileNotFoundError('Add Input dataset '+str(p))
            def run(args,cwd=None,timeout=900):
                args=list(map(str,args));print('Running:',' '.join(args),flush=True)
                with LOG.open('a',encoding='utf-8') as log:
                    log.write('Running: '+' '.join(args)+'\\n');log.flush()
                    process=subprocess.Popen(args,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                        text=True,errors='replace',bufsize=1,start_new_session=True)
                    def drain():
                        for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                    reader=threading.Thread(target=drain,daemon=True);reader.start();started=time.monotonic()
                    try:
                        while process.poll() is None:
                            if time.monotonic()-started>timeout:raise TimeoutError('Fixed phase timeout; incomplete')
                            reader.join(timeout=20)
                            if process.poll() is None:print('Worker active;',round(time.monotonic()-started),'s elapsed. See stage/step lines.',flush=True)
                        reader.join()
                        if process.returncode:raise subprocess.CalledProcessError(process.returncode,args)
                    finally:
                        if process.poll() is None:
                            os.killpg(process.pid,signal.SIGTERM)
                            try:process.wait(timeout=10)
                            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                        reader.join(timeout=5)
            print('Bootstrap started; input paths found.',flush=True)
            '''),cell('code','''
            run(['git','init',REPO])
            run(['git','remote','add','origin','https://github.com/Supzxjee/Image_Captioning.git'],cwd=REPO)
            run(['git','fetch','--depth','1','origin',SOURCE_COMMIT],cwd=REPO)
            run(['git','checkout','--detach','FETCH_HEAD'],cwd=REPO)
            if subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()!=SOURCE_COMMIT:raise ValueError('Wrong source commit')
            for name,h in SOURCE_ASSETS.items():
                if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=h:raise ValueError('Changed source '+name)
            os.environ['PYTHONPATH']=str(REPO)
            HERE=REPO/'experiments/shared_pair_readout'
            run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','pycocoevalcap==1.2','h5py','numpy','Pillow','nltk==3.9.2','future'])
            run([sys.executable,'-m','pip','install','-q','--no-deps','git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'])
            run([sys.executable,'-u','-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"])
            run(['java','-version'])
            CHAIR=REPO.parent/'chair'
            run(['git','init',CHAIR]);run(['git','remote','add','origin','https://github.com/Maxlinn/CHAIR-metric-standalone.git'],cwd=CHAIR)
            run(['git','fetch','--depth','1','origin','4087a26211aa2339b9a76307cb8f0321ef691d0a'],cwd=CHAIR)
            run(['git','checkout','--detach','FETCH_HEAD'],cwd=CHAIR)
            if hashlib.sha256((CHAIR/'chair.pkl').read_bytes()).hexdigest()!='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':raise ValueError('Wrong CHAIR cache')
            CONFIG=REPO.parent/'config.json'
            sys.path.insert(0,str(REPO));sys.path.insert(0,str(HERE))
            from readout_data import POLICY
            cfg=dict(arm=ARM,cache=str(CACHE),checkpoint=str(CHECKPOINT),dataset_json=str(COCO_JSON),
                legacy_prompt=str(LEGACY_PROMPT),visual_cache=str(VISUAL_CACHE),detections=str(DETECTIONS),
                metadata=str(REPO/'experiments/semantic_grounding_method/data/gain_metadata.json.gz'),
                dimensions=str(HERE/'DEV_DIMENSIONS.json'),chair=str(CHAIR),output=str(OUTPUT),
                policy=POLICY,source_commit=SOURCE_COMMIT,source_assets=SOURCE_ASSETS,
                reported_remaining_gpu_hours=REPORTED_REMAINING_HOURS,
                quota_note='User report25h; not multiplied across accounts; GPU phase cap2device-hours/arm, preprocessing/CPUeval extra.')
            CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
            '''),cell('code','''
            success=False
            try:
                run([sys.executable,'-u',HERE/'test_readout_data.py'],cwd=REPO)
                run([sys.executable,'-u',HERE/'metric_preflight.py'],cwd=REPO)
                run([sys.executable,'-u',HERE/'run_readout.py','prepare','--config',CONFIG],cwd=REPO,timeout=1800)
                run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',
                    '--max_restarts=0',HERE/'run_readout.py','worker','--config',CONFIG],cwd=REPO,timeout=3600)
                run([sys.executable,'-u',HERE/'run_readout.py','evaluate','--config',CONFIG],cwd=REPO,timeout=900)
                result=json.loads((OUTPUT/'summary.json').read_text())
                if result['status']!='ENDPOINT_500_EVALUATED_WAIT_MATCHED_I_J':raise RuntimeError('Incomplete endpoint/evaluation')
                success=True
            finally:
                with zipfile.ZipFile(REVIEW,'w',zipfile.ZIP_DEFLATED) as z:
                    if LOG.is_file():z.write(LOG,'notebook.log')
                    z.write(CONFIG,'notebook_config.json')
                    for name in SOURCE_ASSETS:
                        if name.endswith('.py'):z.write(REPO/name,'source/'+name)
                    if OUTPUT.is_dir():
                        for p in OUTPUT.iterdir():
                            if p.suffix in ('.json','.jsonl','.log'):z.write(p,p.name)
                    z.writestr('execution_status.json',json.dumps(dict(success=success,arm=ARM,source_commit=SOURCE_COMMIT,
                        assistant_launched_job=False),indent=2))
                print('SEND SMALL REVIEW ZIP:',REVIEW,flush=True)
                if (OUTPUT/'checkpoint.bin').is_file():print('KEEP CHECKPOINT:',OUTPUT/'checkpoint.bin',flush=True)
                if not success:print('INCOMPLETE; send ZIP/log. No result or method improvement accepted.',flush=True)
            if success:print('DONE',ARM,'; send BOTH I/J reviewZIPs for paired comparison and semantic panel.',flush=True)
            '''),cell('markdown','''
            Giữ checkpoint.bin. Gửi `shared_pair_readout_I_review.zip` và `shared_pair_readout_J_review.zip`.
            ZIP không chứa checkpoint/HDF5. Endpoint hoàn tất chưa có nghĩa phương pháp tốt hơn hoặc đủ để được nhận bài.
            So sánh trên A vừa chạy cùng runtime; nếu khác caption A lịch sử thì báo drift, không thay số lịch sử vào bảng.
            Chưa có chấm semantic panel trong notebook. Không tự tạo Dataset, chọn checkpoint, rerank hoặc chạy seed2.
            ''')]
        path=HERE/f'SHARED_PAIR_READOUT_{arm}_2GPU.ipynb'
        nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),language_info=dict(name='python'),kaggle=dict(isGpuEnabled=True,isInternetEnabled=True)),cells=cells)
        path.write_bytes((json.dumps(nb,ensure_ascii=False,indent=2)+'\n').encode())
        for c in cells:
            if c['cell_type']=='code':compile(c['source'],str(path),'exec')
        assert path.stat().st_size<1000000
        package.append(dict(file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (HERE/'NOTEBOOK_PACKAGE.json').write_bytes((json.dumps(dict(source_commit=commit,source_assets=assets,notebooks=package,
        static_compile=True,local_neural_execution=False,kaggle_launched=False),indent=2)+'\n').encode())
    print(json.dumps(package,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--core',required=True);make(p.parse_args().core)
