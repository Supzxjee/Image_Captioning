"""Generate small pinned C/G Kaggle notebooks; never submit or run them."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def cell(kind,text):
    source=textwrap.dedent(text).strip()+'\n';out=dict(cell_type=kind,metadata={},source=source,id=hashlib.sha256((kind+source).encode()).hexdigest()[:12])
    if kind=='code':out.update(execution_count=None,outputs=[])
    return out

def make(core):
    commit=subprocess.check_output(['git','rev-parse',core],cwd=ROOT,text=True).strip()
    names=['concept_data.py','concept_model.py','prepare_concept.py','run_concept.py','diagnostic_concept.py','evaluate_concept.py','test_concept_data.py','recover_stage.py','test_recover_stage.py','TRAIN_PRESENCE.json.gz','DEV_DIAGNOSTIC_GT.json.gz']
    files=['experiments/concept_mass_grounding/'+n for n in names]
    files+=['experiments/shared_pair_readout/readout_data.py','experiments/shared_pair_readout/metric_preflight.py','experiments/relation_frame_pretrain/frame_data.py',
        'experiments/semantic_grounding_method/data/gain_metadata.json.gz','experiments/coco_grounding_feasibility/MAPPING_POLICY_V11.json']
    files+=['captioning/'+n+'.py' for n in ('__init__','models','config','grounded_relations','decoder_relation_memory','scst','scst_runtime','tokenizer','object_reward','object_consistency','visual_cache','semantic_prompts')]
    assets={p:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+p],cwd=ROOT)).hexdigest() for p in files}
    package=[]
    for arm in ('C','G'):
        cells=[cell('markdown',f'''
            # Concept-mass grounding — {arm} · T4×2 · endpoint500
            **Internet ON · T4×2 · Save & Run All. Mỗi notebook một nhánh, dùng cả hai GPU.**

            C: native A + SCST reward gốc + rollout-KL + giữ xác suất khái niệm đúng.
            G: như C, thêm phạt probability mass của nhóm cách gọi đối tượng chưa được annotation hỗ trợ.
            Chưa có kết quả; đây là ứng viên objective ở mức khái niệm, chưa phải căn chỉnh đúng vùng/instance.

            Add Input: `pair-source-fullframe-ce`, `prompt-cache`, `visual-cache`, `objectdetectionecache`,
            `mscoco-2014` của vuthetam. Không cần RGB, UPT, VSR, checkpoint I/J hoặc dataset mới.
            Bản v2 giữ output cũ và dùng thư mục output mới. Tự tìm stage cũ nếu còn trong Working.
            Stage cũ chưa có row journal phải đọc lại nguồn để xác minh từng row, không tin mặc định4.750rows.
            Đọc lại có thể vẫn tốn I/O; không có cam kết khôi phục tức thì. Dữ liệu/500updates/loss không đổi.
            GT TRAIN10k và GT dev-only diagnostic nhỏ được pin trong Git; nhãn dev không dùng trong training.

            Policy đã sửa trước run: GT/ref absence là nhãn yếu, detector chỉ veto mọi hint hợp lệ có sẵn.
            Không suy vắng vật từ cache; nguồn extractor detector vẫn unknown. Không gọi negative là verified absence.
            Nếu không có usable negative events, G chưa kiểm được objective, không báo đã có đối chứng thành công.

            TRAIN population10k; stage đúng4000images trong lịch500updates, batch8 và fixeddev1000.
            Cả hai học hai decoder layer cuối từ cùng A; không thêm model head, CE warmup hoặc sửa logits inference.
            Giới hạn CPUprepare60phút, GPUphase60phút, CPUevaluation15phút/nhánh; đây là caps, không ETA.
            User báo24giờ còn lại; không tự cộng quota hai tài khoản. Notebook chuẩn bị/CPUmetrics vẫn có thể tiêu GPU quota khi accelerator bật.
            '''),cell('code',f'''
            from pathlib import Path
            ARM={arm!r}
            CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin')
            COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
            LEGACY_PROMPT=Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
            VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
            DETECTIONS=Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
            OLD_PREPARED=Path('/kaggle/working/concept_mass_'+ARM)
            PREVIOUS_PREPARED=OLD_PREPARED if (OLD_PREPARED/'inputs.h5').is_file() and (OLD_PREPARED/'protocol.json').is_file() else None
            OUTPUT=Path('/kaggle/working/concept_mass_'+ARM+'_v2')
            REPORTED_REMAINING_GPU_HOURS=24
            '''),cell('code','''
            import hashlib,json,os,signal,subprocess,sys,tempfile,threading,time,traceback,zipfile
            os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
                PYTHONUNBUFFERED='1',PYTHONHASHSEED='42',TOKENIZERS_PARALLELISM='false',HDF5_USE_FILE_LOCKING='FALSE')
            REPO=Path(tempfile.mkdtemp(prefix='concept-mass-'+ARM+'-'))/'repo'
            LOG=OUTPUT.with_name(OUTPUT.name+'_notebook.log');REVIEW=OUTPUT.with_name(OUTPUT.name+'_review.zip')
            for p in (OUTPUT,LOG,REVIEW):
                if p.exists():raise FileExistsError('Giữ output/log cũ; dùng phiên mới: '+str(p))
            for p in (CHECKPOINT,COCO_JSON,LEGACY_PROMPT,DETECTIONS):
                if not p.is_file():raise FileNotFoundError('Add Input chứa '+str(p))
            if not VISUAL_CACHE.is_dir():raise FileNotFoundError(VISUAL_CACHE)
            def run(args,cwd=None,timeout=900):
                args=list(map(str,args));print('Running:',' '.join(args),flush=True)
                with LOG.open('a',encoding='utf8') as log:
                    log.write('Running: '+' '.join(args)+'\\n');log.flush()
                    process=subprocess.Popen(args,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1,start_new_session=True)
                    def drain():
                        for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                    reader=threading.Thread(target=drain,daemon=True);reader.start();start=time.monotonic()
                    try:
                        while process.poll() is None:
                            if time.monotonic()-start>timeout:raise TimeoutError('Phase cap; incomplete')
                            reader.join(timeout=20)
                            if process.poll() is None:print('Worker active;',round(time.monotonic()-start),'s elapsed; see stage/step logs.',flush=True)
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
            run(['git','init',REPO]);run(['git','remote','add','origin','https://github.com/Supzxjee/Image_Captioning.git'],cwd=REPO)
            run(['git','fetch','--depth','1','origin',SOURCE_COMMIT],cwd=REPO);run(['git','checkout','--detach','FETCH_HEAD'],cwd=REPO)
            if subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()!=SOURCE_COMMIT:raise ValueError('Wrong source commit')
            for name,h in SOURCE_ASSETS.items():
                if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=h:raise ValueError('Changed source '+name)
            os.environ['PYTHONPATH']=str(REPO);HERE=REPO/'experiments/concept_mass_grounding'
            run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','pycocoevalcap==1.2','h5py','numpy','Pillow','nltk==3.9.2','future'])
            run([sys.executable,'-m','pip','install','-q','--no-deps','git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'])
            run([sys.executable,'-u','-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"])
            run(['java','-version']);CHAIR=REPO.parent/'chair'
            run(['git','init',CHAIR]);run(['git','remote','add','origin','https://github.com/Maxlinn/CHAIR-metric-standalone.git'],cwd=CHAIR)
            run(['git','fetch','--depth','1','origin','4087a26211aa2339b9a76307cb8f0321ef691d0a'],cwd=CHAIR);run(['git','checkout','--detach','FETCH_HEAD'],cwd=CHAIR)
            if hashlib.sha256((CHAIR/'chair.pkl').read_bytes()).hexdigest()!='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3':raise ValueError('Changed CHAIR cache')
            sys.path.insert(0,str(REPO));sys.path.insert(0,str(HERE))
            import importlib.util
            policy_spec=importlib.util.spec_from_file_location('concept_mass_policy_'+SOURCE_COMMIT,HERE/'concept_data.py')
            policy_module=importlib.util.module_from_spec(policy_spec);policy_spec.loader.exec_module(policy_module)
            POLICY=policy_module.POLICY
            CONFIG=REPO.parent/'config.json'
            cfg=dict(arm=ARM,checkpoint=str(CHECKPOINT),dataset_json=str(COCO_JSON),legacy_prompt=str(LEGACY_PROMPT),
                visual_cache=str(VISUAL_CACHE),detections=str(DETECTIONS),metadata=str(REPO/'experiments/semantic_grounding_method/data/gain_metadata.json.gz'),
                chair=str(CHAIR),output=str(OUTPUT),policy=POLICY,source_commit=SOURCE_COMMIT,source_assets=SOURCE_ASSETS,
                previous_prepared=str(PREVIOUS_PREPARED) if PREVIOUS_PREPARED is not None else None,
                reported_remaining_gpu_hours=REPORTED_REMAINING_GPU_HOURS,quota_note='User report24h; account distribution/billing not inferred; enabledGPU CPU phases extra')
            CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf8')
            '''),cell('code','''
            success=False
            try:
                run([sys.executable,'-u',HERE/'test_concept_data.py'],cwd=REPO)
                run([sys.executable,'-u',HERE/'test_recover_stage.py'],cwd=REPO)
                run([sys.executable,'-u',REPO/'experiments/shared_pair_readout/metric_preflight.py'],cwd=REPO)
                run([sys.executable,'-u',HERE/'run_concept.py','prepare','--config',CONFIG],cwd=REPO,timeout=3720)
                run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0',
                    HERE/'run_concept.py','worker','--config',CONFIG],cwd=REPO,timeout=3600)
                run([sys.executable,'-u',HERE/'run_concept.py','evaluate','--config',CONFIG],cwd=REPO,timeout=900)
                result=json.loads((OUTPUT/'summary.json').read_text())
                if result['status']!='ENDPOINT_500_EVALUATED_WAIT_MATCHED_C_G':raise RuntimeError('Incomplete evaluation')
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
                    z.writestr('execution_status.json',json.dumps(dict(success=success,arm=ARM,source_commit=SOURCE_COMMIT,assistant_launched_job=False),indent=2))
                print('SEND SMALL REVIEW ZIP:',REVIEW,flush=True)
                if (OUTPUT/'checkpoint.bin').is_file():print('KEEP CHECKPOINT:',OUTPUT/'checkpoint.bin',flush=True)
                if not success:print('INCOMPLETE; send ZIP/log, do not accept as a method result.',flush=True)
            if success:print('DONE',ARM,'; send BOTH C/G reviewZIPs for matched comparison, not automatic promotion.',flush=True)
            '''),cell('markdown','''
            Giữ `checkpoint.bin`. Gửi `concept_mass_C_v2_review.zip` và `concept_mass_G_v2_review.zip`.
            Hai notebook chưa được assistant chạy. Actual gradients/DDP/numeric parity chỉ được kiểm ở preflight user-run.
            Endpoint chỉ là pilot; G cần có tín hiệu tốt hơn A cả CIDEr lẫn CHAIR, có ích so C và giữ object recall.
            Không tự đổi lambda, chọn checkpoint, chạy full/seed2 hoặc upload Dataset.
            ''')]
        cells[2]['source']='SOURCE_COMMIT='+repr(commit)+'\nSOURCE_ASSETS='+repr(assets)+'\n'+cells[2]['source']
        setup=cells[3]['source'];cells[3]=cell('code','try:\n'+textwrap.indent(setup,'    ')+'''
except BaseException:
    with zipfile.ZipFile(REVIEW,'w',zipfile.ZIP_DEFLATED) as z:
        if LOG.is_file():z.write(LOG,'notebook.log')
        z.writestr('execution_status.json',json.dumps(dict(success=False,arm=ARM,stage='bootstrap',source_commit=SOURCE_COMMIT)))
        z.writestr('bootstrap_error.txt',traceback.format_exc())
    print('BOOTSTRAP FAILED; SEND:',REVIEW,flush=True)
    raise
''')
        nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),language_info=dict(name='python'),kaggle=dict(isGpuEnabled=True,isInternetEnabled=True)),cells=cells)
        path=HERE/f'CONCEPT_MASS_{arm}_2GPU.ipynb';path.write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        for c in cells:
            if c['cell_type']=='code':compile(c['source'],str(path),'exec')
        if path.stat().st_size>=1000000:raise ValueError('Notebook too large')
        package.append(dict(file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (HERE/'NOTEBOOK_PACKAGE.json').write_text(json.dumps(dict(source_commit=commit,source_assets=assets,notebooks=package,local_neural_execution=False,kaggle_launched=False),indent=2)+'\n',encoding='utf8')
    print(json.dumps(package,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--core',required=True);make(p.parse_args().core)
