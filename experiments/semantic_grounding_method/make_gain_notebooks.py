"""Create two portable, unexecuted Kaggle notebooks; no ML imports."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('notebook_helpers',ROOT/'experiments/upt_hidden_pair/make_notebooks.py')
nb=importlib.util.module_from_spec(spec);spec.loader.exec_module(nb);nb.HERE=HERE


def build():
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    names=['gain_data.py','gain_model.py','gain_runtime.py','test_gain_data.py','test_gain_model.py']
    files=[f'experiments/semantic_grounding_method/{n}' for n in names]
    files+=['captioning/'+n+'.py' for n in ['models','scst','scst_runtime','object_reward','object_consistency','tokenizer','config','semantic_prompts','visual_cache']]
    hashes={}
    for n in files:
        source=(ROOT/n).read_text(encoding='utf-8');ast.parse(source,filename=n)
        committed=subprocess.check_output(['git','show',commit+':'+n],cwd=ROOT).decode('utf-8').replace('\r\n','\n')
        if source!=committed:raise ValueError('Commit runtime sources before generating: '+n)
        hashes[n]=hashlib.sha256(source.encode()).hexdigest()
    metadata=json.loads((HERE/'METADATA_PACKAGE.json').read_text())
    helper=(HERE/'restore_gain_metadata.py').read_text(encoding='utf-8')
    committed_helper=subprocess.check_output(['git','show',commit+':experiments/semantic_grounding_method/restore_gain_metadata.py'],cwd=ROOT).decode('utf-8').replace('\r\n','\n')
    if helper!=committed_helper:raise ValueError('Commit restoration helper before generating')
    helper_sha=hashlib.sha256(helper.encode()).hexdigest()
    records={}
    for label,arms in [('CH',['C','H']),('T',['T'])]:
        cells=[nb.md(f'''
            # Grounded probability gain — {label} · T4×2
            Hai notebook dùng CÙNG bộ chuẩn bị, cùng khởi tạo,2.000ảnh,500update/nhánh.
            C và H luôn khởi tạo lại từ A + nhánh zero; H KHÔNG tiếp tục checkpoint C.

            **Lần đầu:** chạy C/H ở MODE='prepare' với CPU để chép cache có sẵn.
            Lưu toàn thư mục `grounded_gain_preparation` thành dataset private; gắn dataset
            đó vào cả hai tài khoản. Sau đó đổi C/H sang MODE='train' và chạy hai notebook
            C/H và T song song bằng T4×2, Internet ON, Save & Run All.

            Bước prepare không trích đặc trưng mới hoặc huấn luyện. Không cần UPT mới,
            ảnh gốc hay API token. Các path đầu vào nằm ở cell tiếp theo.
            Chưa có model run/GPU preflight trên máy local; các checks torch chạy ở đây.
            Grounding trên held-out chưa có labelbank: ghi UNAVAILABLE, không coi heatmap
            hay loss train là xác nhận ngữ nghĩa/factuality. Chỉ end checkpoint; không sweep/test/rerank.
            '''),nb.code(f'''
            from pathlib import Path
            MODE={('prepare' if label=='CH' else 'train')!r}
            ARMS={arms!r}
            PREPARED_ROOT=None  # Path thư mục chứa receipt.json; None tự tìm đúng metadata hash.
            AVAILABLE_DEVICE_GPU_HOURS=25.0  # User xác nhận25giờ MỖI tài khoản; dùng làm ngân sách bảo thủ, không cộng hai tài khoản.
            QUOTA_UNIT='device_gpu_hours'
            REPORTED_QUOTA=dict(hours_per_account=25,source='user2026-10-27',display_unit='Kaggle GPU hours; device budget entered conservatively',maximum_session='not supplied')
            CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/pair-source-fullframe-ce/upt_hidden_caption_A/checkpoint.bin')
            COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
            LEGACY_PROMPT=Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
            VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
            DETECTIONS=Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
            PREPARED_OUTPUT=Path('/kaggle/working/grounded_gain_preparation')
            OUTPUT_ROOT=Path('/kaggle/working')
            assert MODE in ('prepare','train')
            assert ARMS=={arms!r},'Preserve matched C/H or T protocol.'
            if MODE=='train':
                assert AVAILABLE_DEVICE_GPU_HOURS is not None and QUOTA_UNIT=='device_gpu_hours','Record verified per-account GPU balance/unit first.'
                assert AVAILABLE_DEVICE_GPU_HOURS >= {4.4 if label=='CH' else 2.2},'Insufficient conservative allocation for this account; do not combine accounts.'
            print('Grounded gain',MODE,ARMS,'— bootstrap started',flush=True)
            '''),nb.code(f'''
            import hashlib,json,os,shutil,signal,subprocess,sys,tempfile,threading,time,traceback,zipfile
            os.environ.update(PYTHONUNBUFFERED='1',HDF5_USE_FILE_LOCKING='FALSE',PYTHONHASHSEED='42',
                CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')
            REPO=Path(tempfile.mkdtemp(prefix='grounded-gain-{label.lower()}-'))/'repo'
            OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
            LOG=OUTPUT_ROOT/'grounded_gain_{label.lower()}_notebook.log'
            PROJECT_COMMIT={commit!r}
            SOURCE_HASHES={hashes!r}
            METADATA_SHA={metadata['metadata_sha256']!r}
            RESTORATION_HELPER_SHA={helper_sha!r}
            def fail(message):
                (OUTPUT_ROOT/'grounded_gain_{label.lower()}_failure.json').write_text(json.dumps(dict(error=str(message),stack=traceback.format_stack())))
                raise RuntimeError(message)
            def run(args,cwd=None,timeout=900):
                args=list(map(str,args));print('\\nRunning:',' '.join(args),flush=True)
                with LOG.open('a',encoding='utf-8') as log:
                    log.write('\\nRunning: '+' '.join(args)+'\\n');log.flush()
                    try:
                        process=subprocess.Popen(args,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                            text=True,errors='replace',bufsize=1,start_new_session=True)
                    except Exception as error:
                        log.write(traceback.format_exc());log.flush();fail(str(error))
                    def drain():
                        for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                    reader=threading.Thread(target=drain,daemon=True);reader.start();start=time.monotonic()
                    try:
                        while process.poll() is None:
                            if time.monotonic()-start>timeout:
                                os.killpg(process.pid,signal.SIGTERM)
                                try:process.wait(timeout=10)
                                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                                fail('INCOMPLETE_COMPUTATIONAL_NO_GO: fixed timeout exceeded')
                            reader.join(timeout=20)
                            if process.poll() is None:print('Worker(s) active; elapsed',round(time.monotonic()-start,1),'s; see stage log.',flush=True)
                        reader.join()
                        if process.returncode:fail(f'Command failed ({{process.returncode}}); preserve {{LOG}}')
                    finally:
                        if process.poll() is None:os.killpg(process.pid,signal.SIGTERM);process.wait()
                        reader.join(timeout=5)
            def preserve_review(output):
                # stdlib fallback must not run source guards or hide the original failure.
                try:
                    output=Path(output);output.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(LOG,output/'notebook.log')
                    with zipfile.ZipFile(str(output)+'_review.zip','w',zipfile.ZIP_DEFLATED) as archive:
                        for p in sorted(output.rglob('*')):
                            if p.is_file() and p.suffix in ('.json','.jsonl','.log','.py','.md'):archive.write(p,p.relative_to(output))
                    print('SEND REVIEW:',str(output)+'_review.zip',flush=True)
                except Exception as error:
                    print('Could not build review ZIP; preserve original output/log:',output,LOG,repr(error),flush=True)
            REPO.mkdir()
            run(['git','init',REPO]);run(['git','remote','add','origin','https://github.com/Supzxjee/Image_Captioning.git'],cwd=REPO)
            run(['git','fetch','--depth','1','origin',PROJECT_COMMIT],cwd=REPO)
            run(['git','checkout','--detach','FETCH_HEAD'],cwd=REPO)
            if subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()!=PROJECT_COMMIT:fail('Wrong project commit')
            os.environ['PYTHONPATH']=str(REPO)
            HERE=REPO/'experiments/semantic_grounding_method'
            METADATA=HERE/'data/gain_metadata.json.gz'
            if hashlib.sha256(METADATA.read_bytes()).hexdigest()!=METADATA_SHA:fail('Metadata digest mismatch')
            for name,expected in SOURCE_HASHES.items():
                if hashlib.sha256((REPO/name).read_bytes()).hexdigest()!=expected:fail('Source digest mismatch: '+name)
            if hashlib.sha256((HERE/'restore_gain_metadata.py').read_bytes()).hexdigest()!=RESTORATION_HELPER_SHA:fail('Restoration helper digest mismatch')
            '''),nb.code('''
            run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','pycocotools','pycocoevalcap==1.2','h5py','numpy','Pillow','nltk==3.9.2','future'])
            run([sys.executable,'-m','pip','install','-q','--no-deps','git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'])
            run([sys.executable,'-u','-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"])
            run(['java','-version'])
            CHAIR=REPO.parent/'chair'
            run(['git','init',CHAIR]);run(['git','remote','add','origin','https://github.com/Maxlinn/CHAIR-metric-standalone.git'],cwd=CHAIR)
            run(['git','fetch','--depth','1','origin','4087a26211aa2339b9a76307cb8f0321ef691d0a'],cwd=CHAIR)
            run(['git','checkout','--detach','FETCH_HEAD'],cwd=CHAIR)
            assert hashlib.sha256((CHAIR/'chair.pkl').read_bytes()).hexdigest()=='73ce11225febb97df38c5bfad60b3ff6cc25852e9df9a1330673155b4be851c3'
            run([sys.executable,'-u',HERE/'test_gain_data.py'],cwd=REPO)
            CONFIG_BASE=dict(checkpoint=str(CHECKPOINT),dataset_json=str(COCO_JSON),metadata=str(METADATA),
                chair=str(CHAIR),sources=SOURCE_HASHES,project_commit=PROJECT_COMMIT)
            RUNTIME=HERE/'gain_runtime.py'
            '''),nb.code('''
            if MODE=='prepare':
                assert not PREPARED_OUTPUT.exists(),'Preserve previous output; change path or use a fresh session.'
                cfg=dict(CONFIG_BASE,legacy_prompt=str(LEGACY_PROMPT),visual_cache=str(VISUAL_CACHE),detections=str(DETECTIONS),
                    prepared_output=str(PREPARED_OUTPUT),output=str(PREPARED_OUTPUT))
                config=REPO.parent/'prepare.json';config.write_text(json.dumps(cfg,indent=2))
                try:
                    run([sys.executable,'-u',RUNTIME,'prepare','--config',config],cwd=REPO,timeout=1800)
                except Exception as error:
                    PREPARED_OUTPUT.mkdir(parents=True,exist_ok=True)
                    (PREPARED_OUTPUT/'failure.json').write_text(json.dumps(dict(status='INCOMPLETE_COMPUTATIONAL_NO_GO',error=str(error),traceback=traceback.format_exc())))
                    raise
                finally:
                    if PREPARED_OUTPUT.is_dir():
                        preserve_review(PREPARED_OUTPUT)
                print('PREPARATION COMPLETE. Save entire folder as a private dataset, attach to both accounts:',PREPARED_OUTPUT,flush=True)
                print('Then switch C/H MODE to train; T already uses train. This pass has NOT trained C/H/T.',flush=True)
            '''),nb.code('''
            if MODE=='train':
                import torch
                assert torch.cuda.device_count()==2,'Choose T4x2 on EACH notebook.'
                assert all('T4' in torch.cuda.get_device_name(i) for i in (0,1)),'Protocol requires the same T4x2 hardware on both accounts.'
                if PREPARED_ROOT is None:
                    matches=[]
                    for p in Path('/kaggle/input').rglob('receipt.json'):
                        try:r=json.loads(p.read_text())
                        except (ValueError,OSError):continue
                        if r.get('metadata_sha256')==METADATA_SHA and r.get('status')=='PREPARATION_COMPLETE':matches.append(p.parent)
                    assert len(matches)==1,f'Attach exactly one complete preparation dataset; found {len(matches)}. Set PREPARED_ROOT explicitly if needed.'
                    PREPARED_ROOT=matches[0]
                sys.path.insert(0,str(HERE))
                from gain_data import checked_cache
                from restore_gain_metadata import restore_metadata_view
                PREPARED_ROOT=restore_metadata_view(PREPARED_ROOT,METADATA,METADATA_SHA)
                checked_cache(PREPARED_ROOT,METADATA_SHA,hash_payloads=False)  # rank0 verifies full payloads before each GPU phase.
                run([sys.executable,'-u',HERE/'test_gain_model.py'],cwd=REPO,timeout=120)
                for arm in ARMS:
                    output=OUTPUT_ROOT/('grounded_gain_'+arm)
                    if (output/'summary.json').is_file():
                        previous=json.loads((output/'summary.json').read_text())
                        expected_receipt=hashlib.sha256((Path(PREPARED_ROOT)/'receipt.json').read_bytes()).hexdigest()
                        assert previous['status']=='COMPLETE_WAIT_MATCHED_CHT' and previous['arm']==arm and previous['updates']==500
                        assert previous['sources']==SOURCE_HASHES and previous['preparation_receipt_sha256']==expected_receipt
                        assert hashlib.sha256((output/'checkpoint.bin').read_bytes()).hexdigest()==previous['checkpoint_sha256']
                        print('Preserve and reuse completed matched arm:',arm,flush=True)
                        continue
                    assert not output.exists(),'Preserve previous arm; use a fresh session/output root.'
                    output.mkdir(parents=True)
                    cfg=dict(CONFIG_BASE,prepared=str(PREPARED_ROOT),output=str(output),arm=arm,
                        quota=dict(available=AVAILABLE_DEVICE_GPU_HOURS,unit=QUOTA_UNIT,reported=REPORTED_QUOTA))
                    restored=Path(PREPARED_ROOT)/'metadata_restoration.json'
                    if restored.is_file():cfg['metadata_restoration']=json.loads(restored.read_text())
                    config=output/'protocol.json';config.write_text(json.dumps(cfg,indent=2))
                    for n in SOURCE_HASHES:
                        target=output/'source'/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/n,target)
                    try:
                        run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0',RUNTIME,'preflight','--config',config],cwd=REPO,timeout=600)
                        cfg['arm_started_unix']=time.time();config.write_text(json.dumps(cfg,indent=2))
                        run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0',RUNTIME,'worker','--config',config],cwd=REPO,timeout=2700)
                        remaining=2700-(time.time()-cfg['arm_started_unix'])
                        assert remaining>0,'Arm time cap exhausted before CPU endpoint scoring.'
                        run([sys.executable,'-u',RUNTIME,'evaluate','--config',config],cwd=REPO,timeout=remaining)
                    except Exception as error:
                        (output/'failure.json').write_text(json.dumps(dict(status='INCOMPLETE_COMPUTATIONAL_NO_GO',error=str(error),arm=arm,traceback=traceback.format_exc())))
                        raise
                    finally:
                        preserve_review(output)
                    print('DONE',arm,'SEND:',str(output)+'_review.zip','KEEP:',output/'checkpoint.bin',flush=True)
            '''),nb.md('''
            Gửi đủ3ZIP review C/H/T; giữ riêng checkpoint.bin và nguyên dataset chuẩn bị.
            Một nhánh hoàn tất chưa đủ kết luận. So sánh pairedCI10.000samples với cùngA/dev1000,
            kiểmT−A>=.01,CI(T−A,T−C,T−H)>0,CHAIRkhôngxấu,recall>=A−.005.
            No-Go dừng recipe, không chứng minh trần kiến trúc. GPU/time/precisionfailure là
            INCOMPLETE, không phải kết quả chất lượng âm. Không sửa tolerance/loss/seed để cứu lượt chạy.
            Không có held-out phrase-grounding gold trong gói này: không tuyên bố causalgrounding
            hoặc đóng góp paper chỉ từ loss/heatmap đẹp. Các giới hạn ghi trong protocol/review.
            ''')]
        name=f'GROUNDED_GAIN_{label}_2GPU.ipynb';nb.write(name,cells)
        records[label]=dict(file=name,bytes=(HERE/name).stat().st_size,sha256=hashlib.sha256((HERE/name).read_bytes()).hexdigest())
    (HERE/'NOTEBOOK_PACKAGE.json').write_text(json.dumps(dict(status='PREPARED_UNEXECUTED',project_commit=commit,
        notebooks=records,sources_sha256=hashes,metadata_sha256=metadata['metadata_sha256'],local_model_execution=False),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':build()
