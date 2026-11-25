"""Bundle pinned successful cache/model/VALID helpers; never execute a notebook."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT/'experiments/clip_patch_baseline/CLIP_PATCH_CSA_SCST_ONLY1000_MERGED_DATASET_2GPU.ipynb'


def literals(source):
    values = {}
    for n in ast.parse(source).body:
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name):
            try:
                values[n.targets[0].id] = ast.literal_eval(n.value)
            except (ValueError, TypeError):
                pass
    return values


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Unexpected anchor: '+old[:90])
    return source.replace(old, new, 1)


def build():
    parent = json.loads(PARENT.read_text(encoding='utf-8'))
    setup, execution = [''.join(parent['cells'][i]['source']) for i in (1,3)]
    constants = literals(setup)
    payloads, origins = {}, {}
    def pinned(commit, path, digest, name):
        raw = subprocess.check_output(['git','show',commit+':'+path], cwd=ROOT)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError('Pinned source SHA differs: '+path)
        payloads[name] = raw.decode('utf-8')
        origins[name] = dict(commit=commit, path=path, sha256=digest)
    for path, name in constants['MAPPING'].items():
        pinned(constants['REVISION'], path, constants['HASHES'][path], name)
    for path, digest in constants['TRAIN_PINS'].items():
        name = 'bank_layers.py' if path == 'experiments/csa_paper/model.py' else Path(path).name
        pinned(constants['TRAIN_REVISION'], path, digest, name)
    for node in ast.walk(ast.parse(execution)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'write_text' and node.args and isinstance(node.args[0], ast.Constant):
            source = node.args[0].value
            target = node.func.value
            if isinstance(source, str) and isinstance(target, ast.BinOp) and isinstance(target.op, ast.Div) and isinstance(target.right, ast.Constant):
                name = target.right.value
                if name in ('csa_per_caption_test_data.py','run_csa_per_caption_test.py','scst_cache.py','scst_dataset_mount.py'):
                    payloads[name] = source
                    origins[name] = dict(parent_notebook=PARENT.name, embedded=True)
    pinned('cd3c8eb372f39df4454a2da27735e00e28f7f394', 'captioning/scst.py',
        '3a9a1f32fb27dae0447066e1388a31461f3f0781d52db5861880b037a572a78e', 'captioning/scst.py')
    payloads['captioning/__init__.py'] = ''
    call = next(n for n in ast.walk(ast.parse(execution)) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'materialize_part0')
    parts, ids_sha = ast.literal_eval(call.args[2]), ast.literal_eval(call.args[3])
    payloads['CACHE_PARTS.json'] = json.dumps(dict(parts=parts, ids_sha256=ids_sha,
        source_sha256=parts[0]['source_sha256']), indent=2)+'\n'
    payloads['csa_per_caption_test_data.py'] = once(payloads['csa_per_caption_test_data.py'],
        "if training!='2.11.0+cu128' or current not in ('2.11.0+cu128','2.11.0+cu130'):",
        "if training not in ('2.11.0+cu128','2.11.0+cu130') or current not in ('2.11.0+cu128','2.11.0+cu130'):")
    payloads['run_csa_per_caption_test.py'] = once(payloads['run_csa_per_caption_test.py'],
        "return ck,d.runtime_policy('2.11.0+cu128',str(torch.__version__))",
        "return ck,d.runtime_policy(read(Path(cfg['output'])/'training_runtime.json')['torch_version'],str(torch.__version__))")
    for name in ('beam_policy.py','full_epoch_data.py','run_beam_epoch.py','PROTOCOL.md'):
        payloads[name] = (HERE/name).read_text(encoding='utf-8')
    hashes = {name:hashlib.sha256(source.encode()).hexdigest() for name,source in payloads.items()}
    for name, source in payloads.items():
        if name.endswith('.py'):
            ast.parse(source, filename=name)
    reference = json.loads((ROOT/'experiments/clip_patch_baseline/CSA_PER_CAPTION_BEST9_RERANK_VALID5000_RESULT.json').read_text(encoding='utf-8'))['metrics']['decoder_rank0']
    reference = {k:v for k,v in reference.items() if isinstance(v,(int,float))}
    header = '''import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,zipfile,shutil
from pathlib import Path
CHECKPOINT_ROOT='/kaggle/input/datasets/ducanh3/upt-coco-full-cache'
CACHE_ROOT='/kaggle/input/datasets/ducanh3/csa-scst-full-karpathy-cache-v1'
COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
RESUME_ROOT='' # Whole previous output, only this exact beam-SCST epoch.
RUN_TAG='run1'
WALL_HOURS=11.5 # Per-invocation stop cap, NOT ETA or billed GPU quota.
OUTPUT=Path(f'/kaggle/working/beam_scst_epoch1_seed42_{RUN_TAG}')
for path in (Path(CHECKPOINT_ROOT),Path(CACHE_ROOT),COCO_JSON):
    if not path.exists():raise FileNotFoundError('Attach/correct input before GPU: '+str(path))
if OUTPUT.exists():raise RuntimeError('Preserve output; use a new run tag/session')
if not 1<=WALL_HOURS<=11.5:raise ValueError('Reviewed invocation cap is at most11.5h')
if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<8*1024**3:raise RuntimeError('Need8GiB scratch/output space')
OUTPUT.mkdir();SOURCE=Path(tempfile.mkdtemp(prefix='beam-scst-epoch1-',dir='/tmp'))
DEPS=SOURCE/'deps';CHAIR=SOURCE/'chair';CONFIG=SOURCE/'config.json';LOG=OUTPUT/'notebook.log'
PARENT_DEADLINE=time.monotonic()+WALL_HOURS*3600
os.environ['PYTHONPATH']=str(SOURCE);os.environ['OMP_NUM_THREADS']='1'
cfg=dict(semantic=True,output=str(OUTPUT),checkpoint_root=CHECKPOINT_ROOT,cache_root=CACHE_ROOT,
    dataset_json=str(COCO_JSON),metadata_root=str(SOURCE/'metadata'),chair=str(CHAIR),
    resume_root=RESUME_ROOT,train_seconds=3600,worker_seconds=1800)
def write_config():
    CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    (OUTPUT/'protocol.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
def redact(line):return line # No secrets or authentication in this notebook.
'''
    header += '\nSOURCE_PAYLOADS='+repr(payloads)+'\nSOURCE_SHA256='+repr(hashes)+'\nREFERENCE_VALID='+repr(reference)+'\n'
    runner = '''complete=False
try:
    preserved=OUTPUT/'source';preserved.mkdir()
    for name,source in SOURCE_PAYLOADS.items():
        target=SOURCE/name
        if not target.resolve().is_relative_to(SOURCE.resolve()):raise ValueError('Unsafe embedded source path')
        raw=source.encode('utf-8')
        if hashlib.sha256(raw).hexdigest()!=SOURCE_SHA256[name]:raise ValueError('Changed embedded source')
        for root in (SOURCE,preserved):
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    (OUTPUT/'source_receipt.json').write_text(json.dumps(dict(sha256=SOURCE_SHA256),indent=2))
    cfg['source_hashes']=SOURCE_SHA256
    write_config()
    run([sys.executable,'-u',SOURCE/'cuda_runtime_guard.py','--output',OUTPUT],60)
    shutil.copyfile(OUTPUT/'cuda_runtime.json',OUTPUT/'training_runtime.json')
    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS,'pycocoevalcap==1.2','nltk==3.9.2','future'],600)
    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',DEPS,'git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'],600)
    os.environ['PYTHONPATH']=str(DEPS)+os.pathsep+str(SOURCE)
    run([sys.executable,'-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"],300)
    sys.path.insert(0,str(SOURCE))
    from full_epoch_data import prepare
    prepare(cfg);write_config()
    environment=json.loads((OUTPUT/'environment.json').read_text())
    os.environ['PATH']=str(Path(environment['java']).parent)+os.pathsep+os.environ['PATH']
    launch=[sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2','--max_restarts=0']
    cfg['train_seconds']=min(2400,PARENT_DEADLINE-time.monotonic()-1800);write_config()
    if cfg['train_seconds']<300:raise TimeoutError('Insufficient time for memory/throughput guard')
    run([*launch,SOURCE/'run_beam_epoch.py','preflight',CONFIG],cfg['train_seconds']+30)
    guards=[json.loads((OUTPUT/f'beam_preflight_rank{rank}.json').read_text()) for rank in (0,1)]
    if not all(g['passed'] for g in guards):raise ValueError('Beam preflight incomplete')
    print('Guard PASS; peakGiB:',[round(g['peak_allocated_bytes']/1024**3,2) for g in guards],
          'projected training hours:',round(max(g['projected_training_seconds'] for g in guards)/3600,2),'not an ETA/quota guarantee',flush=True)
    cfg['train_seconds']=PARENT_DEADLINE-time.monotonic()-1800;write_config()
    if cfg['train_seconds']<600:raise TimeoutError('Insufficient training budget after setup')
    run([*launch,SOURCE/'run_beam_epoch.py','train',CONFIG],cfg['train_seconds']+60)
    trained=json.loads((OUTPUT/'training_complete.json').read_text())
    if not trained['complete'] or trained['train_images']!=113287 or trained['updates']!=3541:raise ValueError('Incomplete one epoch')
    cfg.update(evaluate_pilot=True,eval_update=3541)
    cfg['worker_seconds']=min(1800,PARENT_DEADLINE-time.monotonic()-900);write_config()
    if cfg['worker_seconds']<300:raise TimeoutError('Epoch complete; endpoint saved, VALID phase needs recovery')
    evaluator=SOURCE/'run_csa_per_caption_test.py'
    run([*launch,evaluator,'worker','--config',CONFIG],cfg['worker_seconds']+30)
    for mode,cap in (('tokenize',180),('basic',240),('chair',600)):
        run([sys.executable,'-u',evaluator,mode,'--config',CONFIG],cap)
    metrics={}
    for mode in ('basic','chair'):metrics.update(json.loads((OUTPUT/(mode+'.json')).read_text())['metrics'])
    workers=[json.loads((OUTPUT/f'worker{rank}.json').read_text()) for rank in (0,1)]
    metrics['NLL']=sum(w['NLL_sum'] for w in workers)/sum(w['NLL_tokens'] for w in workers)
    receipt=json.loads((OUTPUT/'pilot_checkpoint_receipt.json').read_text())
    result=dict(status='BEAM_SCST_EPOCH1_VALID_COMPLETE',split='val',images=5000,train_images=113287,updates=3541,
        test_used=False,reranking=False,checkpoint=receipt,metrics=metrics,CE_reference=REFERENCE_VALID,
        delta={k:metrics[k]-v for k,v in REFERENCE_VALID.items() if k in metrics},
        limits=['One epoch, one seed, author-style mechanism on our frozen CLIP/cache architecture.',
                'Accumulation4/local4 differs in dropout RNG from author local16 batch.',
                'No TEST or automatic second epoch; unfavorable CHAIR retained.'])
    (OUTPUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    complete=True
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc());raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,training=True,test_used=False,
        requested_train_images=113287,requested_epochs=1,reranking=False)))
    review=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in OUTPUT.rglob('*'):
            if path.is_file() and path.suffix in ('.json','.jsonl','.txt','.log','.py','.md'):
                archive.write(path,str(path.relative_to(OUTPUT)))
    print('SEND REVIEW:',review,flush=True);print('KEEP ENTIRE OUTPUT:',OUTPUT,flush=True)
    print('LATEST:',OUTPUT/'checkpoint.bin','ENDPOINT:',OUTPUT/'pilot_checkpoint.bin',flush=True)
    print('ONE EPOCH + VALID COMPLETE' if complete else 'INCOMPLETE; preserve output. No automatic restart or fallback.',flush=True)
'''
    intro = '''# Beam-mean SCST — một epoch TRAIN / VALID5000

Người dùng báo16hGPU. GPU T4 ×2, Internet ON, Save & Run All. **Chưa chạy thực tế trên GPU; bộ nhớ/tốc độ được đo trước train.**

Gắn3 Dataset đã có: `ducanh3/upt-coco-full-cache` (best CE9, không phải SCST); `ducanh3/csa-scst-full-karpathy-cache-v1`; `vuthetam/mscoco-2014`. Path nằm ở cell cấu hình. Không cần Secret/API; nguồnCE được kiểm SHA b0c5bde… Không cần visual-cache/prompt175 riêng vì full cache chứa TRAIN/VALID features và ROI.

Đúng113287ảnhTRAIN một lần,3541updates/global32,tail7. Beam5 cógradient/trainmode, baseline reward trung bình5beam, mean20tokenlogp, SCSTthuần/lr4e-6/FP32. Local4/GPU ×4accumulation để giảm memory; đây là áp dụng cơ chế tác giả lên mô hình mình, không tái lập backbone/tokenizer toàn bài. Không CE/rerank/TEST/epoch2.

Preflight kiểm beam/EOS/gradient/tail7, so beam eval với decoder cũ, chạy6updates kiểm/đo cả3shard rồi bỏ; train khởi tạo lại từCE. Nếu OOM hoặc projection vượt14h, dừng gửi review, không tự đổi batch/precision. Cap11.5h **mỗi phiên**, không hứa đủ epoch hay vừa quota16h. Hết cap giữ cả OUTPUT với checkpoint+Adam+RNG+cursor; không train lại từ đầu. Resume đúng recipe bằng RESUME_ROOT và RUN_TAG mới; nếu chỉ còn Save Version, gửi số để khóa APIresume.

VALID endpoint chấm CIDEr/BLEU1/4/ROUGE-L/CHAIRs/i/Recall/NLL; METEOR/SPICE để CPU bổ sung nếu cần. Caption tham chiếu TRAIN dùngreward; VALID chỉ đánh giá. Giữ mọi metric kể cả CHAIR tăng. ReviewZIP không chứa trọng số: giữ toàn bộOutput.
'''
    def cell(kind, source):
        value = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code':
            compile(source,'notebook','exec')
            value.update(execution_count=None, outputs=[])
        return value
    nb = dict(nbformat=4, nbformat_minor=5, metadata=parent['metadata'],
        cells=[cell('markdown',intro),cell('code',header),parent['cells'][2],cell('code',runner)])
    path = HERE/'BEAM_SCST_FULL_EPOCH1_VALID5000_2GPU.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    manifest = dict(parent_notebook=str(PARENT.relative_to(ROOT)),parent_sha256=sha_bytes(PARENT.read_bytes()),
        notebook_sha256=sha_bytes(path.read_bytes()),sha256=hashes,origins=origins,
        source_checkpoint_sha256=parts[0]['source_sha256'],test_used=False,models_executed=False)
    (HERE/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print(path.name,len(payloads),'embedded files; no models executed')


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


if __name__ == '__main__':
    build()
