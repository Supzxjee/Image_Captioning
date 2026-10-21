"""Create three portable unexecuted notebooks. Never runs inference/training."""
import ast
import base64
import hashlib
import json
from pathlib import Path
import textwrap
import zipfile
import zlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COMMIT = '31fdec14486ca8ab7ce364a8ffc596f1f9acdd34'
RUNTIME = ['hidden_data.py', 'hidden_model.py', 'extract_hidden.py', 'run_hidden.py', 'hidden_metric_smoke.py',
           'test_hidden_data.py', 'test_hidden_hook.py', 'test_hidden_model.py', 'test_hidden_ddp.py',
           'test_hidden_hdf5.py', 'excluded_phrase_ids.json']


def sources():
    result = {name: (HERE / name).read_text(encoding='utf-8') for name in RUNTIME}
    for name in ['pilot_data.py', 'pilot_model.py', 'run_pilot.py']:
        result[name] = (ROOT / 'experiments/upt_alignment_pilot' / name).read_text(encoding='utf-8')
    for name in ['captioning/upt_reference.py', 'captioning/scst_runtime.py']:
        result[name] = (ROOT / name).read_text(encoding='utf-8')
    return result


def code(s):
    s = textwrap.dedent(s).strip() + '\n'
    ast.parse(s)
    return dict(cell_type='code', execution_count=None, outputs=[], metadata={}, source=s.splitlines(True))


def md(s):
    return dict(cell_type='markdown', metadata={}, source=textwrap.dedent(s).strip().splitlines(True))


def bootstrap(bundle, name):
    packed = base64.b64encode(zlib.compress(json.dumps(bundle, ensure_ascii=False).encode('utf-8'), 9)).decode()
    hashes = {n: hashlib.sha256(s.encode()).hexdigest() for n, s in bundle.items()}
    return code(f'''
    import base64, hashlib, json, os, shutil, subprocess, sys, tempfile, zlib
    os.environ.update(PYTHONUNBUFFERED='1', HDF5_USE_FILE_LOCKING='FALSE', PYTHONHASHSEED='42',
        CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_XET='1')
    # Isolate each bootstrap: embedded tracked overlays from a prior run are intentional edits.
    REPO = Path(tempfile.mkdtemp(prefix='Image_Captioning-hidden-{name}-'))/'repo'
    LOG = Path('/tmp/upt_hidden_{name}.log')
    PROJECT_COMMIT = '{COMMIT}'
    def run(args, cwd=None):
        args = list(map(str, args))
        print('\\nRunning:', ' '.join(args), flush=True)
        with LOG.open('a', encoding='utf-8') as log:
            log.write('\\nRunning: ' + ' '.join(args) + '\\n'); log.flush()
            with subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, errors='replace', bufsize=1) as process:
                for line in process.stdout:
                    print(line, end='', flush=True); log.write(line); log.flush()
                status = process.wait()
        if status:
            raise RuntimeError(f'Command failed ({{status}}). Saved log: {{LOG}}')
    def checkout(url, path, revision):
        if not path.exists():
            run(['git','clone',url,path])
        assert (path/'.git').is_dir(), path
        assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=path,text=True).strip(), 'Preserve tracked edits; use a fresh runtime path.'
        run(['git','fetch','--depth','1','origin',revision], cwd=path)
        run(['git','checkout','--detach',revision], cwd=path)
    checkout('https://github.com/Supzxjee/Image_Captioning.git', REPO, PROJECT_COMMIT)
    SOURCES = json.loads(zlib.decompress(base64.b64decode('{packed}')).decode('utf-8'))
    SOURCE_HASHES = {hashes!r}
    for name, source in SOURCES.items():
        assert hashlib.sha256(source.encode()).hexdigest() == SOURCE_HASHES[name], name
        destination = REPO/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(source, encoding='utf-8')
    print('Pinned runtime and hash-checked embedded source READY', flush=True)
    ''' )


def write(name, cells):
    for i, cell in enumerate(cells):
        cell['id'] = f'hidden-{i:02d}'
    notebook = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),
        language_info=dict(name='python', version='3.13')), cells=cells)
    payload = json.dumps(notebook, ensure_ascii=False, indent=1).encode('utf-8')
    assert len(payload) < 1_000_000
    (HERE / name).write_bytes(payload)
    print(name, len(payload), 'bytes; unexecuted')


def main():
    bundle = sources()
    for name, source in bundle.items():
        if name.endswith('.py'):
            ast.parse(source)
    cache = [md('''
        # 1. UPT hidden-pair cache — 11.000 ảnh, hai GPU
        Chạy notebook này **trước**. Internet ON, GPU **T4 ×2**, Save & Run All.
        Chỉ trích đặc trưng UPT frozen; không huấn luyện caption, không cần Q-Former checkpoint.
        Full UPT chạy trước; sau đó chọn tối đa8 cặp human→nonhuman có cả hai detection scores **>0,2**.
        Đầu ra HDF5 không phải checkpoint ZIP nên Kaggle không cần giải nén bên trong.
        Giữ **toàn bộ** `upt_hidden_pairs_11k` làm dataset cho cả A và H. Cache cũ/prompt JSON không chứa các vector này.
        ''')]
    cache.append(code('''
        print('UPT HIDDEN CACHE: BOOTSTRAP START', flush=True)
        from pathlib import Path
        COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
        COCO_IMAGES = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/images')
        OUTPUT = Path('/kaggle/working/upt_hidden_pairs_11k')
        LOCAL_UPT_CHECKPOINT = None  # Optional official UPT-R50 HICO checkpoint, NOT Q-Former.
        assert COCO_JSON.is_file() and COCO_IMAGES.is_dir(), 'Attach MSCOCO2014 JSON + images.'
        assert not OUTPUT.exists(), 'Preserve prior output; choose fresh OUTPUT/session.'
        '''))
    cache.append(bootstrap(bundle, 'cache'))
    cache.append(code('''
        run([sys.executable,'-m','pip','install','-q','numpy','Pillow','h5py','scipy','gdown'])
        import torch
        assert torch.cuda.device_count()==2, 'Select GPU T4x2.'
        for name in ('test_hidden_data.py','test_hidden_hook.py','test_hidden_hdf5.py'):
            run([sys.executable,'-u',name],cwd=REPO)
        WORK = Path('/tmp/upt-hidden-reference')
        WORK.mkdir(parents=True,exist_ok=True)
        UPT, POCKET = WORK/'upt', WORK/'pocket'
        checkout('https://github.com/fredzzhang/upt.git',UPT,'192743045e4c5f6827299fdc15809f5b97ac1b8e')
        run(['git','submodule','update','--init','detr','hicodet'],cwd=UPT)
        checkout('https://github.com/fredzzhang/pocket.git',POCKET,'bff287098bb73427fd8d5525b2ac9867fb56629d')
        UPT_CHECKPOINT = Path(LOCAL_UPT_CHECKPOINT) if LOCAL_UPT_CHECKPOINT else WORK/'upt-r50-hico.pt'
        if not UPT_CHECKPOINT.is_file():
            assert LOCAL_UPT_CHECKPOINT is None, UPT_CHECKPOINT
            import gdown
            print('Downloading official UPT-R50 HICO (~220MB)',flush=True)
            gdown.download(id='1lvopTC_JlGHdKOyyEZTUp2VtFfCQ-08z',output=str(UPT_CHECKPOINT),quiet=False)
        cfg=dict(dataset_json=str(COCO_JSON),images=str(COCO_IMAGES),output=str(OUTPUT),upt=str(UPT),
            pocket=str(POCKET),upt_checkpoint=str(UPT_CHECKPOINT),project_commit=PROJECT_COMMIT,
            implementation_sha256=SOURCE_HASHES)
        CONFIG=Path('/tmp/upt_hidden_cache_config.json')
        CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        try:
            run([sys.executable,'-u','extract_hidden.py','run','--config',CONFIG],cwd=REPO)
        finally:
            if OUTPUT.is_dir():
                shutil.copyfile(LOG,OUTPUT/'notebook.log')
                import zipfile
                with zipfile.ZipFile(str(OUTPUT)+'_review.zip','w',zipfile.ZIP_DEFLATED) as archive:
                    for path in sorted(OUTPUT.rglob('*')):
                        if path.is_file() and path.suffix in ('.json','.jsonl','.log','.py'):
                            archive.write(path,path.relative_to(OUTPUT))
        assert (OUTPUT/'cache_receipt.json').is_file(), 'Cache incomplete; preserve logs and return review ZIP.'
        print('KEEP ENTIRE FOLDER AS DATASET:',OUTPUT,flush=True)
        print('SEND SMALL REVIEW ZIP:',str(OUTPUT)+'_review.zip',flush=True)
        '''))
    write('UPT_HIDDEN_CACHE_11K_2GPU.ipynb', cache)
    for arm in ('A', 'H'):
        cells = [md(f'''
            # 2. Caption pilot {arm} — {'baseline huấn luyện tiếp' if arm=='A' else 'baseline + UPT hidden-pair memory'}
            Một nhánh trên **HAI T4 bằng DDP**. Internet ON; Save & Run All.
            Chạy sau khi cache notebook hoàn tất và đã lưu toàn bộ folder thành dataset.
            A/H có thể chạy song song trên hai tài khoản, mỗi notebook T4x2. Chia sẻ/mount đúng các private dataset.
            Cùng baseline, train10k/val1000 đã dùng trước,1250updates,batch8,CE200 rồi CE+.05SCST(CIDErD+.5object_count_v2).
            Không dùng adapter crop/phrase. Giữ prompt legacy của baseline; không thêm verb text UPT.
            ''')]
        cells.append(code(f'''
            print('UPT HIDDEN {arm}: BOOTSTRAP START',flush=True)
            from pathlib import Path
            ARM='{arm}'
            CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/qformer-scst-object-w05-legacy/qformer_scst_object_w05_legacy.pth')
            LEGACY_PROMPT=Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
            VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
            COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
            DETECTIONS=Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
            HIDDEN_CACHE=Path('/kaggle/input/datasets/ducanh2403/upt-hidden-pairs-caption11k/upt_hidden_pairs_11k')
            OUTPUT=Path('/kaggle/working/upt_hidden_caption_{arm}')
            # HIDDEN_CACHE là ví dụ tên dataset; sửa đúng thư mục chứa cache_receipt.json của bạn.
            for path in (CHECKPOINT,LEGACY_PROMPT,COCO_JSON,DETECTIONS,HIDDEN_CACHE/'cache_receipt.json'):
                assert path.is_file(), f'Missing exact input: {{path}}'
            assert VISUAL_CACHE.is_dir(),VISUAL_CACHE
            assert not OUTPUT.exists(),'Use fresh output/session; preserve checkpoints.'
            '''))
        cells.append(bootstrap(bundle, arm.lower()))
        cells.append(code('''
            run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','pycocotools','pycocoevalcap==1.2',
                'h5py','numpy','Pillow','tqdm','nltk==3.9.2','future'])
            run([sys.executable,'-m','pip','install','-q','--no-deps',
                'git+https://github.com/clips/pattern.git@af754685cca3713db0abc4f020f2e94467c19d85'])
            run([sys.executable,'-u','-c',"import nltk;from pattern.en import singularize;assert singularize('umbrellas')=='umbrella';assert nltk.download('punkt');assert nltk.download('punkt_tab')"])
            run(['java','-version'])
            CHAIR=Path('/tmp/hidden-caption-chair')
            checkout('https://github.com/Maxlinn/CHAIR-metric-standalone.git',CHAIR,'4087a26211aa2339b9a76307cb8f0321ef691d0a')
            import torch
            assert torch.cuda.device_count()==2,'Select GPU T4x2 for this one arm.'
            for name in ('test_hidden_data.py','test_hidden_hdf5.py','test_hidden_model.py'):
                run([sys.executable,'-u',name],cwd=REPO)
            run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1',
                '--nproc_per_node=2','--max_restarts=0','test_hidden_ddp.py'],cwd=REPO)
            run([sys.executable,'-u','hidden_metric_smoke.py',CHAIR],cwd=REPO)
            '''))
        cells.append(code('''
            CONFIG=Path('/tmp/upt_hidden_config_'+ARM+'.json')
            cfg=dict(arm=ARM,checkpoint=str(CHECKPOINT),legacy_prompt=str(LEGACY_PROMPT),visual_cache=str(VISUAL_CACHE),
                dataset_json=str(COCO_JSON),detections=str(DETECTIONS),hidden_cache=str(HIDDEN_CACHE),output=str(OUTPUT),
                stage='/tmp/upt_hidden_stage_'+ARM+'.h5',chair=str(CHAIR),project_commit=PROJECT_COMMIT,
                global_batch_size=8,eval_batch_size=16,warmup_updates=200,base_lr=5e-6,branch_lr=1e-4,scst_weight=.05,seed=42,
                adamw_options=dict(weight_decay=.01,betas=[.9,.999],eps=1e-8,amsgrad=False,maximize=False,
                    foreach=None,capturable=False,differentiable=False,fused=None),
                record_optimizer_manifest=True,max_gradient_norm=1.,dropout_policy='model.eval() with gradients enabled',
                implementation_sha256=SOURCE_HASHES,
                locked_numeric_gate=dict(CIDEr_delta_H_minus_A_min=.01,paired_bootstrap95_lower_gt=0.,
                    H_CIDEr_gt_original=True,CHAIRs_increase_max=0.,CHAIRi_increase_max=0.,Recall_drop_max=.01),
                promotion_requires='Matched valid A/H, numeric gates and blind human review of supported relations/instances/extra facts; no automatic promotion.')
            CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
            PROTOCOL=OUTPUT/'protocol.json'
            try:
                run([sys.executable,'-u','run_hidden.py','prepare','--config',CONFIG],cwd=REPO)
                run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1',
                    '--nproc_per_node=2','--max_restarts=0','run_hidden.py','worker','--config',PROTOCOL],cwd=REPO)
                run([sys.executable,'-u','run_hidden.py','evaluate','--config',PROTOCOL],cwd=REPO)
            finally:
                if OUTPUT.is_dir():
                    shutil.copyfile(LOG,OUTPUT/'notebook.log')
                    if PROTOCOL.is_file():
                        run([sys.executable,'-u','run_hidden.py','archive','--config',PROTOCOL],cwd=REPO)
                    else:
                        import zipfile
                        with zipfile.ZipFile(str(OUTPUT)+'_review.zip','w',zipfile.ZIP_DEFLATED) as z:
                            for path in OUTPUT.rglob('*'):
                                if path.is_file() and path.suffix in ('.json','.log','.py'):
                                    z.write(path,path.relative_to(OUTPUT))
            print('SEND REVIEW:',str(OUTPUT)+'_review.zip',flush=True)
            print('KEEP CHECKPOINT:',OUTPUT/'checkpoint.bin',flush=True)
            '''))
        cells.append(md('''
            Gửi ZIP review và giữ riêng checkpoint.bin. Stage HDF5 đặt trong/tmp, không đưa vào dataset checkpoint.
            Không dùng điểm của một nhánh để sửa cấu hình nhánh còn lại. Chờ đủ A/H mới so sánh.
            Development đã dùng trước; một seed không xác nhận độ ổn định; chưa có kết quả mới hoặc novelty đã chứng minh.
            Nếu lỗi trước khi tạo Output, chạy `print(LOG.read_text(encoding='utf-8')[-12000:])` và gửi phần lỗi đầu tiên.
            '''))
        write(f'UPT_HIDDEN_CAPTION_{arm}_2GPU.ipynb', cells)
    with zipfile.ZipFile(HERE / 'UPT_HIDDEN_3_NOTEBOOKS.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for file in sorted(HERE.glob('UPT_HIDDEN_*_2GPU.ipynb')):
            z.write(file, file.name)


if __name__ == '__main__':
    main()
