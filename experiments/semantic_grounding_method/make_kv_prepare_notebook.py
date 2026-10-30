"""Build a small CPU-only notebook from pinned Git blobs; never submit/run it."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREFIX = 'experiments/semantic_grounding_method/'
ASSETS = [PREFIX+n for n in ('kv_data.py', 'prepare_kv_cpu.py', 'test_kv_data.py',
          'test_relation_kv_math.py', 'test_kv_source_fetch.py', 'data/gain_metadata.json.gz')]
ASSETS += ['experiments/relation_frames_inventory/inventory.py',
           'experiments/relation_frames_inventory/sources/source_receipt.json',
           'experiments/relation_frames_inventory/V2_MANIFEST.json.gz',
           'experiments/relation_frame_pretrain/frame_data.py',
           'experiments/concept_mass_grounding/TRAIN_PRESENCE.json.gz',
           'fetch_vcoco_coco_instances.py', 'audit_vcoco_alignment_data.py']
ASSETS += ['captioning/'+n+'.py' for n in ('__init__', 'object_consistency', 'supervised_relations',
                                         'semantic_prompts', 'visual_cache')]


def cell(kind, source):
    source = textwrap.dedent(source).strip()+'\n'
    value = dict(cell_type=kind, metadata={}, source=source,
                 id=hashlib.sha256(source.encode()).hexdigest()[:12])
    if kind == 'code':
        value.update(execution_count=None, outputs=[])
    return value


def make(revision):
    core = subprocess.check_output(['git','rev-parse',revision], cwd=ROOT, text=True).strip()
    assets = {p:hashlib.sha256(subprocess.check_output(['git','show',core+':'+p],cwd=ROOT)).hexdigest()
              for p in ASSETS}
    cells = [cell('markdown', '''
        # K/V pilot — chuẩn bị dữ liệu trên CPU
        **Accelerator: None · Internet ON · Save & Run All. Không cần GPU.**

        Notebook chỉ lọc official TRAIN/nhãn được quan sát, ghép GT với predicted boxes,
        và sao chép CLIP/legacy tensors có sẵn. Không dựng hoặc chạy captioner/encoder;
        checkpoint C0 chỉ được kiểm SHA. Không training/SCST/đánh giá caption ở đây.

        Add Input giống lượt expanded: `concept-mass-c-c0-v111-v112`, `prompt-cache`,
        `visual-cache`, `objectdetectionecache`, `mscoco-2014` của vuthetam.
        Kiểm các đường dẫn ở cell đầu. Repo nằm trong /tmp; Output không chứa toàn source repo.

        Metadata COCO chính thức được tải thêm (~152 MiB compressed của hai member, không ảnh).
        Dữ liệu có thể chưa đủ sau GT/top6/ontology checks: notebook sẽ dừng ở trạng thái
        NEEDS_SOURCE_REVIEW thay vì tự nới tiêu chuẩn. Đây không phải test CUDA/model.
        Không bật GPU để chạy bước này. Giới hạn7200s là cap, không ETA.
        '''), cell('code', '''
        from pathlib import Path
        import tempfile, sys, json, time, hashlib, urllib.request, importlib.util
        import subprocess, threading, queue, os, signal, zipfile
        print('K/V CPU preparation starting — no model, no training.', flush=True)
        CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/concept-mass-c-c0-v111-v112/C0.bin')
        COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
        LEGACY=Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
        VISUAL=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
        DETECTIONS=Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
        OUTPUT=Path('/kaggle/working/relation_kv_preparation_v1')
        LOG=OUTPUT.with_suffix('.log')
        REVIEW=OUTPUT.with_name(OUTPUT.name+'_review.zip')
        for p in (CHECKPOINT,COCO_JSON,LEGACY,DETECTIONS):
            if not p.is_file():raise FileNotFoundError(f'Edit the input path in this cell: {p}')
        if not VISUAL.is_dir():raise FileNotFoundError(VISUAL)
        if OUTPUT.exists() or LOG.exists() or REVIEW.exists():
            raise FileExistsError('Preserve existing output; use a fresh session or edit OUTPUT.')
        WORK=Path(tempfile.mkdtemp(prefix='relation-kv-cpu-'))
        REPO=WORK/'repo'; REPO.mkdir()
        PREP_SECONDS=7200
        print('Inputs found. Select Accelerator=None for this notebook.', flush=True)
        '''), cell('code', f'''
        CORE={core!r}
        ASSETS={assets!r}
        for asset_index, (relative, expected) in enumerate(ASSETS.items(), 1):
            url='https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+CORE+'/'+relative
            with urllib.request.urlopen(url,timeout=60) as response:raw=response.read()
            if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Changed source: '+relative)
            target=REPO/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            if asset_index%5==0 or asset_index==len(ASSETS):print('Verified source files:',asset_index,'/',len(ASSETS),flush=True)
        for package in ('numpy','h5py','torch'):
            if importlib.util.find_spec(package) is None:raise RuntimeError('Kaggle runtime lacks '+package)
        if importlib.util.find_spec('ijson') is None:
            subprocess.run([sys.executable,'-m','pip','install','--quiet','ijson==3.4.0'],check=True)
        HERE=REPO/{PREFIX!r}
        CONFIG=WORK/'config.json'
        cfg=dict(checkpoint=str(CHECKPOINT),dataset_json=str(COCO_JSON),legacy_prompt=str(LEGACY),
                 visual_cache=str(VISUAL),detections=str(DETECTIONS),output=str(OUTPUT),
                 work=str(WORK),source_commit=CORE)
        CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
        print('Verified pinned source:',CORE,'CPU preparation only.',flush=True)
        '''), cell('code', '''
        def run(args, timeout):
            print('Running:', ' '.join(map(str,args)),flush=True)
            process=subprocess.Popen(list(map(str,args)),cwd=REPO,stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT,text=True,bufsize=1,start_new_session=True)
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
                            print(f'CPU worker active; {now-start:.0f}s elapsed. No model/training.',flush=True)
                            heartbeat=now
                        if now-start>timeout:raise TimeoutError('CPU phase exceeded locked cap')
                    code=process.wait()
                    if code:raise subprocess.CalledProcessError(code,args)
                finally:
                    if process.poll() is None:
                        os.killpg(process.pid,signal.SIGTERM)
                        try:process.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                    thread.join(timeout=5)

        try:
            for test in ('test_kv_data.py','test_relation_kv_math.py','test_kv_source_fetch.py'):
                run([sys.executable,'-u',HERE/test],120)
            run([sys.executable,'-u',HERE/'prepare_kv_cpu.py','--config',CONFIG],PREP_SECONDS)
        finally:
            with zipfile.ZipFile(REVIEW,'x',compression=zipfile.ZIP_DEFLATED) as archive:
                for p in sorted(OUTPUT.glob('*.json')):
                    archive.write(p,p.name)
                if LOG.exists():archive.write(LOG,'notebook.log')
                archive.write(CONFIG,'requested_config.json')
                for name in ('kv_data.py','prepare_kv_cpu.py'):
                    archive.write(HERE/name,'source/'+name)
            print('SEND REVIEW ZIP:',REVIEW,flush=True)
            print('KEEP ENTIRE FOLDER:',OUTPUT,flush=True)
        receipt=OUTPUT/'cpu_preparation.json'
        status=json.loads(receipt.read_text()).get('status') if receipt.is_file() else 'INCOMPLETE'
        print('Status:',status,'— model/CUDA have NOT run.',flush=True)
        print('Send the review ZIP before starting D/E training.',flush=True)
        ''')]
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
                    language_info=dict(name='python'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=False)),
                    nbformat=4,nbformat_minor=5)
    target=HERE/'PREPARE_RELATION_KV_CPU.ipynb'
    target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    package=dict(core_commit=core,assets=assets,notebook=target.name,
                 notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                 no_model_execution=True,no_submission=True,accelerator='None')
    (HERE/'KV_CPU_NOTEBOOK_PACKAGE.json').write_text(json.dumps(package,indent=2)+'\n',encoding='utf-8')
    print(target)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--core',default='HEAD')
    make(parser.parse_args().core)
