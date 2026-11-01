"""Author CPU POS/ROI notebook; never execute its model or submit it."""
import argparse
import hashlib
import json
import subprocess

from make_smoke_notebook import HERE, ROOT, PREFIX, RUNNER, cell

FILES = ['prepare_real_assets.py', 'audit_assets.py', 'data.py', 'test_real_assets.py', 'REAL_ASSET_PREPARATION_PROTOCOL.md']
MODEL_URL = 'https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl'
MODEL_SHA = '1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85'


def make(revision):
    commit = subprocess.check_output(['git', 'rev-parse', revision], cwd=ROOT, text=True).strip()
    hashes = {name: hashlib.sha256(subprocess.check_output(['git', 'show', commit+':'+PREFIX+name], cwd=ROOT)).hexdigest() for name in FILES}
    setup = f'''
import hashlib, json, os, queue, signal, subprocess, sys, tempfile, threading, time, traceback, urllib.request, zipfile
from pathlib import Path
COMMIT = {commit!r}
HASHES = {hashes!r}
OUTPUT = Path('/kaggle/working/csa_paper_real_preparation')
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
DETECTIONS = Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
VISUAL_CACHE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')
IMAGE_ROOT = COCO_JSON.parent/'images'
if OUTPUT.exists(): raise RuntimeError('Preserve previous output and use a fresh session.')
OUTPUT.mkdir(parents=True)
SOURCE = Path(tempfile.mkdtemp(prefix='csa-real-prepare-'))
LOG = OUTPUT/'notebook.log'
CONFIG = SOURCE/'config.json'
CONFIG.write_text(json.dumps(dict(dataset_json=str(COCO_JSON), detections=str(DETECTIONS), visual_cache=str(VISUAL_CACHE), image_root=str(IMAGE_ROOT), output=str(OUTPUT))), encoding='utf-8')
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=COMMIT, sha256=HASHES, bootstrap_policy='venv_without_pip_base_pip_python', pos_model_url={MODEL_URL!r}, pos_model_sha256={MODEL_SHA!r}), indent=2), encoding='utf-8')
for name, expected in HASHES.items():
    payload = urllib.request.urlopen(f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{{COMMIT}}/{PREFIX}{{name}}', timeout=90).read()
    if hashlib.sha256(payload).hexdigest() != expected: raise RuntimeError('Source differs: '+name)
    (SOURCE/name).write_bytes(payload)
print('CPU preparation source verified. Accelerator must be None.', flush=True)
'''
    execute = f'''
complete = False
try:
    import numpy as np
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['OPENBLAS_NUM_THREADS'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    venv = SOURCE/'pos-env'
    run([sys.executable, '-m', 'venv', '--without-pip', '--system-site-packages', venv], timeout=180)
    PYTHON = venv/'bin/python'
    if '--python' not in subprocess.check_output([sys.executable, '-m', 'pip', '--help'], text=True):
        raise RuntimeError('Base pip must support --python (pip>=22.3); do not install into global environment.')
    PIP = [sys.executable, '-m', 'pip', '--python', str(PYTHON)]
    constraint = SOURCE/'constraints.txt'
    constraint.write_text('numpy=='+np.__version__+chr(10), encoding='utf-8')
    run(PIP + ['install', '--disable-pip-version-check', '--only-binary=:all:', '-c', constraint, 'spacy==3.8.11'], timeout=900)
    wheel = SOURCE/'en_core_web_sm-3.8.0-py3-none-any.whl'
    payload = urllib.request.urlopen({MODEL_URL!r}, timeout=180).read()
    if hashlib.sha256(payload).hexdigest() != {MODEL_SHA!r}: raise RuntimeError('Official POS model wheel SHA differs')
    wheel.write_bytes(payload)
    run(PIP + ['install', '--no-deps', '--force-reinstall', wheel], timeout=180)
    (OUTPUT/'software_freeze.txt').write_text(subprocess.check_output(PIP + ['freeze', '--all'], text=True), encoding='utf-8')
    run([PYTHON, '-u', SOURCE/'test_real_assets.py'], timeout=120)
    run([PYTHON, '-u', SOURCE/'prepare_real_assets.py', '--config', CONFIG], timeout=7200)
    summary = json.loads((OUTPUT/'summary.json').read_text(encoding='utf-8'))
    receipt = json.loads((OUTPUT/'receipt.json').read_text(encoding='utf-8'))
    if summary.get('status') != 'CPU_PREPARATION_COMPLETE' or receipt.get('status') != 'CPU_PREPARATION_COMPLETE': raise RuntimeError('Preparation incomplete')
    for name, expected in receipt['payload_sha256'].items():
        if Path(name).name != name or hashlib.sha256((OUTPUT/name).read_bytes()).hexdigest() != expected: raise RuntimeError('Output payload differs: '+name)
    complete = True
    print(json.dumps(summary, indent=2), flush=True)
except BaseException:
    (OUTPUT/'notebook_failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
    raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete, caption_model_executed=False, gpu_used=False, semantic_embeddings_created=False, training_ready=False)), encoding='utf-8')
    archive = OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.iterdir()):
            if p.is_file() and p.suffix in ('.json', '.txt', '.log'): z.write(p, OUTPUT.name+'/'+p.name)
    print('SEND REVIEW ZIP:', archive, flush=True)
    if complete: print('KEEP ENTIRE OUTPUT FOLDER AS DATASET:', OUTPUT, flush=True)
    else: print('INCOMPLETE: preserve logs; do not use as prepared training data.', flush=True)
'''
    nb = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')),
        cells=[cell('markdown', '''
        # CSA — danh từ/động từ TRAIN + ROI thật (CPU)

        **Accelerator None, Internet ON**. Add Input `visual-cache`, `objectdetectionecache`,
        `mscoco-2014`. Không cần checkpoint/prompt/UPT/token. Cell đầu có4path nếu mount khác.

        1. Tạo venv không gọi ensurepip; dùng pip gốc với --python để cài spaCy3.8.11
           vào venv tạm, kiểm SHA model en_core_web_sm3.8.0.
        2. Pool raw CLIP patches theo box thật cho256TRAIN+64val; không dùng test.
        3. POS toàn566747captionTRAIN để đếm noun/verb lemma, giữ tần suất≥5.

        Có chạy model POS nhỏ trên CPU; **chưa chạy CLIP text/captioner hoặc huấn luyện**.
        Worker cap2giờ là giới hạn dừng, không ETA. Không bật GPU cho giai đoạn này.
        Save & Run All. Gửi `csa_paper_real_preparation_review.zip`.
        **Giữ toàn bộ folder `csa_paper_real_preparation` để lưu thành Dataset**;
        `regions.npz` không nằm trong ZIP review, không xóa nó. Bước sau mới mã hóa
        các từ bằng CLIP text và kiểm CSA với đặc trưng thật.
        '''), cell('code', setup), cell('code', RUNNER), cell('code', execute)])
    path = HERE/'CSA_PAPER_REAL_PREPARATION_CPU.ipynb'
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    assert path.stat().st_size < 1_000_000
    (HERE/'REAL_PREPARATION_PACKAGE.json').write_text(json.dumps(dict(core_commit=commit, sha256=hashes,
        notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), notebook_executed_by_agent=False,
        user_run_pending=True), indent=2)+'\n', encoding='utf-8')
    print(path.name, path.stat().st_size, 'bytes; unexecuted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('revision'); args = parser.parse_args()
    make(args.revision)
