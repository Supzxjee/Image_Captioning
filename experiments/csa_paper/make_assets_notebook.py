"""Package CPU-only actual-input audit, without executing its notebook."""
import argparse
import hashlib
import json
import subprocess

from make_smoke_notebook import HERE, ROOT, PREFIX, RUNNER, cell

FILES = ['audit_assets.py', 'data.py', 'test_assets.py', 'IMPLEMENTATION_PROTOCOL.md']


def make(revision):
    commit = subprocess.check_output(['git', 'rev-parse', revision], cwd=ROOT, text=True).strip()
    hashes = {name: hashlib.sha256(subprocess.check_output(['git', 'show', commit+':'+PREFIX+name], cwd=ROOT)).hexdigest() for name in FILES}
    setup = f'''
import hashlib, json, os, queue, signal, subprocess, sys, tempfile, threading, time, traceback, urllib.request, zipfile
from pathlib import Path
COMMIT = {commit!r}
HASHES = {hashes!r}
OUTPUT = Path('/kaggle/working/csa_paper_asset_inventory')
COCO_JSON = Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
DETECTIONS = Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
VISUAL_CACHE = Path('/kaggle/input/datasets/ducanh2403/visual-cache')
IMAGE_ROOT = COCO_JSON.parent/'images'
if OUTPUT.exists(): raise RuntimeError('Preserve previous output/ZIP; use fresh runtime.')
OUTPUT.mkdir(parents=True)
SOURCE = Path(tempfile.mkdtemp(prefix='csa-assets-'))
LOG = OUTPUT/'notebook.log'
CONFIG = SOURCE/'config.json'
CONFIG.write_text(json.dumps(dict(dataset_json=str(COCO_JSON), detections=str(DETECTIONS),
    visual_cache=str(VISUAL_CACHE), image_root=str(IMAGE_ROOT), output=str(OUTPUT))), encoding='utf-8')
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=COMMIT, sha256=HASHES), indent=2), encoding='utf-8')
for name, expected in HASHES.items():
    payload = urllib.request.urlopen(f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{{COMMIT}}/{PREFIX}{{name}}', timeout=90).read()
    if hashlib.sha256(payload).hexdigest() != expected: raise RuntimeError('Source hash differs: '+name)
    (SOURCE/name).write_bytes(payload)
print('Pinned CPU audit source ready; GPU not required.', flush=True)
'''
    execute = '''
complete = False
try:
    run([sys.executable, '-u', SOURCE/'test_assets.py'], timeout=120)
    run([sys.executable, '-u', SOURCE/'audit_assets.py', '--config', CONFIG], timeout=1800)
    summary = json.loads((OUTPUT/'summary.json').read_text(encoding='utf-8'))
    complete = summary.get('status') == 'ASSET_INVENTORY_COMPLETE'
    if not complete: raise RuntimeError('Inventory incomplete')
    print(json.dumps(summary, indent=2), flush=True)
except BaseException:
    (OUTPUT/'notebook_failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
    raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete, training_ready=False, model_executed=False)), encoding='utf-8')
    archive = OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file(): z.write(p, OUTPUT.name+'/'+str(p.relative_to(OUTPUT)))
    print('SEND:', archive, flush=True)
    print('INVENTORY COMPLETE; not training-ready.' if complete else 'INCOMPLETE: send ZIP/log.', flush=True)
'''
    nb = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')),
        cells=[cell('markdown', '''
        # CSA — kiểm cache thật và tạo từ vựng TRAIN (CPU)

        **Accelerator None**, Internet ON. Add Input ba dataset:
        `ducanh2403/visual-cache`, `ducanh2403/objectdetectionecache`, `vuthetam/mscoco-2014`.
        Không cần checkpoint, prompt cache, UPT hay API token.

        Đọc metadata/ID/filename của HDF5, không đọc tensor features; kiểm đủ123287ảnh,
        detector records và12headerảnh lấy kích thước; tạo vocabulary từ train+restval.
        Nếu đường mount khác, sửa bốn biến COCO_JSON, DETECTIONS, VISUAL_CACHE, IMAGE_ROOT ở cell đầu.
        Giữ pinned hashes; khác payload thì gửi lỗi, không bỏ guard.

        Chưa POS/CLIP text bank, chưa trích ROI hay train. Không dùng GPU. Worker cap30phút
        là giới hạn dừng, không ETA. Save & Run All; gửi ZIP `csa_paper_asset_inventory_review.zip`.
        '''), cell('code', setup), cell('code', RUNNER), cell('code', execute)])
    path = HERE/'CSA_PAPER_ASSET_INVENTORY_CPU.ipynb'
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    assert path.stat().st_size < 1_000_000
    (HERE/'ASSET_INVENTORY_PACKAGE.json').write_text(json.dumps(dict(core_commit=commit, sha256=hashes,
        notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        user_run_pending=True, model_executed=False), indent=2)+'\n', encoding='utf-8')
    print(path.name, path.stat().st_size, 'bytes; unexecuted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('revision'); args = parser.parse_args()
    make(args.revision)
