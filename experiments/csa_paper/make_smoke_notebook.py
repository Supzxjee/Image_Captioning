"""Author a source-pinned notebook. Does not import Torch or submit Kaggle runs."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREFIX = 'experiments/csa_paper/'
FILES = ['data.py', 'model.py', 'test_data.py', 'runtime_checks.py', 'IMPLEMENTATION_PROTOCOL.md']


def cell(kind, source):
    value = dict(cell_type=kind, metadata={}, source=textwrap.dedent(source).strip() + '\n')
    if kind == 'code':
        ast.parse(value['source'])
        value.update(execution_count=None, outputs=[])
    return value


RUNNER = '''
def run(args, timeout=1800):
    print('Running:', ' '.join(map(str, args)), flush=True)
    process = subprocess.Popen(list(map(str, args)), cwd=SOURCE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1, start_new_session=True)
    messages = queue.Queue()
    def reader():
        try:
            for line in process.stdout: messages.put(line)
        finally: messages.put(None)
    thread = threading.Thread(target=reader, daemon=True); thread.start()
    started = heartbeat = time.monotonic(); ended = False
    with LOG.open('a', encoding='utf-8') as log:
        try:
            while not ended or process.poll() is None or not messages.empty():
                try:
                    line = messages.get(timeout=1)
                    if line is None: ended = True
                    else: print(line, end='', flush=True); log.write(line); log.flush()
                except queue.Empty: pass
                now = time.monotonic()
                if now-heartbeat >= 30:
                    print('Smoke active;', round(now-started), 'seconds', flush=True); heartbeat=now
                if now-started > timeout: raise TimeoutError('Smoke wall cap exceeded')
            if process.wait(): raise subprocess.CalledProcessError(process.returncode, args)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
            thread.join(timeout=5)
'''


def make(revision):
    commit = subprocess.check_output(['git', 'rev-parse', revision], cwd=ROOT, text=True).strip()
    hashes = {name: hashlib.sha256(subprocess.check_output(['git', 'show', commit+':'+PREFIX+name], cwd=ROOT)).hexdigest()
              for name in FILES}
    setup = f'''
import hashlib, json, os, queue, signal, subprocess, sys, tempfile, threading, time, traceback, urllib.request, zipfile
from pathlib import Path
COMMIT = {commit!r}
HASHES = {hashes!r}
OUTPUT = Path('/kaggle/working/csa_paper_core_smoke')
if OUTPUT.exists():
    raise RuntimeError('Output exists. Preserve its review ZIP; use a fresh session for this smoke.')
OUTPUT.mkdir(parents=True)
SOURCE = Path(tempfile.mkdtemp(prefix='csa-paper-core-'))
LOG = OUTPUT/'notebook.log'
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=COMMIT, sha256=HASHES, synthetic_only=True), indent=2), encoding='utf-8')
print('Source commit:', COMMIT, flush=True)
for name, expected in HASHES.items():
    url = f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{{COMMIT}}/{PREFIX}{{name}}'
    payload = urllib.request.urlopen(url, timeout=90).read()
    if hashlib.sha256(payload).hexdigest() != expected: raise RuntimeError('Source hash differs: '+name)
    (SOURCE/name).write_bytes(payload)
print('Pinned source verified. No dataset, model weights or API token needed.', flush=True)
'''
    execution = '''
complete = False
try:
    run([sys.executable, '-u', SOURCE/'test_data.py'], timeout=120)
    run([sys.executable, '-u', SOURCE/'runtime_checks.py', '--output', OUTPUT], timeout=1800)
    summary = json.loads((OUTPUT/'summary.json').read_text(encoding='utf-8'))
    complete = summary.get('status') == 'CORE_SMOKE_COMPLETE'
    if not complete: raise RuntimeError('Missing complete smoke status')
    print(json.dumps(summary, indent=2), flush=True)
except BaseException:
    (OUTPUT/'notebook_failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
    raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete, synthetic_only=True,
        quality_metrics_computed=False, real_data_verified=False, training_authorized=False)), encoding='utf-8')
    archive = OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.rglob('*')):
            if p.is_file(): z.write(p, OUTPUT.name+'/'+str(p.relative_to(OUTPUT)))
    print('SEND REVIEW ZIP:', archive, flush=True)
    print('CORE SMOKE COMPLETE; not a quality result.' if complete else 'INCOMPLETE: send review ZIP/log.', flush=True)
'''
    notebook = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')),
        cells=[cell('markdown', '''
        # CSA paper — kiểm tra core tự viết

        Bật **Internet** để tải source đã khóa SHA. Chọn **GPU T4**; chỉ dùng GPU0,
        không cần hai GPU, dataset, checkpoint, API token, spaCy hoặc tải CLIP.
        Save Version → Save & Run All. Không tự chạy notebook bằng agent.

        Kiểm tensor nhỏ, rồi D512/8 heads/K150/3 decoder layers/vocab10000:
        2 cập nhật XE + 1 cập nhật RL với dữ liệu và reward **giả**. Đây là kiểm mã,
        không phải huấn luyện thực nghiệm; không có CIDEr/CHAIR. Thời gian worker
        tối đa30phút là giới hạn dừng, không phải dự đoán thời gian.

        Gửi `csa_paper_core_smoke_review.zip` dù thành công hoặc lỗi.
        Sau smoke mới kiểm detector/region cache và tạo bank TRAIN thực.
        '''), cell('code', setup), cell('code', RUNNER), cell('code', execution)])
    path = HERE/'CSA_PAPER_CORE_SMOKE.ipynb'
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    assert path.stat().st_size < 1_000_000
    (HERE/'SMOKE_PACKAGE.json').write_text(json.dumps(dict(core_commit=commit, sha256=hashes,
        notebook_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        model_run_by_agent=False, kaggle_submitted=False), indent=2)+'\n', encoding='utf-8')
    print(path, path.stat().st_size, 'bytes; unexecuted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('revision'); args = parser.parse_args()
    make(args.revision)
