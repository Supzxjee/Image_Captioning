"""Build self-contained user-run CPU POS trace; no POS/model/API execution."""
import hashlib
import json

from make_smoke_notebook import HERE, RUNNER, cell


def make():
    packet = (HERE/'LEMMA_TRACE_CASES.json').read_bytes()
    previous = json.loads((HERE/'real_preparation_review_text/summary.json').read_bytes())
    environment = json.dumps(dict(versions={k: previous['versions'][k] for k in ('spacy', 'thinc', 'numpy', 'en-core-web-sm')},
                                 pos_model_files_sha256=previous['pos_model_files_sha256']), sort_keys=True)
    modules = {name: (HERE/name).read_text(encoding='utf-8') for name in ('lemma_trace.py', 'run_lemma_trace.py')}
    hashes = {name: hashlib.sha256(value.encode()).hexdigest() for name, value in modules.items()}
    old = json.loads((HERE/'CSA_PAPER_REAL_PREPARATION_CPU.ipynb').read_text(encoding='utf-8'))
    original = old['cells'][-1]['source']
    marker = "    run([PYTHON, '-u', SOURCE/'test_real_assets.py']"
    assert marker in original
    bootstrap = original[:original.index(marker)]
    constraint = "'numpy=='+np.__version__+chr(10)"
    assert constraint in bootstrap
    bootstrap = bootstrap.replace(constraint, "'numpy==2.1.3'+chr(10)")
    bootstrap = bootstrap.replace("'spacy==3.8.11'", "'spacy==3.8.11', 'thinc==8.3.11'")
    setup = '''
import hashlib, json, os, queue, signal, subprocess, sys, tempfile, threading, time, traceback, urllib.request, zipfile
from pathlib import Path
OUTPUT = Path('/kaggle/working/csa_lemma_trace_check')
if OUTPUT.exists(): raise RuntimeError('Preserve previous ZIP and use a fresh CPU session.')
OUTPUT.mkdir(parents=True)
SOURCE = Path(tempfile.mkdtemp(prefix='csa-lemma-trace-'))
LOG = OUTPUT/'notebook.log'
'''
    setup += 'MODULES='+repr(modules)+'\nMODULE_HASHES='+repr(hashes)+'\nPACKET='+repr(packet.decode('utf-8'))+'\nENVIRONMENT='+repr(environment)+'\n'
    setup += '''
for name, source in MODULES.items():
    payload=source.encode('utf-8')
    if hashlib.sha256(payload).hexdigest()!=MODULE_HASHES[name]: raise RuntimeError('Module hash differs: '+name)
    (SOURCE/name).write_bytes(payload)
(SOURCE/'trace_cases.json').write_text(PACKET, encoding='utf-8')
(SOURCE/'source_environment.json').write_text(ENVIRONMENT, encoding='utf-8')
(OUTPUT/'trace_cases.json').write_text(PACKET, encoding='utf-8')
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(module_sha256=MODULE_HASHES,
    packet_sha256=hashlib.sha256(PACKET.encode('utf-8')).hexdigest(),
    source_environment_sha256=hashlib.sha256(ENVIRONMENT.encode('utf-8')).hexdigest(),
    original_roi_source_version=126, dataset_required=False, bootstrap_policy='venv_without_pip_base_pip_python'), indent=2), encoding='utf-8')
print('45 small cases; no Dataset/API token/raw features needed. CPU only.', flush=True)
'''
    execution = bootstrap + '''    run([PYTHON, '-u', SOURCE/'run_lemma_trace.py', '--output', OUTPUT], timeout=900)
    summary=json.loads((OUTPUT/'summary.json').read_text(encoding='utf-8'))
    complete=summary.get('status')=='LEMMA_TRACE_COMPLETE'
    if not complete: raise RuntimeError('Trace incomplete')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
except BaseException:
    (OUTPUT/'notebook_failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
    raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete, bank_modified=False, roi_modified=False, dataset_created=False)), encoding='utf-8')
    archive=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.iterdir()):
            if p.is_file(): z.write(p,OUTPUT.name+'/'+p.name)
    print('SEND REVIEW ZIP:',archive,flush=True)
    print('TRACE COMPLETE; original lemmas retained.' if complete else 'INCOMPLETE: send ZIP/log.',flush=True)
'''
    nb = dict(nbformat=4, nbformat_minor=5, cells=[cell('markdown', '''
        # Kiểm tra riding → ride hay rid (CPU)

        **Accelerator None, Internet ON. Không cần Add Input, Dataset hay API token.**
        40captionTRAIN có chủ đích +5câu kiểm soát ngôn ngữ; không chạy lại toànTRAIN.
        Cùng spaCy3.8.11/Thinc8.3.11/NumPy2.1.3/model3.8.0 và so khớp dữ liệu model
        với Output126. Xuất nguyên từ, POS, tag, lemma và bảng HTML; không sửa bank/ROI.

        Save & Run All. Gửi `csa_lemma_trace_check_review.zip`.
        Không chạy model caption, GPU, CLIP text, huấn luyện hoặc lưu Dataset.
        Dữ liệu Output126 giữ nguyên; có thể lưu riêng bằng notebook SAVE126 đã gửi.
        '''), cell('code', setup), cell('code', RUNNER), cell('code', execution)],
        metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                      kaggle=dict(isInternetEnabled=True, isGpuEnabled=False)))
    target=HERE/'CSA_LEMMA_TRACE_CPU.ipynb'
    target.write_bytes((json.dumps(nb,ensure_ascii=False,indent=1)+'\n').encode('utf-8'))
    assert target.stat().st_size<1_000_000
    (HERE/'LEMMA_TRACE_PACKAGE.json').write_text(json.dumps(dict(module_sha256=hashes,
        packet_sha256=hashlib.sha256(packet).hexdigest(), environment_sha256=hashlib.sha256(environment.encode()).hexdigest(),
        notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), notebook_bytes=target.stat().st_size,
        notebook_executed_by_agent=False, model_executed_by_agent=False),indent=2)+'\n',encoding='utf-8')
    print(target.name,target.stat().st_size,'bytes;45cases; unexecuted')


if __name__=='__main__':
    make()
