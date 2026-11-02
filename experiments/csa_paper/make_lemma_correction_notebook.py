"""Author CPU subset bank repair; no POS/model/API execution by the agent."""
import hashlib
import json
import subprocess

from make_smoke_notebook import HERE, ROOT, RUNNER, cell

BANK_SHA = '450fc969e0842ca3270d1f899b7e8fe594639260644ad5815a69430fe48264b1'


def make():
    origin = subprocess.check_output(['git', 'rev-parse', '28361ce'], cwd=ROOT, text=True).strip()
    bank_path = 'experiments/csa_paper/real_preparation_review_text/bank_words.json'
    assert hashlib.sha256(subprocess.check_output(['git', 'show', origin+':'+bank_path], cwd=ROOT)).hexdigest() == BANK_SHA
    bank_url = f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{origin}/{bank_path}'
    old = json.loads((HERE/'CSA_LEMMA_TRACE_CPU.ipynb').read_text(encoding='utf-8'))
    execute = old['cells'][-1]['source']
    marker = "    run([PYTHON, '-u', SOURCE/'run_lemma_trace.py'"
    assert marker in execute
    prefix = execute[:execute.index(marker)]
    prefix = prefix.replace('    import numpy as np', '''    if not COCO_JSON.is_file(): raise FileNotFoundError('Add Input vuthetam/mscoco-2014 and check COCO_JSON')
    payload=urllib.request.urlopen(BANK_URL,timeout=120).read()
    if hashlib.sha256(payload).hexdigest()!=BANK_SHA: raise RuntimeError('Original126bank SHA differs')
    (SOURCE/'original_bank_words.json').write_bytes(payload)
    import numpy as np''')
    previous = json.loads((HERE/'real_preparation_review_text/summary.json').read_bytes())
    environment = json.dumps(dict(versions={k: previous['versions'][k] for k in ('spacy', 'thinc', 'numpy', 'en-core-web-sm')},
        pos_model_files_sha256=previous['pos_model_files_sha256']), sort_keys=True)
    modules = {n: (HERE/n).read_text(encoding='utf-8') for n in ('lemma_correction.py', 'run_lemma_correction.py')}
    hashes = {n: hashlib.sha256(s.encode()).hexdigest() for n, s in modules.items()}
    setup = '''
import hashlib, json, os, queue, signal, subprocess, sys, tempfile, threading, time, traceback, urllib.request, zipfile
from pathlib import Path
OUTPUT=Path('/kaggle/working/csa_bank_lemma_correction')
COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
if OUTPUT.exists(): raise RuntimeError('Preserve previous review ZIP; use a fresh CPU session.')
OUTPUT.mkdir(parents=True)
SOURCE=Path(tempfile.mkdtemp(prefix='csa-lemma-correction-'))
LOG=OUTPUT/'notebook.log'
'''
    setup += 'BANK_URL='+repr(bank_url)+'\nBANK_SHA='+repr(BANK_SHA)+'\nMODULES='+repr(modules)+'\nMODULE_HASHES='+repr(hashes)+'\nENVIRONMENT='+repr(environment)+'\n'
    setup += '''
for name, source in MODULES.items():
    payload=source.encode('utf-8')
    if hashlib.sha256(payload).hexdigest()!=MODULE_HASHES[name]: raise RuntimeError('Module SHA differs: '+name)
    (SOURCE/name).write_bytes(payload)
(SOURCE/'source_environment.json').write_text(ENVIRONMENT,encoding='utf-8')
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(module_sha256=MODULE_HASHES,
    original_bank_url=BANK_URL, original_bank_sha256=BANK_SHA, preserved_roi_source_version=126,
    source_environment_sha256=hashlib.sha256(ENVIRONMENT.encode()).hexdigest()),indent=2),encoding='utf-8')
print('CPU subset recount only; original bank and ROI126 remain unchanged.',flush=True)
'''
    execution = prefix + '''    run([PYTHON,'-u',SOURCE/'run_lemma_correction.py','--dataset-json',COCO_JSON,'--output',OUTPUT],timeout=1800)
    summary=json.loads((OUTPUT/'summary.json').read_text(encoding='utf-8'))
    complete=summary.get('status')=='BANK_LEMMA_CORRECTION_COMPLETE'
    if not complete: raise RuntimeError('Correction incomplete')
    receipt=json.loads((OUTPUT/'correction_receipt.json').read_text(encoding='utf-8'))
    if hashlib.sha256((OUTPUT/'bank_words_corrected.json').read_bytes()).hexdigest()!=receipt['corrected_bank_sha256']: raise RuntimeError('Corrected bank SHA differs')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
except BaseException:
    (OUTPUT/'notebook_failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
    raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,original_bank_modified=False,
        roi_modified=False,caption_targets_changed=False,dataset_created=False)),encoding='utf-8')
    archive=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.iterdir()):
            if p.is_file(): z.write(p,OUTPUT.name+'/'+p.name)
    print('SEND REVIEW ZIP:',archive,flush=True)
    print('CORRECTION COMPLETE; new bank file, original126 preserved.' if complete else 'INCOMPLETE: send ZIP/log.',flush=True)
'''
    notebook = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        kaggle=dict(isInternetEnabled=True, isGpuEnabled=False)), cells=[cell('markdown', '''
        # Sửa lemma riding/rides trong bank — CPU

        **Accelerator None, Internet ON. Add Input duy nhất `vuthetam/mscoco-2014`.**
        Không cần Dataset126, visual-cache, detector, API token hoặc GPU.
        Dùng cùng POS/model đã kiểm; chỉ chạy POS lại trên captionTRAIN chứa riding/rides,
        không chạy lại toàn566747caption. Đếm token thực sự chuyển rid→ride dưới điều kiện
        VERB +riding/rides +lemma cũ rid. Không đổi rid/ridding thật, noun, captiontarget, ROI.

        Xuất **bank_words_corrected.json mới**, giữ nguyên bankgốc và Output126.
        Save & Run All; gửi `csa_bank_lemma_correction_review.zip`.
        Chưa mã hóaCLIPtext/huấn luyện; hạn worker30phút là cap, không phải ETA.
        '''), cell('code', setup), cell('code', RUNNER), cell('code', execution)])
    target=HERE/'CSA_BANK_LEMMA_CORRECTION_CPU.ipynb'
    target.write_bytes((json.dumps(notebook,ensure_ascii=False,indent=1)+'\n').encode('utf-8'))
    assert target.stat().st_size<1_000_000
    (HERE/'LEMMA_CORRECTION_PACKAGE.json').write_text(json.dumps(dict(module_sha256=hashes,
        original_bank_commit=origin, original_bank_sha256=BANK_SHA,
        notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), notebook_bytes=target.stat().st_size,
        model_executed_by_agent=False, notebook_executed_by_agent=False),indent=2)+'\n',encoding='utf-8')
    print(target.name,target.stat().st_size,'bytes;unexecuted; originalbankunchanged')


if __name__=='__main__':
    make()
