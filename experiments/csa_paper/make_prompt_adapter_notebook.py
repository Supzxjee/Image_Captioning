"""Author a CPU-only mapping notebook; never run a model or submit Kaggle."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from make_smoke_notebook import cell,RUNNER

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PREFIX='experiments/csa_paper/'
FILES=['prompt_adapter_data.py','prompt_adapter_model.py','run_prompt_mapping_audit.py',
       'test_prompt_adapter_data.py','test_prompt_adapter_model.py','make_prompt_adapter_notebook.py',
       'real_bank_data.py','audit_assets.py','data.py','model.py','SAVE_FULL_ROI_V134_PACKAGE.json']
EXTERNAL=['captioning/__init__.py','captioning/upt_prompts.py','captioning/upt_action_text.py',
          'captioning/selected_relation_prompts.py','captioning/supervised_relations.py']


def make(revision):
    revision=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    mapping={PREFIX+n:n for n in FILES};mapping.update({n:n for n in EXTERNAL})
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',revision+':'+n],cwd=ROOT)).hexdigest() for n in mapping}
    setup='''
import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile,shutil
from pathlib import Path
FULL_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-full-roi-karpathy-v134')
PROMPTS=Path('/kaggle/input/datasets/ducanh2403/upt-prompts-full-all-actions-v2/prompt_cache.json')
DETECTIONS=Path('/kaggle/input/datasets/ducanh2403/objectdetectionecache/objectdetectioncache.json')
OUTPUT=Path('/kaggle/working/csa_prompt_roi_audit_v1')
if OUTPUT.exists():raise RuntimeError('Preserve existing output; use a fresh runtime/output directory.')
for p in (FULL_ROOT,PROMPTS,DETECTIONS):
    if not p.exists():raise FileNotFoundError('Attach the three listed Datasets or correct their mount paths: '+str(p))
SOURCE=Path(tempfile.mkdtemp(prefix='csa-prompt-audit-',dir='/tmp'))
LOG=SOURCE/'notebook.log'
'''
    setup+='\nREVISION='+repr(revision)+'\nMAPPING='+repr(mapping)+'\nHASHES='+repr(hashes)+'\n'
    setup+='''
print('CPU-only prompt/ROI audit. No model download or training.',flush=True)
for original,destination in MAPPING.items():
    url='https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+REVISION+'/'+original
    payload=urllib.request.urlopen(url,timeout=90).read()
    if hashlib.sha256(payload).hexdigest()!=HASHES[original]:raise RuntimeError('Source SHA differs: '+original)
    target=SOURCE/destination;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
(SOURCE/'source_receipt.json').write_text(json.dumps(dict(commit=REVISION,sha256=HASHES),indent=2),encoding='utf-8')
os.environ['PYTHONPATH']=str(SOURCE)
CONFIG=SOURCE/'config.json'
CONFIG.write_text(json.dumps(dict(full_root=str(FULL_ROOT),prompts=str(PROMPTS),detections=str(DETECTIONS),output=str(OUTPUT),wall_seconds=7200)),encoding='utf-8')
'''
    final='''
complete=False
try:
    run([sys.executable,'-u',SOURCE/'test_prompt_adapter_data.py'],timeout=120)
    run([sys.executable,'-u',SOURCE/'run_prompt_mapping_audit.py','--config',CONFIG],timeout=7500)
    complete=json.loads((OUTPUT/'summary.json').read_text())['status']=='PROMPT_ROI_MAPPING_AUDIT_COMPLETE'
finally:
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if LOG.exists():shutil.copy2(LOG,OUTPUT/'notebook.log')
    shutil.copy2(SOURCE/'source_receipt.json',OUTPUT/'source_receipt.json')
    sys.path.insert(0,str(SOURCE))
    from run_prompt_mapping_audit import archive
    archive(OUTPUT)
    print('Mapping audit complete; send review ZIP. This is not a trained caption model.' if complete else 'INCOMPLETE: send review ZIP/log; do not train from partial links.',flush=True)
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        kaggle=dict(isInternetEnabled=True,isGpuEnabled=False)),cells=[
        cell('markdown','# CSA prompt–ROI mapping audit\nBước đầu của pilot đã duyệt. **Accelerator None / Internet ON**. Chỉ kiểm TRAIΝ10k + VALID5000, không dùng TEST, không chạy CLIP/UPT/caption model, không train. Thêm ba Dataset: full ROI134; UPT prompt all-actions-v2; objectdetectionecache. Không cần checkpoint, ảnh JPEG, bank130 hay API Secret. Sửa ba path ở cell đầu nếu mount khác. Giữ toàn bộ output và gửi ZIP review; cần kiểm mapping trước khi tạo cache CLIP prompt và chạy hai nhánh500update.'),
        cell('code',setup),cell('code',RUNNER.replace('Smoke active;','CPU mapping active;').replace('Smoke wall cap exceeded','CPU mapping wall cap exceeded')),
        cell('code',final)])
    target=HERE/'CSA_PROMPT_ROI_AUDIT_CPU.ipynb';target.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(target.name,target.stat().st_size,'bytes; source',revision)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);make(p.parse_args().revision)
