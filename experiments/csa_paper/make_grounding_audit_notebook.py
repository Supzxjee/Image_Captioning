"""Generate pinned CPU audit notebook; never submits or executes Kaggle."""
import argparse
import hashlib
import json
import subprocess
from make_smoke_notebook import HERE,ROOT,PREFIX,RUNNER,cell

FILES=['grounding_labels.py','run_grounding_audit.py','test_grounding_labels.py',
       'SAVE_FULL_ROI_V134_PACKAGE.json','GROUNDED_ATTENTION_AUDIT_PROTOCOL.md']


def make(revision):
    commit=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',commit+':'+PREFIX+n],cwd=ROOT)).hexdigest() for n in FILES}
    setup=f'''
import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile
from pathlib import Path
COMMIT={commit!r}
HASHES={hashes!r}
FULL_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-full-roi-karpathy-v134')
COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
OUTPUT=Path('/kaggle/working/csa_grounding_data_audit_10k')
if OUTPUT.exists():raise RuntimeError('Preserve previous output; use a fresh session.')
if not FULL_ROOT.is_dir() or not COCO_JSON.is_file():raise FileNotFoundError('Attach both datasets; check FULL_ROOT and COCO_JSON before running.')
OUTPUT.mkdir(parents=True)
SOURCE=Path(tempfile.mkdtemp(prefix='csa-grounding-audit-'))
LOG=OUTPUT/'notebook.log'
CONFIG=SOURCE/'config.json'
CONFIG.write_text(json.dumps(dict(full_root=str(FULL_ROOT),dataset_json=str(COCO_JSON),output=str(OUTPUT),work=str(SOURCE/'work'))),encoding='utf-8')
for name,expected in HASHES.items():
    data=urllib.request.urlopen(f'https://raw.githubusercontent.com/Supzxjee/Image_Captioning/{{COMMIT}}/{PREFIX}{{name}}',timeout=90).read()
    if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('Changed source '+name)
    (SOURCE/name).write_bytes(data)
(OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=COMMIT,sha256=HASHES),indent=2),encoding='utf-8')
print('CPU audit source ready; no GPU/checkpoint/API token needed.',flush=True)
'''
    execute='''
complete=False
try:
    run([sys.executable,'-m','pip','install','-q','ijson==3.4.0'],timeout=300)
    run([sys.executable,'-u',SOURCE/'test_grounding_labels.py'],timeout=120)
    run([sys.executable,'-u',SOURCE/'run_grounding_audit.py','--config',CONFIG],timeout=7200)
    result=json.loads((OUTPUT/'summary.json').read_text(encoding='utf-8'))
    complete=result['status']=='CSA_GROUNDING_DATA_AUDIT_COMPLETE'
    if not complete:raise RuntimeError('Audit incomplete')
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
    raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,model_executed=False,training_ready=False,manual_review_pending=True)),encoding='utf-8')
    # Keep full labels/filtered annotations in Output; send only review/sample metadata.
    archive=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUTPUT.iterdir()):
            if p.is_file() and p.name not in ('mentions.jsonl','train10k_instances.json'):z.write(p,OUTPUT.name+'/'+p.name)
    print('SEND REVIEW:',archive,flush=True)
    print('KEEP ENTIRE OUTPUT:',OUTPUT,flush=True)
    print('Audit complete; human review required before GPU.' if complete else 'INCOMPLETE: send ZIP/log.',flush=True)
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3')),
            cells=[cell('markdown','''
            # CSA — kiểm nhãn từ–vùng trước huấn luyện (CPU)
            Accelerator **None**, Internet **ON**. Add Input:
            1. `ducanh2403/csa-full-roi-karpathy-v134`
            2. `vuthetam/mscoco-2014`

            Không cần checkpoint/bank/API token/visual cache gốc. Dùng ROI đã lưu;
            không chạy CLIP/detector. Tải annotation COCO chính thức qua Internet,
            chỉ giữ nhãn của 10.000 TRAIN/restval. Không huấn luyện hoặc chấm TEST.
            Sửa FULL_ROOT/COCO_JSON nếu mount khác; không bỏ kiểm hash.

            Save & Run All. Gửi `csa_grounding_data_audit_10k_review.zip`.
            Giữ toàn bộ folder output, gồm mentions.jsonl và train10k_instances.json,
            để tái sử dụng. Trang review chưa điền nhãn; ảnh cần Internet để hiển thị.
            Hai giờ là cap CPU, không phải ETA. Không bật GPU cho notebook này.
            '''),cell('code',setup),cell('code',RUNNER.replace('Smoke active;','CPU audit active;')),cell('code',execute)])
    target=HERE/'CSA_GROUNDING_DATA_AUDIT_10K_CPU.ipynb'
    target.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    assert target.stat().st_size<1_000_000
    (HERE/'GROUNDING_AUDIT_PACKAGE.json').write_text(json.dumps(dict(core_commit=commit,sha256=hashes,notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),user_run_pending=True),indent=2)+'\n',encoding='utf-8')
    print(target.name,target.stat().st_size,'bytes; unexecuted')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('revision');make(p.parse_args().revision)
