"""Build a version98 CPU-only upload notebook; never execute Kaggle API locally."""
import hashlib,importlib.util,json,textwrap
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
nb=load('notebook_helpers',ROOT/'experiments/upt_hidden_pair/make_notebooks.py');nb.HERE=HERE
old=load('old_api',ROOT/'experiments/upt_hidden_pair/make_dataset_api_notebook.py')
KERNEL_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/98'
VERSION_PREFLIGHT=old.VERSION_PREFLIGHT.replace('/58','/98').replace('v58','v98')

def build():
 source=(HERE/'package_dataset.py').read_text(encoding='utf-8');inventory=json.loads((HERE/'upload_inventory.json').read_text())
 cells=[nb.md('''
 # Lưu stale prompt preparation 2000 — Output version98
 **CPU / Internet ON**, không GPU, không chạy lại cache hoặc train.
 Tải đúng `ducanh2403/2-vi-cai-kia-qua-nhieu/98`, kiểm fingerprint receipt
 và SHA từng payload, rồi tạo dataset Private mới. Không lấy latest khi thiếu file.
 Bật secret chứa API token; SECRET_NAME có thể là KAGGLE_API_TOKEN hoặc tên bạn đã đặt.
 Cần khoảng6GB trống cho tải và bản sao. ZIP review không chứa3HDF5.
 Không ghi token ra Output; không cập nhật/ghi đè dataset đã tồn tại.
 ''')]
 cells.append(nb.code('''
 from pathlib import Path
 KERNEL_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/98'
 OWNER='ducanh2403'
 SLUG='stale-prompt-preparation-2000'
 TITLE='Stale prompt preparation 2000'
 SECRET_NAME='KAGGLE_API_TOKEN'
 DESTINATION=Path('/kaggle/working/stale_prompt_preparation_2000_dataset')
 '''))
 setup='''
 import os,re,subprocess,sys
 from kaggle_secrets import UserSecretsClient
 if not re.fullmatch(r'[a-zA-Z0-9_-]+',OWNER) or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',SLUG):raise ValueError('Invalid OWNER/SLUG.')
 if not 3<=len(SLUG)<=50 or not 6<=len(TITLE)<=50:raise ValueError('Invalid title/slug length.')
 try:
     token=UserSecretsClient().get_secret(SECRET_NAME).strip()
     if not token:raise ValueError('empty')
 except Exception:raise RuntimeError('Enable SECRET_NAME in Add-ons > Secrets; do not paste token in code/chat.') from None
 def run_cli(args,authenticated=True):
     environment=os.environ.copy()
     for key in ('KAGGLE_API_TOKEN','KAGGLE_USERNAME','KAGGLE_KEY'):environment.pop(key,None)
     if authenticated:environment['KAGGLE_API_TOKEN']=token
     environment['PYTHONUNBUFFERED']='1'
     with subprocess.Popen(args,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace') as process:
         for line in process.stdout:print(line.replace(token,'[REDACTED]'),end='',flush=True)
         code=process.wait()
     if code:raise RuntimeError(f'Command failed ({code}); inspect redacted output above. No fallback to latest.')
 '''
 cells.append(nb.code(textwrap.dedent(setup)+f"\nrun_cli([sys.executable,'-m','pip','install','--upgrade','--force-reinstall','--quiet','https://github.com/Kaggle/kaggle-cli/archive/{old.KAGGLE_COMMIT}.zip'],authenticated=False)\nVERSION_PREFLIGHT="+repr(VERSION_PREFLIGHT)+"\nrun_cli([sys.executable,'-u','-c',VERSION_PREFLIGHT])\n"))
 cells.append(nb.code('import hashlib,json\nINVENTORY='+repr(inventory)+'\nSOURCE_CODE='+repr(source)+'\nassert hashlib.sha256(SOURCE_CODE.encode()).hexdigest()=='+repr(hashlib.sha256(source.encode()).hexdigest())+"\nexec(compile(SOURCE_CODE,'package_dataset.py','exec'))\n"))
 cells.append(nb.code('''
 import tempfile,shutil
 if KERNEL_VERSION!='ducanh2403/2-vi-cai-kia-qua-nhieu/98':raise ValueError('This audited notebook pins98. No automatic latest/version fallback.')
 if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<6*1024**3:raise RuntimeError('Need about6GiB free for download and staging.')
 DOWNLOAD=Path(tempfile.mkdtemp(prefix='stale-preparation-v98-',dir='/tmp'))
 names=set(INVENTORY['files'])|{'panel/'+n for n in INVENTORY['panel_files']}
 pattern=r'(^|/)(?:'+'|'.join(re.escape(n) for n in names)+r')$'
 run_cli([sys.executable,'-m','kaggle','kernels','output',KERNEL_VERSION,'-p',str(DOWNLOAD),'--file-pattern',pattern])
 matches=[p.parent for p in DOWNLOAD.rglob('receipt.json') if sha(p)==INVENTORY['files']['receipt.json']['sha256']]
 if len(matches)!=1:raise RuntimeError(f'Version98 returned {len(matches)} matching receipts; STOP. Preserve source output; do not substitute latest.')
 SOURCE=matches[0]
 report=package(SOURCE,DESTINATION,INVENTORY)
 report['source_kernel_version']=KERNEL_VERSION
 (DESTINATION/'transport_receipt.json').write_text(json.dumps(report,indent=2)+'\\n',encoding='utf-8')
 print('Verified3HDF5 + metadata + panel from version98. Ready for private upload.',flush=True)
 '''))
 cells.append(nb.code('CREATE_DATASET_SCRIPT='+repr(old.CREATE_DATASET_SCRIPT)+'\n'+textwrap.dedent('''
 metadata_path=DESTINATION/'dataset-metadata.json'
 metadata=dict(title=TITLE,id=f'{OWNER}/{SLUG}',licenses=[dict(name='other')],description='Private research cache from pinned notebook version98. Frozen original/occluded CLIP visual features and original/neutral semantic prompt embeddings for2000 locked TRAIN images;3HDF5 payloads and verification metadata. Upstream COCO/model rights retained. No trained checkpoint or caption-quality claim. Panel images retain original data terms.')
 if metadata_path.exists():raise RuntimeError('Existing metadata preserved; use a fresh staging destination.')
 try:
     metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
     run_cli([sys.executable,'-u','-c',CREATE_DATASET_SCRIPT,str(DESTINATION)])
 finally:metadata_path.unlink(missing_ok=True)
 print('Creation accepted; wait for processing, do not create repeatedly.',flush=True)
 print(f'https://www.kaggle.com/datasets/{OWNER}/{SLUG}',flush=True)
 try:run_cli([sys.executable,'-m','kaggle','datasets','status',f'{OWNER}/{SLUG}'])
 except RuntimeError:print('Status not confirmed; check dataset page before retrying.',flush=True)
 ''')))
 cells.append(nb.md('''
 Khi Kaggle xử lý xong, kiểm tra đủ `aux_rank0.h5`, `aux_rank1.h5`, `benchmark_inputs.h5`,
 `receipt.json`, `protocol.json`, `manifest.json` ngay gốc dataset. Gửi path gốc đó để điền C/T.
 Panel được giữ dưới tên phẳng `panel__NAME` để tránh upload bỏ qua subfolder;
 panel gốc xem trong ZIP review đã tải. Không có training trong notebook này.
 '''))
 nb.write('SAVE_STALE_PROMPT_2000_DATASET.ipynb',cells)
 path=HERE/'SAVE_STALE_PROMPT_2000_DATASET.ipynb';b=json.loads(path.read_text(encoding='utf-8'));b['metadata']['kaggle']=dict(isInternetEnabled=True,accelerator='none');path.write_text(json.dumps(b,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
 return path
if __name__=='__main__':print(build())
