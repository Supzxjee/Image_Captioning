"""Build a CPU-only private upload notebook for exact Output102. Never call API."""
import hashlib
import importlib.util
import json
from pathlib import Path
import textwrap

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


nb=load('notebook_helpers',ROOT/'experiments/upt_hidden_pair/make_notebooks.py');nb.HERE=HERE
old=load('dataset_api_helpers',ROOT/'experiments/upt_hidden_pair/make_dataset_api_notebook.py')
VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/102'
VERSION_PREFLIGHT=old.VERSION_PREFLIGHT.replace('/58','/102').replace('v58','v102')


def build():
    result=json.loads((HERE/'PREPARATION_RESULT.json').read_text());source=(HERE/'package_gain_dataset.py').read_text(encoding='utf-8')
    inventory=dict(schema='grounded_gain_preparation_transport_v1',kernel_version=VERSION,
        files=dict(result['declared_payload_sha256'],**{'receipt.json':result['receipt_sha256']}),
        source_review_zip_sha256=result['zip_sha256'])
    cells=[nb.md('''
        # Lưu grounded gain preparation — Output102 thành Dataset
        **CPU / Internet ON / Save & Run All. Không GPU, không chạy lại chuẩn bị hay train.**
        Tải đúng `ducanh2403/2-vi-cai-kia-qua-nhieu/102`, giữ nguyên5payload đã kiểm tra.
        Không lấy latest khi thiếu file; không cần Add Input hoặc model checkpoint.
        Cần khoảng5GiB trống cho bản tải và bản sao, gồm toàn bộ `inputs.h5`.
        API token đọc từ Kaggle Secrets; không dán token vào code/chat.
        Chạy bằng tài khoản có quyền xem Output102; OWNER phải khớp tài khoản sở hữu token.
        SECRET_NAME là tên secret trong Add-ons → Secrets, khác tên gợi nhớ của token trên trang Settings.
        Nếu secret của bạn tên `codex`, đổi SECRET_NAME thành `codex` và bật quyền cho notebook này.
        Chỉ tạo dataset **Private mới**, không ghi đè dataset đã tồn tại.
        '''),nb.code('''
        from pathlib import Path
        KERNEL_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/102'
        OWNER='ducanh2403'
        SLUG='grounded-gain-prep-v102'
        TITLE='Grounded gain v102 feature dataset'
        SECRET_NAME='KAGGLE_API_TOKEN'
        DESTINATION=Path('/kaggle/working/grounded_gain_preparation_dataset')
        ''')]
    setup='''
        import os,re,subprocess,sys
        from kaggle_secrets import UserSecretsClient
        if KERNEL_VERSION!='ducanh2403/2-vi-cai-kia-qua-nhieu/102':raise ValueError('This notebook pins102; do not substitute latest or another version.')
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',OWNER) or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',SLUG):raise ValueError('Invalid OWNER/SLUG')
        if not 6<=len(SLUG)<=50 or not 6<=len(TITLE)<=50:raise ValueError('TITLE/SLUG must have6–50characters')
        try:
            token=UserSecretsClient().get_secret(SECRET_NAME).strip()
            if not token:raise ValueError('empty')
        except Exception:raise RuntimeError('Enable the configured SECRET_NAME in Add-ons > Secrets. Do not paste token in code/chat.') from None
        def run_cli(args,authenticated=True):
            environment=os.environ.copy()
            for key in ('KAGGLE_API_TOKEN','KAGGLE_USERNAME','KAGGLE_KEY'):environment.pop(key,None)
            if authenticated:environment['KAGGLE_API_TOKEN']=token
            environment['PYTHONUNBUFFERED']='1'
            with subprocess.Popen(args,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace') as process:
                for line in process.stdout:print(line.replace(token,'[REDACTED]'),end='',flush=True)
                code=process.wait()
            if code:raise RuntimeError(f'Command failed ({code}); see redacted output. No automatic fallback or dataset overwrite.')
        '''
    cells.append(nb.code(textwrap.dedent(setup)+f"\nrun_cli([sys.executable,'-m','pip','install','--upgrade','--force-reinstall','--quiet','https://github.com/Kaggle/kaggle-cli/archive/{old.KAGGLE_COMMIT}.zip'],authenticated=False)\nVERSION_PREFLIGHT="+repr(VERSION_PREFLIGHT)+"\nrun_cli([sys.executable,'-u','-c',VERSION_PREFLIGHT])\n"))
    cells.append(nb.code('import hashlib,json\nINVENTORY='+repr(inventory)+'\nSOURCE_CODE='+repr(source)+
        '\nassert hashlib.sha256(SOURCE_CODE.encode()).hexdigest()=='+repr(hashlib.sha256(source.encode()).hexdigest())+
        "\nexec(compile(SOURCE_CODE,'package_gain_dataset.py','exec'))\n"))
    cells.append(nb.code('''
        import tempfile,shutil
        if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<5*1024**3:raise RuntimeError('Need about5GiB free for download+copy; preserve existing output.')
        DOWNLOAD=Path(tempfile.mkdtemp(prefix='grounded-gain-v102-',dir='/tmp'))
        pattern=r'(^|/)(?:'+'|'.join(re.escape(n) for n in INVENTORY['files'])+r')$'
        print('Downloading full preparation from:',KERNEL_VERSION,flush=True)
        run_cli([sys.executable,'-m','kaggle','kernels','output',KERNEL_VERSION,'-p',str(DOWNLOAD),'--file-pattern',pattern])
        matches=[p.parent for p in DOWNLOAD.rglob('receipt.json') if sha(p)==INVENTORY['files']['receipt.json']]
        if len(matches)!=1:raise RuntimeError(f'Output102 returned {len(matches)} matching receipts; STOP, no latest/version fallback.')
        report=package(matches[0],DESTINATION,INVENTORY)
        print('All5original files verified. Ready for private dataset upload:',DESTINATION,flush=True)
        '''))
    cells.append(nb.code('CREATE_DATASET_SCRIPT='+repr(old.CREATE_DATASET_SCRIPT)+'\n'+textwrap.dedent('''
        metadata_path=DESTINATION/'dataset-metadata.json'
        metadata=dict(title=TITLE,id=f'{OWNER}/{SLUG}',licenses=[dict(name='other')],
            description='Other license: private research preparation from pinned notebook Output102. Original COCO/model rights and upstream terms retained. Exact cached CLIP/original-prompt inputs for2000TRAIN+1000DEV, training-only object targets and common initial head. No trained checkpoint, new feature extraction or caption-quality claim. See original receipt and transport receipt.')
        if metadata_path.exists():raise RuntimeError('Existing metadata preserved; choose fresh staging destination.')
        try:
            metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
            run_cli([sys.executable,'-u','-c',CREATE_DATASET_SCRIPT,str(DESTINATION)])
        finally:metadata_path.unlink(missing_ok=True)
        print('API accepted creation; processing may still be pending. Do not repeatedly create.',flush=True)
        print(f'https://www.kaggle.com/datasets/{OWNER}/{SLUG}',flush=True)
        try:run_cli([sys.executable,'-m','kaggle','datasets','status',f'{OWNER}/{SLUG}'])
        except RuntimeError:print('Status not confirmed; check dataset page before retrying.',flush=True)
        try:run_cli([sys.executable,'-m','kaggle','datasets','files',f'{OWNER}/{SLUG}','--page-size','100'])
        except RuntimeError:print('File list not available yet. Wait for processing; do not rerun create.',flush=True)
        ''')))
    cells.append(nb.md('''
        Sau khi Kaggle xử lý xong, kiểm tra5file gốc: `receipt.json`, `inputs.h5`,
        `object_targets.json`, `gain_metadata.json.gz`, `initial_head.bin` (thêm transportreceipt).
        Chia sẻ dataset Private cho Ngân; gắn cùng dataset vào C/H và T.
        Cache nằm NGAY GỐC dataset, ví dụ:
        `/kaggle/input/datasets/ducanh2403/grounded-gain-prep-v102`.
        Không thêm `/grounded_gain_preparation` nếu file nằm ở gốc.
        C/H đổi MODE='train', T giữ train; hai notebook T4x2 và cùng A/COCOJSON.
        `PREPARED_ROOT=None` tự tìm receipt đúng; không phải chạy lại bước prepare.
        Notebook này chưa được chạy API bởi Codex; chỉ kiểm tra local bằng file/secret giả lập.
        '''))
    nb.write('SAVE_GROUNDED_GAIN_V102_DATASET.ipynb',cells)
    target=HERE/'SAVE_GROUNDED_GAIN_V102_DATASET.ipynb';n=json.loads(target.read_text(encoding='utf-8'))
    n['metadata']['kaggle']=dict(isInternetEnabled=True,accelerator='none')
    target.write_text(json.dumps(n,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return target


if __name__=='__main__':print(build())
