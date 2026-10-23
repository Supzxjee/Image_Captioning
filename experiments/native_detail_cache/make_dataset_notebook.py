"""Create a CPU-only private dataset-upload notebook. No API calls at build time."""
import hashlib
import importlib.util
import json
import textwrap
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('old_api_builder',ROOT/'experiments/upt_hidden_pair/make_dataset_api_notebook.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
spec=importlib.util.spec_from_file_location('notebook_helpers',ROOT/'experiments/upt_hidden_pair/make_notebooks.py')
nb=importlib.util.module_from_spec(spec);spec.loader.exec_module(nb);nb.HERE=HERE
KERNEL_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/89'
VERSION_PREFLIGHT=old.VERSION_PREFLIGHT.replace('/58','/89').replace('v58','v89')


def build():
    source=(HERE/'package_dataset.py').read_text(encoding='utf-8');inventory=json.loads((HERE/'upload_inventory.json').read_text(encoding='utf-8'))
    cells=[nb.md('''
        # Lưu native-detail H/L 15k thành dataset Kaggle riêng
        **CPU, không cần GPU**, Internet ON. Không chạy lại cache hay huấn luyện.
        Tự tải Output **version89** của `ducanh2403/2-vi-cai-kia-qua-nhieu`; **không cần Add Input**.
        Khóa đúng89, không tự chuyển sang latest hay chạy lại notebook nguồn.
        Phải có30 file `.h5` và `cache_receipt.json`; ZIP review không chứa đặc trưng.
        Tải đúng63 tệp cần dùng từ lượt89, xác minh fingerprint cache và toàn bộ hash,
        sao chép sang folder phẳng rồi upload thành dataset **Private mới** bằng API token.
        Log/mã của lần chạy gốc vẫn nằm trong Output nguồn và báo cáo GitHub.

        1. Add-ons → Secrets: thêm/bật secret **KAGGLE_API_TOKEN** chứa token Kaggle API.
        2. Sửa OWNER đúng tài khoản sở hữu token. SECRET_NAME là tên secret, có thể đổi nếu bạn đặt tên khác.
        3. Chạy lần lượt, hoặc **Save & Run All** để giữ bản notebook này.
        Không ghi token ra file/Output; không sửa dataset đã tồn tại. Nếu slug đã có, kiểm tra dataset đó trước.
        Cần khoảng12,6GB dung lượng trống cho bản tải và bản sao; thời gian kiểm tra/copy/upload phụ thuộc I/O và mạng.
        ''')]
    cells.append(nb.code('''
        from pathlib import Path
        KERNEL_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/89'
        OWNER='ducanh2403'
        SLUG='native-detail-hl-15k'
        TITLE='Native detail HL 15k cache'
        SECRET_NAME='KAGGLE_API_TOKEN'
        DESTINATION=Path('/kaggle/working/native_detail_HL_15k_dataset')
        '''))
    cells.append(nb.code("import hashlib,json\nINVENTORY="+repr(inventory)+"\nSOURCE_CODE="+repr(source)+"\nassert hashlib.sha256(SOURCE_CODE.encode()).hexdigest()=="+repr(hashlib.sha256(source.encode()).hexdigest())+"\nexec(compile(SOURCE_CODE,'package_dataset.py','exec'))\n"))
    cells.append(nb.code(textwrap.dedent('''
        import os,re,subprocess,sys
        from kaggle_secrets import UserSecretsClient
        if not re.fullmatch(r'[a-zA-Z0-9_-]+',OWNER) or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',SLUG):raise ValueError('Invalid OWNER/SLUG.')
        if not 3<=len(SLUG)<=50 or not 6<=len(TITLE)<=50:raise ValueError('SLUG length3–50; TITLE length6–50.')
        try:
            token=UserSecretsClient().get_secret(SECRET_NAME).strip()
            if not token:raise ValueError('empty')
        except Exception:
            raise RuntimeError('Enable the configured SECRET_NAME in Add-ons > Secrets. Do not paste your token into chat or a code cell.') from None
        def run_cli(args,authenticated=True):
            environment=os.environ.copy()
            for key in ('KAGGLE_API_TOKEN','KAGGLE_USERNAME','KAGGLE_KEY'):environment.pop(key,None)
            if authenticated:environment['KAGGLE_API_TOKEN']=token
            environment['PYTHONUNBUFFERED']='1'
            with subprocess.Popen(args,env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace') as process:
                for line in process.stdout:print(line.replace(token,'[REDACTED]'),end='',flush=True)
                code=process.wait()
            if code:raise RuntimeError(f'Command failed (exit {code}); see redacted output above. Check token/account/Internet or existing dataset slug.')
        ''')+f"\nrun_cli([sys.executable,'-m','pip','install','--upgrade','--force-reinstall','--quiet','https://github.com/Kaggle/kaggle-cli/archive/{old.KAGGLE_COMMIT}.zip'],authenticated=False)\n"))
    cells.append(nb.code('CREATE_DATASET_SCRIPT='+repr(old.CREATE_DATASET_SCRIPT)+'\n'+textwrap.dedent('''
        metadata_path=DESTINATION/'dataset-metadata.json'
        metadata=dict(title=TITLE,id=f'{OWNER}/{SLUG}',licenses=[dict(name='other')],
            description='Private research feature cache. Original data/model rights and upstream terms retained. Frozen CLIP native/degraded quadrant features for locked COCO train10000 and val5000. No original images. See cache/spec/transport receipts for provenance; no caption-quality claim.')
        if metadata_path.exists():raise RuntimeError('Existing staging metadata preserved; choose a fresh DESTINATION.')
        try:
            metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
            run_cli([sys.executable,'-u','-c',CREATE_DATASET_SCRIPT,str(DESTINATION)])
        finally:
            metadata_path.unlink(missing_ok=True)
        print('API accepted creation; processing may still be pending. Do not rerun create repeatedly.',flush=True)
        print(f'https://www.kaggle.com/datasets/{OWNER}/{SLUG}',flush=True)
        try:run_cli([sys.executable,'-m','kaggle','datasets','status',f'{OWNER}/{SLUG}'])
        except RuntimeError:print('Creation was accepted, but status could not be confirmed. Check the dataset page before retrying.',flush=True)
        ''')))
    cells.append(nb.md('''
        Khi trang dataset xử lý xong, kiểm tra có30HDF5 và `cache_spec.json`, `manifest.json`, `cache_receipt.json`.
        Cache được lưu **ngay gốc dataset**, không có thêm folder `native_detail_HL_15k` ở trong.
        Gắn dataset vào notebook tiếp theo rồi gửi path thật chứa `cache_spec.json`, ví dụ:
        `/kaggle/input/datasets/ducanh2403/native-detail-hl-15k`.
        API chấp nhận upload chưa đồng nghĩa dataset đã sẵn sàng; chưa chạy huấn luyện ở notebook này.
        '''))
    # Resolve credentials/dependencies before the expensive verified6GB copy.
    cells[2],cells[3]=cells[3],cells[2]
    setup=''.join(cells[2]['source'])
    setup+='\nVERSION_PREFLIGHT='+repr(VERSION_PREFLIGHT)+'\n'
    setup+="run_cli([sys.executable,'-u','-c',VERSION_PREFLIGHT])\n"
    cells[2]=nb.code(setup)
    download=nb.code('''
        import tempfile,shutil
        if KERNEL_VERSION!='ducanh2403/2-vi-cai-kia-qua-nhieu/89':raise ValueError('This notebook pins version89; do not silently download another run.')
        total=sum(v['bytes'] for v in INVENTORY['files'].values())
        if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<2*total+512*1024**2:
            raise RuntimeError('Need about12.6GB free for download plus verified upload staging; preserve existing files.')
        DOWNLOAD=Path(tempfile.mkdtemp(prefix='native-detail-v89-',dir='/tmp'))
        pattern=r'(^|/)(?:'+'|'.join(re.escape(n) for n in INVENTORY['files'])+r')$'
        print('Downloading audited cache files from:',KERNEL_VERSION,flush=True)
        run_cli([sys.executable,'-m','kaggle','kernels','output',KERNEL_VERSION,'-p',str(DOWNLOAD),'--file-pattern',pattern])
        fingerprint=INVENTORY['files']['cache_receipt.json']['sha256']
        matches=[p.parent for p in DOWNLOAD.rglob('cache_receipt.json') if sha(p)==fingerprint]
        if len(matches)!=1:
            raise RuntimeError(f'Version89 returned {len(matches)} matching cache receipts; STOP. No latest/version fallback is allowed.')
        SOURCE=locate(INVENTORY,matches[0])
        print('Version89 source fingerprint PASS; next verify every payload hash.',flush=True)
        ''')
    package_cell=nb.code('''
        report=package(SOURCE,DESTINATION,INVENTORY)
        report['source_kernel_version']=KERNEL_VERSION
        (DESTINATION/'transport_receipt.json').write_text(json.dumps(report,indent=2)+'\\n',encoding='utf-8')
        print('Verified version89 cache ready for private API upload.',flush=True)
        ''')
    cells[4:4]=[download,package_cell]
    nb.write('SAVE_NATIVE_DETAIL_HL_15K_DATASET.ipynb',cells)
    target=HERE/'SAVE_NATIVE_DETAIL_HL_15K_DATASET.ipynb'
    notebook=json.loads(target.read_text(encoding='utf-8'));notebook['metadata']['kaggle']=dict(isInternetEnabled=True,accelerator='none')
    target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    return target


if __name__=='__main__':print(build())
