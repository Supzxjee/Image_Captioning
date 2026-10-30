"""Build exact Output115 transport notebook. Never call Kaggle or execute cells."""
import hashlib
import importlib.util
import json
from pathlib import Path
from make_kv_prepare_notebook import cell

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def main():
    cached = json.loads((HERE/'review/kv_preparation_loaded.json').read_bytes())
    raw = cached['members']['cpu_preparation.json'].encode()
    receipt = json.loads(raw)
    inventory = dict(schema='relation_kv_transport_v1',
        kernel_version='ducanh2403/2-vi-cai-kia-qua-nhieu/115',
        source_review_zip_sha256=cached['zip_sha256'],
        files=dict(receipt['payload_sha256'], **{'cpu_preparation.json':hashlib.sha256(raw).hexdigest()}))
    old = json.loads((HERE/'SAVE_GROUNDED_GAIN_V102_DATASET.ipynb').read_bytes())
    setup = next(''.join(c['source']) if isinstance(c['source'],list) else c['source']
                 for c in old['cells'] if c['cell_type']=='code' and 'def run_cli(' in ''.join(c['source']))
    setup = setup.replace('/102','/115').replace('v102','v115').replace('pins102','pins115')
    spec = importlib.util.spec_from_file_location('saved_api_constants',ROOT/'experiments/upt_hidden_pair/make_dataset_api_notebook.py')
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    code = (HERE/'package_kv_dataset.py').read_text(encoding='utf-8')
    cells = [cell('markdown', '''
        # Lưu K/V preparation — đúng Output115 thành Dataset
        **Accelerator=None, Internet ON, Save & Run All. Không chạy lại chuẩn bị/model.**
        Tải đúng version115 của `ducanh2403/2-vi-cai-kia-qua-nhieu`, kiểm full SHA cả8file
        gồm inputs.h5, giữ nguyên byte rồi tạo Dataset Private mới. Không lấy latest.
        Không cần Add Input. Cần khoảng6GiB trống cho bản tải và bản sao.
        Bật Kaggle Secret KAGGLE_API_TOKEN; nếu secret tên codex, đổi SECRET_NAME ở cell đầu.
        Không dán token vào code/chat. OWNER phải là tài khoản có quyền xem source115 và tạo Dataset.
        Dùng giấy phép Other để giữ các điều khoản nguồn; không tự gán CC0 cho cache.
        '''), cell('code', '''
        from pathlib import Path
        KERNEL_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/115'
        OWNER='ducanh2403'
        SLUG='relation-kv-preparation-v115'
        TITLE='Relation KV preparation v115 dataset'
        SECRET_NAME='KAGGLE_API_TOKEN'
        DESTINATION=Path('/kaggle/working/relation_kv_dataset')
        '''), cell('code',setup), cell('code',
        'import hashlib,json\nINVENTORY='+repr(inventory)+'\nSOURCE_CODE='+repr(code)+
        '\nassert hashlib.sha256(SOURCE_CODE.encode()).hexdigest()=='+repr(hashlib.sha256(code.encode()).hexdigest())+
        "\nexec(compile(SOURCE_CODE,'package_kv_dataset.py','exec'))\n"),cell('code','''
        import tempfile,shutil
        if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<6*1024**3:
            raise RuntimeError('Need about6GiB free for download+copy; preserve existing outputs.')
        DOWNLOAD=Path(tempfile.mkdtemp(prefix='relation-kv-v115-',dir='/tmp'))
        pattern=r'(^|/)(?:'+'|'.join(re.escape(n) for n in INVENTORY['files'])+r')$'
        print('Download EXACT source:',KERNEL_VERSION,flush=True)
        run_cli([sys.executable,'-m','kaggle','kernels','output',KERNEL_VERSION,'-p',str(DOWNLOAD),'--file-pattern',pattern])
        matches=[p.parent for p in DOWNLOAD.rglob('cpu_preparation.json') if sha(p)==INVENTORY['files']['cpu_preparation.json']]
        if len(matches)!=1:raise RuntimeError(f'Output115 returned {len(matches)} matching receipts; STOP, no latest fallback.')
        report=package(matches[0],DESTINATION,INVENTORY)
        print('Eight original files verified and copied:',DESTINATION,flush=True)
        '''),cell('code','CREATE_DATASET_SCRIPT='+repr(api.CREATE_DATASET_SCRIPT)+'\n'+'''
metadata_path=DESTINATION/'dataset-metadata.json'
metadata=dict(title=TITLE,id=f'{OWNER}/{SLUG}',licenses=[dict(name='other')],
    description='Private research preparation from exact Output115. Existing COCO CLIP/legacy feature tensors, predicted boxes and VSR observed relation metadata. Upstream data/model terms retained. No original pixels, new encoder, trained model or caption-quality claim. Original source/transport receipts preserved.')
if metadata_path.exists():raise RuntimeError('Preserve existing metadata; use a fresh destination.')
try:
    metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    run_cli([sys.executable,'-u','-c',CREATE_DATASET_SCRIPT,str(DESTINATION)])
finally:metadata_path.unlink(missing_ok=True)
print('API accepted creation; processing may be pending. Do not repeatedly create.',flush=True)
print(f'https://www.kaggle.com/datasets/{OWNER}/{SLUG}',flush=True)
try:run_cli([sys.executable,'-m','kaggle','datasets','status',f'{OWNER}/{SLUG}'])
except RuntimeError:print('Status not confirmed; check dataset page before retrying.',flush=True)
'''),cell('markdown','''
        Dataset có8file gốc và transport_receipt.json ngay thư mục gốc; không chứa repo/log/sourcefiles.
        Nếu giữ mặc định, path mount dự kiến:
        `/kaggle/input/datasets/ducanh2403/relation-kv-preparation-v115`.
        Gửi path thực tế sau khi attach. Chia sẻ Private với Ngân để chạy nhánh thứ hai.
        Giữ Output115 đến khi Dataset xử lý xong. Không cần upload checkpoint mới.
        Notebook chỉ được kiểm metadata/static/mock; Codex chưa gọi API hoặc chạy trên Kaggle.
        ''')]
    notebook = dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=False)))
    target = HERE/'SAVE_RELATION_KV_V115_DATASET.ipynb'
    target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    for c in cells:
        if c['cell_type']=='code':compile(c['source'],target.name,'exec')
    if target.stat().st_size >= 1_000_000:raise ValueError('Notebook too large')
    (HERE/'KV_SAVE_V115_PACKAGE.json').write_text(json.dumps(dict(inventory=inventory,
        notebook_bytes=target.stat().st_size,notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        static_compile=True,kaggle_api_executed=False,model_executed=False),indent=2)+'\n',encoding='utf-8')
    print(target.name,target.stat().st_size,'bytes; compiled; NO API/MODEL EXECUTION')


if __name__ == '__main__':
    main()
