"""Author exact-v126 CPU Dataset transport; never call Kaggle or run models."""
import hashlib
import importlib.util
import json
from pathlib import Path

from make_smoke_notebook import HERE, ROOT, cell


def make():
    cache = HERE/'real_preparation_review_text'
    receipt = json.loads((cache/'receipt.json').read_bytes())
    files = dict(receipt['payload_sha256'])
    for name in ('receipt.json', 'summary.json', 'source_receipt.json', 'software_freeze.txt'):
        files[name] = hashlib.sha256((cache/name).read_bytes()).hexdigest()
    inventory = dict(kernel_version='ducanh2403/2-vi-cai-kia-qua-nhieu/126',
        source_review_zip_sha256=json.loads((HERE/'REAL_PREPARATION_RESULT.json').read_text())['zip_sha256'], files=files)
    old = json.loads((ROOT/'experiments/semantic_grounding_method/SAVE_GROUNDED_GAIN_V102_DATASET.ipynb').read_bytes())
    setup = next(''.join(c['source']) if isinstance(c['source'], list) else c['source']
                 for c in old['cells'] if c['cell_type']=='code' and 'def run_cli(' in ''.join(c['source']))
    setup = setup.replace('/102', '/126').replace('v102', 'v126').replace('pins102', 'pins126')
    spec = importlib.util.spec_from_file_location('dataset_api_constants', ROOT/'experiments/upt_hidden_pair/make_dataset_api_notebook.py')
    api = importlib.util.module_from_spec(spec); spec.loader.exec_module(api)
    code = (HERE/'package_real_dataset.py').read_text(encoding='utf-8')
    cells = [cell('markdown', '''
        # Lưu CSA real preparation — đúng Output126 thành Dataset

        **Accelerator None, Internet ON**, không cần Add Input. Save & Run All.
        Bật Kaggle Secret KAGGLE_API_TOKEN; nếu secret tên codex, sửa SECRET_NAME ở cell đầu.
        Không dán token vào code/chat. OWNER phải có quyền xem source126 và tạo Dataset.
        Chỉ tạo Dataset **Private mới**, không ghi đè dataset đã tồn tại.

        Tải đúng version126, kiểm SHA đầy đủ8file, bao gồm regions.npz; không lấy latest,
        không chạy lại POS/ROI hay mô hình. Giữ nguyên word lists như được tạo:
        mục rid18575 đang cần kiểm tra từ gốc trước bước CLIP text, chưa tự sửa/chuẩn y.
        Giữ Output126 đến khi Dataset xử lý xong.
        '''), cell('code', '''
        from pathlib import Path
        KERNEL_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/126'
        OWNER='ducanh2403'
        SLUG='csa-paper-real-preparation-v126'
        TITLE='CSA real preparation v126 dataset'
        SECRET_NAME='KAGGLE_API_TOKEN'
        DESTINATION=Path('/kaggle/working/csa_real_preparation_dataset')
        '''), cell('code', setup), cell('code', 'import hashlib,json\nINVENTORY='+repr(inventory)+'\nSOURCE_CODE='+repr(code)+
        '\nassert hashlib.sha256(SOURCE_CODE.encode()).hexdigest()=='+repr(hashlib.sha256(code.encode()).hexdigest())+
        "\nexec(compile(SOURCE_CODE,'package_real_dataset.py','exec'))\n"), cell('code', '''
        import tempfile,shutil
        if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<512*1024**2:
            raise RuntimeError('Need512MiB free for staged download/copy; preserve existing files.')
        DOWNLOAD=Path(tempfile.mkdtemp(prefix='csa-real-v126-',dir='/tmp'))
        pattern=r'(^|/)(?:'+'|'.join(re.escape(n) for n in INVENTORY['files'])+r')$'
        print('Downloading EXACT Output:',KERNEL_VERSION,flush=True)
        run_cli([sys.executable,'-m','kaggle','kernels','output',KERNEL_VERSION,'-p',str(DOWNLOAD),'--file-pattern',pattern])
        matches=[p.parent for p in DOWNLOAD.rglob('receipt.json') if sha(p)==INVENTORY['files']['receipt.json']]
        if len(matches)!=1:raise RuntimeError(f'Output126 returned {len(matches)} matching receipts. STOP, no latest fallback.')
        report=package(matches[0],DESTINATION,INVENTORY)
        print('Eight unchanged files verified/copied. Lexical warning retained:',report['lexical_warning'],flush=True)
        '''), cell('code', 'CREATE_DATASET_SCRIPT='+repr(api.CREATE_DATASET_SCRIPT)+'\n'+'''
metadata_path=DESTINATION/'dataset-metadata.json'
metadata=dict(title=TITLE,id=f'{OWNER}/{SLUG}',licenses=[dict(name='other')],
    description='Private research assets preserved from exact Output126: TRAIN noun/verb POS word lists and320-image pooled raw CLIP regions. Upstream model/data terms retained. Source word list linguistic review pending (rid vs ride); not CLIP text embeddings or a trained caption model. No image pixels included. Original receipt/provenance preserved.')
if metadata_path.exists():raise RuntimeError('Preserve existing metadata; choose a fresh destination.')
try:
    metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    run_cli([sys.executable,'-u','-c',CREATE_DATASET_SCRIPT,str(DESTINATION)])
finally:metadata_path.unlink(missing_ok=True)
print('API accepted creation; processing may still be pending. Do not repeatedly create.',flush=True)
print(f'https://www.kaggle.com/datasets/{OWNER}/{SLUG}',flush=True)
try:run_cli([sys.executable,'-m','kaggle','datasets','status',f'{OWNER}/{SLUG}'])
except RuntimeError:print('Status not confirmed; check Dataset page before retrying.',flush=True)
'''), cell('markdown', '''
        Gửi path thực tế sau khi attach Dataset. Mount dự kiến:
        `/kaggle/input/datasets/ducanh2403/csa-paper-real-preparation-v126`.
        Đây chỉ là lưu dữ liệu, không mã hóa CLIP text hoặc sửa bank từ.
        Codex chỉ kiểm metadata/static/mock, chưa gọi API hoặc chạy notebook này.
        ''')]
    notebook = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        kaggle=dict(isInternetEnabled=True, isGpuEnabled=False)))
    target = HERE/'SAVE_CSA_REAL_PREPARATION_V126_DATASET.ipynb'
    target.write_bytes((json.dumps(notebook, ensure_ascii=False, indent=1)+'\n').encode('utf-8'))
    assert target.stat().st_size < 1_000_000
    (HERE/'SAVE_V126_PACKAGE.json').write_text(json.dumps(dict(inventory=inventory,
        notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), notebook_bytes=target.stat().st_size,
        kaggle_api_executed=False, model_executed=False), indent=2)+'\n', encoding='utf-8')
    print(target.name, target.stat().st_size, 'bytes; static compiled; no API/model execution')


if __name__ == '__main__':
    make()
