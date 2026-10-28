"""Package the established private upload workflow for exact Output105; never call API."""
import hashlib
import importlib.util
import json
from pathlib import Path
import textwrap
from make_prepare_notebook import cell

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/105'


def main():
    result=json.loads((HERE/'CACHE_RESULT.json').read_text(encoding='utf-8'))
    package_code=(HERE/'package_frames_dataset.py').read_text(encoding='utf-8')
    inventory=dict(schema='relation_frames_v2_transport_v1',kernel_version=VERSION,
        source_review_zip_sha256=result['zip_sha256'],
        files=dict(result['declared_payload_sha256'],**{'receipt.json':result['receipt_sha256']}))
    # Reuse the successful token-redaction and pinned CLI version-routing setup.
    old=json.loads((ROOT/'experiments/semantic_grounding_method/SAVE_GROUNDED_GAIN_V102_DATASET.ipynb').read_text(encoding='utf-8'))
    setup=next(''.join(c['source']) if isinstance(c['source'],list) else c['source'] for c in old['cells']
        if c['cell_type']=='code' and 'def run_cli(' in ''.join(c['source']))
    setup=setup.replace('/102','/105').replace('v102','v105').replace('pins102','pins105')
    spec=importlib.util.spec_from_file_location('old_dataset_constants',ROOT/'experiments/upt_hidden_pair/make_dataset_api_notebook.py')
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    cells=[cell('markdown','''
        # Lưu cache relation frames v2 — Output105 thành Dataset
        **CPU / Accelerator=None / Internet ON / Save & Run All. Không GPU, không chạy lại chuẩn bị.**
        Tải đúng version105 của `ducanh2403/2-vi-cai-kia-qua-nhieu`, kiểm hash cả5file gốc,
        gồm `inputs.h5`, rồi tạo Dataset Private mới. Không tải latest hoặc tự dùng version khác.
        Không cần Add Input hoặc checkpoint. Cần khoảng2GiB trống cho bản tải và bản sao.
        Thêm/bật Kaggle Secret `KAGGLE_API_TOKEN`. Nếu secret của bạn tên `codex`, sửa SECRET_NAME.
        Tên secret khác tên gợi nhớ của token trên trang Settings. Không dán token vào code hoặc chat.
        OWNER phải khớp tài khoản có quyền xem output105 và tạo Dataset.
        '''),cell('code','''
        from pathlib import Path
        KERNEL_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/105'
        OWNER='ducanh2403'
        SLUG='relation-frames-v2-cache-v105'
        TITLE='Relation frames v2 cache v105 dataset'
        SECRET_NAME='KAGGLE_API_TOKEN'
        DESTINATION=Path('/kaggle/working/relation_frames_v2_dataset')
        '''),cell('code',setup)]
    cells.append(cell('code','import hashlib,json\nINVENTORY='+repr(inventory)+'\nSOURCE_CODE='+repr(package_code)+
        '\nassert hashlib.sha256(SOURCE_CODE.encode()).hexdigest()=='+repr(hashlib.sha256(package_code.encode()).hexdigest())+
        "\nexec(compile(SOURCE_CODE,'package_frames_dataset.py','exec'))\n"))
    cells.append(cell('code','''
        import tempfile,shutil
        if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<2*1024**3:
            raise RuntimeError('Need about2GiB free for download+copy. Preserve existing outputs.')
        DOWNLOAD=Path(tempfile.mkdtemp(prefix='relation-frames-v105-',dir='/tmp'))
        pattern=r'(^|/)(?:'+'|'.join(re.escape(n) for n in INVENTORY['files'])+r')$'
        print('Download EXACT source:',KERNEL_VERSION,flush=True)
        run_cli([sys.executable,'-m','kaggle','kernels','output',KERNEL_VERSION,'-p',str(DOWNLOAD),'--file-pattern',pattern])
        matches=[p.parent for p in DOWNLOAD.rglob('receipt.json') if sha(p)==INVENTORY['files']['receipt.json']]
        if len(matches)!=1:raise RuntimeError(f'Output105 returned {len(matches)} matching receipts. STOP; no latest/version fallback.')
        report=package(matches[0],DESTINATION,INVENTORY)
        print('All5original files and full HDF5 verified; dataset staging ready:',DESTINATION,flush=True)
        '''))
    cells.append(cell('code','CREATE_DATASET_SCRIPT='+repr(api.CREATE_DATASET_SCRIPT)+'\n'+textwrap.dedent('''
        metadata_path=DESTINATION/'dataset-metadata.json'
        metadata=dict(title=TITLE,id=f'{OWNER}/{SLUG}',licenses=[dict(name='other')],
            description='Private research cache from exact notebook Output105. Original COCO/CLIP terms retained. Copies existing visual and original prompt embeddings for612images; paired VSR metadata and predicted boxes. No new feature extraction, trained checkpoint or model-quality claim. Source and transport receipts preserved.')
        if metadata_path.exists():raise RuntimeError('Existing metadata preserved; use fresh staging destination.')
        try:
            metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
            run_cli([sys.executable,'-u','-c',CREATE_DATASET_SCRIPT,str(DESTINATION)])
        finally:metadata_path.unlink(missing_ok=True)
        print('API accepted creation; processing may still be pending. Do not repeatedly create.',flush=True)
        print(f'https://www.kaggle.com/datasets/{OWNER}/{SLUG}',flush=True)
        try:run_cli([sys.executable,'-m','kaggle','datasets','status',f'{OWNER}/{SLUG}'])
        except RuntimeError:print('Status not confirmed. Check dataset page before retrying.',flush=True)
        try:run_cli([sys.executable,'-m','kaggle','datasets','files',f'{OWNER}/{SLUG}','--page-size','100'])
        except RuntimeError:print('File listing unavailable yet; wait for processing, do not rerun create.',flush=True)
        ''')))
    cells.append(cell('markdown','''
        Sau khi xử lý xong, Dataset có5file gốc ở ngay thư mục gốc:
        `inputs.h5`, `relation_frames_manifest.bin`, `images.json`, `detector_boxes.json`, `receipt.json`,
        cộng `transport_receipt.json`. Không có folderImage_Captioning hoặc hàng trăm file source.
        `.bin` là metadata nén nguyên byte, không phải checkpoint; không cần tự giải nén.
        Nếu dùng SLUG mặc định, path dự kiến:
        `/kaggle/input/datasets/ducanh2403/relation-frames-v2-cache-v105`.
        Gửi path mount thực tế khi attach dataset. Chia sẻ Private với Ngân nếu dùng tài khoản thứ hai.
        Không ghi đè Dataset đã có; nếu slug tồn tại, kiểm tra trang trước khi tạo lại.
        Cờ audit=false trong receipt gốc là trạng thái tại lúc copy; review32mẫu đã hoàn tất được lưu riêng.
        Notebook này chưa được Codex gọi API hoặc chạy trên Kaggle.
        '''))
    notebook=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python3',language='python',name='python3'),
        language_info=dict(name='python'),kaggle=dict(isInternetEnabled=True,accelerator='none',isGpuEnabled=False)),cells=cells)
    target=HERE/'SAVE_RELATION_FRAMES_V105_DATASET.ipynb'
    target.write_text(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for c in cells:
        if c['cell_type']=='code':compile(c['source'],str(target),'exec')
    if target.stat().st_size>=1000000:raise ValueError('Notebook too large')
    (HERE/'SAVE_V105_PACKAGE.json').write_text(json.dumps(dict(inventory=inventory,notebook_bytes=target.stat().st_size,
        notebook_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),static_compile=True,
        kaggle_api_executed=False,local_neural_execution=False),indent=2)+'\n',encoding='utf-8')
    print(target.name,target.stat().st_size,'bytes, compiled statically; NO API EXECUTION')


if __name__=='__main__':main()
