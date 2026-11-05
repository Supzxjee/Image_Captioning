"""Author exact175 CPU transport using the established token/upload workflow."""
import ast
import hashlib
import json
from pathlib import Path
from make_smoke_notebook import cell

HERE=Path(__file__).resolve().parent


def make():
    result=json.loads((HERE/'FULL_PROMPT_FRESH_V175_RESULT.json').read_text(encoding='utf-8'))
    inventory=result['transport_inventory'];old=json.loads((HERE/'SAVE_CSA_PROMPT_INPUTS_V169_DATASET.ipynb').read_text(encoding='utf-8'))
    code=[c['source'] for c in old['cells'] if c['cell_type']=='code']
    setup=next(c for c in code if 'def run_cli(' in c).replace('private outputs169/161','private output175')
    version=next(c for c in code if c.startswith('VERSION_PREFLIGHT='))
    preflight=ast.literal_eval(ast.parse(version).body[0].value).replace("('169','161')","('175',)").replace('Exact169/161','Exact175')
    upload=next(c for c in code if c.startswith('CREATE_DATASET_SCRIPT='))
    start=upload.index("    description=");end=upload.index('\ntry:',start)
    upload=upload[:start]+"    description='Private frozen CLIP prompt cache from exact output175: full Karpathy TRAIN113287 and VALID5000, 410 unique text vectors linked to ROI134. No TEST, images or trained caption weights. Upstream data/model terms retained.')"+upload[end:]
    source=(HERE/'package_prompt_inputs.py').read_text(encoding='utf-8')
    config="""from pathlib import Path
OWNER='ducanh2403'
SLUG='csa-prompt-full-karpathy-v175'
TITLE='CSA full Karpathy prompt cache v175'
SECRET_NAME='codex'
DESTINATION=Path('/kaggle/working/csa_prompt_full_karpathy_v175_dataset')
"""
    transport="""DOWNLOAD=Path(tempfile.mkdtemp(prefix='csa-prompt-full175-',dir='/tmp'))
run_cli(CLI+['kernels','output',INVENTORY['cache_version'],'-p',str(DOWNLOAD),'--file-pattern',pattern(INVENTORY['cache_files'])])
report=package_full_cache(DOWNLOAD,DESTINATION,INVENTORY)
print('Verified and copied original175 bytes:',len(report['files_sha256']),'files;410x512 vectors. No model loaded.',flush=True)
"""
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=False)),cells=[
        cell('markdown','# Lưu cache prompt full từ đúng Output175\n**Accelerator None, Internet ON; không cần Add Input.** Bật Secret `codex` của tài khoản sở hữu Output175. Tải đúng175 và kiểm14fileSHA cùng vectors410×512/dtype/norm/cohort trước khi tạo **Dataset Private mới**. Không chạy lại cache/CLIP hoặc load caption checkpoint. Không dùng latest; không xóa/ghi đè output. Sau khi Dataset xử lý xong hãy chia sẻ cho Ngân qua quyền Dataset, không chia sẻ token.'),
        cell('code',config),cell('code',setup),cell('code','VERSION_PREFLIGHT='+repr(preflight)+"\nrun_cli([sys.executable,'-u','-c',VERSION_PREFLIGHT])\n"),
        cell('code','import hashlib\nINVENTORY='+repr(inventory)+'\nSOURCE_CODE='+repr(source)+'\nassert hashlib.sha256(SOURCE_CODE.encode()).hexdigest()=='+repr(hashlib.sha256(source.encode()).hexdigest())+"\nexec(compile(SOURCE_CODE,'package_prompt_inputs.py','exec'))\n"),
        cell('code',transport),cell('code',upload),cell('markdown','Gửi path mount khi xử lý xong. Dự kiến `/kaggle/input/datasets/ducanh2403/csa-prompt-full-karpathy-v175`. Đây chỉ là cache prompt; ROI134 và semantic banks130 vẫn gắn riêng khi huấn luyện từ đầu. Không cần checkpoint161/CSA cũ.')])
    for c in nb['cells']:
        if c['cell_type']=='code':ast.parse(c['source'])
    path=HERE/'SAVE_CSA_PROMPT_FULL_V175_DATASET.ipynb';path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(path.name,path.stat().st_size,'bytes; unexecuted CPU transport')


if __name__=='__main__':make()
