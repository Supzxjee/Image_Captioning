"""Author CPU-only exact169+161 Dataset transport; never execute its cells."""
import hashlib
import importlib.util
import json
from pathlib import Path
from make_smoke_notebook import cell

HERE=Path(__file__).resolve().parent


def make():
    result=json.loads((HERE/'PROMPT_TEXT_CACHE_RESULT.json').read_text(encoding='utf-8'))
    files=dict(result['payload_sha256'])
    for name in ('cache_receipt.json','run_status.json','source_receipt.json','notebook.log'):
        files[name]=hashlib.sha256((HERE/'prompt_text_review_text'/name).read_bytes()).hexdigest()
    inventory=dict(cache_version='ducanh2403/2-vi-cai-kia-qua-nhieu/169',
        checkpoint_version='ducanh2403/2-vi-cai-kia-qua-nhieu/161',cache_files=files,
        checkpoint_sha256='58ce5dc0e30ac26add9bac8e24a2e9e03b91532bfdeece588997395bf9cfc16b')
    spec=importlib.util.spec_from_file_location('dataset_api_constants',HERE.parents[1]/'experiments/upt_hidden_pair/make_dataset_api_notebook.py')
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    source=(HERE/'package_prompt_inputs.py').read_text(encoding='utf-8')
    config='''
from pathlib import Path
OWNER='ducanh2403'
SLUG='csa-prompt-pilot-inputs-v169'
TITLE='CSA prompt pilot assets v169 and v161'
SECRET_NAME='codex' # Tên Secret đã bật, token thuộc tài khoản ducanh2403.
DESTINATION=Path('/kaggle/working/csa_prompt_pilot_inputs_v169_dataset')
'''
    setup=r'''
import json,os,re,subprocess,sys,tempfile,shutil
from kaggle_secrets import UserSecretsClient
if not re.fullmatch(r'[a-zA-Z0-9_-]+',OWNER) or not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',SLUG):raise ValueError('Invalid OWNER/SLUG')
if not 6<=len(SLUG)<=50 or not 6<=len(TITLE)<=50:raise ValueError('Title/slug must have6–50characters')
if DESTINATION.exists():raise RuntimeError('Preserve existing output; choose fresh DESTINATION')
if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<4*1024**3:raise RuntimeError('Need4GiB free for download/copy; no files removed')
try:token=UserSecretsClient().get_secret(SECRET_NAME).strip()
except Exception:raise RuntimeError('Enable the named Secret for the owner of private outputs169/161') from None
if not token:raise RuntimeError('Empty API token')
def run_cli(args,authenticated=True):
    environment=os.environ.copy()
    for k in ('KAGGLE_API_TOKEN','KAGGLE_USERNAME','KAGGLE_KEY'):environment.pop(k,None)
    if authenticated:environment['KAGGLE_API_TOKEN']=token
    environment['PYTHONUNBUFFERED']='1'
    process=subprocess.Popen(list(map(str,args)),env=environment,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace')
    for line in process.stdout:print(line.replace(token,'[REDACTED]'),end='',flush=True)
    if process.wait():raise RuntimeError('API command failed; preserve staged files and read redacted log above')
CLI=[sys.executable,'-c','from kaggle.cli import main; main()']
run_cli([sys.executable,'-m','pip','install','-q','kaggle==2.2.5'],authenticated=False)
'''
    preflight=r'''
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from kaggle.api.kaggle_api_extended import KaggleApi
api=KaggleApi()
for version in ('169','161'):
    with tempfile.TemporaryDirectory() as directory,patch.object(api,'build_kaggle_client') as factory:
        endpoint=factory.return_value.__enter__.return_value.kernels.kernels_api_client.list_kernel_session_output
        endpoint.return_value=SimpleNamespace(files=[],next_page_token=None,log=None)
        api.kernels_output('ducanh2403/2-vi-cai-kia-qua-nhieu/'+version,path=directory)
        request=endpoint.call_args.args[0]
        assert request.version_label=='v'+version and request.user_name=='ducanh2403' and request.kernel_slug=='2-vi-cai-kia-qua-nhieu','Exact version ignored'
print('Exact169/161 API routing preflight PASS; no network in this check.',flush=True)
'''
    transport=r'''
DOWNLOAD=Path(tempfile.mkdtemp(prefix='csa-prompt-inputs-',dir='/tmp'))
CACHE_DOWNLOAD=DOWNLOAD/'cache169';CACHE_DOWNLOAD.mkdir()
WEIGHT_DOWNLOAD=DOWNLOAD/'weight161';WEIGHT_DOWNLOAD.mkdir()
run_cli(CLI+['kernels','output',INVENTORY['cache_version'],'-p',str(CACHE_DOWNLOAD),'--file-pattern',pattern(INVENTORY['cache_files'])])
run_cli(CLI+['kernels','output',INVENTORY['checkpoint_version'],'-p',str(WEIGHT_DOWNLOAD),'--file-pattern',pattern(['best.bin'])])
report=package(CACHE_DOWNLOAD,WEIGHT_DOWNLOAD,DESTINATION,INVENTORY)
print('Verified/copy unchanged:',len(report['files_sha256']),'files. No model loaded.',flush=True)
'''
    upload=r'''
metadata_path=DESTINATION/'dataset-metadata.json'
if metadata_path.exists():raise RuntimeError('Preserve existing metadata; use fresh destination')
metadata=dict(title=TITLE,id=f'{OWNER}/{SLUG}',licenses=[dict(name='other')],
    description='Private research assets: original frozen CLIP prompt text cache from exact notebook169, mapping TRAIN10k and VALID5k only; original CSA SCST+CE best checkpoint from exact161. No TEST or images. Upstream data/model terms retained. No model executed during transport.')
try:
    metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    run_cli([sys.executable,'-u','-c',CREATE_DATASET_SCRIPT,str(DESTINATION)])
finally:metadata_path.unlink(missing_ok=True)
print('API accepted creation; processing may still be active. Check page before retrying.',flush=True)
print(f'https://www.kaggle.com/datasets/{OWNER}/{SLUG}',flush=True)
print('After processing, grant Ngan access in Dataset sharing; do not give anyone your API token.',flush=True)
'''
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        kaggle=dict(isInternetEnabled=True,isGpuEnabled=False)),cells=[
        cell('markdown','# Lưu đầu vào hai pilot prompt — exact169 +161\n**Accelerator None, Internet ON, không cần Add Input.** Chạy bằng tài khoản sở hữu hai Output; bật Secret `codex` (hoặc đổi SECRET_NAME). Tải cache169 và chỉ best.bin161, kiểm SHA mọi file, giữ nguyên binary; tạo **Dataset Private mới**. Không chạy CLIP/Torch/train. Chia sẻ Dataset cho Ngân sau khi xử lý xong để hai notebook gắn cùng input; không dùng token của bạn trong notebook của Ngân. Không ghi đè dataset/output đã tồn tại.'),
        cell('code',config),cell('code',setup),
        cell('code','VERSION_PREFLIGHT='+repr(preflight)+"\nrun_cli([sys.executable,'-u','-c',VERSION_PREFLIGHT])\n"),
        cell('code','import hashlib\nINVENTORY='+repr(inventory)+'\nSOURCE_CODE='+repr(source)+
             '\nassert hashlib.sha256(SOURCE_CODE.encode()).hexdigest()=='+repr(hashlib.sha256(source.encode()).hexdigest())+
             "\nexec(compile(SOURCE_CODE,'package_prompt_inputs.py','exec'))\n"),
        cell('code',transport),cell('code','CREATE_DATASET_SCRIPT='+repr(api.CREATE_DATASET_SCRIPT)+'\n'+upload),
        cell('markdown','Sau khi Dataset xử lý xong, gửi path mount thực tế. Dự kiến `/kaggle/input/datasets/ducanh2403/csa-prompt-pilot-inputs-v169`. Dataset này gồm cache + một trọng số; **ROI134 và dataset COCO vẫn gắn riêng** khi train. Đây chỉ là bảo toàn đầu vào, không phải kết quả hai pilot.')])
    path=HERE/'SAVE_CSA_PROMPT_INPUTS_V169_DATASET.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(path.name,path.stat().st_size,'bytes; unexecuted CPU transport')


if __name__=='__main__':make()
