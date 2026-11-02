"""Build exact-version CPU transport notebook; does not call Kaggle."""
import hashlib
import importlib.util
import json
from make_smoke_notebook import HERE, ROOT, cell

def make():
    reviewed=HERE/'real_bank_smoke_review_text'
    receipt=json.loads((reviewed/'cache_receipt.json').read_text(encoding='utf-8'))
    files=dict(receipt['payload_sha256'])
    for name in ('cache_receipt.json','smoke_summary.json','notebook.log','source_receipt.json'):
        files[name]=hashlib.sha256((reviewed/name).read_bytes()).hexdigest()
    inventory=dict(kernel_version='ducanh2403/2-vi-cai-kia-qua-nhieu/130',files=files,
        source_review_zip_sha256=json.loads((HERE/'REAL_BANK_SMOKE_RESULT.json').read_text(encoding='utf-8'))['zip_sha256'])
    old=json.loads((HERE/'SAVE_CSA_REAL_PREPARATION_V126_DATASET.ipynb').read_text(encoding='utf-8'))
    setup=next(c['source'] for c in old['cells'] if c['cell_type']=='code' and 'def run_cli(' in c['source'])
    setup=setup.replace('/126','/130').replace('v126','v130').replace('pins126','pins130')
    spec=importlib.util.spec_from_file_location('dataset_api_constants',ROOT/'experiments/upt_hidden_pair/make_dataset_api_notebook.py')
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    code=(HERE/'package_bank_dataset.py').read_text(encoding='utf-8')
    cells=[cell('markdown','''# Save frozen CSA semantic banks — exact Output130
Accelerator **None**, Internet **ON**. No Add Input. Enable Secret `KAGGLE_API_TOKEN`; if named `codex`, change SECRET_NAME below. Never paste the token in code/chat.
This only downloads and saves the completed cache. No CLIP/CSA rerun. New Private Dataset; preserve Output130 until processing finishes.
'''),cell('code', '''
from pathlib import Path
KERNEL_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/130'
OWNER='ducanh2403'
SLUG='csa-semantic-banks-v130'
TITLE='CSA semantic banks v130 dataset'
SECRET_NAME='KAGGLE_API_TOKEN'
DESTINATION=Path('/kaggle/working/csa_semantic_banks_v130_dataset')
'''),cell('code',setup),cell('code','import hashlib,json\nINVENTORY='+repr(inventory)+'\nSOURCE_CODE='+repr(code)+
        '\nassert hashlib.sha256(SOURCE_CODE.encode()).hexdigest()=='+repr(hashlib.sha256(code.encode()).hexdigest())+
        "\nexec(compile(SOURCE_CODE,'package_bank_dataset.py','exec'))"),cell('code','''
import tempfile,shutil
if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<256*1024**2:
    raise RuntimeError('Need256MiB free; preserve existing files.')
DOWNLOAD=Path(tempfile.mkdtemp(prefix='csa-bank-v130-',dir='/tmp'))
pattern=r'(^|/)(?:'+'|'.join(re.escape(n) for n in INVENTORY['files'])+r')$'
print('Downloading EXACT Output:',KERNEL_VERSION,flush=True)
run_cli([sys.executable,'-m','kaggle','kernels','output',KERNEL_VERSION,'-p',str(DOWNLOAD),'--file-pattern',pattern])
matches=[p.parent for p in DOWNLOAD.rglob('cache_receipt.json') if sha(p)==INVENTORY['files']['cache_receipt.json']]
if len(matches)!=1:raise RuntimeError(f'Output130 returned {len(matches)} matching receipts. STOP; no latest fallback.')
report=package(matches[0],DESTINATION,INVENTORY)
print('All six original payloads verified/copied unchanged.',flush=True)
'''),cell('code','CREATE_DATASET_SCRIPT='+repr(api.CREATE_DATASET_SCRIPT)+'\n'+'''
metadata_path=DESTINATION/'dataset-metadata.json'
metadata=dict(title=TITLE,id=f'{OWNER}/{SLUG}',licenses=[dict(name='other')],
    description='Private research cache from exact notebook Output130: frozen CLIP512D vectors for5269nouns/1594verbs and provenance. Upstream model/data terms retained. No trained caption checkpoint. Original ROI/vocabulary remain in csa-real-v126-lemma-fixed.')
if metadata_path.exists():raise RuntimeError('Preserve existing metadata; choose a fresh destination.')
try:
    metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    run_cli([sys.executable,'-u','-c',CREATE_DATASET_SCRIPT,str(DESTINATION)])
finally:metadata_path.unlink(missing_ok=True)
print('Creation accepted; Dataset may still be processing. Do not repeatedly create.',flush=True)
print(f'https://www.kaggle.com/datasets/{OWNER}/{SLUG}',flush=True)
try:run_cli([sys.executable,'-m','kaggle','datasets','status',f'{OWNER}/{SLUG}'])
except RuntimeError:print('Status unconfirmed; check Dataset page before retrying.',flush=True)
'''),cell('markdown','''Send the actual mount after adding the completed Dataset.
Expected `/kaggle/input/datasets/ducanh2403/csa-semantic-banks-v130`.
Keep the earlier `csa-real-v126-lemma-fixed` Dataset too: it contains ROI features/vocabulary needed for training.
''')]
    nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=False)))
    path=HERE/'SAVE_CSA_BANK_V130_DATASET.ipynb';path.write_bytes((json.dumps(nb,indent=1,ensure_ascii=False)+'\n').encode('utf-8'))
    (HERE/'SAVE_BANK_V130_PACKAGE.json').write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8')
    print(path,path.stat().st_size)

if __name__=='__main__':make()
