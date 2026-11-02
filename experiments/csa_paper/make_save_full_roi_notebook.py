"""Author exact134 CPU Dataset transport; no Kaggle calls or models."""
import hashlib
import importlib.util
import json
from make_smoke_notebook import HERE,ROOT,cell

def make():
    cache=HERE/'full_roi_review_text';manifest=json.loads((cache/'manifest.json').read_text(encoding='utf-8'))
    receipt=json.loads((cache/'receipt.json').read_text(encoding='utf-8'));files={}
    for entry in manifest['shards']:files.update(entry['files'])
    files['vocabulary.json']=receipt['vocabulary_sha256']
    for p in sorted(cache.iterdir()):files[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    assert len(files)==247
    inventory=dict(kernel_version='ducanh2403/2-vi-cai-kia-qua-nhieu/134',files=files,
        source_review_zip_sha256=json.loads((HERE/'FULL_ROI_RESULT.json').read_text(encoding='utf-8'))['zip_sha256'])
    old=json.loads((HERE/'SAVE_CSA_BANK_V130_DATASET.ipynb').read_text(encoding='utf-8'))
    setup=next(c['source'] for c in old['cells'] if c['cell_type']=='code' and 'def run_cli(' in c['source'])
    setup=setup.replace('/130','/134').replace('v130','v134').replace('pins130','pins134')
    spec=importlib.util.spec_from_file_location('dataset_api_constants',ROOT/'experiments/upt_hidden_pair/make_dataset_api_notebook.py')
    api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    code=(HERE/'package_full_roi_dataset.py').read_text(encoding='utf-8')
    cells=[cell('markdown','''# Save full CSA ROI — exact Output134
Accelerator NONE, Internet ON; no Add Input. Enable Secret `KAGGLE_API_TOKEN`; if named `codex`, edit SECRET_NAME below.
Downloads~1.75GB from exact134, checks all247originalfileSHAs and per-image metadata, creates NEW Private Dataset. No pooling/model/training. Keep sourceOutput134 until processing finishes.
'''),cell('code','''
from pathlib import Path
KERNEL_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/134'
OWNER='ducanh2403'
SLUG='csa-full-roi-karpathy-v134'
TITLE='CSA full ROI Karpathy v134 dataset'
SECRET_NAME='KAGGLE_API_TOKEN'
DESTINATION=Path('/kaggle/working/csa_full_roi_v134_dataset')
'''),cell('code',setup),cell('code','import hashlib,json\nINVENTORY='+repr(inventory)+'\nSOURCE_CODE='+repr(code)+
        '\nassert hashlib.sha256(SOURCE_CODE.encode()).hexdigest()=='+repr(hashlib.sha256(code.encode()).hexdigest())+
        "\nexec(compile(SOURCE_CODE,'package_full_roi_dataset.py','exec'))"),cell('code','''
import tempfile,shutil
if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<8*1024**3:
    raise RuntimeError('Need8GiB free for download/copy; preserve existing files.')
DOWNLOAD=Path(tempfile.mkdtemp(prefix='csa-full-roi-v134-',dir='/tmp'))
pattern=r'(^|/)(?:'+'|'.join(re.escape(n) for n in INVENTORY['files'])+r')$'
print('Downloading EXACT Output:',KERNEL_VERSION,flush=True)
run_cli([sys.executable,'-m','kaggle','kernels','output',KERNEL_VERSION,'-p',str(DOWNLOAD),'--file-pattern',pattern])
matches=[p.parent for p in DOWNLOAD.rglob('receipt.json') if sha(p)==INVENTORY['files']['receipt.json']]
if len(matches)!=1:raise RuntimeError(f'Output134 returned {len(matches)} matching fullROI receipts. STOP; no latest fallback.')
report=package(matches[0],DESTINATION,INVENTORY)
print('247fileSHA + actual per-image metadata verified:',report['actual_per_image_metadata_verified'],flush=True)
'''),cell('code','CREATE_DATASET_SCRIPT='+repr(api.CREATE_DATASET_SCRIPT)+'\n'+'''
metadata_path=DESTINATION/'dataset-metadata.json'
metadata=dict(title=TITLE,id=f'{OWNER}/{SLUG}',licenses=[dict(name='other')],
    description='Private research cache from exactOutput134: rawCLIP768D ROIfeatures for113287KarpathyTRAIN+5000val,833473regions,originalvocab/references/provenance. No test/pixelimages/trainedcaptioncheckpoint. Upstreamtermsretained;semanticbanks remainseparate csa-semantic-banks-v130.')
if metadata_path.exists():raise RuntimeError('Preserve existing metadata; choose a fresh destination.')
try:
    metadata_path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    run_cli([sys.executable,'-u','-c',CREATE_DATASET_SCRIPT,str(DESTINATION)])
finally:metadata_path.unlink(missing_ok=True)
print('Creation accepted; Dataset may still be processing. Do not repeatedly create.',flush=True)
print(f'https://www.kaggle.com/datasets/{OWNER}/{SLUG}',flush=True)
try:run_cli([sys.executable,'-m','kaggle','datasets','status',f'{OWNER}/{SLUG}'])
except RuntimeError:print('Status unconfirmed; check Dataset page before retrying.',flush=True)
'''),cell('markdown','''Send the actual mount after Dataset processing completes. Expected:
`/kaggle/input/datasets/ducanh2403/csa-full-roi-karpathy-v134`.
Keep `csa-semantic-banks-v130` too. This saves cache only; full training is not started.
''')]
    nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=False)))
    path=HERE/'SAVE_CSA_FULL_ROI_V134_DATASET.ipynb';path.write_bytes((json.dumps(nb,indent=1,ensure_ascii=False)+'\n').encode('utf-8'))
    (HERE/'SAVE_FULL_ROI_V134_PACKAGE.json').write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8');print(path,path.stat().st_size)

if __name__=='__main__':make()
