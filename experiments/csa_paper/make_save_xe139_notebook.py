"""Author CPU139 transport, using established API/version preflight. No API calls."""
import hashlib
import json
from make_smoke_notebook import HERE,cell

def make():
    cache=json.loads((HERE/'review/full_xe10_v2_loaded.json').read_text(encoding='utf-8'));f=cache['files']
    files={n:hashlib.sha256((json.dumps(f[n],indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()).hexdigest()
           for n in ('result.json','progress.json','input_identity.json','best_receipt.json','checkpoint_receipt.json')}
    files.update({'best.bin':f['best_receipt.json']['sha256'],'checkpoint.bin':f['checkpoint_receipt.json']['sha256'],
                  'csa_full_xe_10_v2_review.zip':cache['zip_sha256']})
    inventory=dict(kernel_version='ducanh2403/2-vi-cai-kia-qua-nhieu/139',files=files)
    old=json.loads((HERE/'SAVE_CSA_FULL_ROI_V134_DATASET.ipynb').read_text(encoding='utf-8'))
    setup=next(c['source'] for c in old['cells'] if c['cell_type']=='code' and 'def run_cli(' in c['source'])
    setup=setup.replace('/134','/139').replace('v134','v139').replace('pins134','pins139')
    upload=next(c['source'] for c in old['cells'] if c['cell_type']=='code' and 'CREATE_DATASET_SCRIPT=' in c['source'])
    upload=upload.replace("description='Private research cache from exactOutput134: rawCLIP768D ROIfeatures for113287KarpathyTRAIN+5000val,833473regions,originalvocab/references/provenance. No test/pixelimages/trainedcaptioncheckpoint. Upstreamtermsretained;semanticbanks remainseparate csa-semantic-banks-v130.'",
        "description='Private checkpoints from exact notebook139: CSA XE10 latest and validation-best epoch9, original binary bytes, receipts and complete small review ZIP. Retains upstream model/data terms. No model/training rerun.'")
    code=(HERE/'package_xe139.py').read_text(encoding='utf-8')
    cells=[cell('markdown','''# Preserve CSA full XE10 — exact Output139
**Accelerator NONE, Internet ON, no Add Input.** Enable Secret `KAGGLE_API_TOKEN`; if named `codex`, edit SECRET_NAME.
Downloads best.bin (validated epoch9) and checkpoint.bin (trained epoch10), receipts and original review ZIP from exact139.
Checks every original byte SHA and copies unchanged into a NEW Private Dataset. Never load weights/run models.
Keep source139 until Dataset finishes processing. The review ZIP contains all other logs/captions/references.
'''),cell('code',"""
from pathlib import Path
KERNEL_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/139'
OWNER='ducanh2403'
SLUG='csa-full-xe10-v139'
TITLE='CSA full XE10 v139 checkpoints dataset'
SECRET_NAME='KAGGLE_API_TOKEN'
DESTINATION=Path('/kaggle/working/csa_full_xe10_v139_dataset')
"""),cell('code',setup),cell('code','import hashlib,json\nINVENTORY='+repr(inventory)+'\nSOURCE_CODE='+repr(code)+
        '\nassert hashlib.sha256(SOURCE_CODE.encode()).hexdigest()=='+repr(hashlib.sha256(code.encode()).hexdigest())+
        "\nexec(compile(SOURCE_CODE,'package_xe139.py','exec'))"),cell('code',"""
import tempfile,shutil
if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<8*1024**3:raise RuntimeError('Need8GiB free for download and unchanged copy.')
DOWNLOAD=Path(tempfile.mkdtemp(prefix='csa-xe139-',dir='/tmp'))
pattern=r'(^|/)(?:'+'|'.join(re.escape(n) for n in INVENTORY['files'])+r')$'
print('Downloading EXACT:',KERNEL_VERSION,flush=True)
run_cli([sys.executable,'-m','kaggle','kernels','output',KERNEL_VERSION,'-p',str(DOWNLOAD),'--file-pattern',pattern])
report=package(DOWNLOAD,DESTINATION,INVENTORY)
print('Both checkpoints and original review verified/copied unchanged.',flush=True)
"""),cell('code',upload),cell('markdown','''Send the actual mount after creation finishes. Expected:
`/kaggle/input/datasets/ducanh2403/csa-full-xe10-v139`.
Keep ROI134 and semantic banks130 separately for any later continuation. This notebook preserves artifacts only; it does not extend training.
''')]
    nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=False)))
    p=HERE/'SAVE_CSA_XE10_V139_DATASET.ipynb';p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    (HERE/'SAVE_XE139_PACKAGE.json').write_text(json.dumps(inventory,indent=2)+'\n',encoding='utf-8')
    print(p,p.stat().st_size)

if __name__=='__main__':make()
