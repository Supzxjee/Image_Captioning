"""Author CPU exact181 transport; no model/API/notebook execution."""
import ast,json
from pathlib import Path
from make_smoke_notebook import cell
HERE=Path(__file__).resolve().parent

def make():
    report=json.loads((HERE/'PROMPT_TEST_PREPARATION_RESULT.json').read_text(encoding='utf-8'));inventory=report['transport_inventory']
    old=json.loads((HERE/'SAVE_CSA_PROMPT_INPUTS_V169_DATASET.ipynb').read_text(encoding='utf-8'));codes=[c['source'] for c in old['cells'] if c['cell_type']=='code']
    setup=next(c for c in codes if 'def run_cli(' in c).replace('private outputs169/161','private output181')
    version=next(c for c in codes if c.startswith('VERSION_PREFLIGHT='));preflight=ast.literal_eval(ast.parse(version).body[0].value).replace("('169','161')","('181',)").replace('Exact169/161','Exact181')
    upload=next(c for c in codes if c.startswith('CREATE_DATASET_SCRIPT='));start=upload.index('    description=');end=upload.index('\ntry:',start)
    upload=upload[:start]+"    description='Private exact181 TEST5000 ROI and structured prompt cache, frozen CLIP vectors. No caption weights or training. Upstream data/model terms retained.')"+upload[end:]
    config="""from pathlib import Path
OWNER='ducanh2403'
SLUG='csa-prompt-test-cache-v181'
TITLE='CSA prompt TEST cache v181'
SECRET_NAME='codex'
DESTINATION=Path('/kaggle/working/csa_prompt_test_cache_v181_dataset')
"""
    helper=(HERE/'package_prompt_inputs.py').read_text(encoding='utf-8')
    transport="""import numpy as np
INVENTORY="""+repr(inventory)+"""
DOWNLOAD=Path(tempfile.mkdtemp(prefix='prompt-test181-',dir='/tmp'))
run_cli(CLI+['kernels','output',INVENTORY['cache_version'],'-p',str(DOWNLOAD),'--file-pattern',pattern(INVENTORY['cache_files'])])
root=locate(DOWNLOAD,'receipt.json',INVENTORY['cache_files']['receipt.json']).parent
for name,digest in INVENTORY['cache_files'].items():
    if Path(name).name!=name or not (root/name).is_file() or sha(root/name)!=digest:raise ValueError('Missing/changed181 payload:'+name)
receipt=json.loads((root/'receipt.json').read_text(encoding='utf-8'))
if receipt['status']!='PROMPT_TEST_PREPARATION_COMPLETE' or any(INVENTORY['cache_files'].get(n)!=h for n,h in receipt['payload_sha256'].items()):raise ValueError('Wrong cache receipt')
vectors=np.load(root/'prompt_embeddings.npy',allow_pickle=False)
if vectors.shape!=(258,512) or vectors.dtype!=np.float32 or not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors,axis=1),1,atol=2e-6,rtol=0):raise ValueError('Wrong text vectors')
rows=[json.loads(line) for line in (root/'prompt_links.jsonl').read_text(encoding='utf-8').splitlines()]
if len(rows)!=5000 or len({r['coco_id'] for r in rows})!=5000 or any(r['group']!='test' for r in rows):raise ValueError('Wrong TEST5000 prompt coverage')
DESTINATION.mkdir(parents=True)
for name,digest in INVENTORY['cache_files'].items():
    shutil.copy2(root/name,DESTINATION/name)
    if sha(DESTINATION/name)!=digest:raise ValueError('Copied bytes changed:'+name)
(DESTINATION/'transport_receipt.json').write_text(json.dumps(dict(status='EXACT181_PRESERVED',source=INVENTORY['cache_version'],files_sha256=INVENTORY['cache_files'],models_executed=False),indent=2),encoding='utf-8')
print('All10 payloads verified and copied unchanged; TEST5000/258vectors. No model executed.',flush=True)
"""
    cells=[cell('markdown','# Lưu TEST cache từ đúng Output181\n**Accelerator None, Internet ON**, bật Secret `codex` của ducanh2403. Không cần Add Input/checkpoint. Tải đúng181, kiểm10SHA và vectors258x512 rồi tạo Dataset Private mới. Không chạy lại cache/model; không dùng latest. Sau khi Dataset xử lý xong, chia sẻ quyền cho Ngân và gửi path mount.'),cell('code',config),cell('code',setup),cell('code','VERSION_PREFLIGHT='+repr(preflight)+"\nrun_cli([sys.executable,'-u','-c',VERSION_PREFLIGHT])\n"),cell('code',helper),cell('code',transport),cell('code',upload)]
    nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),kaggle=dict(isInternetEnabled=True,isGpuEnabled=False)),cells=cells)
    path=HERE/'SAVE_CSA_PROMPT_TEST_V181_DATASET.ipynb';path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8');print(path.name,path.stat().st_size)

if __name__=='__main__':make()
