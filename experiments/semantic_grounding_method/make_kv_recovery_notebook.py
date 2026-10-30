"""Package CPU-only recovery of exact Output116; never call Kaggle here."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from make_kv_prepare_notebook import cell

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FILES=['kv_data.py','kv_pilot_data.py','kv_pilot_evaluate.py','recover_kv_summary_cpu.py',
       'test_kv_json_regression.py','KV_SAVE_V115_PACKAGE.json']


def make(core):
    core=subprocess.check_output(['git','rev-parse',core],cwd=ROOT,text=True).strip()
    assets={'experiments/semantic_grounding_method/'+n:hashlib.sha256(subprocess.check_output(
        ['git','show',core+':experiments/semantic_grounding_method/'+n],cwd=ROOT)).hexdigest() for n in FILES}
    name='experiments/relation_frames_inventory/inventory.py'
    assets[name]=hashlib.sha256(subprocess.check_output(['git','show',core+':'+name],cwd=ROOT)).hexdigest()
    old=json.loads((HERE/'SAVE_RELATION_KV_V115_DATASET.ipynb').read_bytes())
    setup=next(c['source'] for c in old['cells'] if c['cell_type']=='code' and 'def run_cli(' in c['source'])
    setup=setup.replace('/115','/116').replace('v115','v116').replace('pins115','pins116')
    cells=[cell('markdown','''
        # Khôi phục kết quả D — đúng Output116 — CPU, không train lại
        **Accelerator=None · Internet ON · Save & Run All. Không GPU, không load mô hình.**
        Add Input duy nhất: relation-kv-preparation-v115. Bật Secret KAGGLE_API_TOKEN;
        nếu secret tên codex thì sửa SECRET_NAME. Không dán token vào code/chat.
        Tải đúng version116, gồm checkpoint chỉ để kiểm SHA (không torch.load).
        Kiểm đủ2worker ENDPOINT_500_COMPLETE, captions1000/held99, cached CIDEr/CHAIR và metadata.
        Giữ nguyên file nguồn, tạo thư mục recovery mới; không gọi prepare/worker/training.
        Nếu thiếu artifact thì dừng và xuất ZIP/log, không tự dùng version khác hoặc train lại.
        '''),cell('code','''
        from pathlib import Path
        KERNEL_VERSION='ducanh2403/2-vi-cai-kia-qua-nhieu/116'
        OWNER='ducanh2403';SLUG='unused-recovery';TITLE='CPU summary recovery'
        SECRET_NAME='KAGGLE_API_TOKEN'
        PREPARED_ROOT=Path('/kaggle/input/datasets/ducanh2403/relation-kv-preparation-v115')
        OUTPUT=Path('/kaggle/working/relation_kv_D_recovered_v116')
        REVIEW=OUTPUT.with_name(OUTPUT.name+'_review.zip')
        if OUTPUT.exists() or REVIEW.exists():raise FileExistsError('Preserve old recovery; use a fresh session/path')
        if not (PREPARED_ROOT/'metadata.json').is_file():raise FileNotFoundError('Add Input relation-kv-preparation-v115')
        print('CPU recovery only. Training/model/GPU will NOT run.',flush=True)
        '''),cell('code',setup),cell('code',f'''
        import tempfile,hashlib,json,urllib.request,zipfile,traceback,os
        CORE={core!r}
        ASSETS={assets!r}
        WORK=Path(tempfile.mkdtemp(prefix='kv-recovery116-'));REPO=WORK/'repo';REPO.mkdir()
        HERE=REPO/'experiments/semantic_grounding_method'
        DOWNLOAD=WORK/'download';DOWNLOAD.mkdir()
        try:
            for relative,expected in ASSETS.items():
                with urllib.request.urlopen('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+CORE+'/'+relative,timeout=60) as response:raw=response.read()
                if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Changed recovery source '+relative)
                target=REPO/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
            os.environ['PYTHONPATH']=str(REPO)
            run_cli([sys.executable,'-u',HERE/'test_kv_json_regression.py'],authenticated=False)
            print('Downloading exact Output116, saved artifacts only.',flush=True)
            run_cli([sys.executable,'-m','kaggle','kernels','output',KERNEL_VERSION,'-p',str(DOWNLOAD),
                '--file-pattern',r'(^|/)([^/]+\.jsonl?|[^/]+\.log|checkpoint\.bin)$'])
            candidates=[p.parent for p in DOWNLOAD.rglob('protocol.json')
                if json.loads(p.read_bytes()).get('arm')=='D' and (p.parent/'checkpoint.bin').is_file()]
            if len(candidates)!=1:raise RuntimeError('Output116 lacks a unique D endpoint folder; STOP, no training retry')
            run_cli([sys.executable,'-u',HERE/'recover_kv_summary_cpu.py','--source',str(candidates[0]),
                '--output',str(OUTPUT),'--prepared',str(PREPARED_ROOT),'--version',KERNEL_VERSION,'--code-commit',CORE],authenticated=False)
        finally:
            with zipfile.ZipFile(REVIEW,'x',zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(OUTPUT.glob('*')):
                    if path.suffix in ('.json','.jsonl','.log'):archive.write(path,path.name)
                for path in DOWNLOAD.rglob('*.json'):
                    if path.name.startswith(('failure_','worker_')):archive.write(path,'download_diagnostics/'+path.name)
                archive.writestr('recovery_execution.json',json.dumps(dict(source_version=KERNEL_VERSION,
                    recovery_code=CORE,summary_created=(OUTPUT/'summary.json').is_file(),training_repeated=False,model_loaded=False)))
                if sys.exc_info()[0] is not None:archive.writestr('recovery_error.txt',traceback.format_exc())
            print('SEND RECOVERY ZIP:',REVIEW,flush=True)
        '''),cell('markdown','''
        Gửi `relation_kv_D_recovered_v116_review.zip`. Giữ Output116/checkpoint gốc.
        Bản này không tải raw visual/legacy cache, không chạy GPU, không huấn luyện hoặc chọn checkpoint mới.
        ''')]
    notebook=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(
        kernelspec=dict(display_name='Python 3',language='python',name='python3'),
        language_info=dict(name='python'),kaggle=dict(isGpuEnabled=False,isInternetEnabled=True)))
    target=HERE_LOCAL/'RECOVER_RELATION_KV_D_V116_CPU.ipynb'
    for c in cells:
        if c['cell_type']=='code':compile(c['source'],target.name,'exec')
    target.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(target.name,target.stat().st_size,'bytes; cells compiled, NOT executed')


HERE_LOCAL=HERE
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--core',default='HEAD');make(p.parse_args().core)
