"""Generate pinned user-run notebook. Does not execute notebook/model code."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import textwrap
from make_kv_prepare_notebook import cell
from make_kv_pilot_notebooks import RUNNER

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREFIX = 'experiments/semantic_grounding_method/'
FILES = [PREFIX+n for n in ('response15_data.py', 'run_response15_probe.py', 'test_response15_data.py',
                            'RESPONSE15_ASSET_RECEIPT.json')]
FILES += ['captioning/'+n+'.py' for n in ('__init__', 'config', 'models', 'grounded_relations',
          'decoder_relation_memory', 'semantic_prompts', 'scst_runtime')]


def make(revision):
    core = subprocess.check_output(['git', 'rev-parse', revision], cwd=ROOT, text=True).strip()
    hashes = {n: hashlib.sha256(subprocess.check_output(['git', 'show', core+':'+n], cwd=ROOT)).hexdigest()
              for n in FILES}
    cells = [cell('markdown', """
        # C0 — semantic response15 diagnostic · v2 identity-scope fix
        **Internet ON, GPU, Save & Run All. Chưa huấn luyện.**
        Dùng GPU0; nếu chọn T4×2 thì GPU1 không dùng. Không DDP.

        Add Input:
        1. Dataset private chứa semantic_response_confirmed15_assets.zip (3.46MB), hoặc nội dung đã giải nén.
        2. concept-mass-c-c0-v111-v112 (file C0.bin).
        3. prompt-cache (prompt_clip_tokens_cache.pt).

        Không cần MSCOCO images/UPT/visual-cache/detector. Gói15cặp đã chứa30ảnh nguồn và nhãn bạn xác nhận.
        Hai ảnh dùng cùng prompt gốc: edited side là fixed-prompt diagnostic, không phải pipeline tạo lại prompt.
        Notebook kiểm15cặp, không đo tăngCIDEr/giảmCHAIR, không sinh checkpoint hoặc train.
        Kết thúc gửi semantic_response_C0_15_v2_review.zip; ZIP có log kể cả khi fail.
        """), cell('code', """
        from pathlib import Path
        CHECKPOINT=Path('/kaggle/input/datasets/ducanh2403/concept-mass-c-c0-v111-v112/C0.bin')
        LEGACY_PROMPT=Path('/kaggle/input/datasets/ducanh2403/prompt-cache/prompt_clip_tokens_cache.pt')
        # Tự tìm marker/ZIP. Nếu có nhiều bản, điền root của đúng Dataset.
        ASSETS_INPUT=Path('/kaggle/input')
        OUTPUT=Path('/kaggle/working/semantic_response_C0_15_v2')
        """), cell('code', textwrap.dedent(f"""
        CORE={core!r}
        SOURCE_ASSETS={hashes!r}
        import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile
        os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
            PYTHONUNBUFFERED='1',PYTHONHASHSEED='42',TOKENIZERS_PARALLELISM='false')
        LOG=OUTPUT.with_name(OUTPUT.name+'_notebook.log')
        REVIEW=OUTPUT.with_name(OUTPUT.name+'_review.zip')
        for p in (OUTPUT,LOG,REVIEW):
            if p.exists():raise FileExistsError('Preserve previous output; choose a fresh path/session: '+str(p))
        OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        WORK=Path(tempfile.mkdtemp(prefix='response15-'))
        REPO=WORK/'repo';REPO.mkdir()
        HERE=REPO/{PREFIX!r}
        CONFIG=WORK/'config.json'
        print('Starting frozen15-pair diagnostic; source',CORE,flush=True)
        """) + RUNNER), cell('code', """
        success=False
        try:
            for p in (CHECKPOINT,LEGACY_PROMPT):
                if not p.is_file():raise FileNotFoundError('Add Input or correct path: '+str(p))
            for n,expected in SOURCE_ASSETS.items():
                with urllib.request.urlopen('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+CORE+'/'+n,timeout=60) as f:raw=f.read()
                if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Wrong pinned source: '+n)
                p=REPO/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
            os.environ['PYTHONPATH']=str(REPO)
            run([sys.executable,'-u',HERE/'test_response15_data.py'],timeout=120)
            # Torch comes from the Kaggle image; never reinstall or replace it here.
            run([sys.executable,'-m','pip','install','-q','transformers==4.57.1','Pillow'],timeout=600)
            cfg=dict(checkpoint=str(CHECKPOINT),legacy_prompt=str(LEGACY_PROMPT),assets_input=str(ASSETS_INPUT),
                     output=str(OUTPUT),scratch=str(WORK),source_commit=CORE,source_assets=SOURCE_ASSETS,training=False)
            CONFIG.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
            run([sys.executable,'-u',HERE/'run_response15_probe.py','--config',CONFIG],timeout=2400)
            result=json.loads((OUTPUT/'summary.json').read_bytes())
            success=result['status']=='DIAGNOSTIC_COMPLETE' and result['pairs']==15
            if not success:raise RuntimeError('Incomplete diagnostic; do not report a subset as15pairs')
            print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
        finally:
            mode='a' if REVIEW.exists() else 'x'
            with zipfile.ZipFile(REVIEW,mode,zipfile.ZIP_DEFLATED) as z:
                if LOG.exists():z.write(LOG,'notebook.log')
                if CONFIG.exists():z.write(CONFIG,'notebook_config.json')
                for n in SOURCE_ASSETS:
                    if (REPO/n).is_file():z.write(REPO/n,'source/'+n)
                z.writestr('notebook_status.json',json.dumps(dict(success=success,training=False,source_commit=CORE)))
                if not success:z.writestr('notebook_error.txt',traceback.format_exc())
            print('SEND REVIEW ZIP:',REVIEW,flush=True)
            if not success:print('INCOMPLETE: send this ZIP/log; no model conclusion.',flush=True)
        """)]
    notebook = dict(nbformat=4, nbformat_minor=5, metadata=dict(
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        kaggle=dict(isGpuEnabled=True, language='python', sourceType='notebook')), cells=cells)
    target = HERE/'SEMANTIC_RESPONSE_C0_15_DIAGNOSTIC.ipynb'
    raw = (json.dumps(notebook, ensure_ascii=False, indent=1)+'\n').encode('utf-8')
    if len(raw) >= 1_000_000:
        raise ValueError('Kaggle notebook source too large')
    target.write_bytes(raw)
    print('Generated', target.name, len(raw), 'bytes; pinned', core, '; not executed')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--revision', default='HEAD')
    make(p.parse_args().revision)
