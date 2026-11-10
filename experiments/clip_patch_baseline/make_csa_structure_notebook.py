"""Build pinned paper-structure smoke notebook; no neural or Kaggle execution."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT/'experiments/csa_paper'))
from make_smoke_notebook import RUNNER, cell


def make(revision):
    revision = subprocess.check_output(['git', 'rev-parse', revision], cwd=ROOT, text=True).strip()
    names = ('csa_patch_model.py', 'csa_structure_data.py', 'csa_structure_runtime.py', 'run_csa_structure_smoke.py',
             'test_csa_structure_data.py', 'semantic_data.py', 'semantic_patch_model.py', 'memory_fusion.py',
             'patch_data.py', 'patch_model.py', 'patch_beam.py', 'full_data.py', 'test_semantic_data.py',
             'CSA_STRUCTURE_PROTOCOL.md', 'make_csa_structure_notebook.py')
    mapping = {'experiments/clip_patch_baseline/'+n: n for n in names}
    mapping.update({'experiments/csa_paper/'+n: n for n in ('data.py', 'real_bank_data.py', 'cuda_runtime_guard.py',
                   'SAVE_BANK_V130_PACKAGE.json', 'FULL_PROMPT_FRESH_V175_RESULT.json')})
    mapping.update({'experiments/csa_paper/model.py': 'bank_layers.py',
                    'experiments/csa_paper/lemma_correction_review_text/bank_words_corrected.json': 'bank_words_corrected.json'})
    hashes = {n: hashlib.sha256(subprocess.check_output(['git', 'show', revision+':'+n], cwd=ROOT)).hexdigest() for n in mapping}
    setup = f'''import hashlib,json,os,queue,signal,subprocess,sys,tempfile,threading,time,traceback,urllib.request,zipfile,shutil
from pathlib import Path
REVISION={revision!r}
MAPPING={mapping!r}
HASHES={hashes!r}
'''+'''VISUAL_CACHE=Path('/kaggle/input/datasets/ducanh2403/visual-cache')
COCO_JSON=Path('/kaggle/input/datasets/vuthetam/mscoco-2014/dataset_coco.json')
BANK_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-semantic-banks-v130')
PROMPT_ROOT=Path('/kaggle/input/datasets/ducanh2403/csa-prompt-full-karpathy-v175')
OUTPUT=Path('/kaggle/working/clip_patch_csa_structure_smoke_v1')
for p in (VISUAL_CACHE,COCO_JSON,BANK_ROOT,PROMPT_ROOT):
    if not p.exists():raise FileNotFoundError('Attach/correct all four inputs BEFORE GPU: '+str(p))
if OUTPUT.exists():raise RuntimeError('Preserve prior output; use fresh session/output path')
if min(shutil.disk_usage('/tmp').free,shutil.disk_usage('/kaggle/working').free)<2*1024**3:raise RuntimeError('Need2GiB for small TRAIN stage/logs')
OUTPUT.mkdir(parents=True);SOURCE=Path(tempfile.mkdtemp(prefix='csa-structure-',dir='/tmp'));LOG=OUTPUT/'notebook.log'
PARENT_DEADLINE=time.monotonic()+4500 #75min STOPcap including setup; not ETA/billing.
os.environ['PYTHONPATH']=str(SOURCE);os.environ['OMP_NUM_THREADS']='1'
CONFIG=SOURCE/'config.json'
cfg=dict(output=str(OUTPUT),stage=str(SOURCE/'stage'),dataset_json=str(COCO_JSON),visual_cache=str(VISUAL_CACHE),
         bank_root=str(BANK_ROOT),prompt_root=str(PROMPT_ROOT),updates=200,prepare_seconds=2100,worker_seconds=2100)
def write_config():
    CONFIG.write_text(json.dumps(cfg,indent=2));(OUTPUT/'protocol.json').write_text(json.dumps(cfg,indent=2))
write_config()
'''
    runner = RUNNER.replace("print('Running:',", "if time.monotonic()>=PARENT_DEADLINE:raise TimeoutError('Parent cap exceeded')\n    timeout=min(timeout,PARENT_DEADLINE-time.monotonic())\n    print('Running:',").replace('Smoke active;', 'CSA structure active;')
    execution = '''complete=False
try:
    for name,destination in MAPPING.items():
        remaining=PARENT_DEADLINE-time.monotonic()
        if remaining<=0:raise TimeoutError('Source download parent cap')
        raw=urllib.request.urlopen('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+REVISION+'/'+name,timeout=min(90,remaining)).read()
        if hashlib.sha256(raw).hexdigest()!=HASHES[name]:raise ValueError('Changed source: '+name)
        (SOURCE/destination).write_bytes(raw)
    (OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=REVISION,sha256=HASHES),indent=2))
    run([sys.executable,'-u',SOURCE/'test_csa_structure_data.py'],120)
    run([sys.executable,'-u',SOURCE/'cuda_runtime_guard.py','--output',OUTPUT],60)
    run([sys.executable,'-u',SOURCE/'run_csa_structure_smoke.py','prepare','--config',CONFIG],2100)
    cfg['worker_seconds']=min(2100,PARENT_DEADLINE-time.monotonic()-120)
    if cfg['worker_seconds']<300:raise TimeoutError('Insufficient remaining parent budget; preserve log/stage')
    write_config()
    run([sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=2',
         '--max_restarts=0',SOURCE/'run_csa_structure_smoke.py','worker','--config',CONFIG],cfg['worker_seconds']+30)
    complete=json.loads((OUTPUT/'result.json').read_text())['status']=='CLIP_PATCH_CSA_STRUCTURE_SMOKE_COMPLETE'
except BaseException:
    (OUTPUT/'failure.txt').write_text(traceback.format_exc());raise
finally:
    (OUTPUT/'run_status.json').write_text(json.dumps(dict(complete=complete,full_training_complete=False,test_used=False,validation_used=False,SCST=False,reranking=False)))
    review=OUTPUT.with_name(OUTPUT.name+'_review.zip')
    with zipfile.ZipFile(review,'w',zipfile.ZIP_DEFLATED) as z:
        for p in OUTPUT.rglob('*'):
            if p.is_file() and p.suffix in ('.json','.jsonl','.txt','.log'):z.write(p,str(p.relative_to(OUTPUT)))
    print('SEND REVIEW:',review,flush=True)
    print('SMOKE COMPLETE; no full training/CIDEr result.' if complete else 'INCOMPLETE; preserve log/output, no automatic retry.',flush=True)
'''
    intro = '''# Restore CSA architecture on CLIP patches — T4×2 TRAIN smoke

InternetON, Accelerator **GPU T4×2**, Save & Run All. Attach FOUR existing inputs:
- `ducanh2403/visual-cache`
- `vuthetam/mscoco-2014`
- `ducanh2403/csa-semantic-banks-v130`
- `ducanh2403/csa-prompt-full-karpathy-v175`

No API Secret, checkpoint or new CLIP/UPT extraction. Fresh initialization. Budget reported6h yours/30h Ngân; this notebook is bounded to75min including setup, not a full10epoch run or billing estimate.

Reuse our paper-core:2complete object alignment/fusion layers →1object-guided relation layer →TWO separate memories →3layer parallel adaptive decoder with independent per-word gates. Dim512/heads8/FFN2048/dropout.2. Remove old single-memory mixture and strength gate. The two memories are packaged for beam expansion and split BEFORE the two decoder cross-attentions.

Disclosed adaptation: frozenCLIP-B16 patches and CLIP-text banks replace the paper's detector/RegionCLIP. Our image-specific UPT prompt adapters remain AFTER the core encoder and modify the two streams separately. Thus this is CSA structural inheritance + topic-specific prompt extension, not exact full-system replication. No paper-score claim, no new loss.

Only256fixedTRAINimages/200XEupdates/global32/FP32/Adam. Check two-stage semantic carry, independent stream response, causal mask, actual DDP gradient parity, prompt-mask locality, TRAIN-probe loss decrease, beam5 score replay, throughput and GPU memory. No VALID/TEST/CIDEr/SCST/reranking. No full33GiB copy; selectedTRAINpatches only. Source/cache SHA verification remains.

Send `clip_patch_csa_structure_smoke_v1_review.zip`. This smoke does NOT prove better CIDEr and is not a full-training checkpoint. Full-run scheduling follows measured results; no automatic restart, full run or Torch reinstall. Author checks are NumPy/mock/AST only; actual neural checks run here on Kaggle.
'''
    nb = dict(nbformat=4, nbformat_minor=5, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
              kaggle=dict(isInternetEnabled=True, isGpuEnabled=True)), cells=[cell('markdown', intro), cell('code', setup), cell('code', runner), cell('code', execution)])
    path = HERE/'CLIP_PATCH_CSA_STRUCTURE_SMOKE_2GPU.ipynb'
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n', encoding='utf-8')
    print(path.name, path.stat().st_size, 'bytes;', len(mapping), 'pinned assets;', revision)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--revision', required=True); make(p.parse_args().revision)
