"""Reuse the working CSA smoke notebook setup; pin the sharing benchmark assets."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]


def make(revision):
    revision = subprocess.check_output(['git','rev-parse',revision], cwd=ROOT, text=True).strip()
    # Immutable published scaffold; source/data contracts are unchanged.
    raw = subprocess.check_output(['git','show','98d9a1d:experiments/clip_patch_baseline/CLIP_PATCH_CSA_STRUCTURE_SMOKE_2GPU.ipynb'], cwd=ROOT)
    nb = json.loads(raw); setup = nb['cells'][1]['source']; old = {}
    for n in ast.parse(setup).body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('REVISION','MAPPING','HASHES'):
            old[n.targets[0].id] = ast.literal_eval(n.value)
    mapping = dict(old['MAPPING'])
    for n in ('csa_sharing_data.py','csa_shared_model.py','run_csa_sharing_benchmark.py','test_csa_sharing_data.py','test_csa_accelerate_contract.py','gradient_agreement.py','test_gradient_agreement.py','CSA_SHARING_PROTOCOL.md','make_csa_sharing_notebook.py'):
        mapping['experiments/clip_patch_baseline/'+n] = n
    hashes = {n:hashlib.sha256(subprocess.check_output(['git','show',revision+':'+n], cwd=ROOT)).hexdigest() for n in mapping}
    setup = setup.replace(repr(old['REVISION']),repr(revision)).replace('MAPPING='+repr(old['MAPPING']), 'MAPPING='+repr(mapping)).replace('HASHES='+repr(old['HASHES']), 'HASHES='+repr(hashes))
    setup = setup.replace('clip_patch_csa_structure_smoke_v1','csa_feature_sharing_accelerate_v3').replace('updates=200,','updates=40,warmup=10,')
    setup += "\nACCEL_DEPS=SOURCE/'accelerate-deps'\nos.environ['PYTHONPATH']=str(ACCEL_DEPS)+os.pathsep+str(SOURCE)\n"
    nb['cells'][1]['source'] = setup
    execution = nb['cells'][3]['source'].replace('test_csa_structure_data.py','test_csa_sharing_data.py').replace('run_csa_structure_smoke.py','run_csa_sharing_benchmark.py').replace('CLIP_PATCH_CSA_STRUCTURE_SMOKE_COMPLETE','CSA_FEATURE_SHARING_ACCELERATE_BENCHMARK_COMPLETE').replace('SMOKE COMPLETE; no full training/CIDEr result.','ACCELERATE SHARING BENCHMARK COMPLETE; no full training/CIDEr result.')
    anchor="    run([sys.executable,'-u',SOURCE/'cuda_runtime_guard.py','--output',OUTPUT],60)"
    install='''
    run([sys.executable,'-u',SOURCE/'test_csa_accelerate_contract.py'],120)
    run([sys.executable,'-m','pip','download','-q','--no-deps','--only-binary=:all:','--dest',SOURCE,'accelerate==1.15.0'],180)
    wheel=SOURCE/'accelerate-1.15.0-py3-none-any.whl'
    if hashlib.sha256(wheel.read_bytes()).hexdigest()!='97eacca0b73e45cb867dbf8c5d5d4dc32219544300e0c8992c7334dc2ef33cec':raise ValueError('Changed Accelerate wheel')
    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',ACCEL_DEPS,wheel],180)
    probe="import accelerate;from pathlib import Path;assert accelerate.__version__=='1.15.0';assert Path(accelerate.__file__).resolve().is_relative_to(Path("+repr(str(ACCEL_DEPS))+"));print('Pinned isolatedAccelerate',accelerate.__version__,flush=True)"
    run([sys.executable,'-c',probe],120)
    (OUTPUT/'accelerate_install.json').write_text(json.dumps(dict(version='1.15.0',wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),isolated_target=str(ACCEL_DEPS),dependencies_reinstalled=False)))
'''
    if execution.count(anchor)!=1:raise ValueError('Changed setup scaffold')
    execution=execution.replace(anchor,anchor+'\n'+install)
    execution=execution.replace(anchor,"    run([sys.executable,'-u',SOURCE/'test_gradient_agreement.py'],120)\n"+anchor)
    nb['cells'][3]['source'] = execution
    nb['cells'][0]['source'] = '''# CSA batch-level feature sharing — Accelerate / FP32 / T4×2

InternetON, GPU T4×2, Save & Run All. Attach the SAME four Datasets: visual-cache, vuthetam/mscoco-2014, csa-semantic-banks-v130, csa-prompt-full-karpathy-v175. No Secret/checkpoint/newcache. Lastbudget6hown/30hNgân. Parent75min stopcap includes setup, not ETA/billing.

Same restored2OSA+1ORA/dual-memory3layerdecoder/promptadapters. Unique images encode ONCE per GPU local batch, then both memories index_select back to caption order. Gradients sum across all captions; decoder still processes all32globalcaptions. No parameters added, no FP16. This supersedes the previous native-DDP benchmark.

Accelerate1.15.0 manages devices/DDP/model+optimizer preparation/backward/clipping. FP32/no mixed precision, accumulation1. Torchrun launches two processes only. NumPy event batches are already rank-sliced: no Accelerate DataLoader sharding. Explicit free_memory between fresh passes prevents retaining previous models. Wheel pinned bySHA, isolated --no-deps install; never reinstallTorch/HFhub. Runtime version/DDP/precision recorded, import checked before staging. No speedup promise versus nativeDDP.

Two TRAIN256/1280event orders: original random captions and image-grouped. Each order has ordinary/shared/shared/ordinary passes,40updates/pass,10warmup+30timed;320updates total with fresh identical initialization every pass. Matched event IDs WITHIN each order; all1280events consumed per pass. Random batches may have few repeated images and sharing may be slower. Record actual encoded-image count, synchronized slowest-rank latency and peak memory.

Numerical guards BEFORE timing: dropoutOFF, ordinary/shared logits and DDP token loss/all parameter gradients versus full-global ordinary reference; include unequal16/11tail27. Encoder row hook verifies sharing actually occurs. Training speed uses dropout.2 ON: shared encoder dropout becomes correlated across captions of an image. Training trajectories are not claimed identical. Grouped sampling changes caption order and must be chosen explicitly for future full training.

V2 stopped at the first ORDINARY gradient guard, before sharing/timing. V3 keeps model/precision/optimizer/samplers/timing unchanged, uses ordinary rank-shaped microbatch gradient reference (same global token denominator) and scale-aware per-tensor diagnostics. Full-global ordinary scalar loss still checked. RelativeL2 limit.005 is stricter than V2.01; maxabsolute limit scales with reference magnitude plus3e-7. Detailed diagnostics saved even if rejected; no assumed GPU PASS. Do not label V2 a performance result.

No VALID/TEST/CIDEr/SCST/rerank/fulltraining/checkpoint. Author only NumPy/AST, actual Accelerate/GPU parity/performance pending. Send csa_feature_sharing_accelerate_v3_review.zip. No auto restart, cap extension or Torch reinstall.
'''
    for c in nb['cells']:
        if c['cell_type']=='code': ast.parse(c['source'])
    path = HERE/'CLIP_PATCH_CSA_FEATURE_SHARING_ACCELERATE_V3_2GPU.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(path.name,path.stat().st_size,'bytes;',len(mapping),'pins;',revision)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--revision',required=True); make(p.parse_args().revision)
