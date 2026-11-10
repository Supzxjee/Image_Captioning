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
    for n in ('csa_sharing_data.py','csa_shared_model.py','run_csa_sharing_benchmark.py','test_csa_sharing_data.py','CSA_SHARING_PROTOCOL.md','make_csa_sharing_notebook.py'):
        mapping['experiments/clip_patch_baseline/'+n] = n
    hashes = {n:hashlib.sha256(subprocess.check_output(['git','show',revision+':'+n], cwd=ROOT)).hexdigest() for n in mapping}
    setup = setup.replace(repr(old['REVISION']),repr(revision)).replace('MAPPING='+repr(old['MAPPING']), 'MAPPING='+repr(mapping)).replace('HASHES='+repr(old['HASHES']), 'HASHES='+repr(hashes))
    setup = setup.replace('clip_patch_csa_structure_smoke_v1','csa_feature_sharing_benchmark_v1').replace('updates=200,','updates=40,warmup=10,')
    nb['cells'][1]['source'] = setup
    execution = nb['cells'][3]['source'].replace('test_csa_structure_data.py','test_csa_sharing_data.py').replace('run_csa_structure_smoke.py','run_csa_sharing_benchmark.py').replace('CLIP_PATCH_CSA_STRUCTURE_SMOKE_COMPLETE','CSA_FEATURE_SHARING_BENCHMARK_COMPLETE').replace('SMOKE COMPLETE; no full training/CIDEr result.','SHARING BENCHMARK COMPLETE; no full training/CIDEr result.')
    nb['cells'][3]['source'] = execution
    nb['cells'][0]['source'] = '''# CSA batch-level feature sharing — FP32/DDP T4×2 benchmark

InternetON, GPU T4×2, Save & Run All. Attach the SAME four Datasets: visual-cache, vuthetam/mscoco-2014, csa-semantic-banks-v130, csa-prompt-full-karpathy-v175. No Secret/checkpoint/newcache. Lastbudget6hown/30hNgân. Parent75min stopcap includes setup, not ETA/billing.

Same restored2OSA+1ORA/dual-memory3layerdecoder/promptadapters. Unique images encode ONCE per GPU local batch, then both memories index_select back to caption order. Gradients sum across all captions; decoder still processes all32globalcaptions. No parameters added, no FP16/Accelerate.

Two TRAIN256/1280event orders: original random captions and image-grouped. Each order has ordinary/shared/shared/ordinary passes,40updates/pass,10warmup+30timed;320updates total with fresh identical initialization every pass. Matched event IDs WITHIN each order; all1280events consumed per pass. Random batches may have few repeated images and sharing may be slower. Record actual encoded-image count, synchronized slowest-rank latency and peak memory.

Numerical guards BEFORE timing: dropoutOFF, ordinary/shared logits and DDP token loss/all parameter gradients versus full-global ordinary reference; include unequal16/11tail27. Encoder row hook verifies sharing actually occurs. Training speed uses dropout.2 ON: shared encoder dropout becomes correlated across captions of an image. Training trajectories are not claimed identical. Grouped sampling changes caption order and must be chosen explicitly for future full training.

No VALID/TEST/CIDEr/SCST/rerank/fulltraining/checkpoint. Author only NumPy/AST, actual GPU parity/performance pending. Send csa_feature_sharing_benchmark_v1_review.zip. No auto restart, cap extension or Torch reinstall.
'''
    for c in nb['cells']:
        if c['cell_type']=='code': ast.parse(c['source'])
    path = HERE/'CLIP_PATCH_CSA_FEATURE_SHARING_BENCHMARK_2GPU.ipynb'
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(path.name,path.stat().st_size,'bytes;',len(mapping),'pins;',revision)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--revision',required=True); make(p.parse_args().revision)
