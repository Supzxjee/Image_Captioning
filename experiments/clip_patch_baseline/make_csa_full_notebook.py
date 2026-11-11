"""Build full Accelerate per-caption CSA notebook from the working full-cache scaffold."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def check_local_imports(mapping,revision):
    """Check reachable repository-local imports using AST, without importing neural code."""
    sources={Path(dst).stem:subprocess.check_output(['git','show',revision+':'+src],cwd=ROOT).decode('utf-8-sig') for src,dst in mapping.items() if dst.endswith('.py')}
    inventory={p.stem for folder in (HERE,ROOT/'experiments/csa_paper') for p in folder.glob('*.py')}
    pending=['run_csa_full','score_csa_full','csa_patch_model','csa_full_runtime'];seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        if name not in sources:raise ValueError('Missing staged local module: '+name)
        seen.add(name)
        for node in ast.walk(ast.parse(sources[name])):
            names=([a.name.split('.')[0] for a in node.names] if isinstance(node,ast.Import) else [node.module.split('.')[0]] if isinstance(node,ast.ImportFrom) and node.module else [])
            for target in names:
                if target in sources or target in inventory:
                    if target not in sources:raise ValueError('Missing staged local module: '+target+' imported by '+name)
                    pending.append(target)
    print('Static local import closure PASS:',len(seen),'modules')
    return seen


def make(revision):
    revision=subprocess.check_output(['git','rev-parse',revision],cwd=ROOT,text=True).strip()
    scaffold='98d9a1d:experiments/clip_patch_baseline/CLIP_PATCH_BANK_PROMPT_SCRATCH_XE10_2GPU.ipynb'
    nb=json.loads(subprocess.check_output(['git','show',scaffold],cwd=ROOT));old={}
    for n in ast.parse(nb['cells'][1]['source']).body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('REVISION','MAPPING','HASHES'):old[n.targets[0].id]=ast.literal_eval(n.value)
    mapping=dict(old['MAPPING'])
    for n in ('memory_fusion.py','csa_full_data.py','csa_full_runtime.py','run_csa_full.py','score_csa_full.py','test_csa_full_data.py','csa_patch_model.py','csa_sharing_data.py','gradient_agreement.py','test_gradient_agreement.py','CSA_SHARED_FULL_XE10_PROTOCOL.md','make_csa_full_notebook.py'):
        mapping['experiments/clip_patch_baseline/'+n]=n
    check_local_imports(mapping,revision)
    hashes={n:hashlib.sha256(subprocess.check_output(['git','show',revision+':'+n],cwd=ROOT)).hexdigest() for n in mapping}
    setup=nb['cells'][1]['source'].replace(repr(old['REVISION']),repr(revision)).replace('MAPPING='+repr(old['MAPPING']),'MAPPING='+repr(mapping)).replace('HASHES='+repr(old['HASHES']),'HASHES='+repr(hashes))
    setup=setup.replace('clip_patch_bank_prompt_xe10_v1','csa_structure_per_caption_full_xe10_v1').replace('WALL_HOURS=10.5','WALL_HOURS=11.5').replace("shutil.disk_usage('/kaggle/working').free<4*1024**3","shutil.disk_usage('/kaggle/working').free<8*1024**3").replace('and4GiB Working','and8GiB Working').replace('cfg=dict(semantic=True,','cfg=dict(csa_per_caption_full=True,semantic=True,')
    setup+="\nACCEL_DEPS=SOURCE/'accelerate-deps'\nos.environ['PYTHONPATH']=str(ACCEL_DEPS)+os.pathsep+str(SOURCE)\n"
    nb['cells'][1]['source']=setup
    execution=nb['cells'][3]['source'].replace("SOURCE/'run_full.py'","SOURCE/'run_csa_full.py'").replace("SOURCE/'score_full.py'","SOURCE/'score_csa_full.py'").replace('CLIP_PATCH_FULL_XE10_COMPLETE','CSA_PER_CAPTION_FULL_XE10_COMPLETE').replace('BANK+PROMPT FULL10 COMPLETE; VALID only, no TEST/SCST/rerank.','CSA PER-CAPTION FULL10 COMPLETE; VALID only, no TEST/SCST/rerank.')
    execution=execution.replace("os.environ['PYTHONPATH']=str(DEPS)+os.pathsep+str(SOURCE)","os.environ['PYTHONPATH']=str(ACCEL_DEPS)+os.pathsep+str(DEPS)+os.pathsep+str(SOURCE)")
    anchor="    run([sys.executable,'-u',SOURCE/'cuda_runtime_guard.py','--output',OUTPUT],60)"
    install='''
    run([sys.executable,'-m','pip','download','-q','--no-deps','--only-binary=:all:','--dest',SOURCE,'accelerate==1.15.0'],180)
    wheel=SOURCE/'accelerate-1.15.0-py3-none-any.whl'
    if hashlib.sha256(wheel.read_bytes()).hexdigest()!='97eacca0b73e45cb867dbf8c5d5d4dc32219544300e0c8992c7334dc2ef33cec':raise ValueError('Changed Accelerate wheel')
    run([sys.executable,'-m','pip','install','-q','--no-deps','--target',ACCEL_DEPS,wheel],180)
    run([sys.executable,'-c',"import accelerate;assert accelerate.__version__=='1.15.0';print('PinnedAccelerate',accelerate.__version__,flush=True)"],120)
    run([sys.executable,'-c',"from csa_patch_model import CSAPatchCaptioner;from csa_full_runtime import make_model;print('Neural module imports PASS; no model instantiated',flush=True)"],120)
    (OUTPUT/'accelerate_install.json').write_text(json.dumps(dict(version='1.15.0',wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),dependencies_reinstalled=False)))
'''
    if execution.count(anchor)!=1:raise ValueError('Changed full setup scaffold')
    execution=execution.replace(anchor,"    run([sys.executable,'-u',SOURCE/'test_csa_full_data.py'],120)\n    run([sys.executable,'-u',SOURCE/'test_gradient_agreement.py'],120)\n"+anchor+'\n'+install)
    # Score CPU metrics only on complete10, preserve partial training output on cap.
    nb['cells'][3]['source']=execution
    nb['cells'][0]['source']='''# Full Karpathy TRAIN — CSA structure + prompt + Accelerate, per-caption encoding

**Each caption gets its own encoder forward path. The full-data preflight compares ordinary DDP and ordinary reference gradients on both batch sizes. A fresh Save Version stages the cache again.**

**GPU T4×2, InternetON, Save & Run All. Use the account lastreported30h; ownlast6h is insufficient for requestedfull10.** Attach SAME fourDatasets: visual-cache, vuthetam/mscoco-2014, csa-semantic-banks-v130, csa-prompt-full-karpathy-v175. No Secret/oldcheckpoint/new semantic cache.

Fresh10epochXE,113287TRAIN/all566747captionevents per epoch,17711updates/epoch/global32(final16+11). Keep restored2OSA+1ORA/dual-memory3layerdecoder/dropout.2/promptadapters. Accelerate1.15.0 pinned/no-deps install, FP32/noAMP/no compile/accum1, image-grouped sampler42+epoch, encode each caption event separately. Grouped sampling differs from the random-caption baseline.

Fullrecipe: Adam(.9,.99)/eps1e-8/wd0/clip1; warmup1epoch8e-6→1e-4 then9epochcosine→1e-5, matching previouspatchXE10 horizon. Not originalpaper15XE+30RL/RegionCLIP/detector. No smokeweights, SCST/TEST/rerank.

Reuse fullrawcache via sequentiallocalFP16stage (~33.17GiB)+semanticarrays; need40GiB/tmp+8GiBWorking. No CLIP/UPT inference. Parent11.5hSTOPcap includes setup, not ETA/billing/session-fit guarantee. Preparation90min. Actualfull-loader Accelerategradientparity and64discarded trainupdates; forced20tokenTRAINbeam timing onCOPY avoids earlyEOS underestimate. Mainworker startsfreshseed42.

Requested10epochs progresses within cap; no requirement that all10fit before ANYtraining. Currentepoch+VALIDbudget guard, atomiccheckpointevery500updates/phases/epochs withAdam/RNG/cursor, explicitresume viaRESUME_ROOT for SAMEsource/data/recipe only. No automaticrestart/capextension/LRchange. Ifcapped, preserveentireoutput; nextsession restages /tmp and resumesexactcursor, not freshmodel. Keepbest.bin ANDcheckpoint.bin/receipts.

VALID5000beam5/max20/NLL/CIDEr after EVERYepoch, logmeanlength/shortcaptioncount; best=maxVALIDCIDEr, earlierties. Savedbest/endcaps score BLEU1/4,ROUGE-L,CHAIRs/i,Recall atcompletion. METEOR/SPICE/TEST later. User inspects firstepoch progress, full10horizon fixed. No referencecaption selection.

Send csa_structure_per_caption_full_xe10_v1_review.zip. Author onlyNumPy/mock/AST checks; actualfullIO/GPU/Accelerate/model/scorers user-run. Prior grouped ordinary benchmark suggests ~76min training/epoch, excludesfullIO/staging/VALID/scorers/checkpoints; no guaranteed ETA. A partialrun is NOT a full10result.
'''
    for cell in nb['cells']:
        if cell['cell_type']=='code':ast.parse(cell['source'])
    path=HERE/'CLIP_PATCH_CSA_PER_CAPTION_FULL_XE10_ACCELERATE_2GPU.ipynb';path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print(path.name,path.stat().st_size,'bytes;',len(mapping),'pins;',revision)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--revision',required=True);make(p.parse_args().revision)
