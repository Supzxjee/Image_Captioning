"""Build user-run, source-pinned historical FP32 XE10 repeats."""
import ast
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def pinned(name):
    return subprocess.check_output([
        'git', 'show', '5eb012279e6fd0b5e47a56710ed5e3efa7449d2f:experiments/clip_patch_baseline/'+name
    ], cwd=HERE).decode('utf-8')


def replace_once(source, old, new):
    if source.count(old)!=1:
        raise ValueError('Unexpected source marker: '+old)
    return source.replace(old,new,1)


def sources():
    runner = pinned('run_csa_full.py')
    runner = replace_once(runner, "    rank=int(os.environ['LOCAL_RANK'])",
        "    seed=cfg['seed']\n    if type(seed)!=int or seed not in (42,43,44):raise ValueError('Expected seed42/43/44')\n    rank=int(os.environ['LOCAL_RANK'])")
    runner = runner.replace('torch.manual_seed(42)', 'torch.manual_seed(seed)')
    runner = runner.replace('seed=42,', 'seed=seed,')
    runner = runner.replace("sampler='image_grouped_42_plus_epoch'", "sampler=f'image_grouped_{seed}_plus_epoch'")
    runner = runner.replace('random.seed(42+rank);np.random.seed(42+rank);torch.manual_seed(4242+rank);torch.cuda.manual_seed(4242+rank)',
                            'random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed*101+rank);torch.cuda.manual_seed(seed*101+rank)')
    runner = runner.replace('order_for_epoch(data,epoch)', 'order_for_epoch(data,epoch,seed)')
    runner = replace_once(runner,
        "        if projected>deadline-time.monotonic():checkpoint();raise TimeoutError('Measured current epoch+VALID projection exceeds cap; initial checkpoint saved, resume explicitly')",
        "        # Projections are informational; the hard deadline below saves exact progress.")
    runner = replace_once(runner,
        "                            if int(stop):checkpoint();raise TimeoutError('Current epoch+VALID projection exceeds remaining cap; saved for explicitresume')",
        "                            if int(stop) and rank==0:print('Current-epoch estimate exceeds remaining cap; continuing with periodic checkpoints and hard deadline.',flush=True)")
    sampler = pinned('csa_full_data.py')
    sampler = sampler.replace('train_images, epoch):', 'train_images, epoch, seed=42):')
    sampler = sampler.replace('np.random.default_rng(42+epoch)', 'np.random.default_rng(seed+epoch)')
    sampler = sampler.replace('def order_for_epoch(data,epoch):','def order_for_epoch(data,epoch,seed=42):')
    sampler = sampler.replace('base.TRAIN_IMAGES,epoch)', 'base.TRAIN_IMAGES,epoch,seed)')
    runtime = pinned('csa_full_runtime.py').replace('order_for_epoch(data,1)', "order_for_epoch(data,1,cfg.get('seed',42))")
    scoring = pinned('score_csa_full.py').replace("sampler='image_grouped_42_plus_epoch'", "sampler=f\"image_grouped_{cfg['seed']}_plus_epoch\",seed=cfg['seed']")
    result = dict(zip(('run_csa_full.py','csa_full_data.py','csa_full_runtime.py','score_csa_full.py'),
                      (runner,sampler,runtime,scoring),strict=True))
    for name,code in result.items():
        ast.parse(code,filename=name)
    return result


def notebook(seed):
    nb = json.loads((HERE/'CLIP_PATCH_CSA_PER_CAPTION_FULL_XE10_ACCELERATE_2GPU.ipynb').read_text(encoding='utf-8'))
    def put(index,source):
        if nb['cells'][index]['cell_type']=='code':
            compile(source,f'cell{index}','exec')
            nb['cells'][index]['outputs']=[]
            nb['cells'][index]['execution_count']=None
        nb['cells'][index]['source']=source.splitlines(keepends=True)
    put(0,f"""# Historical per-caption FP32 XE10, seed {seed}
GPU T4 x2, Internet on, the four input datasets below. Save & Run All. Fresh initialization by default.

This notebook reads mounted H5 patches, encodes each caption separately, and preserves the original global batch32 and per-step loss logging. Best checkpoint is selected by VALID5000 CIDEr after each epoch. The 11.5-hour invocation cap preserves resumable progress.

For continuation, attach the entire prior output and set RESUME_ROOT to its folder containing checkpoint.bin, checkpoint_receipt.json, evaluations, and best.bin. Set RUN_TAG to a new value for each invocation. The source, seed, data and software must match. Keep the full output, including weights, after each run.
""")
    setup=''.join(nb['cells'][1]['source'])
    setup=setup.replace("RESUME_ROOT='' # Optional preserved full-run output from THIS exact recipe. Blank starts fresh.",
        f"SEED={seed}\nRESUME_ROOT='' # Exact prior output folder of this seed, when continuing.\nRUN_TAG='run1'")
    setup=setup.replace("OUTPUT=Path('/kaggle/working/csa_structure_per_caption_full_xe10_v1')",
        "OUTPUT=Path(f'/kaggle/working/csa_per_caption_h5_seed{SEED}_{RUN_TAG}')")
    setup=setup.replace("if shutil.disk_usage('/tmp').free<40*1024**3 or shutil.disk_usage('/kaggle/working').free<8*1024**3:raise RuntimeError('Need40GiB temporary scratch and8GiB Working; sequential local stage required')",
        "if shutil.disk_usage('/tmp').free<4*1024**3 or shutil.disk_usage('/kaggle/working').free<8*1024**3:raise RuntimeError('Need 4 GiB temporary scratch and 8 GiB Working')")
    setup=setup.replace("cfg=dict(csa_per_caption_full=True,", "cfg=dict(seed=SEED,csa_per_caption_full=True,")
    setup=setup.replace("feature_storage='local_fp16_memmap_v2',prepare_seconds=5400", "prepare_seconds=1800")
    setup+='\nSEED_SOURCES = '+repr(sources())+'\n'
    setup+="""
if RESUME_ROOT:
    root=Path(RESUME_ROOT)
    if not (root/'checkpoint_receipt.json').is_file():
        raise FileNotFoundError('RESUME_ROOT must contain checkpoint.bin and checkpoint_receipt.json')
    if json.loads((root/'identity.json').read_text())['seed']!=SEED:
        raise ValueError('Resume seed differs')
"""
    put(1,setup)
    run=''.join(nb['cells'][3]['source'])
    run=replace_once(run,
        "    for name,destination in MAPPING.items():download('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+REVISION+'/'+name,HASHES[name],SOURCE/destination)\n    (OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=REVISION,sha256=HASHES),indent=2))",
        """    upstream={path:HASHES[path] for path,destination in MAPPING.items() if destination not in SEED_SOURCES}
    for path,digest in upstream.items():
        download('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+REVISION+'/'+path,digest,SOURCE/MAPPING[path])
    for name,source in SEED_SOURCES.items():
        (SOURCE/name).write_bytes(source.encode('utf-8'))
    embedded_sha={name:hashlib.sha256((SOURCE/name).read_bytes()).hexdigest() for name in SEED_SOURCES}
    (OUTPUT/'source_receipt.json').write_text(json.dumps(dict(commit=REVISION,sha256=upstream,embedded_sha256=embedded_sha),indent=2))""")
    run=run.replace("SOURCE/'run_csa_full.py','prepare','--config',CONFIG],5400", "SOURCE/'run_csa_full.py','prepare','--config',CONFIG],1800")
    run=run.replace('epochs_requested=10,test_used=False', 'seed=SEED,epochs_requested=10,test_used=False')
    put(3,run)
    return nb


if __name__=='__main__':
    for seed in (43,44):
        path=HERE/f'CSA_PER_CAPTION_H5_SEED{seed}_XE10_2GPU.ipynb'
        path.write_text(json.dumps(notebook(seed),indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
        print(path.name)
