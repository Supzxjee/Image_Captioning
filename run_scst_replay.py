"""Manual same-GPU, fresh-process repeat of the first 32 SCST+CE updates."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import zipfile


def compare_runs(first, second):
    """Exact replay comparison; elapsed time alone is excluded."""
    def read(root, name): return json.loads((Path(root)/name).read_text(encoding='utf-8'))
    differences = []
    for name in ('runtime.json', 'initial_model_identity.json', 'replay_summary.json'):
        if read(first, name) != read(second, name): differences.append({'artifact': name})
    for side, root in [('first', first), ('second', second)]:
        runtime = read(root, 'runtime.json')
        if not (runtime.get('deterministic_algorithms') and runtime.get('sdp_math')
                and not runtime.get('warn_only') and runtime.get('cublas_workspace') == ':4096:8'
                and runtime.get('gpu')):
            differences.append({'side': side, 'error': 'Strict CUDA runtime not verified'})
        identity, summary = read(root, 'initial_model_identity.json'), read(root, 'replay_summary.json')
        if (not identity.get('state_sha256') or summary.get('steps') != 32
                or summary.get('train_images_for_cider_idf') != 20000
                or not summary.get('final_model_sha256') or not summary.get('optimizer_sha256')):
            differences.append({'side': side, 'error': 'Missing replay state identities or incorrect budget/IDF'})
    a, b = read(first, 'history.json'), read(second, 'history.json')
    if len(a) != 32 or len(b) != 32: differences.append({'artifact': 'history.json', 'error': 'Expected 32 updates each'})
    required = {'step', 'loss', 'ce_loss', 'scst_loss', 'grad_norm', 'sample_cider_reward',
                'greedy_cider_reward', 'batch_indices_sha256', 'visual_sha256', 'captions_sha256',
                'prompts_sha256', 'caption_masks_sha256', 'prompt_masks_sha256', 'sample_tokens_sha256',
                'greedy_tokens_sha256', 'model_after_sha256', 'rng_before_sha256', 'rng_after_sha256'}
    for step, (left, right) in enumerate(zip(a, b), 1):
        for side, record in [('first', left), ('second', right)]:
            missing = sorted(required-set(record))
            if missing: differences.append({'step': step, 'side': side, 'missing': missing})
            if record.get('step') != step: differences.append({'step': step, 'side': side, 'error': 'Invalid step'})
            if any(not math.isfinite(v) for v in record.values() if isinstance(v, (int, float))):
                differences.append({'step': step, 'side': side, 'error': 'Nonfinite numeric field'})
        keys = (set(left) | set(right))-{'seconds'}
        changed = sorted(k for k in keys if k not in left or k not in right or left[k] != right[k])
        if changed: differences.append({'step': step, 'changed_fields': changed})
    return {'status': 'PASS' if not differences else 'FAIL', 'exact_replay': not differences,
            'expected_steps': 32, 'differences': differences, 'ignored_fields': ['seconds'],
            'scope': 'Same GPU, same runtime, first 32 updates only; no caption quality or full-run guarantee.'}


def run(args):
    from run_scst_comparison import cohorts
    from run_hoi_caption_validation import sha, save
    from captioning.visual_cache import VisualCache
    repo = Path(__file__).resolve().parent
    output = Path(args.output).resolve()
    if output.exists(): raise FileExistsError(f'Choose a fresh output: {output}')
    paths = {key: str(Path(getattr(args, key)).resolve()) for key in
             ('checkpoint', 'prompt_cache_path', 'dataset_json_path', 'base_path', 'visual_cache')}
    expected = json.loads((repo/'audit/hoi_validation_input_identity.json').read_text())['input_sha256']
    hashes = {}
    for key in ('checkpoint', 'prompt_cache_path', 'dataset_json_path'):
        if not Path(paths[key]).is_file(): raise FileNotFoundError(paths[key])
        print('Checking identity:', key, flush=True)
        hashes[key] = sha(paths[key])
        if hashes[key] != expected[key]: raise ValueError(f'Input identity mismatch: {key}')
    for key in ('base_path', 'visual_cache'):
        if not Path(paths[key]).is_dir(): raise FileNotFoundError(paths[key])
    groups = cohorts(json.loads(Path(paths['dataset_json_path']).read_text(encoding='utf-8')), 20000, 1000)
    cache = VisualCache([paths['visual_cache']], id_key='coco_id')
    try: cache.require_ids([r['coco_id'] for r in groups['train']], 'Replay training cohort')
    finally: cache.close()
    cfg = {**paths, 'groups': groups, 'batch_size': 8, 'lr': 5e-6, 'seed': 42,
           'ce_anchor_comparison': True, 'scst_weight': .05, 'ce_weight': 1.,
           'strict_reproducibility': True, 'replay_steps': 32, 'input_sha256': hashes,
           'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
           'variant': 'same_gpu_scst_ce_replay_32', 'test_used': False,
           'reward': 'Unchanged CIDEr-D with IDF from all selected 20000 train images; no object reward',
           'evaluation': 'Exact replay only; no validation caption generation or metric-based selection'}
    output.mkdir(parents=True)
    save(output/'protocol.json', cfg)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '0', 'PYTHONUNBUFFERED': '1',
           'HDF5_USE_FILE_LOCKING': 'FALSE', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONHASHSEED': '42'}
    try:
        for repeat in ('repeat_1', 'repeat_2'):
            folder = output/repeat
            folder.mkdir()
            protocol = folder/'protocol.json'
            save(protocol, {**cfg, 'output': str(folder)})
            print(f'\n{repeat}: 32 updates on the SAME GPU 0, fresh process', flush=True)
            with (folder/'worker.log').open('w', encoding='utf-8') as log:
                with subprocess.Popen([sys.executable, '-u', str(repo/'run_scst_comparison.py'),
                        '--worker', 'scst_ce', '--protocol', str(protocol)], cwd=repo, env=env,
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as worker:
                    for line in worker.stdout:
                        print(line, end='', flush=True); log.write(line); log.flush()
                    if worker.wait(): raise RuntimeError(f'{repeat} failed. See worker.log; no deterministic fallback.')
        result = compare_runs(output/'repeat_1/scst_ce', output/'repeat_2/scst_ce')
        save(output/'replay_comparison.json', result)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        save(output/'failure.json', {'error': repr(error), 'status': 'INCOMPLETE', 'exact_replay': False})
        raise
    finally:
        archive = str(output)+'_review.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
            for file in sorted(output.rglob('*')):
                if file.is_file() and file.suffix in ('.json', '.log'):
                    bundle.write(file, file.relative_to(output))
        print('RETURN FILE:', archive, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('checkpoint', 'prompt-cache-path', 'dataset-json-path', 'base-path', 'visual-cache', 'output'):
        parser.add_argument('--'+flag, required=True)
    run(parser.parse_args())
