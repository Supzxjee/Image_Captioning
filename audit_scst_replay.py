"""Recheck a user-run replay review ZIP without executing its contents."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile
from run_scst_replay import compare_runs


def audit(source):
    source = Path(source)
    with zipfile.ZipFile(source) as bundle:
        def read(name): return json.loads(bundle.read(name))
        protocol = read('protocol.json')
        expected = json.loads((Path(__file__).parent/'audit/hoi_validation_input_identity.json').read_text())['input_sha256']
        for key in ('checkpoint', 'prompt_cache_path', 'dataset_json_path'):
            if protocol['input_sha256'][key] != expected[key]: raise ValueError(f'Unrecognized input hash: {key}')
        locked = {'code_commit': 'ac672acabbd2c79ab21898129f354a651ac65a3d',
                  'batch_size': 8, 'lr': 5e-6, 'seed': 42, 'scst_weight': .05, 'ce_weight': 1.,
                  'ce_anchor_comparison': True, 'strict_reproducibility': True,
                  'replay_steps': 32, 'test_used': False, 'variant': 'same_gpu_scst_ce_replay_32'}
        for key, value in locked.items():
            if protocol.get(key) != value: raise ValueError(f'Unexpected replay setting: {key}')
        train, val = protocol['groups']['train'], protocol['groups']['val']
        if len(train) != 20000 or len(val) != 1000: raise ValueError('Incorrect cohort lengths')
        for field in ('image_id', 'coco_id'):
            if len({r[field] for r in train+val}) != 21000: raise ValueError(f'Overlapping/duplicate {field}')
        records = []
        with tempfile.TemporaryDirectory() as temporary:
            for repeat in ('repeat_1', 'repeat_2'):
                worker = read(repeat+'/protocol.json')
                if {k: v for k, v in worker.items() if k != 'output'} != protocol:
                    raise ValueError('Worker protocol differs beyond output directory')
                root = Path(temporary)/repeat
                root.mkdir()
                for filename in ('history.json', 'runtime.json', 'initial_model_identity.json', 'replay_summary.json'):
                    (root/filename).write_text(json.dumps(read(repeat+'/scst_ce/'+filename)), encoding='utf-8')
                history = json.loads((root/'history.json').read_text())
                summary = json.loads((root/'replay_summary.json').read_text())
                if summary['final_model_sha256'] != history[-1]['model_after_sha256']:
                    raise ValueError('Final model hash differs from last update')
                if summary['checkpoint_saved'] or not summary['diagnostic_only']:
                    raise ValueError('Unexpected saved checkpoint or nondiagnostic result')
                log_records = [json.loads(line[len('scst_ce '):])
                               for line in bundle.read(repeat+'/worker.log').decode().splitlines()
                               if line.startswith('scst_ce {')]
                if log_records != history: raise ValueError('Worker log differs from recorded history')
                for i, row in enumerate(history):
                    if row['images_seen'] != (i+1)*8: raise ValueError('Incorrect update count')
                    if abs(row['loss']-(row['ce_loss']+.05*row['scst_loss'])) > 1e-6:
                        raise ValueError('Logged objective arithmetic mismatch')
                    if i and row['rng_before_sha256'] != history[i-1]['rng_after_sha256']:
                        raise ValueError('CUDA RNG chain mismatch')
                records.append({'repeat': repeat, 'steps': len(history),
                                'training_trace_seconds': history[-1]['seconds'], **summary})
            result = compare_runs(Path(temporary)/'repeat_1', Path(temporary)/'repeat_2')
        if result != read('replay_comparison.json'): raise ValueError('Recomputed comparison differs from supplied result')
        runtime = read('repeat_1/scst_ce/runtime.json')
        return {'source_file': source.name, 'source_bytes': source.stat().st_size,
                'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'recomputed_comparison': result, 'protocol_sha256': hashlib.sha256(bundle.read('protocol.json')).hexdigest(),
                'protocol_settings_verified': locked, 'cohort_lengths': {'train': len(train), 'val': len(val)},
                'input_sha256': protocol['input_sha256'], 'runtime': runtime, 'runs': records,
                'checks': ['worker protocols differ only in output path', 'disjoint unique cohort IDs',
                           'all logged update records match JSON histories', 'CE+.05SCST arithmetic',
                           'CUDA RNG state continuity', 'final model hash equals step32 hash',
                           'exact 32-step comparison independently recomputed'],
                'limits': ['Review artifacts rechecked locally; original input bytes, model tensors and optimizer tensors not included.',
                           'Same-GPU first32 steps only; no cross-device/full2500-step guarantee.',
                           'No CIDEr/CHAIR quality evaluation; does not prove cause of previous run divergence.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = audit(args.source)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps(result['recomputed_comparison'], indent=2))
