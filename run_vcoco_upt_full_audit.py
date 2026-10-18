"""Prepare a frozen UPT relation audit for manual Kaggle execution, two GPUs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

from audit_vcoco_alignment_data import digest
from captioning.vcoco_relation_audit import fit_thresholds, evaluate

UPT = '192743045e4c5f6827299fdc15809f5b97ac1b8e'
POCKET = 'bff287098bb73427fd8d5525b2ac9867fb56629d'
CHECKPOINT = '12f7f84c7c8d5bcc38f53470f57c5cc30ff2f025726e8bcc27addeb0578a258c'


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def run(args):
    repo = Path(__file__).resolve().parent
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Use a new OUTPUT; preserve earlier journals and failure reports.')
    for directory, revision in [(args.upt, UPT), (args.pocket, POCKET),
                                (args.upt / 'detr', '820d17abc697e8dd5f49d4cd1fdbde412545bac5'),
                                (args.upt / 'hicodet', '9dd59c1f6ee2ddc011567765d0bc644858391fea')]:
        if subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=directory, text=True).strip() != revision:
            raise ValueError('Reference revision differs.')
        if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=directory, text=True).strip():
            raise ValueError('Tracked reference source is modified.')
    if digest(args.checkpoint) != CHECKPOINT:
        raise ValueError('Wrong frozen HICO-DET checkpoint.')
    summary_path = args.grounded_data / 'box_join_summary.json'
    summary = json.loads(summary_path.read_text())
    if digest(args.instances) != summary['sources']['coco_subset_sha256']:
        raise ValueError('Instances differ from audited source.')
    records = []
    for group in ('relation_calibration', 'relation_dev'):
        path = args.grounded_data / f'{group}_images.json'
        if digest(path) != summary['output_sha256'][path.name]:
            raise ValueError('Grounded data changed.')
        records += [{**row, 'group': group} for row in json.loads(path.read_text())]
    if len(records) != 3374 or len({r['image_id'] for r in records}) != len(records):
        raise ValueError('Expected 507 calibration + 2867 development distinct images.')
    for row in records:
        path = args.base_path / row['filepath'] / row['filename']
        if not path.is_file():
            raise FileNotFoundError(path)
    import torch
    if torch.cuda.device_count() < 2:
        raise RuntimeError('Select T4 x2; no automatic CPU or single-GPU run.')
    cfg = {'variant': 'vcoco_upt_full_native_score_audit_v2' if args.score_mode == 'native' else 'vcoco_upt_full_native_audit_v1',
           'score_mode': args.score_mode,
           'upt': str(args.upt.resolve()), 'pocket': str(args.pocket.resolve()),
           'checkpoint': str(args.checkpoint.resolve()), 'instances': str(args.instances.resolve()),
           'base_path': str(args.base_path.resolve()), 'output': str(output), 'images': records,
           'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
           'checkpoint_sha256': CHECKPOINT, 'source_summary_sha256': digest(summary_path),
           'instances_sha256': digest(args.instances),
           'policy': 'Per-predicate lowest score with >=90% known precision and >=5 known accepts on Calibration; both known outcomes required; otherwise abstain.',
           'score': ('Official UPT postprocessing scores (sigmoid verb logit times subject/object detector priors), holding/riding only'
                     if args.score_mode == 'native' else 'sigmoid UPT raw verb logit, holding/riding only; native HICO object-verb ontology'),
           'detector': 'UPT native DETR, not the cached YOLO detector in previous coverage report',
           'matching': 'All valid noncrowd GT, class-aware IoU>=.5 one-to-one greedy; matching occurs after native proposals.',
           'unknown': 'Unlinked target, unmatched actor and missing action labels stay unknown; no role-swap negatives.',
           'training': False, 'caption_evaluation': False, 'manual_execution_only': True}
    output.mkdir(parents=True)
    config_path = output / 'protocol.json'
    save(config_path, cfg)
    processes, handles, positions = [], [], [0, 0]
    try:
        for rank in range(2):
            log = (output / f'worker_{rank}.log').open('wb')
            handles.append(log)
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=str(rank), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2')
            processes.append(subprocess.Popen([sys.executable, '-u', str(repo / 'vcoco_upt_audit_worker.py'),
                                                '--config', str(config_path), '--rank', str(rank)],
                                               cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT))
        while True:
            statuses = [p.poll() for p in processes]
            for rank in range(2):
                with (output / f'worker_{rank}.log').open('rb') as stream:
                    stream.seek(positions[rank]); data = stream.read(); positions[rank] = stream.tell()
                if data:
                    print(f'[GPU {rank}] ' + data.decode('utf-8', errors='replace'), flush=True)
            if any(code is not None and code != 0 for code in statuses):
                raise RuntimeError(f'GPU worker failed: {statuses}; see worker logs.')
            if all(code == 0 for code in statuses):
                break
            print('Frozen UPT workers active; no training or DDP.', flush=True)
            time.sleep(15)
        rows = []
        for rank in range(2):
            values = [json.loads(line) for line in (output / f'upt_{rank}.jsonl').read_text().splitlines()]
            if [r['image_id'] for r in values] != [r['image_id'] for r in records[rank::2]]:
                raise ValueError('Incomplete or reordered shard.')
            rows += values
        cal = [r for r in rows if r['group'] == 'relation_calibration']
        dev = [r for r in rows if r['group'] == 'relation_dev']
        policy = fit_thresholds(cal)
        save(output / 'locked_calibration_policy.json', policy)
        result = {'status': 'FROZEN_RELATION_AUDIT_COMPLETE', 'calibration_images': len(cal),
                  'development_images': len(dev), 'policy': policy,
                  'calibration': evaluate(cal, policy), 'development': evaluate(dev, policy),
                  'protocol_sha256': digest(config_path), 'policy_sha256': digest(output / 'locked_calibration_policy.json'),
                  'limits': ['Known-label precision does not establish precision on unknown target substitutions.',
                             'Existing V-COCO development has been inspected historically; not a fresh independent benchmark.',
                             'HICO/COCO pretraining overlap is not exhaustively audited.',
                             'No caption gain, new trained model or novelty claim.'],
                  'training': False, 'caption_metrics': False}
        save(output / 'run_summary.json', result)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        save(output / 'failure.json', {'status': 'INCOMPLETE', 'error': repr(error)})
        raise
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            p.wait()
        for handle in handles:
            handle.close()
        with zipfile.ZipFile(str(output) + '_review.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(output.iterdir()):
                if path.is_file() and path.suffix in ('.json', '.jsonl', '.log'):
                    archive.write(path, path.name)
        print('RETURN:', str(output) + '_review.zip', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--score-mode', choices=['raw', 'native'], default='raw')
    for key in ('grounded-data', 'instances', 'upt', 'pocket', 'checkpoint', 'base-path', 'output'):
        parser.add_argument('--' + key, type=Path, required=True)
    run(parser.parse_args())
