"""Recover a completed endpoint from saved outputs. No model/HDF5/training imports."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from kv_pilot_data import HERE,digest,save
from kv_pilot_evaluate import evaluate


def recover(source,output,prepared,version,code_commit):
    source,output,prepared=Path(source).resolve(),Path(output).resolve(),Path(prepared).resolve()
    if output.exists() or source==output or source in output.parents:raise FileExistsError('Preserve existing/source output')
    original=json.loads((source/'protocol.json').read_bytes())
    if original['arm']!='D' or version!='ducanh2403/2-vi-cai-kia-qua-nhieu/116':
        raise ValueError('Recovery pins D / Output116; no latest fallback')
    required=['protocol.json','preparation.json','checkpoint.bin']
    for rank in (0,1):
        required += [f'{kind}_rank{rank}.json' for kind in ('worker','C0','end','held','reader_off','intervention')]
    for condition in ('C0','end'):
        required += [f'{condition}_{kind}.json' for kind in ('per_image_cider','chair_input','chair_output')]
    for name in required:
        if not (source/name).is_file():raise ValueError('Missing completed artifact '+name+'; send review ZIP, do not retrain')
    for rank in (0,1):
        worker=json.loads((source/f'worker_rank{rank}.json').read_bytes())
        if worker['status']!='ENDPOINT_500_COMPLETE' or worker['updates']!=500:
            raise ValueError('GPU endpoint not verified complete; recovery does not launch training')
    inventory=json.loads((HERE/'KV_SAVE_V115_PACKAGE.json').read_bytes())['inventory']['files']
    for name in ('metadata.json','auxiliary_facts.json','cpu_preparation.json'):
        if digest(prepared/name)!=inventory[name]:raise ValueError('Prepared metadata mismatch '+name)
    output.mkdir(parents=True)
    hashes={}
    for path in source.iterdir():
        if path.is_file() and (path.suffix in ('.json','.jsonl','.log') or path.name=='checkpoint.bin'):
            shutil.copyfile(path,output/path.name)
            hashes[path.name]=digest(path)
            if digest(output/path.name)!=hashes[path.name]:raise ValueError('Copy mismatch '+path.name)
    recovery=dict(source_kernel_version=version,original_worker_source_commit=original['source_commit'],
        recovery_code_commit=code_commit,original_artifact_sha256=hashes,training_repeated=False,
        model_loaded=False,gpu_used=False,official_cached_scores_reused=True)
    save(output/'recovery_receipt.json',recovery)
    cfg=dict(original,output=str(output),prepared_root=str(prepared),recovery=recovery)
    evaluate(cfg,reuse_scores=True)
    if digest(output/'checkpoint.bin')!=hashes['checkpoint.bin']:raise ValueError('Checkpoint bytes changed')
    print('CPU RECOVERY COMPLETE. No training/model/GPU was run.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--source',required=True);p.add_argument('--output',required=True);p.add_argument('--prepared',required=True)
    p.add_argument('--version',required=True);p.add_argument('--code-commit',required=True)
    a=p.parse_args();recover(a.source,a.output,a.prepared,a.version,a.code_commit)
