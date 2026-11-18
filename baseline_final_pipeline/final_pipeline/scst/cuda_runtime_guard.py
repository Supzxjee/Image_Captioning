"""Metadata-only CUDA preflight; Torch is imported only in the user-run probe."""
import argparse
import json
from pathlib import Path


def validate_runtime(info):
    if not info['cuda_build'] or not info['set_device_symbol']:
        raise RuntimeError('Loaded PyTorch lacks CUDA support. Select Accelerator GPU T4 x2 and restart the Kaggle session. If T4 x2 was already selected, send cuda_runtime.json; do not blindly reinstall Torch.')
    if not info['cuda_available'] or info['device_count']!=2:
        raise RuntimeError('Exactly two CUDA GPUs must be visible. Select GPU T4 x2, restart the session and check Save Version settings. See cuda_runtime.json.')
    if not info['nccl_available']:raise RuntimeError('This PyTorch runtime lacks NCCL. Send cuda_runtime.json before changing packages.')


def check_runtime(output,rank=None):
    import torch
    import torch.distributed as dist
    info=dict(torch_version=str(torch.__version__),torch_file=str(torch.__file__),cuda_build=torch.version.cuda,
        set_device_symbol=hasattr(torch._C,'_cuda_setDevice'),cuda_available=bool(torch.cuda.is_available()),
        device_count=int(torch.cuda.device_count()),nccl_available=bool(dist.is_nccl_available()),models_executed=False)
    path=Path(output)/('cuda_runtime.json' if rank is None else f'cuda_runtime_rank{rank}.json')
    path.write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
    print('CUDA runtime:',json.dumps(info),flush=True)
    validate_runtime(info)
    print('CUDA runtime PASS: two GPUs and NCCL; no model loaded.',flush=True)
    return info


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();check_runtime(a.output)
