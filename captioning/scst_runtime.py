"""Opt-in numeric controls and trace identities; no training on import."""
import hashlib
from importlib import metadata
import os
import platform
import sys
import torch


def configure_strict_runtime(enabled):
    if not enabled: return
    workspace = ':4096:8'
    current = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if current != workspace and torch.cuda.is_initialized():
        raise RuntimeError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization.')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = workspace
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # A single attention backend also limits no-grad/grad policy-path differences.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'): torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.mha.set_fastpath_enabled(False)


def tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((str(value.dtype), tuple(value.shape))).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def model_digest(model):
    digest = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        digest.update(key.encode()); digest.update(tensor_digest(value).encode())
    return digest.hexdigest()


def runtime_manifest():
    packages = {}
    for name in ('numpy', 'torchvision', 'transformers', 'nltk', 'h5py'):
        try: packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError: packages[name] = None
    gpu = None
    if torch.cuda.is_available():
        gpu = {'name': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0))}
    return {'python': sys.version, 'platform': platform.platform(), 'torch': str(torch.__version__),
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(), 'gpu': gpu, 'packages': packages,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark, 'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'mha_fastpath': torch.backends.mha.get_fastpath_enabled(),
            'sdp_math': torch.backends.cuda.math_sdp_enabled(), 'sdp_flash': torch.backends.cuda.flash_sdp_enabled(),
            'sdp_mem_efficient': torch.backends.cuda.mem_efficient_sdp_enabled(),
            'cublas_workspace': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'python_hash_seed': os.environ.get('PYTHONHASHSEED'),
            'scope': 'Controls reduce known nondeterminism; no cross-device/release bitwise guarantee.'}
