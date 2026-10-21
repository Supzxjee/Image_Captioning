"""Loader/buffer adaptation of the pinned, unchanged CSA forward equations.

Reference: CrossmodalGroup/CSA-Net, models/caption/csa_net.py at 0cc570b.
Reference code and banks are downloaded from the authors, not vendored here.
"""
import ast
import hashlib
from pathlib import Path
from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F

REFERENCE_SHA = 'd6bc6d8072c03cb7d42bc8ffcc48e48cefb03b1b128e081b608902d845b6ded1'
BANKS = {
    'coco_nouns_emb.pth': ('9d2a783f6db5c5afa055484bd41a86ba1feb5365af791dbda8875c5687479ed2', (4764, 1024)),
    'coco_verbs_emb.pth': ('7fc0eb7812bbbe0a004490330318dd64a0ea1f61e7a00dc3102a0fdf61bb40fb', (975, 1024)),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_banks(directory):
    banks = {}
    for name, (expected, shape) in BANKS.items():
        path = Path(directory) / name
        if sha(path) != expected:
            raise ValueError('Bank hash differs: ' + name)
        tensor = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(tensor, Tensor) or tuple(tensor.shape) != shape or tensor.dtype != torch.float32:
            raise ValueError('Bank shape/type differs: ' + name)
        if not bool(torch.isfinite(tensor).all()) or tensor.requires_grad:
            raise ValueError('Bank must be finite and frozen: ' + name)
        banks[name] = tensor
    return banks


def reference_class(path, banks):
    """Retain author definitions; bridge only the two constructor torch.load calls.

    Extracting definitions avoids importing matplotlib/package detector dependencies.
    The bridge maps the authors' saved CUDA tensor to CPU before .to(device).
    No forward, temperature, gate, initialization or attention AST is changed.
    """
    if sha(path) != REFERENCE_SHA:
        raise ValueError('Pinned reference source differs.')
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    names = {'CascadeSemanticAlignmentNetwork', 'TransformerEncoderLayer_SelfAtten',
             'TransformerEncoderLayer_CrossAtten', '_get_activation_fn'}
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    if {n.name for n in nodes} != names:
        raise ValueError('Reference definitions missing.')
    changes = 0
    class LoadBridge(ast.NodeTransformer):
        def visit_Call(self, node):
            nonlocal changes
            self.generic_visit(node)
            if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'torch' and node.func.attr == 'load':
                changes += 1
                node.func = ast.copy_location(ast.Name(id='_load_bank', ctx=ast.Load()), node.func)
            return node
    module = ast.Module(body=nodes, type_ignores=[])
    module = ast.fix_missing_locations(LoadBridge().visit(module))
    if changes != 2:
        raise ValueError('Expected exactly two constructor bank loads.')
    namespace = dict(torch=torch, nn=nn, F=F, Tensor=Tensor, Optional=Optional,
                     _load_bank=lambda name: banks[Path(name).name].clone())
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['CascadeSemanticAlignmentNetwork']


class CSACore(nn.Module):
    def __init__(self, reference_path, banks):
        super().__init__()
        self.core = reference_class(reference_path, banks)()
        for name in ('object_weight', 'relation_weight'):
            tensor = getattr(self.core, name)
            delattr(self.core, name)
            self.core.register_buffer(name, tensor, persistent=True)

    def forward(self, queries):
        if queries.ndim != 3 or queries.shape[-1] != 512:
            raise ValueError('Expected (B,N,512) Q-Former queries.')
        # Returned author mask is retained for parity, NOT used as padding validity.
        return self.core({'reg_feat': queries})
