"""Stage only frozen TRAIN-only v130 banks; no prompt cache or model calls."""
from pathlib import Path
import time
import numpy as np
from real_bank_data import read, save, sha
from semantic_data import fixed_banks


def stage_banks(cfg, deadline):
    if not cfg.get('bank_alignment') or cfg.get('semantic') or 'prompt_root' in cfg:
        raise ValueError('Expected M2-only, without image-specific prompt staging')
    root = Path(cfg['stage'])
    meta = read(root/'manifest.json')
    if meta.get('semantic') or meta.get('bank_alignment'):
        raise ValueError('Expected a fresh visual stage before bank staging')
    banks, package, manifest = fixed_banks(cfg['bank_root'])
    seals = {}
    for kind, rows in [('noun', 5269), ('verb', 1594)]:
        if time.monotonic() > deadline:
            raise TimeoutError('Bank-only staging deadline')
        if banks[kind].shape != (rows, 512):
            raise ValueError('Changed v130 bank dimensions')
        name = 'sem_'+kind+'_bank.npy'
        np.save(root/name, banks[kind])
        seals[name] = sha(root/name)
    meta['files_sha256'].update(seals)
    meta['bank_alignment'] = dict(method='bank_only_v130', bank_cache=package['files'],
        text_model=manifest['model'], text_revision=manifest['revision'],
        TRAIN_only=True, image_specific_prompts=False, files_sha256=seals)
    save(root/'manifest.json', meta)
    save(Path(cfg['output'])/'stage_summary.json', meta)
    save(Path(cfg['output'])/'bank_stage.json', meta['bank_alignment'])
    print('M2-only banks sealed; no prompt/ROI/detector staging.', flush=True)
