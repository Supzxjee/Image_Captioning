"""Run in a NEW Kaggle notebook attached to the successful E3 TRAIN output."""
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path

import torch

SOURCE = Path('/kaggle/input/notebooks/ducanh2403/kltn-mscoco/pair_grounded_relation_32q_2l_dual_gpu_3ep/checkpoints/model_h1_2_crossattn_epoch_3.pth')
OUTPUT = Path('/kaggle/working/e3_checkpoint_store')
DESTINATION = OUTPUT / SOURCE.name

assert SOURCE.is_file(), (
    f'Khong co checkpoint: {SOURCE}. Notebook MOI phai Add Input version TRAIN E3 '
    'thanh cong, khong phai version validation. Copy File Path va sua SOURCE neu mount khac.'
)
OUTPUT.mkdir(parents=True, exist_ok=True)
assert shutil.disk_usage(OUTPUT).free > SOURCE.stat().st_size + 1024**3, 'Not enough output disk space.'

payload = torch.load(SOURCE, map_location='cpu', weights_only=False)
signature = {key: payload.get(key) for key in (
    'epoch', 'visual_adapter', 'num_visual_queries', 'qformer_layers',
    'grounded_relation_alignment', 'relation_alignment_weight', 'relation_alignment_margin',
)}
assert signature['epoch'] == 3, signature
assert signature['visual_adapter'] == 'qformer', signature
assert signature['num_visual_queries'] == 32, signature
assert signature['qformer_layers'] == 2, signature
assert signature['grounded_relation_alignment'] is True, signature
assert 'encoder_state_dict' in payload and 'decoder_state_dict' in payload
assert payload.get('grounded_relation_metadata', {}).get('variant') == 'instance_pair_grounded_relation_v1'
del payload
gc.collect()
print('E3 checkpoint metadata PASS:', signature, flush=True)

temporary = DESTINATION.with_suffix('.pth.partial')
source_hash = hashlib.sha256()
total, copied = SOURCE.stat().st_size, 0
started, last_log = time.monotonic(), 0
with SOURCE.open('rb') as reader, temporary.open('wb') as writer:
    while True:
        chunk = reader.read(16 * 1024 * 1024)
        if not chunk:
            break
        writer.write(chunk)
        source_hash.update(chunk)
        copied += len(chunk)
        elapsed = time.monotonic() - started
        if elapsed - last_log >= 10 or copied == total:
            print(f'Copy checkpoint: {copied / total:.1%}', flush=True)
            last_log = elapsed
assert copied == total and temporary.stat().st_size == total

destination_hash = hashlib.sha256()
with temporary.open('rb') as reader:
    while True:
        chunk = reader.read(16 * 1024 * 1024)
        if not chunk:
            break
        destination_hash.update(chunk)
assert source_hash.hexdigest() == destination_hash.hexdigest(), 'Copied checkpoint hash mismatch.'
temporary.replace(DESTINATION)

history = SOURCE.parent.parent / 'train_history_h1_2.json'
if history.is_file():
    shutil.copyfile(history, OUTPUT / history.name)

manifest = {
    'source': str(SOURCE), 'checkpoint': DESTINATION.name,
    'bytes': total, 'sha256': destination_hash.hexdigest(), 'signature': signature,
    'purpose': 'Independent E3 checkpoint output for diagnostic and evaluation notebooks',
}
(OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('\nE3 CHECKPOINT STORE COMPLETE', flush=True)
print(json.dumps(manifest, indent=2), flush=True)
print('Save Version output:', OUTPUT, flush=True)
