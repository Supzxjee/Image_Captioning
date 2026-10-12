"""Retrieve checkpoint from Kaggle notebook version 194 into independent output."""
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch

SOURCE_NOTEBOOK = 'ducanh2403/kltn-mscoco/194'
KAGGLE_CLI_COMMIT = '9c57dd74c29904c40ff7879eed33644be89d2e2f'
DOWNLOAD = Path('/tmp/e3_checkpoint_version_194')
RELATIVE_CHECKPOINT = Path('pair_grounded_relation_32q_2l_dual_gpu_3ep/checkpoints/model_h1_2_crossattn_epoch_3.pth')
SOURCE = DOWNLOAD / RELATIVE_CHECKPOINT
OUTPUT = Path('/kaggle/working/e3_checkpoint_store')
DESTINATION = OUTPUT / SOURCE.name

if not SOURCE.is_file():
    environment = os.environ.copy()
    if not environment.get('KAGGLE_API_TOKEN'):
        from kaggle_secrets import UserSecretsClient
        try:
            environment['KAGGLE_API_TOKEN'] = UserSecretsClient().get_secret('KAGGLE_API_TOKEN')
        except Exception:
            raise RuntimeError(
                'Add-ons > Secrets: add and enable KAGGLE_API_TOKEN from '
                'https://www.kaggle.com/settings/api. Enable Internet and run again.'
            ) from None
    # PyPI kaggle 2.2.4 parses /194 but ignores it in output/files API requests.
    # Pin official upstream code that sends version_label='v194' instead.
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-q', '--force-reinstall', '--no-deps',
        f'https://github.com/Kaggle/kaggle-cli/archive/{KAGGLE_CLI_COMMIT}.zip',
    ], check=True)
    subprocess.run([
        sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'kagglesdk>=0.1.37,<1.0',
    ], check=True)
    subprocess.run([
        sys.executable, '-c',
        'import inspect; from kaggle.api.kaggle_api_extended import KaggleApi; '
        'assert "request.version_label" in inspect.getsource(KaggleApi.kernels_output), '
        '"Installed CLI ignores output version"; '
        'assert "request.version_label" in inspect.getsource(KaggleApi.kernels_list_files), '
        '"Installed CLI ignores files version"; '
        'print("Kaggle CLI version forwarding PASS", flush=True)',
    ], env=environment, check=True)
    executable = shutil.which('kaggle')
    assert executable, 'Kaggle CLI executable was not found after installation.'
    DOWNLOAD.mkdir(parents=True, exist_ok=True)
    print('OUTPUT FILES REPORTED BY KAGGLE FOR VERSION 194:', flush=True)
    subprocess.run([
        executable, 'kernels', 'files', SOURCE_NOTEBOOK, '--csv', '--page-size', '200',
    ], env=environment, check=True)
    print('Downloading ONLY E3 epoch-3 checkpoint from version 194...', flush=True)
    subprocess.run([
        executable, 'kernels', 'output', SOURCE_NOTEBOOK, '-p', str(DOWNLOAD),
        '--file-pattern', r'model_h1_2_crossattn_epoch_3[.]pth$',
    ], env=environment, check=True)
    environment.pop('KAGGLE_API_TOKEN', None)

if not SOURCE.is_file():
    flat_source = DOWNLOAD / RELATIVE_CHECKPOINT.name
    if flat_source.is_file():
        SOURCE = flat_source

if not SOURCE.is_file():
    print('LOCAL DOWNLOAD DIRECTORY CONTENTS:', flush=True)
    for entry in DOWNLOAD.iterdir():
        print(entry.name, flush=True)

assert SOURCE.is_file(), (
    f'Checkpoint was not downloaded to a known location: {SOURCE}. '
    'Send OUTPUT FILES REPORTED BY KAGGLE FOR VERSION 194 and the download log. '
    'Do not switch to latest version; we must verify the actual version-194 filename.'
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
    'kaggle_cli_commit': KAGGLE_CLI_COMMIT,
    'source_notebook': SOURCE_NOTEBOOK, 'source_version': 194,
    'source': str(SOURCE), 'checkpoint': DESTINATION.name,
    'bytes': total, 'sha256': destination_hash.hexdigest(), 'signature': signature,
    'purpose': 'Independent E3 checkpoint output for diagnostic and evaluation notebooks',
}
(OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('\nE3 CHECKPOINT STORE COMPLETE', flush=True)
print(json.dumps(manifest, indent=2), flush=True)
print('Save Version output:', OUTPUT, flush=True)
