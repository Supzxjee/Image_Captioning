"""Paste into SAME Kaggle session after the missing-generator failure; no training."""
import ast
import hashlib
import json
import urllib.request

_fix_commit='8a93845a15add168e995aa9abca80df32aa6f6ab'
_fix_name='make_grounding_pilot_notebooks.py'
_fix_sha='1eb7476a58fecc57ec9d50607d38e08d2e3e251c9d36ef52987bd0ba60bd6035'
if globals().get('COMMIT')!=_fix_commit or 'SOURCE' not in globals() or 'OUTPUT' not in globals():
    raise RuntimeError('This cell is only for the original XE/GROUND notebook session; otherwise use the updated notebook.')
if (OUTPUT/'checkpoint.bin').exists():
    raise RuntimeError('A training checkpoint exists; preserve it and use the normal resume workflow.')
_fix_payload=urllib.request.urlopen('https://raw.githubusercontent.com/Supzxjee/Image_Captioning/'+_fix_commit+'/experiments/csa_paper/'+_fix_name,timeout=90).read()
if hashlib.sha256(_fix_payload).hexdigest()!=_fix_sha:raise ValueError('Recovery source hash differs')
ast.parse(_fix_payload.decode('utf-8'))
(SOURCE/_fix_name).write_bytes(_fix_payload)
if (OUTPUT/'failure.txt').is_file():
    _fix_previous=OUTPUT/'pre_packaging_fix_failure.txt'
    if _fix_previous.exists():raise RuntimeError('Previous failure already preserved; do not overwrite it.')
    (OUTPUT/'failure.txt').rename(_fix_previous)
(OUTPUT/'packaging_recovery.json').write_text(json.dumps(dict(source_commit=_fix_commit,added_file=_fix_name,sha256=_fix_sha,training_executed_by_recovery=False),indent=2),encoding='utf-8')
print('Added verified AST-test source only. Run the EXISTING final execution cell again; do not rerun setup or change data/loss.',flush=True)
