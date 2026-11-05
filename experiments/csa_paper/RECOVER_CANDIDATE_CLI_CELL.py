# USER-RUN CELL: replace the failed final cell; keep existing setup and output.
import ast
import hashlib
import json
import time
import urllib.request
from pathlib import Path

required={'SOURCE','OUTPUT','run','sys','SECRET_NAME','FULL_ROOT','CHECKPOINT_ROOT','KERNEL_VERSION','WALL_HOURS','CONFIG','CHAIR','LOG'}
if not required<=set(globals()):
    raise RuntimeError('Original notebook setup is no longer in memory; use the updated notebook in a fresh runtime.')
if str(OUTPUT)!='/kaggle/working/csa_ce161_candidates_valid5000' or KERNEL_VERSION!='ducanh2403/2-vi-cai-kia-qua-nhieu/161':
    raise RuntimeError('This cell repairs only the CE161 candidate diagnostic.')
if (OUTPUT/'prepared.json').exists() or list(OUTPUT.glob('candidates_rank*')):
    raise RuntimeError('Preparation/decode has already started; preserve that output, do not retry it with this cell.')
url='https://raw.githubusercontent.com/Supzxjee/Image_Captioning/43f48da/experiments/csa_paper/make_candidate_notebook.py'
expected='1af2208030b31297fe97aab729f87bf901bd0a05db9711ca2c76aa29967bb326'
raw=urllib.request.urlopen(url,timeout=90).read()
if hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('Recovery source SHA differs.')
tree=ast.parse(raw.decode('utf-8'))
values=[ast.literal_eval(x.value) for x in ast.walk(tree) if isinstance(x,ast.Assign)
    and any(isinstance(t,ast.Name) and t.id=='execute' for t in x.targets)]
if len(values)!=1:raise ValueError('Expected one pinned replacement execution cell.')
execution=values[0];ast.parse(execution)
for name in ('failure.txt','run_status.json'):
    previous=OUTPUT/name;preserved=OUTPUT/('before_cli_fix_'+name)
    if previous.is_file() and not preserved.exists():preserved.write_bytes(previous.read_bytes())
(OUTPUT/'cli_recovery.json').write_text(json.dumps(dict(source=url,sha256=expected,
    execution_sha256=hashlib.sha256(execution.encode()).hexdigest(),scope='pre-decode transport only; original source_receipt retained'),indent=2),encoding='utf-8')
START=time.monotonic();DEADLINE=START+WALL_HOURS*3600
print('Repairing pre-decode CLI only; exact161/SHA unchanged; original failure preserved.',flush=True)
exec(compile(execution,'<candidate-cli-recovery>','exec'),globals())
