# USER-RUN: insert a new Kaggle cell, run it, then rerun ONLY the original last cell.
import os, sys, json, hashlib
from pathlib import Path
from kaggle_secrets import UserSecretsClient
from prompt_text_cache import checked_source, RECEIPT_SHA

if any(k not in globals() for k in ('SOURCE_ROOT','SOURCE','OUTPUT','run','SECRET_NAME')):
    raise RuntimeError('Need the existing failed notebook session; otherwise import the updated V2 notebook.')
if (OUTPUT/'cache_receipt.json').exists() or (OUTPUT/'cache_manifest.json').exists():
    raise RuntimeError('Cache/model phase already produced artifacts; preserve them, do not restart automatically.')
old_output=OUTPUT
new_output=Path('/kaggle/working/csa_prompt_text_cache_v2')
if new_output.exists():raise RuntimeError('Preserve existing V2 output; do not overwrite.')
receipts=[p for p in Path(SOURCE_ROOT).rglob('receipt.json') if hashlib.sha256(p.read_bytes()).hexdigest()==RECEIPT_SHA]
if len(receipts)!=1:raise RuntimeError('Need the exact already-downloaded167 receipt.')
root=receipts[0].parent
receipt=json.loads(receipts[0].read_text(encoding='utf-8'))
expected=receipt['payload_sha256']['review_examples.json']
missing=root/'review_examples.json'
if not missing.exists():
    token=UserSecretsClient().get_secret(SECRET_NAME).strip()
    if not token:raise RuntimeError('Empty API Secret')
    env=os.environ.copy()
    for key in ('KAGGLE_API_TOKEN','KAGGLE_USERNAME','KAGGLE_KEY'):env.pop(key,None)
    env['KAGGLE_API_TOKEN']=token
    run([sys.executable,'-c','from kaggle.cli import main; main()','kernels','output',
         'ducanh2403/2-vi-cai-kia-qua-nhieu/167','-p',str(SOURCE_ROOT),
         '--file-pattern',r'(^|/)review_examples\.json$'],timeout=300,environment=env)
    del token,env
if not missing.is_file() or hashlib.sha256(missing.read_bytes()).hexdigest()!=expected:
    raise RuntimeError('The downloaded review_examples.json does not match the original167 receipt.')
LOCAL_SOURCE_ROOT=str(checked_source(SOURCE_ROOT)[0])
OUTPUT=new_output
print('All original167 payload hashes PASS. Preserved failed output:',old_output,flush=True)
print('Rerun ONLY the original last cell. It will reuse downloaded metadata and write:',OUTPUT,flush=True)
