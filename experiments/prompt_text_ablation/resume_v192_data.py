"""Exact v192 output verifier, stdlib only; never imports a model or optimizer."""
import hashlib
import json
from pathlib import Path


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_resume(root, locked):
    root=Path(root)
    receipts=sorted(root.rglob('checkpoint_receipt.json'))
    if len(receipts)!=1:
        raise ValueError('Need one exact v192 output folder containing latest checkpoint receipt')
    folder=receipts[0].parent
    for name,expected in locked['metadata_sha256'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe metadata filename')
        if not (folder/name).is_file() or digest(folder/name)!=expected:
            raise ValueError('Missing/changed v192 metadata: '+name)
    read=lambda name:json.loads((folder/name).read_text(encoding='utf-8'))
    receipt,best,state,identity=(read(n) for n in ('checkpoint_receipt.json','best_receipt.json','progress.json','identity.json'))
    if identity!=locked['identity'] or identity['seed']!=42 or identity.get('prompt_text_mode')!='zero':
        raise ValueError('Wrong source, seed, zero-text method, or training identity')
    cursor={k:state[k] for k in ('epoch','batch_cursor','phase','updates','caption_events','best_epoch')}
    expected=dict(epoch=8,batch_cursor=17456,phase='train',updates=141433,caption_events=4525821,best_epoch=7)
    if cursor!=expected or len(state['history'])!=7:
        raise ValueError('Wrong v192 cursor; never restart the arm from a best-only snapshot')
    if receipt['updates']!=141433 or receipt['sha256']!=locked['checkpoint_sha256']:
        raise ValueError('Wrong latest v192 checkpoint receipt')
    if best['epoch']!=7 or best['sha256']!=locked['best_sha256'] or best['CIDEr']!=state['best_CIDEr']:
        raise ValueError('Wrong saved best7 receipt')
    for name,key in (('checkpoint.bin','checkpoint_sha256'),('best.bin','best_sha256')):
        if not (folder/name).is_file() or digest(folder/name)!=locked[key]:
            raise ValueError('Missing/changed exact v192 weights: '+name)
    if digest(folder/'training.jsonl')!=receipt['training_log_sha256']:
        raise ValueError('Changed original training log')
    for row in state['history']:
        path=folder/'evaluations'/f"epoch_{row['epoch']:02d}"
        for name,expected in row['files_sha256'].items():
            if Path(name).name!=name or digest(path/name)!=expected:
                raise ValueError('Changed completed validation history')
    return folder
