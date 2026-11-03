"""Explicit Output139→15epoch identity migration; no model imports."""
import hashlib
import json
from real_bank_data import HERE,read
from full_xe_data import validate_state
from run_best9_decode import saved_text


def check_resume_identity(old,current,digest,state):
    validate_state(state,total_epochs=current['epochs'])
    if old==current:return 'identical-current-run'
    raw,_=saved_text();pinned=json.loads(raw['input_identity.json'])
    package=read(HERE/'SAVE_XE139_PACKAGE.json')['files']
    if hashlib.sha256(raw['input_identity.json']).hexdigest()!=package['input_identity.json']:raise ValueError('Changed legacy identity receipt')
    if current['epochs']!=15 or digest!=package['checkpoint.bin'] or old!=pinned:
        raise ValueError('Resume identity differs; only exact latest139 can migrate to15 epochs')
    if (state['epoch'],state['batch_cursor'],state['phase'],state['updates'])!=(11,0,'train',177110):raise ValueError('Not completed10epoch checkpoint')
    excluded={'source','schema','epochs'}
    if {k:v for k,v in old.items() if k not in excluded}!={k:v for k,v in current.items() if k not in excluded}:
        raise ValueError('Resume operational inputs/software differ')
    allowed={'run_full_xe.py','full_xe_data.py'}
    for n,h in old['source'].items():
        if n not in allowed and current['source'].get(n)!=h:raise ValueError('Unapproved inherited source change: '+n)
    return 'exact139-latest10-to15'
