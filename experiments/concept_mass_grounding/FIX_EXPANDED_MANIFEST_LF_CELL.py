# Paste into ONE new Kaggle cell in the still-live failed evaluation session.
# Then rerun ONLY the final evaluation cell. No model is run by this repair cell.
import hashlib
import json
from pathlib import Path

OLD_MANIFEST_SHA='31559d3f68a133fe9d923489875df21b641b84b8238d3a805ec54de9a99af661'
LF_MANIFEST_SHA='8f68ab9c45ab9c2d11618e837b8d945f20c0c3148c3b1705ff352cdf244a6fab'
UPSTREAM='13daa4f'

if not SOURCE_COMMIT.startswith(UPSTREAM):raise RuntimeError('This cell repairs only the original13daa4f package. Use the updated notebook for a new session.')
if OUTPUT.exists():raise RuntimeError('Preparation already created an output folder. Preserve it and send the log; do not overwrite.')
_manifest=HERE/'EXPANDED_EVAL_MANIFEST.json'
_raw=_manifest.read_bytes()
if hashlib.sha256(_raw).hexdigest()!=LF_MANIFEST_SHA:raise RuntimeError('Unexpected manifest bytes; STOP, not the known LF packaging issue.')
_old_serialization=json.dumps(json.loads(_raw),ensure_ascii=False,indent=2)+'\n'
if hashlib.sha256(_old_serialization.replace('\n','\r\n').encode()).hexdigest()!=OLD_MANIFEST_SHA:raise RuntimeError('Manifest is not byte-equivalent to the originally locked CRLF JSON.')
for _name,_want in SOURCE_ASSETS.items():
    if hashlib.sha256((REPO/_name).read_bytes()).hexdigest()!=_want:raise RuntimeError('Other tracked source changed: '+_name)

_cfg=json.loads(CONFIG.read_text(encoding='utf-8'))
_relative='experiments/concept_mass_grounding/expanded_data.py'
_module=REPO/_relative
_code=_module.read_bytes()
_old_line=("MANIFEST_SHA='"+OLD_MANIFEST_SHA+"'").encode()
_new_line=("MANIFEST_SHA='"+LF_MANIFEST_SHA+"'").encode()
if 'technical_amendment' not in _cfg:
    if _code.count(_old_line)!=1:raise RuntimeError('Unexpected source constant; STOP.')
    _fixed=_code.replace(_old_line,_new_line)
    compile(_fixed,str(_module),'exec')
    _before=CONFIG.with_name(CONFIG.name+'.before_manifest_lf_fix.json')
    with _before.open('x',encoding='utf-8') as _f:_f.write(CONFIG.read_text(encoding='utf-8'))
    _module.write_bytes(_fixed)
    SOURCE_ASSETS=dict(SOURCE_ASSETS)
    SOURCE_ASSETS[_relative]=hashlib.sha256(_fixed).hexdigest()
    _cfg['source_assets']=SOURCE_ASSETS
    _cfg['technical_amendment']=dict(kind='manifest_crlf_to_git_lf_pin_only',upstream_commit=SOURCE_COMMIT,
        file=_relative,old_source_sha256=hashlib.sha256(_code).hexdigest(),new_source_sha256=SOURCE_ASSETS[_relative],
        old_manifest_sha256=OLD_MANIFEST_SHA,new_manifest_sha256=LF_MANIFEST_SHA,
        cohort_changed=False,checkpoint_changed=False,evaluation_policy_changed=False)
    CONFIG.write_text(json.dumps(_cfg,indent=2)+'\n',encoding='utf-8')
elif _cfg['technical_amendment'].get('kind')!='manifest_crlf_to_git_lf_pin_only' or _new_line not in _code:
    raise RuntimeError('Unexpected existing amendment; STOP.')
cfg=_cfg
# Preserve the original failed review ZIP; the final cell will write a new one.
REVIEW=OUTPUT.with_name(OUTPUT.name+'_manifest_lf_fix_review.zip')
print('LF manifest pin repaired; exact4000IDs/order and all scientific settings unchanged.',flush=True)
print('Now rerun ONLY the final evaluation cell. No Restart or source/setup rerun needed.',flush=True)
