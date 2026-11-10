"""Best8 strength-gated TEST contract; reuse unchanged patch/prompt staging."""
from pathlib import Path
from real_bank_data import read,sha
import test199_data as shared
import semantic_test_data as semantic
from semantic_data import record_arrays

CHECKPOINT_SHA='456681d45b8b9ce8e3d34f3452c5f6b7bebdef21fd2ad3e6f344790ccbe4ccf9'
VERSION='ducanh2403/kltnnnnnff/12'
runtime_policy=semantic.runtime_policy
prompt_inputs=semantic.prompt_inputs


def validate_lock(value):
    base=shared.lock()
    if any(value[k]!=base[k] for k in ('expected_test_ids','test_id_sha256','beam_width','max_new_tokens','length_normalization','reranking')):raise ValueError('Changed fixed TEST cohort/decode')
    if value['source_version']!=VERSION or value['checkpoint_sha256']!=CHECKPOINT_SHA or value['epoch']!=8 or value['updates']!=141688 or value['method']!='bank_prompt_strength_v1':raise ValueError('Need strength best8, not finalepoch10 or previousbankprompt')
    if value['training_commit']!='cc6a20ea7f7f3263a7a637100ab1166dd6ff6a1b':raise ValueError('Wrong strength training source')
    return value


def lock():
    value=shared.lock();value.update(schema='strength_best8_test_v1',source_version=VERSION,checkpoint_sha256=CHECKPOINT_SHA,
        training_commit='cc6a20ea7f7f3263a7a637100ab1166dd6ff6a1b',epoch=8,updates=141688,method='bank_prompt_strength_v1')
    return validate_lock(value)


def checkpoint(root):
    root=Path(root);files=[root] if root.is_file() else sorted(root.rglob('best.bin'))
    matches=[p for p in files if p.is_file() and sha(p)==CHECKPOINT_SHA]
    if not matches:raise ValueError('Need exact12 best8 SHA456681d4; no latest/final/other model fallback')
    return matches[0]


def validate_checkpoint_state(ck):
    s,i=ck['state'],ck['identity'];expected=lock()
    if s['epoch']!=9 or s['batch_cursor']!=0 or s['phase']!='train' or s['updates']!=141688 or s['caption_events']!=4533976 or s['best_epoch']!=8 or len(s['history'])!=8:raise ValueError('Wrong best8 checkpoint cursor')
    if i['source']['commit']!=expected['training_commit'] or i['method']!=expected['method'] or i['epochs']!=10 or i['global_batch']!=32:raise ValueError('Wrong strength checkpoint identity')
    if not {'noun_bank','verb_bank','strength_gate.weight','strength_gate.bias'}<=set(ck['model']):raise ValueError('Missing frozen banks or learned strength gate')
    return expected


def prepare(cfg):
    rows,texts,vectors,seals=prompt_inputs(cfg)
    for row in rows:record_arrays(row,texts)
    shared.prepare(cfg,locked=lock(),weight_path=checkpoint(cfg['checkpoint_root']))
    semantic.stage_prompts(cfg,rows,texts,vectors,seals)


def load(cfg,verify=True):
    result=shared.load(cfg,verify=verify,validator=validate_lock)
    if result[2].get('method')!='bank_prompt_patch_v1':raise ValueError('Incomplete semantic TEST inputs')
    return result
