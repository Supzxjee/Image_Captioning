"""Data-only contracts for selected fresh-XE models and one full image epoch."""
from pathlib import Path
import shutil
import numpy as np
from real_bank_data import read,save,sha
from full_cache_data import stage_full
from full_xe_data import validation_references
from scst_full_data import full_schedule,local_batch,image_events,UPDATES,TRAIN_IMAGES
from scst_pilot_data import event_schedule,DATA_SHA,MODEL_SHA
from prompt_scratch_data import checked_cache,checked_join

SOURCES={
 'CONTROL':dict(version='ducanh2403/2-vi-cai-kia-qua-nhieu/178',epoch=10,
   sha256='3dbc24409259f2e1a062c674fd5cec495e12aa168bcc1b747c4c5fc1beec1c46',cider=.9945869315089895),
 'PROMPT':dict(version='ducanh3/notebook7bc83c4c76/58',epoch=9,
   sha256='371f8d8ad4635617d0ca2a111aeba1fa81894631475d629ef81c31a71f26dc96',cider=.9884528187965552)}
DETECTION_SHA='498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c'
PHASES={'original':0,'end':UPDATES}
EVAL_NAMES=('val_rank0.json','val_rank1.json','candidates_rank0.json','candidates_rank1.json','worker0.json','worker1.json','cider_replay.json')


def source_checkpoint(root,arm):
    expected=SOURCES[arm]['sha256'];root=Path(root)
    choices=[root] if root.is_file() else sorted(root.rglob('*.bin'))
    matches=[p for p in choices if p.name in ('best.bin','checkpoint.bin') and sha(p)==expected]
    if not matches:raise ValueError('Need selected '+arm+' best.bin from '+SOURCES[arm]['version']+'; SHA '+expected)
    return matches[0]


def check_source(ck,arm,stage,prompt_files):
    identity=ck['identity'];epoch=SOURCES[arm]['epoch'];state=ck['state']
    if identity['schema']!='csa_prompt_scratch_xe10_v1' or identity['arm']!=arm or identity['epochs']!=10:
        raise ValueError('Not the selected fresh-XE arm')
    if state['updates']!=epoch*17711 or state['caption_events']!=epoch*566747 or state['best_epoch']!=epoch:
        raise ValueError('Wrong source XE cursor')
    if identity['stage_sha256']!=stage or identity['prompt_files_sha256']!=prompt_files:
        raise ValueError('Source/staged ROI or prompt175 identities differ')
    if not identity['feature_sharing'] or identity['common_init_sha256']!='d6dad13dc69bf5fe9355e525e5240b518082d897ca82a53db4f71e37966c9791':
        raise ValueError('Wrong fresh-XE initialization')
    if not ck['model'] or any(not k.startswith('model.') for k in ck['model']):raise ValueError('Wrong ScratchCaptioner state format')


def validate_state(state):
    step=state['updates'];phases=state['validated']
    if type(step)!=int or state['image_events']!=image_events(step) or phases!=list(PHASES)[:len(phases)]:raise ValueError('Invalid SCST cursor/phases')
    if step and 'original' not in phases:raise ValueError('Missing ORIGINAL before training')
    if 'end' in phases and step!=UPDATES:raise ValueError('Premature END')
    expected={f'evaluations/{p}/{n}' for p in phases for n in EVAL_NAMES}
    if set(state['validation_files'])!=expected:raise ValueError('Saved evaluation coverage differs')


def copy_evaluations(root,out,files):
    for name,h in files.items():
        parts=Path(name).parts
        if len(parts)!=3 or parts[0]!='evaluations' or parts[1] not in PHASES or parts[2] not in EVAL_NAMES:raise ValueError('Unsafe evaluation path')
        source=Path(root)/name;target=Path(out)/name
        if sha(source)!=h:raise ValueError('Missing/changed resume evaluation:'+name)
        target.parent.mkdir(parents=True,exist_ok=True)
        if source.resolve()!=target.resolve():shutil.copy2(source,target)


def validate_candidates(rows,expected):
    from saved_quality_data import check_rows
    check_rows([dict(coco_id=r['coco_id'],caption=r['candidates'][0]['caption']) for r in rows],expected)
    for row in rows:
        beams=row['candidates']
        if len(beams)!=5 or any(not isinstance(c['caption'],str) or not np.isfinite(c['logprob']) for c in beams):raise ValueError('Need5finite beams')
        if [c['logprob'] for c in beams]!=sorted((c['logprob'] for c in beams),reverse=True):raise ValueError('Unsorted beam scores')


def prepare(cfg):
    arm=cfg['arm'];out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    if cfg['updates']!=UPDATES:raise ValueError('Only one full TRAIN epoch approved')
    checkpoint=source_checkpoint(cfg['checkpoint_root'],arm)
    if sha(cfg['dataset_json'])!=DATA_SHA:raise ValueError('Changed COCO JSON')
    refs={};held=set()
    for r in read(cfg['dataset_json'])['images']:
        i=int(Path(r['filename']).stem.split('_')[-1])
        if r['split'] in ('train','restval'):
            if i in refs:raise ValueError('Duplicate TRAIN ID')
            refs[i]=[s['raw'] for s in r['sentences'][:5]]
        else:held.add(i)
    if len(refs)!=TRAIN_IMAGES or set(refs)&held or any(len(v)!=5 for v in refs.values()):raise ValueError('TRAIN reward coverage differs')
    data,vocab,stage=stage_full(cfg['full_root'],cfg['stage'])
    root,links,_,_,inventory=checked_cache(cfg['prompt_root']);checked_join(data,links,read(root/'cohort.json'))
    _,ids=full_schedule(list(refs));events=event_schedule(data,ids);present=ids>=0
    if not np.array_equal(events>=0,present) or not np.array_equal(data['ids'][data['caption_image'][events[present]]],ids[present]):raise ValueError('TRAIN schedule join differs')
    save(out/'training_references.json',{str(i):v for i,v in refs.items()})
    save(out/'validation_references.json',validation_references(cfg['full_root']));save(out/'vocabulary.json',vocab)
    np.save(out/'events.npy',events);np.save(out/'coco_ids.npy',ids)
    names=('training_references.json','validation_references.json','vocabulary.json','events.npy','coco_ids.npy')
    save(out/'prepared.json',dict(status='PROMPT_SCST_PREPARED',arm=arm,source=SOURCES[arm],checkpoint=str(checkpoint),
        files_sha256={n:sha(out/n) for n in names},stage_sha256=stage['files_sha256'],prompt_files=inventory['cache_files'],
        updates=UPDATES,image_events=TRAIN_IMAGES,test_used=False))
    print('SCST staged:',arm,UPDATES,'updates /',TRAIN_IMAGES,'unique TRAIN / final4+3; no TEST.',flush=True)
