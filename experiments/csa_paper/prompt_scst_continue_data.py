"""Data contracts for one additional epoch from the exact PROMPT60 checkpoint."""
from pathlib import Path
import shutil
import numpy as np
from real_bank_data import read,save,sha
from full_cache_data import stage_full
from full_xe_data import validation_references
from scst_full_data import full_schedule,local_batch,image_events,UPDATES,TRAIN_IMAGES,MID
from scst_pilot_data import event_schedule,DATA_SHA
from prompt_scratch_data import checked_cache,checked_join
from prompt_scst_data import EVAL_NAMES,validate_candidates,DETECTION_SHA

SOURCE=read(Path(__file__).with_name('PROMPT60_SOURCE.json'))
SOURCES={'PROMPT':dict(version='ducanh3/notebook7bc83c4c76/60',
 sha256='be2759b4a6f57af9bf8baeb13ee097bae2d8703d265b0791d9d39bb9c3dcbc63',
 cider=1.0583651905121678,xe_epoch=9,completed_scst_epochs=1)}
PHASES={'original':0,'mid':MID,'end':UPDATES}

def source_checkpoint(root,arm):
    if arm!='PROMPT':raise ValueError('Only PROMPT continuation approved')
    root=Path(root);choices=[root] if root.is_file() else sorted(root.rglob('checkpoint.bin'))
    matches=[p for p in choices if sha(p)==SOURCES[arm]['sha256']]
    if not matches:raise ValueError('Need exact PROMPT60 checkpoint.bin SHA '+SOURCES[arm]['sha256'])
    return matches[0]

def check_source(ck,arm,stage,prompt_files):
    if arm!='PROMPT' or ck['identity']!=SOURCE['identity.json']:raise ValueError('Wrong PROMPT60 identity')
    receipt=SOURCE['checkpoint_receipt.json']
    if ck['state']!={k:receipt[k] for k in ('updates','image_events','validated','validation_files')}:raise ValueError('Wrong completed source cursor')
    prepared=ck['identity']['prepared']
    if prepared['stage_sha256']!=stage or prepared['prompt_files']!=prompt_files:raise ValueError('Changed ROI/prompt inputs')
    if not ck['model'] or 'base.nouns' not in ck['model'] or any(k.startswith('model.') for k in ck['model']):raise ValueError('Not a PromptCSA SCST state')
    if not ck.get('optimizer') or len(ck.get('rng_by_rank',[]))!=2:raise ValueError('Need Adam and two source RNG states')

def next_schedule(data,train_ids):
    pool,previous=full_schedule(train_ids);old=event_schedule(data,previous)
    prior={int(i):int(e) for i,e in zip(previous.ravel(),old.ravel()) if i>=0}
    padded=np.full(UPDATES*8,-1,np.int64);padded[:TRAIN_IMAGES]=np.random.default_rng(43).permutation(pool)
    ids=padded.reshape(UPDATES,8);events=np.full_like(ids,-1)
    for step,row in enumerate(ids):
        for j,i in enumerate(row):
            if i<0:continue
            previous_event=prior[int(i)];image=int(data['caption_image'][previous_event])
            lo,hi=np.searchsorted(data['caption_image'],[image,image+1])
            if data['group'][image]!=0:raise ValueError('Heldout in continuation')
            events[step,j]=lo+(previous_event-lo+1)%(hi-lo)
    return events,ids

def validate_state(state):
    step=state['updates'];phases=state['validated']
    if type(step)!=int or state['image_events']!=image_events(step) or phases!=list(PHASES)[:len(phases)]:raise ValueError('Invalid continuation cursor/phases')
    if step and 'original' not in phases:raise ValueError('Missing original')
    if step>MID and 'mid' not in phases:raise ValueError('Missing MID')
    if any(step<PHASES[p] for p in phases):raise ValueError('Premature evaluation')
    if set(state['validation_files'])!={f'evaluations/{p}/{n}' for p in phases for n in EVAL_NAMES}:raise ValueError('Incomplete validation seal')

def copy_evaluations(root,out,files):
    for name,h in files.items():
        parts=Path(name).parts
        if len(parts)!=3 or parts[0]!='evaluations' or parts[1] not in PHASES or parts[2] not in EVAL_NAMES:raise ValueError('Unsafe evaluation path')
        source=Path(root)/name;target=Path(out)/name
        if sha(source)!=h:raise ValueError('Changed evaluation '+name)
        target.parent.mkdir(parents=True,exist_ok=True)
        if source.resolve()!=target.resolve():shutil.copy2(source,target)

def select(metrics):
    if set(metrics)!=set(PHASES):raise ValueError('Need original/MID/END')
    base=metrics['original'];eligible=[]
    for p in PHASES:
        m=metrics[p]
        if not all(np.isfinite(m[k]) for k in ('CIDEr','CHAIRs','CHAIRi','Recall')):raise ValueError('Nonfinite metric')
        if m['CHAIRs']<=base['CHAIRs'] and m['CHAIRi']<=base['CHAIRi'] and m['Recall']>=base['Recall']:eligible.append(p)
    return max(eligible,key=lambda p:(metrics[p]['CIDEr'],-PHASES[p])),eligible

def prepare(cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True);arm=cfg['arm']
    if arm!='PROMPT' or cfg['updates']!=UPDATES:raise ValueError('Only one added PROMPT epoch approved')
    checkpoint=source_checkpoint(cfg['checkpoint_root'],arm)
    if sha(cfg['dataset_json'])!=DATA_SHA:raise ValueError('Changed COCO JSON')
    refs={};held=set()
    for r in read(cfg['dataset_json'])['images']:
        i=int(Path(r['filename']).stem.split('_')[-1])
        if r['split'] in ('train','restval'):
            if i in refs:raise ValueError('Duplicate TRAIN')
            refs[i]=[s['raw'] for s in r['sentences'][:5]]
        else:held.add(i)
    if len(refs)!=TRAIN_IMAGES or set(refs)&held or any(len(v)!=5 for v in refs.values()):raise ValueError('TRAIN reward coverage differs')
    data,vocab,stage=stage_full(cfg['full_root'],cfg['stage'])
    root,links,_,_,inventory=checked_cache(cfg['prompt_root']);checked_join(data,links,read(root/'cohort.json'))
    events,ids=next_schedule(data,list(refs));present=ids>=0
    if not np.array_equal(events>=0,present) or not np.array_equal(data['ids'][data['caption_image'][events[present]]],ids[present]):raise ValueError('Schedule join differs')
    save(out/'training_references.json',{str(i):v for i,v in refs.items()})
    save(out/'validation_references.json',validation_references(cfg['full_root']));save(out/'vocabulary.json',vocab)
    np.save(out/'events.npy',events);np.save(out/'coco_ids.npy',ids)
    names=('training_references.json','validation_references.json','vocabulary.json','events.npy','coco_ids.npy')
    for n in names[:3]:
        if sha(out/n)!=SOURCE['identity.json']['prepared']['files_sha256'][n]:raise ValueError('Source60 reference/vocab differs:'+n)
    save(out/'prepared.json',dict(status='PROMPT_SCST_CONTINUE_PREPARED',arm=arm,source=SOURCES[arm],checkpoint=str(checkpoint),
        files_sha256={n:sha(out/n) for n in names},stage_sha256=stage['files_sha256'],prompt_files=inventory['cache_files'],
        updates=UPDATES,image_events=TRAIN_IMAGES,test_used=False,epoch_order_seed=43,CE_reference_rotation=1))
    print('Continuation staged:14161 added updates /113287unique TRAIN; no TEST.',flush=True)
