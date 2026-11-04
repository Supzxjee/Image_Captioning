"""Full-one-epoch schedule and metadata selection, no neural imports."""
import hashlib
from pathlib import Path
import shutil
import numpy as np
from real_bank_data import read,save,sha

TRAIN_IMAGES=113287
UPDATES=14161
MID=7081
PHASE_STEPS={'original':0,'mid':MID,'end':UPDATES}


def full_schedule(train_ids):
    ids=[int(i) for i in train_ids]
    if len(ids)!=TRAIN_IMAGES or len(set(ids))!=TRAIN_IMAGES or any(i<0 for i in ids):raise ValueError('Need113287unique TRAIN IDs')
    pool=sorted(ids,key=lambda i:hashlib.sha256(f'42:{i}'.encode()).hexdigest())
    ordered=np.random.default_rng(42).permutation(pool)
    padded=np.full(UPDATES*8,-1,np.int64);padded[:TRAIN_IMAGES]=ordered
    return pool,padded.reshape(UPDATES,8)


def image_events(step):
    if type(step)!=int or not 0<=step<=UPDATES:raise ValueError('Invalid full1 cursor')
    return min(step*8,TRAIN_IMAGES)


def local_batch(events,ids,step,rank):
    if rank not in (0,1):raise ValueError('Expected two ranks')
    ev=events[step,rank*4:rank*4+4];co=ids[step,rank*4:rank*4+4]
    if not np.array_equal(ev>=0,co>=0):raise ValueError('Sentinel masks differ')
    selected=ev>=0
    if not selected.any():raise ValueError('Empty rank not expected for full1 final4/3batch')
    return ev[selected],co[selected]


def validate_state(state):
    step=state['updates'];phases=state['validated']
    if state['image_events']!=image_events(step) or phases!=list(PHASE_STEPS)[:len(phases)]:raise ValueError('Invalid full1 cursor/phases')
    if step>0 and 'original' not in phases:raise ValueError('Original evaluation must precede training')
    if step>MID and 'mid' not in phases:raise ValueError('Missing completed MID evaluation')
    if any(step<PHASE_STEPS[n] for n in phases):raise ValueError('Future evaluation in checkpoint')
    expected={f'evaluations/{n}/{f}' for n in phases for f in ('val_rank0.json','val_rank1.json','worker0.json','worker1.json')}
    if set(state['validation_files'])!=expected:raise ValueError('Evaluation receipt coverage differs')


def selected_name(metrics):
    if set(metrics)!=set(PHASE_STEPS):raise ValueError('Score all original/mid/end phases')
    base=metrics['original'];eligible=[]
    for name in PHASE_STEPS:
        m=metrics[name]
        if not all(np.isfinite(m[k]) for k in ('CIDEr','CHAIRs','CHAIRi','Recall')):raise ValueError('Nonfinite selection metric')
        if m['CHAIRs']<=base['CHAIRs'] and m['CHAIRi']<=base['CHAIRi'] and m['Recall']>=base['Recall']-.005:eligible.append(name)
    return max(eligible,key=lambda n:(metrics[n]['CIDEr'],-PHASE_STEPS[n])),eligible


def copy_evaluations(root,out,files):
    root=Path(root);out=Path(out)
    for name,h in files.items():
        parts=Path(name).parts
        if len(parts)!=3 or parts[0]!='evaluations' or parts[1] not in PHASE_STEPS or parts[2] not in ('val_rank0.json','val_rank1.json','worker0.json','worker1.json'):
            raise ValueError('Invalid saved evaluation path')
        if sha(root/name)!=h:raise ValueError('Missing/changed resume evaluation:'+name)
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
        if (root/name).resolve()!=target.resolve():shutil.copy2(root/name,target)


def score(cfg):
    from run_scst_pilot import score as score_phase
    from scst_pilot_data import exact_checkpoint,CHECKPOINT_SHA
    out=Path(cfg['output']);state=read(out/'progress.json');metrics={}
    validate_state(state)
    if state['updates']!=UPDATES or state['image_events']!=TRAIN_IMAGES or set(state['validated'])!=set(PHASE_STEPS):raise ValueError('Full1/evaluations incomplete')
    copy_evaluations(out,out,state['validation_files'])
    for name,step in PHASE_STEPS.items():
        target=out/'evaluations'/name
        for n in ('validation_references.json','vocabulary.json'):
            shutil.copy2(out/n,target/n)
        score_phase({**cfg,'output':str(target),'evaluation_step':step})
        metrics[name]=read(target/'result.json')['metrics']
    selected,eligible=selected_name(metrics)
    paths={'original':exact_checkpoint(cfg['checkpoint_root']),'mid':out/'mid.bin','end':out/'checkpoint.bin'}
    receipts={'original':{'sha256':CHECKPOINT_SHA},'mid':read(out/'mid_receipt.json'),'end':read(out/'checkpoint_receipt.json')}
    for name in ('mid','end'):
        if receipts[name]['updates']!=PHASE_STEPS[name] or receipts[name]['image_events']!=image_events(PHASE_STEPS[name]):raise ValueError('Wrong selected checkpoint cursor')
    for name,path in paths.items():
        if sha(path)!=receipts[name]['sha256']:raise ValueError('Selection weightSHA differs:'+name)
    shutil.copy2(paths[selected],out/'best.bin')
    if sha(out/'best.bin')!=receipts[selected]['sha256']:raise ValueError('Selected copy differs')
    lock=dict(selected_name=selected,eligible_names=eligible,checkpoint_sha256=receipts[selected]['sha256'],
        best_path=str(out/'best.bin'),rule='maxCIDEr with CHAIRs/i<=original and recall>=original-.005; earlier ties; original fallback',
        test_used=False,metrics_sha256={n:sha(out/'evaluations'/n/'result.json') for n in PHASE_STEPS})
    save(out/'locked_selection.json',lock);save(out/'best_receipt.json',receipts[selected])
    save(out/'result.json',dict(status='CSA_SCST_FULL1_COMPLETE',arm=cfg['arm'],updates=UPDATES,image_events=TRAIN_IMAGES,
        validation_images=5000,test_used=False,metrics=metrics,selection=lock))
    print('FULL1 COMPLETE',cfg['arm'],'selected',selected,metrics,flush=True)
