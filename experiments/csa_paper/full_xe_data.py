"""Full XE schedule, validation contracts and resume checks; no neural imports."""
import math
from pathlib import Path
import numpy as np
from real_bank_data import read,sha
from sharing_data import grouped_order

TRAIN_EVENTS=566747
STEPS=math.ceil(TRAIN_EVENTS/32)
EPOCHS=10

def learning_rate(step):
    if not 1<=step<=EPOCHS*STEPS:raise ValueError('Update outside approved ten epochs')
    return 8e-6+(1e-4-8e-6)*min((step-1)/(3*STEPS-1),1)

def epoch_order(data,epoch):
    if not 1<=epoch<=EPOCHS:raise ValueError('Epoch outside1..10')
    return grouped_order(data,TRAIN_EVENTS,seed=42+epoch)

def validation_references(root):
    root=Path(root)
    if not (root/'manifest.json').is_file():
        expected=read(Path(__file__).parent/'SAVE_FULL_ROI_V134_PACKAGE.json')['files']['receipt.json']
        matches=[p.parent for p in root.rglob('receipt.json') if sha(p)==expected]
        if len(matches)!=1:raise ValueError('Expected unique verified ROI root')
        root=matches[0]
    rows=[]
    for entry in read(root/'manifest.json')['shards']:
        if entry['group']=='val':
            rows.extend(read(root/next(n for n in entry['files'] if n.endswith('.json')))['images'])
    if len(rows)!=5000 or len({x['coco_id'] for x in rows})!=5000 or sum(len(x['captions']) for x in rows)!=25010:
        raise ValueError('Wrong validation reference coverage')
    return [dict(coco_id=int(x['coco_id']),references=x['captions']) for x in rows]

def validate_predictions(rows,refs):
    expected={x['coco_id'] for x in refs}
    if len(rows)!=len(refs) or len({x['coco_id'] for x in rows})!=len(rows) or {x['coco_id'] for x in rows}!=expected:
        raise ValueError('Missing/duplicate/foreign validation prediction')
    if any(not isinstance(x['caption'],str) or not math.isfinite(x['score']) for x in rows):raise ValueError('Invalid prediction')

def validate_state(state):
    epoch=state['epoch'];cursor=state['batch_cursor'];phase=state['phase']
    if type(epoch)!=int or type(cursor)!=int or not 1<=epoch<=EPOCHS+1 or not 0<=cursor<=STEPS or phase not in ('train','validate'):
        raise ValueError('Bad resume cursor')
    if epoch==EPOCHS+1 and (cursor or phase!='train'):raise ValueError('Bad completed cursor')
    if phase=='validate' and cursor!=STEPS:raise ValueError('Validation without complete training epoch')
    if state['updates']!=(epoch-1)*STEPS+cursor:raise ValueError('Update/cursor mismatch')
    if state['caption_events']!=(epoch-1)*TRAIN_EVENTS+min(cursor*32,TRAIN_EVENTS):raise ValueError('Caption exposure mismatch')
    if [x['epoch'] for x in state['history']]!=list(range(1,epoch)):raise ValueError('Missing evaluation history')

def is_best(score,previous):
    if not math.isfinite(score):raise ValueError('Nonfinite CIDEr')
    return previous is None or score>previous
