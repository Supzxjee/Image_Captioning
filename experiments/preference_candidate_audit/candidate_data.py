"""TRAIN-only candidate feasibility rules. No model imports."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from evidence_data import mentions, digest
from captioning.object_consistency import ALIASES

A_SHA='630a4b6d4bcef84515bbd44a52b6474811377bda53caa9722c5806795fa7f901'
BASE_SHA='d2796e30b4e7454798fac5c513be714dca69918b423910233aee95b161e0e81e'
DATA_SHA='2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
LEGACY_SHA='d7910e9bf87e3aad86d806c77ec336b34793e3bef68a3f662fdcb2e9456ca38d'
DET_SHA='498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c'
ARCHIVE_SHA='ebf880fcc0ae23388057abc84ce796b64ee1f0f6c16f804d1cdbf98e73fb78fd'
RECEIPT_SHA='e315b1148e0034aaec64ea5bdf69f89a3dbea5ecf17183a3d30dc1bd5dcbaa49'


def save(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');temp.replace(path)


def sample_rows(groups, dataset, count=2000):
    if len(groups['train'])!=10000 or len(groups['val'])!=5000:raise ValueError('Wrong locked groups.')
    allrows=groups['train']+groups['val']
    if len({r['coco_id'] for r in allrows})!=15000:raise ValueError('Overlapping/duplicate image IDs.')
    for r in groups['train']:
        item=dataset['images'][r['image_id']]
        if item['split'] not in ('train','restval') or item['cocoid']!=r['coco_id'] or item['filename']!=r['filename']:raise ValueError('Not matching TRAIN data.')
        if len(item['sentences'][:5])!=5:raise ValueError('Five references required.')
    return sorted(groups['train'],key=lambda r:hashlib.sha256(('preference-bank-train-v1:'+r['filename']).encode()).hexdigest())[:count]


def draw_seed(coco_id, draw):
    return int.from_bytes(hashlib.sha256(f'preference-bank-v1:{coco_id}:{draw}'.encode()).digest()[:8],'little')%(2**63-1)


def choose_pair(candidates, support, salient, policy, conflict=False):
    if conflict or not salient:return None
    unique=[];seen=set()
    for c in candidates:
        if not math.isfinite(c['CIDEr']):raise ValueError('Nonfinite CIDEr.')
        text=' '.join(c['caption'].lower().split())
        if not c['valid'] or not text or text in seen:continue
        seen.add(text);good,uncertain=mentions(text,ALIASES,policy)
        if uncertain:continue
        unique.append(dict(c,objects=sorted(good),unsupported=sorted(good-set(support)),
                           salient_recall=len(good&set(salient))/len(salient),words=len(text.split())))
    pairs=[]
    for winner in unique:
        if winner['unsupported']:continue
        for loser in unique:
            ratio=winner['words']/loser['words']
            gap=winner['CIDEr']-loser['CIDEr']
            if gap>=.10 and .8<=ratio<=1.25 and winner['salient_recall']>=loser['salient_recall']:
                pairs.append(dict(winner=winner,loser=loser,gap=gap))
    return min(pairs,key=lambda p:(-p['gap'],p['winner']['draw'],p['loser']['draw'])) if pairs else None


def oracle(candidates, support, salient, policy, conflict=False):
    """A hypothetical selector; never present this as a new model's prediction."""
    greedy=candidates[0];eligible=[]
    if salient and not conflict:
        g,uncertain=mentions(greedy['caption'],ALIASES,policy)
        baseline_recall=len(g&set(salient))/len(salient)
        for c in candidates:
            good,bad=mentions(c['caption'],ALIASES,policy)
            if (c['valid'] and not bad and not good-set(support)
                    and len(good&set(salient))/len(salient)>=baseline_recall
                    and .8<=len(c['caption'].split())/max(1,len(greedy['caption'].split()))<=1.25):eligible.append(c)
    return max([greedy]+eligible,key=lambda c:c['CIDEr'])
