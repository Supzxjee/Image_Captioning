"""Frozen concept/event policy and lexical bookkeeping; NumPy/metadata only."""
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from captioning.object_consistency import ALIASES,COCO_LABELS
from experiments.shared_pair_readout.readout_data import DATA_SHA,DETECTION_SHA,LEGACY_SHA,CHAIR_SHA,BASE_SHA,CHECKPOINT_SHA,META_SHA

HERE=Path(__file__).resolve().parent
POLICY=dict(version='concept-mass-cg-v2',seed=42,updates=500,batch=8,max_len=30,lr=5e-6,weight_decay=.01,
    kl_weight=.1,keep_weight=.1,negative_weight=.1,object_reward_weight=.5,gradient_clip=1.,
    detector_veto=0.0,prepare_seconds=3600,gpu_seconds=3600,evaluate_seconds=900,
    diagnostic_classes_minimum=5,diagnostic_prefix='a photo of a',event_chunk=64,
    negative_policy='gt_ref_absence_available_detector_veto_v2',checkpoint_selection='endpoint500 only')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()

def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def normalize(word):return re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$','',word.lower())

def prefix_free(paths):
    result=[]
    for path in sorted(set(tuple(p) for p in paths),key=lambda p:(len(p),p)):
        if not path or any(i<4 for i in path):raise ValueError('Invalid alias event token')
        if not any(path[:len(p)]==p for p in result):result.append(path)
    return result

def lexicon(vocab,rules):
    final={};plain={}
    for word,identifier in vocab.items():
        if identifier<4:continue
        norm=normalize(word)
        # Admit punctuation only at phrase end; never a comma inside a multiword alias.
        if norm and re.fullmatch(r'[a-z0-9]+[.,!?;:]*',word):final.setdefault(norm,[]).append(identifier)
        if word==norm and norm:plain.setdefault(norm,[]).append(identifier)
    aliases=[(list(words),label) for words,label in ALIASES]
    candidates={c:[] for c in COCO_LABELS}
    for words,label in aliases:
        if ' '.join(words) in rules['ambiguity_exclusions']:continue
        choices=[(final if i==len(words)-1 else plain).get(w,[]) for i,w in enumerate(words)]
        if all(choices):candidates[label].extend(itertools.product(*choices))
    events={c:[list(p) for p in prefix_free(paths)] for c,paths in candidates.items()}
    return dict(aliases=aliases,events=events,rules=rules,
        idx2word={str(i):w for w,i in vocab.items()},classes=list(COCO_LABELS))

def mentions(words,lex):
    normalized=[normalize(w) if w and not w.startswith('<') else '' for w in words]
    used=set();out=[];rules=lex['rules'];sentence=' '+' '.join(normalized)+' '
    depiction=any(' '+cue+' ' in sentence for cue in rules['depiction_cues'])
    for alias,label in lex['aliases']:
        width=len(alias)
        for start in range(len(words)-width+1):
            positions=set(range(start,start+width));end=start+width
            if positions&used or normalized[start:end]!=alias:continue
            used|=positions
            unsafe=(depiction or ' '.join(alias) in rules['ambiguity_exclusions'] or
                end<len(words) and normalized[end] in rules['compound_suffixes'] or
                start>0 and normalized[start-1] in rules['artificial_prefixes'] and label!='teddy bear')
            out.append(dict(start=start,end=end,category=label,uncertain=bool(unsafe)))
    return sorted(out,key=lambda m:m['start'])

def reference_support(references,lex):
    hints=set();safe=set()
    for ref in references:
        for m in mentions(ref.lower().split(),lex):
            hints.add(m['category'])
            if not m['uncertain']:safe.add(m['category'])
    return hints,safe

def token_events(sequences,targets,lex,negative):
    events=[];counts=dict(mentions=0,supported=0,unknown=0,weak_negative=0,empty_detector=0,candidates=0,vetoes=0)
    for b,(sequence,target) in enumerate(zip(sequences,targets)):
        tokens=[]
        for token in sequence[1:]:
            if token==2:break
            tokens.append(lex['idx2word'].get(str(int(token)),'') if token>=4 else '')
        for mention in mentions(tokens,lex):
            c=mention['category'];counts['mentions']+=1
            actual=tuple(int(i) for i in sequence[1+mention['start']:1+mention['end']])
            inside=any(actual[:len(path)]==tuple(path) for path in lex['events'][c] if len(path)<=POLICY['max_len']-1-mention['start'])
            if mention['uncertain'] or not inside:counts['unknown']+=1;continue
            if negative:
                if c in target['supported']:counts['supported']+=1;continue
                counts['candidates']+=1
                if c in target['unknown']:
                    counts['vetoes']+=1;counts['unknown']+=1;continue
                if not target['negative_eligible']:counts['unknown']+=1;continue
                counts['weak_negative']+=1
            elif c not in target['positive_anchor_classes']:continue
            events.append(dict(batch=b,position=mention['start'],category=c))
    return events,counts

def order_indices():return np.random.default_rng(42).permutation(10000)[:4000].reshape(500,8)

def validate_rows(train,dev):
    if len(train)!=10000 or len(dev)!=1000:raise ValueError('Wrong locked train/dev sizes')
    if len({r['coco_id'] for r in train})!=10000 or {r['coco_id'] for r in train}&{r['coco_id'] for r in dev}:raise ValueError('Duplicate or leaking IDs')

def log_union_numpy(paths,probability):
    """Tiny independent probability-tree oracle for metadata tests/preflight."""
    terms=[]
    for path in prefix_free(paths):
        mass=1.
        for i,token in enumerate(path):mass*=probability(path[:i],token)
        terms.append(mass)
    return math.log(sum(terms))

class Inputs:
    def __init__(self,out):
        import h5py
        self.meta=json.loads((Path(out)/'metadata.json').read_bytes());self.f=h5py.File(Path(out)/'inputs.h5','r')
        if not self.f.attrs.get('complete') or self.f['coco_ids'][:].tolist()!=[r['coco_id'] for r in self.meta['rows']]:raise ValueError('Bad stage identity')
        self.arrays={k:self.f[k] for k in ('visual','prompt','mask')}
    def batch(self,indices,device):
        import torch
        return {k:torch.from_numpy(np.stack([v[int(i)] for i in indices])).to(device=device,dtype=torch.bool if k=='mask' else torch.float32) for k,v in self.arrays.items()}
    def close(self):self.f.close()
