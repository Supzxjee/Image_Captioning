"""Frozen policy and NumPy metrics. No model imports or execution."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import numpy as np

A_SHA = '630a4b6d4bcef84515bbd44a52b6474811377bda53caa9722c5806795fa7f901'
BASE_SHA = 'd2796e30b4e7454798fac5c513be714dca69918b423910233aee95b161e0e81e'
DATA_SHA = '2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
LEGACY_SHA = 'd7910e9bf87e3aad86d806c77ec336b34793e3bef68a3f662fdcb2e9456ca38d'
NOUN_SHA = 'bde873ba77124333488b263a052419dc80e412c1e089178cdf423d2836e1b146'
CLIP_REV = '57c216476eefef5ab752ec549e440a49ae4ae5f3'
POLICY = dict(version='attention-geometry-diagnostic-v1',max_images=200,min_images=100,
    split='locked A TRAIN10k only',selection='hash-first eligible; no model scores',
    minimum_classes=2,axis_gap=1/14,bootstrap=10000,seed=42,
    spatial_map='mean heads last Q cross-attn; mean heads prompt-to-Q; compose INCLUDING CLS then condition on patches',
    phrase_map='mean all retained CLIP subwords of first conservative alias occurrence per class',
    eligibility='sealed v1.1 unique-instance noun-box classes intersect unchanged legacy prompt; >=2 classes and >=1 separated-axis pair',
    primary='image mean of phrase (GT fractional patch mass minus GT image area)',
    controls=['uniform','image_mean','within_image_full_map_derangement'],
    gate='>=100 images; paired95 lower >0 versus image_mean AND derangement for localization AND axis order',
    scope='TRAIN diagnostic, weak noun-class/instance linkage, not caption improvement or predicate correctness')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def order(value):return hashlib.sha256(('attention-geometry-v1:'+str(value)).encode()).hexdigest()


def prompt_text(entry):
    texts=[entry[k] for k in ('prompt','text','prompt_text') if isinstance(entry.get(k),str) and entry[k].strip()]
    if not texts:return None
    if len(set(texts))!=1:raise ValueError('Conflicting text fields in original prompt entry.')
    return texts[0]


def phrase_positions(text,aliases,tokenized):
    """Character spans map to actual CLIP positions; reject partial/truncated phrases."""
    found=[];occupied=set();offsets=tokenized['offset_mapping'];mask=tokenized['attention_mask']
    for alias,label in sorted(aliases,key=lambda x:(-len(x[0]),x[0])):
        pattern=r'(?<!\w)'+r'\s+'.join(re.escape(w) for w in alias)+r'(?!\w)'
        for match in re.finditer(pattern,text.lower()):
            start,end=match.span()
            if any(i in occupied for i in range(start,end)):continue
            positions=[i for i,(a,b) in enumerate(offsets) if mask[i] and b>a and a<end and b>start]
            covered={j for i in positions for j in range(*offsets[i])}
            if not positions or any(j not in covered for j in range(start,end) if not text[j].isspace()):continue
            if any(offsets[i][0]<start or offsets[i][1]>end for i in positions):continue
            occupied.update(range(start,end))
            found.append(dict(category=label,surface=text[start:end],char_span=[start,end],positions=positions))
    return sorted(found,key=lambda r:r['char_span'])


def axis_pairs(phrases):
    pairs=[]
    for i,a in enumerate(phrases):
        for j in range(i+1,len(phrases)):
            b=phrases[j];x,y=np.asarray(a['box']),np.asarray(b['box'])
            for axis in (0,1):
                sign=1 if y[axis]-x[axis+2]>=POLICY['axis_gap'] else -1 if x[axis]-y[axis+2]>=POLICY['axis_gap'] else 0
                if sign:pairs.append(dict(i=i,j=j,axis=axis,sign=sign))
    return pairs


def derangement(n,key):
    if n<2:raise ValueError('Within-image control needs two distinct classes.')
    # Hash-fixed cyclic shift: no fixed point and every complete map used exactly once.
    shift=1+int(order(key)[:12],16)%(n-1)
    return np.roll(np.arange(n),shift)


def summaries(maps,phrases,pairs):
    p=np.asarray(maps,dtype=np.float64)
    if p.shape!=(len(phrases),196) or not np.isfinite(p).all() or (p<0).any():raise ValueError('Invalid maps.')
    if not np.allclose(p.sum(-1),1,rtol=0,atol=1e-6):raise ValueError('Maps are not spatial probability distributions.')
    yy,xx=np.meshgrid((np.arange(14)+.5)/14,(np.arange(14)+.5)/14,indexing='ij')
    coords=np.stack([xx.ravel(),yy.ravel()],axis=-1);mu=p@coords
    sigma=np.sqrt(np.maximum((p@coords**2)-mu**2,0))
    fraction=np.asarray([q['mask'] for q in phrases]);area=fraction.mean(-1)
    mass=(p*fraction).sum(-1);inside=[]
    for center,q in zip(mu,phrases):
        b=q['box'];inside.append(float(b[0]<=center[0]<=b[2] and b[1]<=center[1]<=b[3]))
    signs=[]
    for pair in pairs:
        delta=mu[pair['j'],pair['axis']]-mu[pair['i'],pair['axis']]
        signs.append(0.5 if abs(delta)<=1e-8 else float(np.sign(delta)==pair['sign']))
    entropy=-(p*np.log(np.maximum(p,1e-30))).sum(-1)/np.log(196)
    by_direction={}
    for pair,correct in zip(pairs,signs):
        key=('x' if pair['axis']==0 else 'y')+('+' if pair['sign']>0 else '-')
        by_direction.setdefault(key,[]).append(correct)
    return dict(mass=float(mass.mean()),excess_mass=float((mass-area).mean()),centroid_hit=float(np.mean(inside)),
        axis_accuracy=float(np.mean(signs)),axis_ties=int(sum(abs(mu[t['j'],t['axis']]-mu[t['i'],t['axis']])<=1e-8 for t in pairs)),
        by_direction={k:dict(count=len(v),accuracy=float(np.mean(v))) for k,v in by_direction.items()},
        phrases=[dict(mu=m.tolist(),sigma=s.tolist(),entropy=float(h),mass=float(v),excess_mass=float(v-a),
                      centroid_hit=hit,area=float(a),size_bin='small' if a<.05 else 'medium' if a<.2 else 'large')
                 for m,s,h,v,a,hit in zip(mu,sigma,entropy,mass,area,inside)])


def evaluate_image(maps,phrases,pairs,key):
    maps=np.asarray(maps,dtype=np.float64)
    controls=dict(true=maps,uniform=np.full_like(maps,1/196),
        image_mean=np.broadcast_to(maps.mean(0),maps.shape),
        within_image_full_map_derangement=maps[derangement(len(maps),key)])
    return {name:summaries(value,phrases,pairs) for name,value in controls.items()}


def paired_ci(values):
    v=np.asarray(values,dtype=float)
    if len(v)<2:return dict(mean=float(v.mean()) if len(v) else None,ci95=None,n=len(v))
    rng=np.random.default_rng(POLICY['seed']);means=[]
    for _ in range(POLICY['bootstrap']//100):
        means.extend(v[rng.integers(0,len(v),(100,len(v)))].mean(-1).tolist())
    return dict(mean=float(v.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),n=len(v))


def aggregate(rows):
    if not rows:raise ValueError('No inference rows.')
    contrasts={}
    for metric in ('excess_mass','centroid_hit','axis_accuracy'):
        contrasts[metric]={name:paired_ci([r['metrics']['true'][metric]-r['metrics'][name][metric] for r in rows])
            for name in POLICY['controls']}
    required=[contrasts[m][c]['ci95'] for m in ('excess_mass','axis_accuracy') for c in POLICY['controls'][1:]]
    passed=len(rows)>=POLICY['min_images'] and all(x is not None and x[0]>0 for x in required)
    bysize={};entropy=[];hits=[]
    for r in rows:
        for p in r['metrics']['true']['phrases']:
            entropy.append(p['entropy']);hits.append(p['centroid_hit']);bysize.setdefault(p['size_bin'],[]).append(p['excess_mass'])
    correlation=None
    if np.std(entropy)>0 and np.std(hits)>0:correlation=float(np.corrcoef(entropy,hits)[0,1])
    direction={}
    for c in ('true',*POLICY['controls']):
        direction[c]={}
        for k in ('x+','x-','y+','y-'):
            v=[r['metrics'][c]['by_direction'][k] for r in rows if k in r['metrics'][c]['by_direction']]
            if v:direction[c][k]=dict(images=len(v),pairs=sum(t['count'] for t in v),image_mean_accuracy=float(np.mean([t['accuracy'] for t in v])))
    return dict(status='PREMISE_SIGNAL_GO_DESIGN_ONLY' if passed else 'NOT_YET_GO',images=len(rows),
        contrasts=contrasts,means={c:{m:float(np.mean([r['metrics'][c][m] for r in rows])) for m in ('mass','excess_mass','centroid_hit','axis_accuracy')} for c in ('true',*POLICY['controls'])},
        phrases_by_size={k:dict(count=len(v),mean_excess_mass=float(np.mean(v))) for k,v in bysize.items()},
        entropy_centroid_hit_correlation=correlation,axis_order_by_direction=direction,
        interpretation='Not a caption result, predicate evaluation, independent test, or automatic approval to train.')
