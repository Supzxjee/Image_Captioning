"""Model-free lexical and intervention policy. Never infer absence from masking."""
import re
import hashlib
import numpy as np

ANNOTATION_URL='https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip'
ANNOTATION_SHA='031296bbc80c45a1d1f76bf9a90ead27e94e99ec629208449507a4917a3bf009'
ANNOTATION_BYTES=252872794
TEXT_SHA='df88b7cafac854b98b1609e5899beab681aa60cd155e5dda5cb25bc320f3f88d'
POLICY=dict(version='stale-prompt-prep-v1',max_aux_images=2000,one_target_per_image=True,
    neutral_word='object',mask_fill=[123,117,104],min_mask_fraction=.01,max_mask_fraction=.40,
    mask='union of ALL annotated instances of target class; reject class if any crowd/invalid instance',
    alias='unambiguous one-caption-token aliases; longest whole-phrase prompt matching',
    selection='TRAIN10k hash first target per image then hash first <=2000 images, metadata only',
    panel_images=16,frozen_score_images=128,benchmark_batches=50,aux_every=5,global_batch=8,
    max_preparation_seconds=1800,max_arm_seconds=7200,projection_safety_factor=1.5,
    loss=dict(output_KL=.1,neutral_CE=.1,visual_contrast=.1,stale=.1,visual_margin=.2,stale_margin=.1),
    trainable='last QFormer layer; prompt projection/MHA/gate/LN; entire decoder; same C/T',
    learning_rate=5e-6,full_pilot_updates=1250,dropout='eval mode with gradients',
    neutral_window='P0 uses only original retained text window; require exact P1 retained-token replay; no new tail context or P0 overflow',
    scope='Preparation, frozen128 scores, no-step forward/backward cost profiles only; no full pilot or checkpoint')


def order(value):return hashlib.sha256(('stale-prompt-v1:'+str(value)).encode()).hexdigest()


def mentions(text,aliases):
    occupied=set();found=[]
    for words,label in sorted(aliases,key=lambda x:(-len(x[0]),-len(' '.join(x[0])),x[0],x[1])):
        pattern=r'(?<!\w)'+r'\s+'.join(re.escape(w) for w in words)+r'(?!\w)'
        for match in re.finditer(pattern,text.lower()):
            a,b=match.span()
            if any(i in occupied for i in range(a,b)):continue
            occupied.update(range(a,b));found.append((a,b,label))
    return sorted(found)


def neutralize(text,label,aliases):
    positions=[(a,b) for a,b,c in mentions(text,aliases) if c==label]
    if not positions:return None
    for a,b in reversed(positions):
        prefix=text[:a]
        # Keep an immediate indefinite article grammatical for the fixed neutral noun.
        prefix=re.sub(r'\ba(\s+)$',r'an\1',prefix,flags=re.I)
        text=prefix+POLICY['neutral_word']+text[b:]
    if any(c==label for a,b,c in mentions(text,aliases)):raise ValueError('Target mention remained after neutralization.')
    return text


def lexical_ids(vocab,aliases,ambiguous):
    mapping={}
    for words,label in aliases:
        if len(words)==1 and words[0] not in ambiguous:mapping.setdefault(words[0],set()).add(label)
    result={}
    for word,idx in vocab.items():
        normalized=re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$','',word)
        labels=mapping.get(normalized,set())
        if len(labels)==1 and idx>3:result.setdefault(next(iter(labels)),[]).append(idx)
    return {k:sorted(set(v)) for k,v in result.items()}


def reference_target(text,label,aliases,vocab,alias_ids):
    matches=[(a,b) for a,b,c in mentions(text,aliases) if c==label]
    if not matches:return None
    a,b=matches[0]
    for index,word in enumerate(re.finditer(r'\S+',text)):
        if word.start()<=a and word.end()>=b:
            outside=text[word.start():a]+text[b:word.end()]
            if re.search(r'[a-z0-9]',outside,re.I):return None
            value=word.group().lower();token=vocab.get(value,3)
            if index>=29 or len(text[a:b].split())!=1 or token not in alias_ids:return None
            return dict(position=index,token_id=token)
    return None


def visible_window(text,encoded):
    end=max((b for (a,b),valid in zip(encoded['offset_mapping'],encoded['attention_mask']) if valid and b>a),default=0)
    return text[:end]


def union_mask(annotations,height,width,decode):
    if not annotations or any(a.get('iscrowd',0) or not a.get('segmentation') for a in annotations):return None
    result=np.zeros((height,width),dtype=bool)
    for a in annotations:
        m=np.asarray(decode(a['segmentation'],height,width))
        if m.shape!=(height,width) or not np.isin(m,[0,1]).all() or not m.any():return None
        result|=m.astype(bool)
    fraction=float(result.mean())
    if not POLICY['min_mask_fraction']<=fraction<=POLICY['max_mask_fraction']:return None
    return result


def contrast_values(s11,s10,s01,s00):
    # Diagnostic-only scalars; benchmark loss implements the one-sided stop-gradient separately.
    return dict(factual_prompt_boost=s11-s10,occluded_prompt_boost=s01-s00,
        prompt_visual_interaction=(s11-s10)-(s01-s00),
        visual_contrast_original=s11-s01,visual_contrast_neutral=s10-s00,
        stale_hinge=max(0.,s01-s00-POLICY['loss']['stale_margin']))
