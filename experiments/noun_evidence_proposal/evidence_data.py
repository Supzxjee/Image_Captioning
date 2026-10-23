"""Pure metadata and lexical rules; no model imports."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
import numpy as np

A_SHA='630a4b6d4bcef84515bbd44a52b6474811377bda53caa9722c5806795fa7f901'
A_BASE_SHA='d2796e30b4e7454798fac5c513be714dca69918b423910233aee95b161e0e81e'
DATA_SHA='2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'


def normalize(word):return re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$','',word)
def encoded(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def save(path,value):Path(path).write_bytes(encoded(value))


def mentions(text,aliases,policy):
    words=[normalize(w) for w in text.lower().split()];used=set();positive=set();uncertain=set()
    depiction=any(' '+cue+' ' in ' '+' '.join(words)+' ' for cue in policy['depiction_cues'])
    present=set(words)
    for alias,label in aliases:
        if alias[0] not in present:continue
        for start in range(len(words)-len(alias)+1):
            end=start+len(alias)
            if used.intersection(range(start,end)) or tuple(words[start:end])!=tuple(alias):continue
            used.update(range(start,end))
            unsafe=(' '.join(alias) in policy['ambiguity_exclusions'] or depiction or
                end<len(words) and words[end] in policy['compound_suffixes'] or
                start>0 and words[start-1] in policy['artificial_prefixes'] and label!='teddy bear')
            (uncertain if unsafe else positive).add(label)
    return positive,uncertain-positive


def labels(annotation_classes,captions,aliases,policy,classes):
    positive=set(annotation_classes);uncertain=set()
    for caption in captions:
        good,bad=mentions(caption,aliases,policy);positive|=good;uncertain|=bad
    uncertain-=positive
    return ([int(c in positive) for c in classes],[int(c not in uncertain) for c in classes])


def lexicon(vocab,aliases,policy,classes):
    by_word=defaultdict(set)
    for words,label in aliases:
        if len(words)==1 and words[0] not in policy['ambiguity_exclusions']:by_word[words[0]].add(label)
    mapping={w:next(iter(c)) for w,c in by_word.items() if len(c)==1}
    ids=[];categories=[];norm_ids=defaultdict(list)
    for word,identifier in vocab.items():
        normalized=normalize(word);norm_ids[normalized].append(identifier)
        if identifier>=4 and normalized in mapping:
            ids.append(identifier);categories.append(classes.index(mapping[normalized]))
    order=np.argsort(ids);ids=[ids[i] for i in order];categories=[categories[i] for i in order]
    guards=[]
    for words,label in aliases:
        if len(words)<2 or words[-1] not in mapping or mapping[words[-1]]==label:continue
        columns=[i for i,identifier in enumerate(ids) if normalize(next(w for w,n in vocab.items() if n==identifier))==words[-1]]
        prefix=[norm_ids.get(w,[]) for w in reversed(words[:-1])]
        if columns and all(prefix):guards.append(dict(prefix_ids_reversed=prefix,columns=columns,compound=list(words)))
    return dict(candidate_ids=ids,candidate_classes=categories,guards=guards,
        vocabulary_sha256=hashlib.sha256(json.dumps(vocab,sort_keys=True).encode()).hexdigest(),
        candidate_surfaces={str(i):word for word,i in vocab.items() if i in set(ids)},
        classes=classes,policy='Single unambiguous alias; prefix-only known-compound guard; no future token/POS.')


def validate_package(directory,expected):
    directory=Path(directory)
    if digest(directory/'receipt.json')!=expected:raise ValueError('Different noun evidence receipt.')
    receipt=json.loads((directory/'receipt.json').read_text(encoding='utf-8'))
    for name,sha in receipt['payload_sha256'].items():
        if Path(name).name!=name or digest(directory/name)!=sha:raise ValueError('Changed label payload: '+name)
    return receipt
