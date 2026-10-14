"""CPU-only diagnostic: lexical caption coverage, not grounded relation truth."""
import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path
import torch
from captioning.relation_memory_pilot import filename_index, action_entry

VERBS = {'holding': r'hold|holds|holding|held', 'carrying': r'carry|carries|carrying|carried',
         'riding': r'ride|rides|riding|rode', 'wearing': r'wear|wears|wearing|wore',
         'eating': r'eat|eats|eating|ate', 'sitting_on': r'sit|sits|sitting|seated',
         'talking_on': r'talk|talks|talking'}
ALIASES = {'person': ['person','people','man','men','woman','women','boy','girl','child','children'],
           'cell phone': ['cell phone','phone','cellphone'], 'cup': ['cup','cups','mug','mugs'],
           'skis': ['ski','skis'], 'sports ball': ['ball','balls'],
           'baseball bat': ['bat','bats'], 'wine glass': ['wine glass','wine glasses','glass of wine']}


def lexical_claim(caption, predicate, obj):
    """Cooccurrence only; does NOT establish actor/target identity or entailment."""
    text = caption.lower()
    aliases = ALIASES.get(obj, [obj, obj+'s'])
    verb = bool(re.search(r'\b(?:'+VERBS.get(predicate, re.escape(predicate.replace('_',' ')))+r')\b', text))
    noun = any(re.search(r'\b'+re.escape(label)+r'\b', text) for label in aliases)
    negated = bool(re.search(r'\b(?:not|never|no)\b', text))
    return dict(verb=verb, object=noun, joint_cooccurrence=verb and noun,
                potential_negation=negated, semantic_truth='not assessed')


def summarize_claims(claims, captions):
    result=[]
    for claim in claims:
        matches=[lexical_claim(c,claim['relation'],claim['object']) for c in captions]
        result.append(dict(relation=claim['relation'],object=claim['object'],
            reference_object_mentioned=any(m['object'] for m in matches),
            reference_verb_mentioned=any(m['verb'] for m in matches),
            reference_joint_cooccurrence=any(m['joint_cooccurrence'] for m in matches)))
    return result


def main():
    p=argparse.ArgumentParser()
    for name in ('dataset-json','training-cache','manifest','selected','pilot-review','output'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    manifest=json.loads(Path(a.manifest).read_text(encoding='utf-8'))
    dataset_bytes=Path(a.dataset_json).read_bytes()
    if hashlib.sha256(dataset_bytes).hexdigest()!=manifest['dataset_sha256']:
        raise ValueError('Dataset JSON differs from pilot training source')
    records={r['filename']:r for r in json.loads(dataset_bytes)['images']}
    bundle=torch.load(a.training_cache,map_location='cpu',weights_only=True)
    if hashlib.sha256(Path(a.training_cache).read_bytes()).hexdigest()!=manifest['training_cache_sha256']:
        raise ValueError('Training relation cache differs from pilot source')
    cache=filename_index(bundle['data'])
    selected=json.loads(Path(a.selected).read_text(encoding='utf-8'))
    if hashlib.sha256(Path(a.selected).read_bytes()).hexdigest()!=manifest['selected_sha256']:
        raise ValueError('Selected JSON differs from pilot source')
    train=[]
    for name in manifest['train_filenames']:
        r=records[name];entry=action_entry(cache[name])
        captions=[s['raw'] for s in r['sentences'][:5]]
        claims=[c for i,c in enumerate(entry['relations']) if entry['mask'][i]]
        train.append(dict(filename=name,claims=summarize_claims(claims,captions)))
    review_path=Path(a.pilot_review)
    if review_path.is_dir():
        evaluation={arm:json.loads((review_path/arm/'evaluation_12/records.json').read_text(encoding='utf-8'))
                    for arm in ('legacy','direct')}
    else:
        with zipfile.ZipFile(review_path) as z:
            evaluation={arm:json.loads(z.read(arm+'/evaluation_12/records.json')) for arm in ('legacy','direct')}
    candidate=[]
    for arm,rows in evaluation.items():
        for row in rows:
            name=row['filename'];captions=[s['raw'] for s in records[name]['sentences'][:5]]
            claims=selected[name]['relations']
            candidate.append(dict(arm=arm,filename=name,reference_claims=summarize_claims(claims,captions),
                candidate_claims=[dict(caption=c['caption'],claims=summarize_claims(claims,[c['caption']]))
                    for c in row['conditions']['on']['candidates']]))
    flat=[c for row in train for c in row['claims']]
    summary=dict(train_images=len(train),train_claims=len(flat),
        claims_with_reference_object=sum(c['reference_object_mentioned'] for c in flat),
        claims_with_reference_verb=sum(c['reference_verb_mentioned'] for c in flat),
        claims_with_joint_reference_cooccurrence=sum(c['reference_joint_cooccurrence'] for c in flat),
        notes='Limited lexical diagnostic. No parse/instance identity/negation entailment. Missing '
              'words are not false relations; synonym mismatch possible. Do not train negatives '
              'or rerank from this diagnostic. Evaluation remains inspected development.')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    for name,value in [('summary.json',summary),('training_coverage.json',train),('candidate_coverage.json',candidate)]:
        (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    with zipfile.ZipFile(str(out)+'_review.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in out.glob('*.json'):z.write(f,f.name)
    print(json.dumps(summary,indent=2));print('Review ZIP:',str(out)+'_review.zip')


if __name__=='__main__':main()
