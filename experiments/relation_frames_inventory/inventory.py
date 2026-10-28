"""Metadata-only VSR feasibility counts. Never imports a neural runtime."""
import argparse
import ast
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
CAPTION_SHA = '2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
RELATIONS = ('on', 'under', 'above', 'below', 'on top of', 'beneath', 'in',
             'inside', 'within', 'outside', 'contains', 'touching', 'next to',
             'near', 'far from', 'far away from')
POLICY = dict(version='vsr-relation-frames-metadata-v1', relations=RELATIONS,
              validators_min=3, dissent_allowed=0,
              allowed_caption_splits=['train', 'restval'],
              split_rule='sha256(vsr-frames-v1:image_id) mod 5; 0 heldout, else train',
              required_pairs=dict(train=256, heldout=64),
              required_relations=4, per_relation=dict(train=16, heldout=8),
              source_split_scope='pooled official random splits; derived image-disjoint cohort')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def normalize(text):
    return ' '.join(text.lower().strip().rstrip('.').split())


def parse_assertion(row, categories):
    q, relation = normalize(row['caption']), row['relation']
    for subject in sorted(categories, key=len, reverse=True):
        # The release uses declarative templates; ambiguous parses fail closed.
        for prefix in (f'the {subject} is {relation} the ',
                       f'the {subject} {relation} the '):
            if q.startswith(prefix) and q[len(prefix):] in categories:
                obj = q[len(prefix):]
                if subject != obj:
                    return q, subject, obj
    return None


def unanimous(row):
    if row['label'] not in (0, 1):
        raise ValueError('Unexpected label')
    true = ast.literal_eval(row['vote_true_validator_id'])
    false = ast.literal_eval(row['vote_false_validator_id'])
    if not isinstance(true, list) or not isinstance(false, list):
        raise ValueError('Invalid validator list')
    yes, no = (true, false) if row['label'] else (false, true)
    return len(set(yes)) >= 3 and not no


def partition(image_id):
    return 'heldout' if int(sha(f'vsr-frames-v1:{image_id}'.encode()), 16) % 5 == 0 else 'train'


def summarize(rows):
    groups = defaultdict(lambda: {0: set(), 1: set()})
    for row in rows:
        groups[row['statement']][row['label']].add(row['image_id'])
    pairs = Counter()
    products = Counter()
    conflicting = 0
    for statement, labels in groups.items():
        both = labels[0] & labels[1]
        conflicting += len(both)
        neg, pos = labels[0] - both, labels[1] - both
        relation = next(r['relation'] for r in rows if r['statement'] == statement)
        pairs[relation] += min(len(neg), len(pos))
        products[relation] += len(neg) * len(pos)
    return dict(records=len(rows), images=len({r['image_id'] for r in rows}),
                statements=len(groups), one_to_one_pair_capacity=sum(pairs.values()),
                all_cross_products=sum(products.values()), by_relation=dict(sorted(pairs.items())),
                conflicting_image_statement_labels=conflicting,
                note='Upper bounds before full GT uniqueness, detector matching, pool caps and image-reuse policy; Cartesian pairs are not independent examples.')


def self_check():
    cats={'cat', 'dining table', 'dog'}
    a=dict(caption='The cat is under the dining table.', relation='under', label=1,
           vote_true_validator_id='[1,2,3]', vote_false_validator_id='[]')
    assert parse_assertion(a,cats)==('the cat is under the dining table','cat','dining table')
    assert unanimous(a)
    assert not unanimous(dict(a,vote_false_validator_id='[4]'))
    assert not unanimous(dict(a,vote_true_validator_id='[1,1,2]'))
    assert parse_assertion(dict(a,caption='The cat is under the cat.'),cats) is None
    rows=[dict(statement='q',label=y,image_id=i,relation='under') for y,i in [(1,1),(1,1),(0,2),(0,3)]]
    assert summarize(rows)['one_to_one_pair_capacity']==1
    assert summarize(rows)['all_cross_products']==2
    assert partition(42)==partition(42)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--instances',type=Path,required=True)
    args=parser.parse_args()
    self_check()
    raw=args.dataset.read_bytes()
    if sha(raw)!=CAPTION_SHA:
        raise ValueError('Caption dataset differs from pinned training data')
    caption=json.loads(raw)
    split={r['cocoid']:r['split'] for r in caption['images']}
    if len(split)!=len(caption['images']):
        raise ValueError('Duplicate caption COCO identity')
    instances=json.loads(args.instances.read_bytes())
    categories={r['name'] for r in instances['categories']}
    gt=defaultdict(Counter)
    cat_by_id={r['id']:r['name'] for r in instances['categories']}
    gt_images={r['id'] for r in instances['images']}
    for ann in instances['annotations']:
        gt[ann['image_id']][cat_by_id[ann['category_id']]]+=1
    with gzip.open(HERE.parent/'semantic_grounding_method/data/gain_metadata.json.gz','rt',encoding='utf-8') as f:
        vocab=set(json.load(f)['word2idx'])
    receipt=json.loads((HERE/'sources/source_receipt.json').read_text())
    stages=defaultdict(list)
    rejects=Counter()
    original_splits=defaultdict(set)
    seen=set()
    for source in ('train','dev','test'):
        source_raw=(HERE/f'sources/{source}.jsonl').read_bytes()
        if sha(source_raw)!=receipt['files'][source]['sha256']:
            raise ValueError('Source hash mismatch')
        for line in source_raw.splitlines():
            r=json.loads(line)
            iid=int(Path(r['image']).stem)
            original_splits[iid].add(source)
            record=dict(image_id=iid,statement=normalize(r['caption']),relation=r['relation'],label=r['label'])
            stages['all'].append(record)
            if not unanimous(r):rejects['not_three_unanimous']+=1;continue
            stages['unanimous'].append(record)
            if r['relation'] not in RELATIONS:rejects['relation_outside_fixed_policy']+=1;continue
            stages['allowed_relations'].append(record)
            parsed=parse_assertion(r,categories)
            if parsed is None:rejects['not_exact_distinct_coco_template']+=1;continue
            record.update(subject=parsed[1],object=parsed[2])
            stages['parsed_coco'].append(record)
            if any(w not in vocab for w in record['statement'].split()):rejects['statement_oov']+=1;continue
            stages['in_vocab'].append(record)
            if split.get(iid) not in ('train','restval'):
                rejects['caption_'+split.get(iid,'missing')]+=1;continue
            stages['caption_train_only'].append(record)
            ident=(iid,record['statement'],r['label'])
            if ident in seen:rejects['duplicate_image_statement_label']+=1;continue
            seen.add(ident)
            stages['deduplicated'].append(record)
    result={name:summarize(rows) for name,rows in stages.items()}
    derived={s:[r for r in stages['deduplicated'] if partition(r['image_id'])==s] for s in ('train','heldout')}
    derived_summary={s:summarize(rows) for s,rows in derived.items()}
    assert not {r['image_id'] for r in derived['train']} & {r['image_id'] for r in derived['heldout']}
    observed=[]
    for r in stages['deduplicated']:
        if r['image_id'] in gt_images and gt[r['image_id']][r['subject']]==gt[r['image_id']][r['object']]==1:
            observed.append(r)
    requirements={}
    for s, d in derived_summary.items():
        requirements[s]=dict(total_capacity_sufficient=d['one_to_one_pair_capacity']>=POLICY['required_pairs'][s],
                             qualifying_relations=[k for k,v in d['by_relation'].items() if v>=POLICY['per_relation'][s]])
    common=set(requirements['train']['qualifying_relations']) & set(requirements['heldout']['qualifying_relations'])
    upper_bound_pass=all(r['total_capacity_sufficient'] for r in requirements.values()) and len(common)>=4
    report=dict(status='METADATA_UPPER_BOUND_PASS_PENDING_GT_DETECTOR' if upper_bound_pass else 'METADATA_POLICY_NO_GO',
        policy=POLICY,sources=receipt,caption_dataset_sha256=CAPTION_SHA,
        caption_split_counts=dict(Counter(split.values())),stages=result,rejections=dict(rejects),
        derived=derived_summary,requirements=requirements,common_qualifying_relations=sorted(common),
        official_random_split_images_reused=sum(len(s)>1 for s in original_splits.values()),
        local_gt=dict(available_images=len(gt_images),unique_instance_records=len(observed),
                      missing_image_records=sum(r['image_id'] not in gt_images for r in stages['deduplicated']),
                      scope='VCOCO subset only; missing is unknown, never evidence of missing object'),
        not_checked=['Full COCO instance uniqueness','Predicted-box IoU and coverage','RGB/features availability',
                     'Model scores, neural execution, novelty, caption improvement'],
        local_neural_execution=False)
    (HERE/'RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (HERE/'metadata_candidates.json').write_text(json.dumps(derived,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','derived','requirements','common_qualifying_relations','local_gt')},indent=2))


if __name__=='__main__':
    main()
