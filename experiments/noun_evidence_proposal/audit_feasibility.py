"""Pre-policy data audit for noun-evidence proposal; no torch/models/notebook execution."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import re
import runpy

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
COCO=ROOT/'experiments/coco_grounding_feasibility'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):return json.loads(path.read_text(encoding='utf-8'))
def normalize(word):return re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$','',word)


def main():
    source=ROOT/'captioning/object_consistency.py'
    aliases=runpy.run_path(str(source))['ALIASES']
    policy=load(COCO/'MAPPING_POLICY_V11.json')
    grouped=defaultdict(set)
    for words,label in aliases:
        if len(words)==1 and words[0] not in policy['ambiguity_exclusions']:grouped[words[0]].add(label)
    mapping={w:next(iter(labels)) for w,labels in grouped.items() if len(labels)==1}
    compounds=[(words,label) for words,label in aliases if len(words)>1]
    def blocked(prefix,word,label):
        tokens=[normalize(w) for w in prefix]+[normalize(word)]
        return any(len(tokens)>=len(words) and tuple(tokens[-len(words):])==words and other!=label
                   for words,other in compounds)
    examples=[('a hot','dog','dog',True),('a playful','dog','dog',False),
        ('a hot','dogs','dog',True),('a cell','phone','cell phone',False)]
    for prefix,word,label,expected in examples:assert blocked(prefix.split(),word,label)==expected
    path=COCO/'review/noun_box_train10k_v1_1/spans.json'
    records=load(path);before=Counter();after=Counter();skipped=[]
    masks=load(COCO/'review/noun_box_train10k_v1_1/mask_bank.json');regions={}
    for row in records:
        box=masks[row['mask_key']]
        assert box['coco_id']==row['coco_id'] and box['category']==row['category']
        regions[(row['coco_id'],box['annotation_id'],row['category'])]=box
    region_counts=Counter(box['category'] for box in regions.values())
    for row in records:
        if len(row['raw_span'])!=1:continue
        word=normalize(row['raw_span'][0]);label=mapping.get(word)
        if label!=row['category']:continue
        before[label]+=1
        prefix=row['caption'].lower().split()[:row['decoder_logit_positions'][0]]
        if blocked(prefix,word,label):skipped.append(row['span_id'])
        else:after[label]+=1
    total_nonperson=sum(row['category']!='person' for row in records)
    nonperson=sum(n for label,n in after.items() if label!='person')
    classes50=sum(label!='person' and n>=50 for label,n in after.items())
    a=load(ROOT/'experiments/vg_region_caption/review/loaded_A.json')['json']
    train={r['image_id'] for r in a['groups.json']['train']}
    secondary={r['image_id'] for r in a['groups.json']['val']}
    data_path=Path('C:/KhoaLuanTotNghiep/Code/output/vcoco_alignment_week1/sources/dataset_coco.json')
    assert sha(data_path)=='2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
    data=load(data_path)
    val={i for i,r in enumerate(data['images']) if r['split']=='val'}
    assert len(train)==10000 and len(val)==5000 and len(secondary)==1000 and secondary<=val and not train&val
    old=load(ROOT/'audit/caption_candidate_headroom_val5000.json')
    assert old['split']=='val' and old['images']==5000
    # These are class/surface candidates, not a final vocabulary whitelist or POS parser.
    stats={}
    for name,items in [('train10k',[data['images'][i] for i in sorted(train)]),('val5000',[data['images'][i] for i in sorted(val)])]:
        captions=[s['raw'] for row in items for s in row['sentences'][:5]]
        hits=[sum(normalize(w) in mapping and not blocked(words[:t],w,mapping[normalize(w)])
                   for t,w in enumerate(words)) for words in [s.lower().split() for s in captions]]
        stats[name]=dict(images=len(items),captions=len(captions),captions_with_candidate=sum(n>0 for n in hits),mentions=sum(hits))
    generated=[row['caption'].lower().split() for row in a['on_predictions.json']]
    hits=[sum(normalize(w) in mapping and not blocked(words[:t],w,mapping[normalize(w)]) for t,w in enumerate(words)) for words in generated]
    stats['A_generated_dev1000']=dict(captions=1000,captions_with_candidate=sum(n>0 for n in hits),mentions=sum(hits))
    result=dict(status='PRE_POLICY_DATA_FEASIBILITY_PASS_CHECKPOINT_BYTES_PENDING',model_run=False,train_notebook_created=False,
        expected_A_checkpoint_sha256=a['run_summary.json']['checkpoint_sha256'],
        user_reports_A_checkpoint_available_on_Kaggle=True,exact_A_bytes_locally_verified=False,
        alias_surfaces=len(mapping),covered_classes=len(after),manifest_spans=len(records),
        before_guard=sum(before.values()),after_guard=sum(after.values()),fraction=sum(after.values())/len(records),
        nonperson_spans=total_nonperson,covered_nonperson=nonperson,nonperson_fraction=nonperson/total_nonperson,
        nonperson_classes_with_at_least50=classes50,
        deduplicated_region_supervision=dict(regions=len(regions),images=len({k[0] for k in regions}),
            person_regions=region_counts['person'],person_fraction=region_counts['person']/len(regions),classes=len(region_counts)),
        feasibility_gates=dict(nonperson_at_least_half=nonperson/total_nonperson>=.5,nonperson_classes50_at_least15=classes50>=15),
        compound_blocked_manifest_spans=skipped,compound_examples_checked=examples,class_counts=dict(after),coverage=stats,
        split=dict(train=10000,primary_val=5000,secondary_old_dev=1000,train_primary_overlap=0,primary_historically_reused=True,test_used=False),
        sources_sha256=dict(manifest=sha(path),alias_code=sha(source),mapping_policy=sha(COCO/'MAPPING_POLICY_V11.json'),dataset_json=sha(data_path)),
        limits=['Pre-policy coverage audit; not a final whitelist, training implementation or caption performance result.',
            'Finite known-compound guard is not a perfect semantic/POS parser; globally contextual CLIP patches are not pure local evidence.',
            'Validation5000 has been used previously; a subset cannot restore independent test status.'])
    (HERE/'FEASIBILITY.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','after_guard','nonperson_fraction','nonperson_classes_with_at_least50','feasibility_gates','coverage','split']},indent=2))


if __name__=='__main__':main()
