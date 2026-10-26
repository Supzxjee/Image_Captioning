"""Package metadata from the audited 138-image review. No model or tokenizer execution."""
from collections import Counter,defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'experiments/attention_geometry_diagnostic'))
from diagnostic_data import save,digest
from utility_data import choose_pair,q_choice,POLICY


def main():
    audit=ROOT/'experiments/attention_geometry_diagnostic'
    loaded=json.loads((audit/'review/loaded_second.json').read_text(encoding='utf-8'))
    cfg=json.loads(loaded['protocol.json']);receipt=json.loads(loaded['receipt.json'])
    assert receipt['status']=='DIAGNOSTIC_COMPLETE' and receipt['images']==138
    oldrows=sum([json.loads(loaded[f'rows_rank{rank}.json']) for rank in (0,1)],[])
    spanfile=ROOT/'experiments/coco_grounding_feasibility/review/noun_box_train10k_v1_1/spans.json'
    assert digest(spanfile)=='6e4ec31245a63ea2a1f50f60435fdf526dd2fd6e5f09b835543866ec33fafb7b'
    spans=defaultdict(lambda:defaultdict(list))
    for s in json.loads(spanfile.read_text(encoding='utf-8')):spans[s['coco_id']][s['category']].append(s)
    rows=[];counts=Counter()
    for rank in (0,1):
        path=audit/f'review/maps_rank{rank}.npz'
        assert digest(path)==receipt['payload_sha256'][path.name]
        with np.load(path,allow_pickle=False) as maps:
            for r in oldrows[rank*69:(rank+1)*69]:
                i,j=choose_pair(r);receiver,donor=r['phrases'][i],r['phrases'][j]
                chosen=q_choice(maps['q_full_'+str(r['coco_id'])],receiver,donor)
                counts['q_distinguishable' if chosen['distinguishable'] else 'q_indistinguishable']+=1
                targets={role:spans[r['coco_id']][p['category']] for role,p in [('receiver',receiver),('donor',donor)]}
                rows.append(dict(coco_id=r['coco_id'],filename=r['filename'],
                    image_id=next(s['image_id'] for s in cfg['selected'] if s['coco_id']==r['coco_id']),
                    prompt=r['prompt'],input_ids=next(s['input_ids'] for s in cfg['selected'] if s['coco_id']==r['coco_id']),
                    receiver=receiver,donor=donor,selection=chosen,targets=targets))
    rows.sort(key=lambda r:r['image_id'])
    payload=dict(policy=POLICY,rows=rows,prior_receipt_sha256=hashlib.sha256(loaded['receipt.json'].encode()).hexdigest(),
        prior_review_sha256='b387d808172f272612e61793bb9e5c3fd608e40a4fa0e663e931eadcc473d20f',
        original_text_sha256=json.loads(loaded['text_source.json'])['sha256'],
        first_mention_source='sealed conservative v1.1 spans; recheck against actual first class mention/reference and tokenizer on Kaggle',
        source_span_sha256=digest(spanfile),metadata_counts=dict(counts),test_used=False)
    save(HERE/'cohort.json',payload)
    print(json.dumps(dict(images=len(rows),counts=dict(counts),cohort_sha256=digest(HERE/'cohort.json'))))


if __name__=='__main__':main()
