"""Count observed ambiguity in existing head/name labels; no new NLP/model run."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def main():
    source=ROOT/'experiments/coco_grounding_feasibility/review/noun_box_train10k_v1_1/dataset'
    receipt=json.loads((source/'manifest_receipt.json').read_text())
    for name,expected in receipt['payload_sha256'].items():
        assert Path(name).name==name and hashlib.sha256((source/name).read_bytes()).hexdigest()==expected
    spans=json.loads((source/'spans.json').read_text())
    masks=json.loads((source/'mask_bank.json').read_text())
    old=json.loads((ROOT/'experiments/stale_prompt_pilot/review/C_loaded.json').read_text(encoding='utf-8'))
    vocab=json.loads(old['texts']['protocol.json'])['word2idx'];groups=defaultdict(dict);raw_prefixes=defaultdict(set)
    for s in spans:
        words=s['caption'].lower().split();ids=[vocab['<bos>']]+[vocab.get(w,vocab['<unk>']) for w in words]
        positions=s['target_token_positions'];assert [ids[i] for i in positions]==s['token_ids']
        first=min(positions);key=(s['coco_id'],tuple(ids[:first]));phrase=' '.join(s['raw_span']).lower().strip(' .,;:!?')
        annotation_id=masks[s['mask_key']]['annotation_id']
        groups[key][(phrase,annotation_id)]=dict(phrase=phrase,annotation_id=annotation_id,category=s['category'])
        raw_prefixes[key].add(' '.join(words[:first-1]))
    multi={k:v for k,v in groups.items() if len(v)>1}
    multibox={k:v for k,v in multi.items() if len({a for _,a in v})>1}
    distribution=Counter(len(v) for v in groups.values())
    result=dict(date='2026-10-27',scope='Existing head/name spans only, before modifier expansion; metadata counts, not a model metric or representative language ambiguity rate.',
        spans=len(spans),prefix_groups=len(groups),pair_count_distribution=dict(sorted(distribution.items())),
        singleton_prefix_groups=sum(len(v)==1 for v in groups.values()),multi_pair_prefix_groups=len(multi),
        multi_pair_fraction=len(multi)/len(groups),multi_box_prefix_groups=len(multibox),multi_box_fraction=len(multibox)/len(groups),
        images_with_multi_pair_prefix=len({k[0] for k in multi}),images_with_multi_box_prefix=len({k[0] for k in multibox}),
        same_box_lexical_alternatives_only=len(multi)-len(multibox),
        raw_prefix_collision_groups=sum(len(raw_prefixes[k])>1 for k in groups),
        multi_pairs_with_UNK_prefix=sum(vocab['<unk>'] in k[1] for k in multi),
        receipt_sha256=hashlib.sha256((source/'manifest_receipt.json').read_bytes()).hexdigest(),
        policy='Exact A token-ID prefix before first head/name token; deduplicate normalized raw head/name plus annotation ID; no generated captions or scores.',
        examples=[dict(coco_id=k[0],prefix=list(raw_prefixes[k]),pairs=list(v.values())) for k,v in sorted(multibox.items(),key=lambda x:hashlib.sha256(str(x[0]).encode()).hexdigest())[:8]],
        limits=['Expanded modifiers may change prefix grouping; these are not full-NP counts.','Repeated or same-box aliases do not establish cross-region semantic binding need.','No count demonstrates originality or caption benefit.'])
    (HERE/'PREFIX_SET_INVENTORY.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='examples'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
