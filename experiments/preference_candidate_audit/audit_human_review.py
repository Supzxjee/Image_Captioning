"""Validate user-confirmed AI-assisted labels without inferring missing judgments."""
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def summarize(review,key,fingerprint,ai):
    if review.get('schema')!=1 or review.get('manifest_sha256')!=fingerprint:raise ValueError('Wrong review schema/fingerprint.')
    if review.get('reviewer_type')!='human_verified_ai_assisted' or review.get('human_verified') is not True or not review.get('reviewer','').strip():raise ValueError('Missing human-assisted provenance.')
    labels=review.get('annotations',{});verified=review.get('verified',{})
    if len(key)!=30 or set(labels)!=set(key) or set(verified)!=set(key):raise ValueError('Wrong/missing pair IDs.')
    if any(v is not True for v in verified.values()):raise ValueError('Unverified pairs remain.')
    if any(v not in ('X','Y','tie','unclear') for v in labels.values()):raise ValueError('Missing/unknown choices.')
    agree=[i for i in key if labels[i]==key[i]]
    oppose=[i for i in key if labels[i] in ('X','Y') and labels[i]!=key[i]]
    ties=[i for i in key if labels[i]=='tie'];unclear=[i for i in key if labels[i]=='unclear']
    changes=[dict(coco_id=int(i),AI=ai[i],human=labels[i]) for i in key if labels[i]!=ai[i]]
    return dict(status='HUMAN_ASSISTED_REVIEW_COMPLETE',records=30,reviewer_type=review['reviewer_type'],
        manifest_sha256=fingerprint,choices=dict(Counter(labels.values())),
        agrees_with_selected_preference=len(agree),opposes_selected_preference=len(oppose),ties=len(ties),unclear=len(unclear),
        agreement_fraction=len(agree)/30,agreement_among_decisive=len(agree)/max(1,len(agree)+len(oppose)),
        agrees_coco_ids=list(map(int,agree)),opposes_coco_ids=list(map(int,oppose)),ties_coco_ids=list(map(int,ties)),unclear_coco_ids=list(map(int,unclear)),
        changes_from_AI=changes,minimum_agreement_required=27,
        agreement_gate_pass=len(agree)>=27,
        decision='STOP_CURRENT_AUTOMATIC_PAIR_RECIPE' if len(agree)<27 else 'Agreement gate passes; still inspect systematic errors before any expansion.',
        limits=['Human confirmed AI-prefilled labels; not an independent unassisted review.',
                'Small deterministic30-pair screen, not a population label-accuracy estimate.',
                'Ties/unclear are retained and do not become wins; denominator remains30.',
                'No systematic-error absence is inferred from empty user notes.'])


def main():
    source=Path('C:/Users/leduc/Downloads/candidate_pair_assisted_verified_review.json')
    raw=source.read_bytes();review=json.loads(raw)
    loaded=json.loads((HERE/'review/loaded.json').read_text(encoding='utf-8'));files=loaded['files']
    key=json.loads(files['review_key.json']);fingerprint=hashlib.sha256(files['eligible_pairs.json'].encode()).hexdigest()
    ai=json.loads((HERE/'review/assistant_preannotations.json').read_text(encoding='utf-8'))['annotations']
    result=summarize(review,key,fingerprint,ai);result['review_sha256']=hashlib.sha256(raw).hexdigest()
    (HERE/'review/human_assisted_review_original.json').write_bytes(raw)
    (HERE/'HUMAN_REVIEW_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
