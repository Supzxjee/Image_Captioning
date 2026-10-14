"""Filter explicit human-reviewed target contrasts; preserve unknown/conflicting cases."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path


def filter_annotations(review,mapping):
    if review.get('schema_version')!=1 or review.get('manifest_sha256')!=mapping['manifest_sha256']:
        raise ValueError('Review schema/manifest mismatch.')
    if review.get('reviewer_type')!='human_self_reported' or not review.get('reviewer','').strip():
        raise ValueError('Explicit named human review required; self-reported provenance is not independent certification.')
    index={c['case_id']:c for c in mapping['cases']};annotations=review.get('annotations',{})
    if set(annotations)-set(index):raise ValueError('Unknown case IDs in review.')
    accepted=[];rejected=[];counts=Counter()
    for case in mapping['cases']:
        a=annotations.get(case['case_id'],{});positive=case['positive_side'];alternate='B' if positive=='A' else 'A'
        for side in ('A','B'):
            if a.get(side+'_status','') not in ('','supported','contradicted','unclear'):raise ValueError('Invalid relation label.')
        for field in ('subject_box_status','A_box_status','B_box_status'):
            if a.get(field,'') not in ('','clear','incorrect','unclear'):raise ValueError('Invalid box label.')
        if not all(a.get(k) for k in ('subject_box_status','A_box_status','B_box_status','A_status','B_status')):reason='incomplete'
        elif not all(a[k]=='clear' for k in ('subject_box_status','A_box_status','B_box_status')):reason='localization_not_clear'
        elif a[positive+'_status']=='contradicted':reason='source_positive_conflict'
        elif a[positive+'_status']!='supported':reason='reference_unconfirmed'
        elif a[alternate+'_status']=='supported':reason='both_supported_no_negative'
        elif a[alternate+'_status']!='contradicted':reason='alternative_unknown'
        elif not a.get('notes','').strip():reason='contradiction_missing_evidence_note'
        else:reason='eligible'
        counts[reason]+=1
        if reason=='eligible':accepted.append({**case,'reviewer':review['reviewer'],'evidence_note':a['notes'],
                                              'positive_label':1,'alternate_label':0})
        else:rejected.append({'case_id':case['case_id'],'reason':reason})
    return {'training_performed':False,'counts':dict(counts),'eligible_contrasts':accepted,'excluded_cases':rejected,
            'notes':'Human reviewer self-reported;notindependent consensus. Only confirmed source-positive / alternative-contradicted clear boxes eligible. Original labels unchanged;uncertain/conflicting cases notnegative.'}


def main():
    p=argparse.ArgumentParser()
    for name in ('review','package','output'):p.add_argument('--'+name,required=True)
    args=p.parse_args();package=Path(args.package)
    mapping=json.loads((package/'private_mapping.json').read_text(encoding='utf-8'))
    review=json.loads(Path(args.review).read_text(encoding='utf-8-sig'))
    if hashlib.sha256((package/'cases.json').read_bytes()).hexdigest()!=mapping['manifest_sha256']:raise ValueError('Case manifest changed.')
    if hashlib.sha256(Path('audit/interaction_region_train_pairs.json').read_bytes()).hexdigest()!=mapping['source_pairs_sha256']:
        raise ValueError('Training pair source changed; resolve provenance before importing.')
    result=filter_annotations(review,mapping)
    result['provenance']={'review_sha256':hashlib.sha256(Path(args.review).read_bytes()).hexdigest(),
                          'manifest_sha256':mapping['manifest_sha256'],'source_pairs_sha256':mapping['source_pairs_sha256']}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result['counts'],indent=2))


if __name__=='__main__':main()
