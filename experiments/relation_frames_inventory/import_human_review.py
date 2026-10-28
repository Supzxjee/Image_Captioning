"""Validate AI-assisted human review and explicit chat supplements; no model execution."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
FIELDS=dict(subject_box={'clear','incorrect','uncertain'},object_box={'clear','incorrect','uncertain'},
            assertion={'supported','contradicted','uncertain'})


def digest(raw):return hashlib.sha256(raw).hexdigest()


def validate(selection,draft,report,supplement):
    if report.get('schema_version')!=1:raise ValueError('Unknown schema')
    for name in ('manifest_sha256','audit_sha256'):
        if report.get(name)!=selection[name] or draft[name]!=selection[name]:raise ValueError('Wrong review identity')
    expected={r['review_id']:r for r in selection['records']}
    if len(expected)!=32 or set(report.get('annotations',{}))!=set(expected):raise ValueError('Wrong review record set')
    if not isinstance(report.get('reviewer'),str) or not report['reviewer'].strip():raise ValueError('Reviewer name missing')
    if report.get('reviewer_type') not in ('human_self_reported','assistant_proposed_pending_human_verification','human_incomplete'):
        raise ValueError('Unknown provenance')
    effective={k:dict(a) for k,a in report['annotations'].items()}
    for k,a in effective.items():
        if any(a.get(field) not in allowed for field,allowed in FIELDS.items()):raise ValueError('Missing/invalid field: '+k)
        if type(a.get('confirmed_by_human')) is not bool or not isinstance(a.get('notes'),str):raise ValueError('Invalid confirmation/notes')
    supplemented=[];revisions=[]
    for item in supplement.get('confirmations',[]):
        key=item['review_id']
        if key not in effective or item.get('confirmed_by_human') is not True:raise ValueError('Invalid supplemental confirmation')
        if not isinstance(item.get('human_reply'),str) or not item['human_reply'].strip():raise ValueError('Missing human evidence')
        if any(name in item and item[name]!=expected[key][name] for name in ('index','image_id')):raise ValueError('Wrong supplemental record identity')
        if item.get('action','confirmation')=='assertion_revision':
            if item.get('origin')!='explicit_user_chat_revision' or item.get('previous_assertion')!=effective[key]['assertion'] or item.get('assertion') not in FIELDS['assertion']:
                raise ValueError('Invalid explicit answer revision')
            revision=dict(index=expected[key]['index'],review_id=key,previous_assertion=effective[key]['assertion'],
                          assertion=item['assertion'],human_reply=item['human_reply'])
            effective[key]['assertion']=item['assertion'];effective[key]['chat_assertion_revision']=revision
            revisions.append(revision)
        else:
            if item.get('action','confirmation')!='confirmation' or item.get('origin')!='explicit_user_chat_reply':raise ValueError('Missing human evidence')
            if any(effective[key][f]!=item[f] for f in FIELDS):raise ValueError('Supplement changes an answer instead of confirming it')
        if key in supplemented:raise ValueError('Duplicate supplement')
        effective[key]['confirmed_by_human']=True;supplemented.append(key)
    pending=[];uncertain=[];disagreements=[];box_issues=[];changes=[]
    for key,row in expected.items():
        a=effective[key]
        identity={k:row[k] for k in ('index','review_id','image_id','split','statement','source_label')}
        if not a['confirmed_by_human']:pending.append(identity);continue
        if a['assertion']=='uncertain':uncertain.append(identity)
        elif (a['assertion']=='supported')!=bool(row['source_label']):disagreements.append(identity)
        if any(a[field]!='clear' for field in ('subject_box','object_box')):box_issues.append(identity)
        delta={f:dict(assistant=draft['annotations'][key][f],human=a[f]) for f in FIELDS if a[f]!=draft['annotations'][key][f]}
        if delta:changes.append(dict(identity,changed_fields=delta))
    confirmed=[a for a in effective.values() if a['confirmed_by_human']]
    summary=dict(status='HUMAN_REVIEW_INCOMPLETE' if pending else 'HUMAN_REVIEW_COMPLETE_WITH_UNCERTAINTY' if uncertain else 'HUMAN_REVIEW_COMPLETE',
        reviewer=report['reviewer'],raw_export_type=report['reviewer_type'],raw_file_confirmations=sum(a['confirmed_by_human'] for a in report['annotations'].values()),
        chat_supplements=len(supplemented),explicit_assertion_revisions=revisions,effective_confirmations=len(confirmed),records=32,
        confirmed_assertions=dict(Counter(a['assertion'] for a in confirmed)),
        pending=pending,uncertain=uncertain,definite_source_disagreements=disagreements,box_issues=box_issues,changes_from_assistant=changes,
        training_approved=False,manifest_modified=False,
        limits=['One AI-assisted self-reported human reviewer, not independent double annotation',
                'Uncertain judgments are preserved; no relabel, exclusion or false-label promotion',
                'Agreement with VSR is not proof of perfect labels or whole-cohort correctness',
                'Cache-copy receipt and model preflight are not provided by this review'])
    return summary,effective


def main():
    p=argparse.ArgumentParser();p.add_argument('review',type=Path);p.add_argument('--supplement',type=Path,required=True);args=p.parse_args()
    raw=args.review.read_bytes();report=json.loads(raw)
    supplement_raw=args.supplement.read_bytes();supplement=json.loads(supplement_raw)
    if supplement['review_export_sha256']!=digest(raw):raise ValueError('Supplement belongs to another export')
    selection=json.loads((HERE/'AUDIT32_SELECTION.json').read_text(encoding='utf-8'))
    draft=json.loads((HERE/'ASSISTANT_AUDIT32.json').read_text(encoding='utf-8'))
    result,effective=validate(selection,draft,report,supplement)
    result.update(review_export_sha256=digest(raw),supplement_sha256=digest(supplement_raw),
                  manifest_sha256=selection['manifest_sha256'],audit_sha256=selection['audit_sha256'],local_neural_execution=False)
    (HERE/'review').mkdir(exist_ok=True)
    preserved=HERE/'review/human_export_original.json'
    if preserved.exists() and preserved.read_bytes()!=raw:raise FileExistsError('Preserve previous export under another revision')
    preserved.write_bytes(raw)
    (HERE/'HUMAN_AUDIT32_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Effective human answers are distinct from both source labels and the unchanged raw export.
    (HERE/'HUMAN_AUDIT32_EFFECTIVE.json').write_text(json.dumps(dict(reviewer=report['reviewer'],
        reviewer_type='human_self_reported_with_explicit_chat_supplement',review_export_sha256=digest(raw),
        supplement_sha256=digest(supplement_raw),manifest_sha256=selection['manifest_sha256'],
        audit_sha256=selection['audit_sha256'],training_approved=False,annotations=effective),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
