"""Use explicit per-group human confirmations; never promote a whole mixed AI export."""
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
MONTAGE_IDS={487943,333575}  # Observed image montages, not natural same-time pair contrasts.


def classify(group,answer):
    if answer is None or answer.get('confirmed_by_human') is not True:return ['not_human_confirmed']
    for key,allowed in [('truth',{'supported','contradicted','unclear','unreviewed'}),('boxes',{'clear','incorrect','unclear','unreviewed'})]:
        value=answer.get(key);assert isinstance(value,list) and len(value)==2
        assert all(isinstance(row,list) and len(row)==2 and set(row)<=allowed for row in value),key
    assert len(answer['detail_224'])==2 and set(answer['detail_224'])<={'yes','no','unclear','unreviewed'}
    reasons=[]
    if group['events'][0]['image_id'] in MONTAGE_IDS:reasons.append('photo_montage_not_natural_same_time_scene')
    if answer['truth'] not in ([['supported','contradicted'],['contradicted','supported']],[['contradicted','supported'],['supported','contradicted']]):reasons.append('not_a_clear_two_way_contrast')
    if any(v!='clear' for row in answer['boxes'] for v in row):reasons.append('boxes_not_all_clear')
    # detail224 and inherited assistant ambiguity flags do NOT veto confirmed human choices.
    return reasons


def lock(manifest,report,fit,proposal_sha):
    assert report['schema_version']==1 and report['groups_sha256']==manifest['groups_sha256']
    assert report['assistant_proposal_sha256']==proposal_sha and report['reviewer'].strip()
    assert report['label_origin']=='assistant_draft_then_explicit_human_confirmation'
    assert report['reviewer_type'] in ('human_self_reported','assistant_proposed_pending_human_verification')
    known={g['group_id'] for g in manifest['groups']};assert set(report['annotations'])<=known
    selected=[];audit=[];counts={'skateboard_support':0,'surfboard_support':0}
    reconfirm=None
    for index,g in enumerate(manifest['groups'],1):
        a=report['annotations'].get(g['group_id']);reasons=classify(g,a)
        if g['group_id']=='fit_reconfirm_235799':reconfirm=(g,a,reasons);continue
        family=g['family'];assert family in counts
        take=not reasons and counts[family]<4
        if take:
            counts[family]+=1
            selected.append(dict(g,positive_indices=[row.index('supported') for row in a['truth']],
                verified_truth_matrix=a['truth'],verified_boxes=a['boxes'],human_confirmed=True,
                label_origin='assistant_draft_then_explicit_human_confirmation',split='new_adaptation_probe'))
        audit.append(dict(index=index,group_id=g['group_id'],image_id=g['events'][0]['image_id'],family=family,
            human_confirmed=bool(a and a.get('confirmed_by_human') is True),selected=take,
            reasons=reasons or ([] if take else ['eligible_reserve_after_first_four'])))
    assert reconfirm is not None
    rg,ra,reasons=reconfirm;assert ra and ra.get('confirmed_by_human') is True,'Fit reconfirmation is not complete'
    retained=not reasons
    fitkeys=fit['partitions']['fit'].copy()
    original=next(g for g in fit['groups'] if g['group_id'] in fitkeys and any(e['image_id']==235799 for e in g['events']))
    if retained:assert original['positive_indices']==[r.index('supported') for r in ra['truth']],'Reconfirmed fit targets changed; protocol forbids silent relabel'
    else:fitkeys.remove(original['group_id'])
    summary=dict(status='HUMAN_LABEL_GATE_PASS' if set(counts.values())=={4} else 'INSUFFICIENT_PROBE_DO_NOT_TRAIN',
        per_group_human_confirmed=sum(a.get('confirmed_by_human') is True for a in report['annotations'].values()),
        exported_groups=len(report['annotations']),whole_export_type=report['reviewer_type'],probe_counts=counts,probe_groups=len(selected),
        fit_group_235799_retained=retained,fit_groups=len(fitkeys),audit=audit,training_approved=False,
        pending=['Actual HDF5 metadata','Pinned encoder feature compatibility before training'],
        limits=['One AI-assisted self-reported human review, no inter-reviewer agreement','Only explicitly confirmed groups used; remaining AI entries stay unverified',
                'Old AI notes/flags may disagree with revised choices; confirmed truth/box fields take precedence',
                'Montage exclusion is structural image review before new model outputs, not performance selection',
                'New to adaptation only; caption backbone may have seen train/restval images'])
    ids=lambda groups:{e['image_id'] for g in groups for e in g['events']}
    fitgroups=[g for g in fit['groups'] if g['group_id'] in fitkeys]
    olddev=[g for g in fit['groups'] if g['group_id'] in fit['partitions']['adaptation_holdout']]
    assert not ids(selected)&(ids(fitgroups)|ids(olddev)|set(manifest['excluded_known_ids']))
    assert len(ids(selected))==len(selected)
    return summary,dict(status='LABELS_LOCKED_TRAINING_GATES_PENDING',fit_groups=fitgroups,exposed_development_groups=olddev,new_probe_groups=selected,
                        selection='First four human-confirmed valid natural-scene groups/family in original fixed hash order;224visibility not a filter',training_approved=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);a=p.parse_args()
    raw=a.review.read_bytes();report=json.loads(raw)
    read=lambda path:json.loads(path.read_text(encoding='utf-8'))
    proposal=hashlib.sha256((HERE/'ASSISTANT_PROPOSED_REVIEW.json').read_bytes()).hexdigest()
    summary,data=lock(read(HERE/'CANDIDATE_MANIFEST.json'),report,read(HERE.parent/'phrase_decoder_pilot/locked_manifest.json'),proposal)
    receipt=dict(human_export_sha256=hashlib.sha256(raw).hexdigest(),assistant_proposal_sha256=proposal,groups_sha256=report['groups_sha256'])
    summary.update(receipt);data.update(receipt)
    (HERE/'HUMAN_REVIEW_RESULT.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    if summary['status']=='HUMAN_LABEL_GATE_PASS':(HERE/'LOCKED_DATA.json').write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    (HERE/'review/human_verified_export_original.json').write_bytes(raw)
    print(summary['status'],summary['probe_counts'],'confirmed',summary['per_group_human_confirmed'],'fit',summary['fit_groups'])
