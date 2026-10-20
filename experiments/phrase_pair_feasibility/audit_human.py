"""Audit a received human review and package feasible groups without approving training."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil
from data import identity
from review_result import summarize
HERE=Path(__file__).resolve().parent


def audit(source,previous):
    manifest=json.loads((HERE/'candidate_manifest.json').read_text(encoding='utf-8'))
    assert identity(manifest['groups'])==manifest['groups_sha256']
    review=json.loads(source.read_text(encoding='utf-8'));result=summarize(manifest,review)
    assert result['summary']['groups_complete']==30,'This final audit requires all30 groups completed'
    prior=json.loads(previous.read_text(encoding='utf-8'));assert prior['groups_sha256']==review['groups_sha256']
    changed=[key for key,value in prior['annotations'].items() if review['annotations'].get(key)!=value]
    groups={g['group_id']:g for g in manifest['groups']};eligible=[];rows=[]
    for outcome in result['groups']:
        group=groups[outcome['group_id']];annotation=review['annotations'][group['group_id']]
        index=manifest['groups'].index(group)+1
        reason=('not_a_clear_two_way_contrast' if not outcome['semantic_contrast'] else
                'box_not_clear' if not outcome['boxes_clear'] else
                '224_detail_not_clear_for_both_events' if not outcome['detail_224_visible'] else 'feasible_for_technical_pilot')
        rows.append(dict(outcome,index=index,family=group['family'],same_image=group['same_image'],reason=reason,
            human_truth_matrix=annotation['truth'],human_box_status=annotation['boxes'],human_detail_224=annotation['detail_224']))
        if reason=='feasible_for_technical_pilot':
            item=copy.deepcopy(group)
            item.update(split='UNASSIGNED_FEASIBILITY',human_truth_matrix=annotation['truth'],
                truth_status='HUMAN_SELF_REPORTED_NOT_ADJUDICATED',training_eligible=False,
                target_description_indices=[row.index('supported') for row in annotation['truth']],
                human_box_status=annotation['boxes'],human_detail_224=annotation['detail_224'])
            eligible.append(item)
    events=[e for g in eligible for e in g['events']]
    result.update(status='RECEIVED_HUMAN_REVIEW_AUDITED',groups_sha256=manifest['groups_sha256'],
        source_review_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),reviewer_type='human_self_reported',
        independent_reviewers=1,changed_earlier_group_ids=changed,groups=rows,
        additional=dict(eligible_groups=len(eligible),eligible_event_views=len(events),eligible_unique_images=len({e['image_id'] for e in events}),
            eligible_same_image_groups=sum(g['same_image'] for g in eligible),eligible_families=dict(Counter(g['family'] for g in eligible)),
            eligible_selected_upt_pairs_found=sum(e['upt_selected_objects']['pair_found_at_iou05'] for e in events),
            eligible_groups_both_selected_upt_pairs_found=sum(all(e['upt_selected_objects']['pair_found_at_iou05'] for e in g['events']) for g in eligible),
            eligible_zero_patch_event_views=sum(min(e['patch_centers'].values())==0 for e in events)),
        decision='Stop this annotation round. Feasibility supports a small technical pilot, not an efficacy study. No training/model launch; split remains unassigned.')
    result['limits']+=['Purposefully selected feasibility examples; all reviewed before architecture/protocol finalization, not an independent benchmark',
        'No exhaustive near-duplicate or all-history exposure audit; caption train/restval potentially seen by backbone',
        'UPT match measures selected JSON objects only; not raw proposal recall or action correctness',
        'One bicycle group, no eligible chair/horse groups; not broad relation coverage']
    bundle=dict(schema_version=1,status='FEASIBLE_GROUPS_UNASSIGNED_NO_TRAINING_APPROVAL',
        parent_groups_sha256=manifest['groups_sha256'],review_sha256=result['source_review_sha256'],
        groups=eligible,groups_sha256=identity(eligible),training_approved=False,test_used=False,
        note='Reviewed feasibility examples only. Lock adaptation partitions and protocol separately; never treat these as a newly unseen benchmark.')
    (HERE/'HUMAN_REVIEW_30_RESULT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    (HERE/'FEASIBLE_GROUPS_21.json').write_text(json.dumps(bundle,indent=2,ensure_ascii=False),encoding='utf-8')
    shutil.copy2(source,HERE/'review/human_review_30_original.json')
    print(json.dumps(dict(summary=result['summary'],additional=result['additional'],changed_earlier_group_ids=changed),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--previous',type=Path,required=True)
    a=p.parse_args();audit(a.source,a.previous)
