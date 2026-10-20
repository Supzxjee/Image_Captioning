"""Validate self-reported human reviews; unreviewed/unclear never become negatives."""
import argparse
import json
from pathlib import Path


def summarize(manifest,report):
    assert report['schema_version']==1 and report['groups_sha256']==manifest['groups_sha256'],'Wrong manifest identity'
    assert report['reviewer_type']=='human_self_reported' and report['reviewer'].strip()
    known={g['group_id']:g for g in manifest['groups']};reviews=report['annotations']
    assert set(reviews)<=set(known),'Unknown group'
    result=dict(groups_total=len(known),groups_with_any_response=len(reviews),groups_complete=0,
        clear_semantic_contrasts=0,contrasts_with_clear_boxes=0,contrasts_with_clear_boxes_and_visible_224=0,
        same_image_clear_contrasts=0,both_descriptions_supported_events=0,unreviewed_groups=len(known)-len(reviews),
        training_approved=False,status='HUMAN_FEASIBILITY_REVIEW_ONLY')
    groups=[]
    for key,a in reviews.items():
        for field,allowed in (('truth',{'unreviewed','supported','contradicted','unclear'}),('boxes',{'unreviewed','clear','incorrect','unclear'})):
            assert len(a[field])==2 and all(len(row)==2 and set(row)<=allowed for row in a[field]),field
        assert len(a['detail_224'])==2 and set(a['detail_224'])<={'unreviewed','yes','no','unclear'}
        assert isinstance(a['notes'],str)
        complete=all(v!='unreviewed' for field in ('truth','boxes') for row in a[field] for v in row) and 'unreviewed' not in a['detail_224']
        t=a['truth'];contrast=t in ([['supported','contradicted'],['contradicted','supported']], [['contradicted','supported'],['supported','contradicted']])
        boxes=all(v=='clear' for row in a['boxes'] for v in row);visible=a['detail_224']==['yes','yes']
        result['groups_complete']+=complete;result['clear_semantic_contrasts']+=contrast
        result['contrasts_with_clear_boxes']+=contrast and boxes
        result['contrasts_with_clear_boxes_and_visible_224']+=contrast and boxes and visible
        result['same_image_clear_contrasts']+=contrast and boxes and known[key]['same_image']
        result['both_descriptions_supported_events']+=sum(row==['supported','supported'] for row in t)
        groups.append(dict(group_id=key,complete=complete,semantic_contrast=contrast,boxes_clear=boxes,detail_224_visible=visible))
    return dict(summary=result,groups=groups,limits=['Self-reported human labels; no inter-reviewer adjudication',
        'Pixel visibility at224 and patch counts do not prove CLIP features preserve relation evidence',
        'Feasibility review is not a training/probe split or model result'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    a=p.parse_args();print(json.dumps(summarize(json.loads(a.manifest.read_text(encoding='utf-8')),json.loads(a.review.read_text(encoding='utf-8'))),indent=2))
