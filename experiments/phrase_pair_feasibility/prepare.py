"""Offline candidate audit preparation; no tensor/model execution, no automatic semantic labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from captioning.supervised_relations import decode_action,normalized_box
from data import collect,choose,identity,proposal_match


def main(base,prompt_zip,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    paths=dict(actions_train=base/'sources/vcoco_train.json',actions_val=base/'sources/vcoco_val.json',
        instances=base/'sources/instances_vcoco_trainval2014.json',dataset=base/'sources/dataset_coco.json',
        F=ROOT/'experiments/verified_same_image/verified_manifest.json',
        old_prompt_review=base/'upt_prompt_prototype_v1_review/prompt_review_50.json',
        old_probe=ROOT/'experiments/semantic_evidence_probe/probe_manifest.json')
    values={k:load(p) for k,p in paths.items()}
    excluded={r['coco_id'] for rows in values['F']['splits'].values() for r in rows}
    excluded|={r['coco_id'] for r in values['old_probe']['rows']}
    by_name={r['filename']:r for r in values['dataset']['images']}
    excluded|={by_name[r['filename']]['cocoid'] for r in values['old_prompt_review'] if r['filename'] in by_name}
    captions={r['cocoid']:r for r in values['dataset']['images']}
    pools,filtered=collect({paths[k].name:values[k] for k in ('actions_train','actions_val')},values['instances'],captions,
        excluded,decode_action,normalized_box)
    groups,available=choose(pools,30)
    assert len(groups)==30,('Not enough candidates; do not pad or weaken labels silently',len(groups),available)
    with zipfile.ZipFile(prompt_zip) as z:
        assert z.testzip() is None
        raw=z.read('prompt_cache.json');cache=json.loads(raw)
    prompt_sha=hashlib.sha256(raw).hexdigest();del raw
    for group in groups:
        for event in group['events']:
            event['upt_selected_objects']=proposal_match(event,cache.get(event['filename'],{}))
    del cache
    manifest=dict(schema_version=1,status='CANDIDATES_READY_NO_HUMAN_LABELS',groups=groups,groups_sha256=identity(groups),
        source_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},prompt_json_sha256=prompt_sha,
        excluded_known_review_image_ids=sorted(excluded),selection='Fixed hash, family round-robin, same-image first within family; no image reused across groups; no model error selection',
        no_training=True,test_used=False,policy='All four truth cells require human review. Annotation/proxy descriptions are candidates, not gold. No split allocation for future training.',
        prior_exposure_limit='Known F/probe/50-prompt IDs excluded. Other historical exposures/near-duplicates not exhaustively checked. All images already belong to caption train/restval.')
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    events=[e for g in groups for e in g['events']]
    summary=dict(status=manifest['status'],groups=30,event_views=60,unique_images=len({e['image_id'] for e in events}),
        same_image_groups=sum(g['same_image'] for g in groups),families=dict(Counter(g['family'] for g in groups)),
        candidate_pool=available,filtered_pool_notes=filtered,excluded_known_review_images=len(excluded),
        support=dict(zero_patch_event_views=sum(min(e['patch_centers'].values())==0 for e in events),
                     under_4_patch_event_views=sum(min(e['patch_centers'].values())<4 for e in events)),
        upt_selected_object_pair_matches_at_iou05=sum(e['upt_selected_objects']['pair_found_at_iou05'] for e in events),
        upt_scope='Selected objects in prompt JSON only. Missing pair does not imply raw detector missed it. Selected-only coverage is not semantic accuracy.',
        human_groups_completed=0,human_usable_groups=None,training_started=False,
        next='Review at most first 10 groups initially. Stop and inspect feasibility before expanding; no training until truth matrices and quality are reviewed.')
    (output/'preparation_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--prompt-zip',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.base,a.prompt_zip,a.output)
