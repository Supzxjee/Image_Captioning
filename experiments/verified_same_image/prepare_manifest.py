"""Prepare fit-only relation supervision and untouched calibration/probe partitions."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'experiments/semantic_evidence_probe')]
from captioning.supervised_relations import decode_action,normalized_box
from captioning.upt_prompts import relation_text
from probe_data import digest,identity
from verified_data import construct,counts,training_schedule


def main():
    source=Path('C:/KhoaLuanTotNghiep/Code/output/vcoco_alignment_week1')
    paths=dict(actions=source/'sources/vcoco_train.json',instances=source/'sources/instances_vcoco_trainval2014.json',
               dataset=source/'sources/dataset_coco.json',groups=source/'inventory_v1/locked_image_groups.json',
               probe=ROOT/'experiments/semantic_evidence_probe/probe_manifest.json')
    data={k:json.loads(p.read_text(encoding='utf-8')) for k,p in paths.items()}
    assert digest(paths['actions'])=='2267afe7adc0a6c3e8ac90fb35f6d3a5bd5cd3c7edebf0622d0cbb7a67fddf19'
    assert digest(paths['dataset'])=='2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
    captions={r['cocoid']:r for r in data['dataset']['images']}
    probe=data['probe']['rows']; forbidden={r['coco_id'] for r in probe}
    fit,compatibility=construct(data['actions'],data['instances'],captions,data['groups']['relation_fit'],forbidden,None,decode_action,normalized_box,relation_text)
    calibration,_=construct(data['actions'],data['instances'],captions,data['groups']['relation_calibration'],forbidden,compatibility,decode_action,normalized_box,relation_text)
    splits=dict(fit=fit,calibration=calibration,probe=probe)
    ids={k:{r['coco_id'] for r in v} for k,v in splits.items()}
    assert not ids['fit']&ids['calibration'] and not ids['fit']&ids['probe'] and not ids['calibration']&ids['probe']
    assert all(r['caption_split'] in ('train','restval') for rows in splits.values() for r in rows)
    manifest=dict(schema_version=1,method='verified_same_image_F',splits=splits,counts={k:counts(v) for k,v in splits.items()},
                  sources_sha256={k:digest(p) for k,p in paths.items()},fit_compatibility=sorted([list(x) for x in compatibility]),
                  split_sha256={k:identity(v) for k,v in splits.items()},schedule=training_schedule(fit),test_used=False,
                  policy='Fit uses explicit actor/action zero only. Missing role is UNKNOWN. Predicate pairs balance both directions within object class/action pair. No probe/calibration loss.',
                  caption_overlap='All auxiliary images are caption train/restval, historically seen by the backbone; relation partitions are disjoint, not unseen caption tests.',
                  object_instance_scope='Wrong-object probe is transfer only; no unconfirmed object negatives are synthesized.')
    output=Path(__file__).parent/'verified_manifest.json';output.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest['counts'],indent=2))


if __name__=='__main__':main()
