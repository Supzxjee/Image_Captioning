"""Seal the one approved correction, preserve fixed16, package seven JSON files for Kaggle."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zipfile
import numpy as np
from manifest_policy_v11 import POLICY,context_exclusion
from prepare_manifest import rectangle_mask

HERE=Path(__file__).resolve().parent
OLD=HERE/'review/noun_box_train10k_v1'
NEW=HERE/'review/noun_box_train10k_v1_1'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def encoded(value):return (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()


def main():
    before=json.loads((OLD/'spans.json').read_text(encoding='utf-8'))
    rows=json.loads((NEW/'spans.json').read_text(encoding='utf-8'))
    masks=json.loads((NEW/'mask_bank.json').read_text(encoding='utf-8'))
    schedule=json.loads((NEW/'supervision_schedule.json').read_text(encoding='utf-8'))
    result=json.loads((NEW/'preparation_summary.json').read_text(encoding='utf-8'))
    assert all(result['gates'].values()) and result['policy']==POLICY
    for name,expected in result['payload_sha256'].items():assert sha((NEW/name).read_bytes())==expected,name
    old_ids={r['span_id'] for r in before};new_ids={r['span_id'] for r in rows}
    assert len(new_ids)==len(rows) and new_ids<=old_ids
    assert len(schedule)==10000 and len({r['coco_id'] for r in schedule})==10000
    scheduled=[s for r in schedule for s in r['span_ids']]
    assert set(scheduled)==new_ids and len(scheduled)==len(rows)
    byid={r['span_id']:r for r in rows}
    for s in schedule:
        assert all(byid[identifier]['coco_id']==s['coco_id'] for identifier in s['span_ids'])
    for key,box in masks.items():
        expected=rectangle_mask(box['bbox_xywh'],*box['image_size'])
        assert expected is not None
        np.testing.assert_allclose(expected['patch_area_fraction'],box['patch_area_fraction'],rtol=0,atol=1e-12)
        assert 4<=box['patch_area_equivalents'] and box['image_fraction']<=.5
    for r in rows:
        box=masks[r['mask_key']];assert box['category']==r['category'] and box['coco_id']==r['coco_id']
        assert r['target_token_positions']==[p+1 for p in r['decoder_logit_positions']]
        assert len(r['raw_span'])==len(r['token_ids'])==len(r['decoder_logit_positions'])
        assert not set(r['token_ids'])&{0,1,2,3}
    removed=Counter()
    for r in before:
        if r['span_id'] in new_ids:continue
        raw=r['caption'].lower().split();norm=[re.sub(r'^[^a-z0-9]+|[^a-z0-9]+$','',t) for t in raw]
        start,end=r['span_words']
        reason='ambiguous_surface' if r['surface'] in POLICY['ambiguity_exclusions'] else context_exclusion(norm,start,end,r['category'],raw_tokens=raw)
        if reason is None:
            assert r['category']=='person';reason='person_cap'
        removed[reason]+=1
    assert sum(removed.values())==len(before)-len(rows)
    initial=json.loads((HERE/'MANUAL16_INITIAL_RESULT.json').read_text(encoding='utf-8'))
    sample=json.loads((NEW/'manual16.json').read_text(encoding='utf-8'))
    assert [r['span_id'] for r in sample]==[r['span_id'] for r in initial['records']]
    for r,old in zip(sample,initial['records']):
        assert r['v11_kept']==(r['span_id'] in new_ids)
        if old['status']=='mapping_ambiguity':assert not r['v11_kept'] and r['v11_rejection_reason']
    effects=dict(status='FIXED16_GENERAL_RULE_CORRECTIONS_VERIFIED',reviewer_type='assistant_not_human_gold',
        original16_reused=True,new_sample_selected=False,known_problem_cases_removed=True,
        kept=sum(r['v11_kept'] for r in sample),removed=sum(not r['v11_kept'] for r in sample),
        note='13 compatible class-level cases retained; train/suitcase ambiguities removed; valid green-apple example also removed by conservative color rule. Not a precision estimate.',
        records=[dict(order=i+1,span_id=r['span_id'],coco_id=r['coco_id'],category=r['category'],
            kept=r['v11_kept'],rejection=r['v11_rejection_reason']) for i,r in enumerate(sample)])
    provenance=result['source_provenance']
    payloads={name:(NEW/name).read_bytes() for name in ['policy.json','spans.json','mask_bank.json','supervision_schedule.json']}
    payloads['source_provenance.json']=encoded(provenance)
    payloads['manual_review_effects.json']=encoded(effects)
    receipt=dict(schema='coco_noun_box_train10k_v11',status='DATA_READY_FOR_PILOT_DESIGN',
        payload_sha256={n:sha(raw) for n,raw in payloads.items()},
        retained_images=result['retained_images'],retained_spans=len(rows),
        retained_classes=result['retained_classes'],classes_with_50_spans=result['classes_with_at_least_50_spans'],
        person_spans=result['person_spans'],person_fraction=result['person_fraction'],
        main_caption_images=10000,main_caption_labels_changed=False,
        gates=result['gates'],vocabulary_sha256=result['vocabulary_sha256'],
        manual_mapping_review='Fixed16 reused; known ambiguities removed with general text rules, not per-image edits.',
        annotations_scope='TRAIN10k only; no held-out supervision',weak_supervision_not_gold=True,
        schedule_scope=result['schedule_scope'],test_used=False,model_run=False)
    payloads['manifest_receipt.json']=encoded(receipt)
    directory=NEW/'dataset';directory.mkdir(exist_ok=True)
    for name,raw in payloads.items():
        path=directory/name
        if path.exists():assert path.read_bytes()==raw,'Do not overwrite a sealed dataset.'
        else:path.write_bytes(raw)
    archive=HERE/'review/coco_noun_box_train10k_v11_dataset.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name in sorted(payloads):
            info=zipfile.ZipInfo(name,date_time=(2026,10,22,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,payloads[name])
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and set(z.namelist())==set(payloads)
        for name,raw in payloads.items():assert sha(z.read(name))==sha(raw)
    final=dict(status='DATA_READY_FOR_PILOT_DESIGN_NO_MODEL_RUN',policy_version=POLICY['version'],
        initial_v1=dict(images=4618,spans=len(before)),
        final_counts={k:receipt[k] for k in ['retained_images','retained_spans','retained_classes','classes_with_50_spans','person_spans','person_fraction']},
        exact_v1_to_v11_span_removals=dict(removed),gates=result['gates'],manual_effects=effects,
        data_manifest_receipt_sha256=sha(payloads['manifest_receipt.json']),dataset_archive_sha256=sha(archive.read_bytes()),
        dataset_archive_bytes=archive.stat().st_size,dataset_file_count=len(payloads),
        source_provenance=provenance,weak_supervision_not_gold=True,model_run=False,
        next='Upload the seven JSON files as a Kaggle dataset; pilot recipe and notebooks to be locked with Astra before training.')
    (HERE/'FULL_DATA_READY.json').write_bytes(encoded(final))
    print(json.dumps({k:final[k] for k in ['status','final_counts','exact_v1_to_v11_span_removals','dataset_archive_bytes','dataset_file_count']},indent=2))
    print('DATASET ZIP:',archive)


if __name__=='__main__':main()
