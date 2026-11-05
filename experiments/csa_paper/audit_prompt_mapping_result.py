"""Review cached user-run text/metadata, never import Torch or rerun mapping models."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import numpy as np
from real_bank_data import read,save,sha
from prompt_adapter_data import PROMPT_SHA,overlap,geometry,box
from captioning.upt_prompts import relation_text

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def run():
    folder=HERE/'prompt_mapping_review_text';checks=0
    def check(value):
        nonlocal checks
        if not value:raise AssertionError('Failed stored-result check '+str(checks+1))
        checks+=1
    summary=read(folder/'summary.json');receipt=read(folder/'receipt.json');source=read(folder/'source_receipt.json')
    cached=read(folder/'saved_text_receipt.json');cohort=read(folder/'cohort.json');examples=read(folder/'review_examples.json')
    check(summary['status']==receipt['status']=='PROMPT_ROI_MAPPING_AUDIT_COMPLETE')
    check(summary['images']==15000 and not any(summary[k] for k in ('test_used','training','model_executed','ready_to_train')))
    check(source['commit']=='14c5bf0e6b19480d9666db4af517e930967c4ea1' and len(source['sha256'])==16)
    import hashlib
    for name,want in source['sha256'].items():
        payload=subprocess.check_output(['git','show',source['commit']+':'+name],cwd=ROOT)
        check(hashlib.sha256(payload).hexdigest()==want)
    check(receipt['source_inputs_sha256']['prompts']==PROMPT_SHA)
    check(receipt['source_inputs_sha256']['detections']=='498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c')
    check(receipt['roi_package_sha256']==source['sha256']['experiments/csa_paper/SAVE_FULL_ROI_V134_PACKAGE.json'])
    for name in ('summary.json','cohort.json','review_examples.json'):
        check(sha(folder/name)==receipt['payload_sha256'][name]==cached['files'][name]['sha256'])
    check(set(receipt['payload_sha256'])=={'summary.json','cohort.json','review_examples.json','prompt_links.jsonl','prompt_texts.json'})
    train,val=cohort['train'],cohort['val']
    check(len(train)==len(set(train))==10000 and len(val)==len(set(val))==5000 and not set(train)&set(val) and cohort['test']==[])
    old=read(HERE/'scst_ce_review_text/pilot_stage.json')
    check(train==old['pool_ids'])
    valid_ids={r['coco_id'] for r in read(HERE/'rerank163_review_text/inference_inputs.json')}
    check(set(val)==valid_ids)
    check((summary['IoU_threshold'],summary['object_cap'],summary['relation_cap'],summary['unique_texts'])==(.5,8,3,318))
    totals=Counter()
    for group,size in [('train',10000),('val',5000)]:
        c=summary['counts'][group]
        check(c['images']==size and sum(c.get('source_'+mode,0) for mode in ('object_relation','object_only','no_objects'))==size)
        check(c['objects']==sum(c.get(k,0) for k in ('mapped_objects','unmatched_objects','ambiguous_objects','colliding_objects')))
        check(c['mapped_objects']==c['kept_objects']+c['capped_objects'])
        check(c['relations']==c['mapped_relations']+c['unmatched_relations'])
        check(c['with_object']+c['without_mapped_prompt']==size and c['with_relation']<=c['source_object_relation']<=c['with_object']+c['without_mapped_prompt'])
        check(abs(c['mapped_object_percent']-100*c['mapped_objects']/c['objects'])<1e-10)
        check(abs(c['mapped_relation_percent']-100*c['mapped_relations']/c['relations'])<1e-10)
        totals.update({k:v for k,v in c.items() if not k.endswith('_percent')})
    check(len(examples)==len({r['coco_id'] for r in examples})==40)
    buckets=Counter()
    for row in examples:
        group=row['group'];check(row['coco_id'] in set(cohort[group]))
        check(set(row)=={'coco_id','filename','group','roi_count','source_mode','objects','relations','counts'})
        check(int(Path(row['filename']).stem.rsplit('_',1)[1])==row['coco_id'])
        objects=row['objects'];relations=row['relations'];by_instance={r['instance_id']:r for r in objects}
        check(len(objects)==len(by_instance)==row['counts']['kept_objects']<=8 and len(relations)==row['counts']['mapped_relations']<=3)
        check(len({o['roi'] for o in objects})==len(objects))
        for obj in objects:
            check(type(obj['roi'])==int and 0<=obj['roi']<row['roi_count'])
            check(abs(float(overlap(obj['bbox'],[obj['roi_bbox']])[0])-obj['IoU'])<1e-12 and obj['IoU']>=.5)
            check(0<=obj['confidence']<=1 and np.isfinite(obj['confidence']))
            check(obj['text']==obj['label'] if obj['label'] in ('skis','scissors') else obj['text']==('an ' if obj['label'][0] in 'aeiou' else 'a ')+obj['label'])
            box(obj['bbox']);box(obj['roi_bbox'])
        for rel in relations:
            s,o=by_instance[rel['subject_instance']],by_instance[rel['object_instance']]
            check(s['label']=='person' and s['roi']==rel['subject_roi'] and o['roi']==rel['object_roi'] and s['roi']!=o['roi'])
            check(np.isfinite([rel['score'],rel['threshold']]).all() and 0<=rel['threshold']<=rel['score']<=1)
            check(rel['text']==relation_text(rel['predicate'],o['label']))
            check(np.allclose(geometry(s['bbox'],o['bbox']),rel['geometry'],atol=1e-12,rtol=0))
        buckets[(group,row['source_mode'],bool(relations))]+=1
    check(all(v==5 for v in buckets.values()))
    log=(folder/'notebook.log').read_text(encoding='utf-8')
    check('Ran 7 tests' in log and 'mapped 15000 /15000' in log and 'Traceback' not in log and 'FAILED' not in log)
    totals=dict(totals)
    result=dict(status='PROMPT_MAPPING_STORED_REVIEW_PASS',checks=checks,source=source['commit'],
        user_reported_output_source='ducanh2403/2-vi-cai-kia-qua-nhieu/167',zip_sha256=cached['zip_sha256'],
        counts=summary['counts'],totals=totals,images=15000,unique_texts=318,seconds=summary['seconds'],
        mapped_relation_percent=100*totals['mapped_relations']/totals['relations'],
        mapped_object_percent=100*totals['mapped_objects']/totals['objects'],
        images_with_prompt_percent=100*totals['with_object']/15000,
        images_with_relation_percent=100*totals['with_relation']/15000,
        object_retained_after_cap_percent=100*totals['kept_objects']/totals['objects'],
        sampled_examples=40,examples_visually_verified=False,test_used=False,model_executed=False,
        missing_payloads_in_review=['prompt_links.jsonl','prompt_texts.json','ROI NPZ','original prompt/detector binaries'],
        next_recommendation='Reuse existing mapped ROI for a bounded pilot; retain fallback for missing links. Cache frozenCLIP text + user-run neural smoke before training.',
        limits=['Mapping is box/label agreement, not relation truth or decoder grounding.',
                'Full15000 link/prompt-text payload hashes are user-run receipts; absent review, not independently reread.',
                'Object mapping loses about23%; relation availability only about25% of images. No caption-quality result.',
                'Source reused VALID and one project; no newTEST or model execution by reviewer.'])
    save(HERE/'PROMPT_MAPPING_RESULT.json',result)
    print(json.dumps(dict(checks=checks,totals=totals,ratios={k:result[k] for k in ('mapped_relation_percent','mapped_object_percent','images_with_prompt_percent','images_with_relation_percent','object_retained_after_cap_percent')}),indent=2))


if __name__=='__main__':run()
