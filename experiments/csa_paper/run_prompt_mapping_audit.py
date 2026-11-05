"""USER-RUN CPU phase: audit UPT-to-CSA region links, never execute a model."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import traceback
import zipfile
import numpy as np
from real_bank_data import read,save,sha
from audit_assets import detection_index,DETECTION_SHA
from prompt_adapter_data import PROMPT_SHA,cohort,detector_regions,join_entry,inference_record

HERE=Path(__file__).resolve().parent


def mapping_cohort(train,val,scope='pilot'):
    if scope not in ('pilot','full'):raise ValueError('Unknown mapping scope')
    if len(train)!=113287 or len(set(train))!=113287 or len(val)!=5000 or len(set(val))!=5000 or set(train)&set(val):
        raise ValueError('Wrong full Karpathy metadata coverage')
    pool=cohort(train) if scope=='pilot' else sorted(train)
    return dict(train=pool,val=val,test=[],selection='hash42 TRAIN10k; fullVALID5000; no relation-based selection' if scope=='pilot' else 'all Karpathy TRAIN113287; fullVALID5000; no relation-based selection')


def audit(cfg):
    started=time.monotonic();out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    print('CPU prompt/ROI audit: no Torch, CLIP, UPT, training, or TEST evaluation.',flush=True)
    expected=read(HERE/'SAVE_FULL_ROI_V134_PACKAGE.json')['files'];root=Path(cfg['full_root'])
    candidates=[p.parent for p in root.rglob('manifest.json') if sha(p)==expected['manifest.json']]
    if len(candidates)!=1:raise ValueError('Need exact full ROI134 manifest under FULL_ROOT')
    root=candidates[0];inputs={}
    for key,want in [('prompts',PROMPT_SHA),('detections',DETECTION_SHA)]:
        p=Path(cfg[key]);print('Hashing pinned input:',key,flush=True)
        if not p.is_file() or sha(p)!=want:raise ValueError('Missing/changed '+key+'; preserve identity, no substitute')
        inputs[key]=want
    for n in ('receipt.json','manifest.json','vocabulary.json'):
        if sha(root/n)!=expected[n]:raise ValueError('Changed ROI identity: '+n)
    manifest=read(root/'manifest.json')
    if manifest['status']!='FULL_ROI_PREPARATION_COMPLETE' or (manifest['train_images'],manifest['val_images'],manifest['test_images'])!=(113287,5000,0):
        raise ValueError('Wrong Karpathy split/coverage')
    print('Loading frozen prompt JSON and detector metadata.',flush=True)
    prompts=read(cfg['prompts']);detections=detection_index(read(cfg['detections']))
    metadata=[];shards=[];seen=set()
    for entry in manifest['shards']:
        name=next(n for n in entry['files'] if n.endswith('.json'))
        if entry['files'][name]!=expected[name] or sha(root/name)!=expected[name]:raise ValueError('Changed shard metadata')
        rows=read(root/name)['images'];group=entry['group']
        for row in rows:
            identity=row['coco_id']
            if type(identity)!=int or identity in seen or row['group']!=group or group not in ('train','val'):
                raise ValueError('Duplicate or held-out ID/group')
            if row['source_split'] not in (('train','restval') if group=='train' else ('val',)):raise ValueError('Split leakage')
            seen.add(identity)
        metadata.extend(rows);shards.append((entry,rows))
    train=[r['coco_id'] for r in metadata if r['group']=='train'];val=[r['coco_id'] for r in metadata if r['group']=='val']
    if len(train)!=113287 or len(val)!=5000:raise ValueError('Metadata coverage differs')
    groups=mapping_cohort(train,val,cfg.get('scope','pilot'));pool=groups['train'];wanted=set(pool+val);expected_images=len(wanted)
    save(out/'cohort.json',groups)
    totals={g:Counter() for g in ('train','val')};samples={};texts=set();count=0
    with (out/'prompt_links.jsonl').open('w',encoding='utf-8',newline='\n') as target:
        for number,(entry,rows) in enumerate(shards,1):
            selected=[(i,r) for i,r in enumerate(rows) if r['coco_id'] in wanted]
            if not selected:continue
            name=next(n for n in entry['files'] if n.endswith('.npz'))
            if entry['files'][name]!=expected[name] or sha(root/name)!=expected[name]:raise ValueError('Changed ROI shard '+name)
            with np.load(root/name,allow_pickle=False) as arrays:
                boxes=arrays['boxes'];offsets=arrays['offsets'];ids=arrays['ids']
                if len(offsets)!=len(rows)+1 or len(ids)!=len(rows) or offsets[0]!=0 or offsets[-1]!=len(boxes):raise ValueError('Malformed ROI offsets')
                for index,meta in selected:
                    if int(ids[index])!=meta['coco_id']:raise ValueError('ROI/image ID mismatch')
                    filename=meta['filename']
                    if filename not in prompts or filename not in detections:raise ValueError('Missing inference source '+filename)
                    lo,hi=map(int,offsets[index:index+2])
                    if not 1<=hi-lo<=150:raise ValueError('ROI count differs')
                    regions=detector_regions(detections[filename],meta['width'],meta['height'],boxes[lo:hi],meta['region_kinds'])
                    record=inference_record(meta,join_entry(prompts[filename],regions))
                    target.write(json.dumps(record,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
                    group=meta['group'];counts=totals[group]
                    counts.update(record['counts']);counts['images']+=1;counts['source_'+record['source_mode']]+=1
                    counts['with_object']+=bool(record['objects']);counts['with_relation']+=bool(record['relations'])
                    counts['without_mapped_prompt']+=not (record['objects'] or record['relations'])
                    for obj in record['objects']:texts.add(obj['text'])
                    for rel in record['relations']:texts.add(rel['text'])
                    # A small deterministic stratified preview, not a precision estimate.
                    bucket=(group,record['source_mode'],bool(record['relations']))
                    samples.setdefault(bucket,[])
                    if len(samples[bucket])<5:samples[bucket].append(record)
                    count+=1
            print('ROI metadata shard',number,'/',len(shards),'mapped',count,'/',expected_images,flush=True)
            if time.monotonic()-started>cfg.get('wall_seconds',7200):raise TimeoutError('CPU audit cap; preserve partial output')
    if count!=expected_images or any(totals[g]['images']!=n for g,n in [('train',len(pool)),('val',5000)]):raise ValueError('Incomplete cohort')
    for group,c in totals.items():
        c['mapped_object_percent']=100*c.get('mapped_objects',0)/max(c.get('objects',0),1)
        c['mapped_relation_percent']=100*c.get('mapped_relations',0)/max(c.get('relations',0),1)
    save(out/'prompt_texts.json',sorted(texts))
    save(out/'review_examples.json',[r for bucket in sorted(samples) for r in samples[bucket]])
    summary=dict(status='PROMPT_ROI_MAPPING_AUDIT_COMPLETE',images=count,test_used=False,training=False,model_executed=False,
        scope=('TRAIN10k' if len(pool)==10000 else 'TRAINfull113287')+' + fullVALID5000; mapping coverage, not prompt truth or caption quality',counts={g:dict(c) for g,c in totals.items()},
        unique_texts=len(texts),IoU_threshold=.5,object_cap=8,relation_cap=3,seconds=time.monotonic()-started,
        ready_to_train=False,next_step='Review mapping coverage/modes and example pairs, then lock pilot cache/training protocol. No automatic training.')
    save(out/'summary.json',summary)
    names=('prompt_links.jsonl','prompt_texts.json','cohort.json','review_examples.json','summary.json')
    save(out/'receipt.json',dict(status=summary['status'],source_inputs_sha256=inputs,
        roi_package_sha256=sha(HERE/'SAVE_FULL_ROI_V134_PACKAGE.json'),payload_sha256={n:sha(out/n) for n in names}))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def archive(output):
    out=Path(output);dest=out.with_name(out.name+'_review.zip')
    with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
        for name in ('summary.json','receipt.json','cohort.json','review_examples.json','source_receipt.json','failure.json','notebook.log'):
            if (out/name).is_file():z.write(out/name,name)
    print('SEND REVIEW:',dest,'\nKEEP ENTIRE OUTPUT:',out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);args=p.parse_args();config=read(args.config)
    try:audit(config)
    except Exception:
        out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
        save(out/'failure.json',dict(status='INCOMPLETE',traceback=traceback.format_exc(),model_executed=False));raise
    finally:archive(config['output'])
