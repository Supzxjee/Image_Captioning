"""User-run CPU-only copy of 4000 scheduled TRAIN rows plus fixeddev1000."""
from collections import defaultdict
import gzip
import json
from pathlib import Path
import shutil
import time
import numpy as np
from concept_data import *

def prepare(cfg):
    import h5py
    import torch
    from captioning.visual_cache import VisualCache
    from captioning.semantic_prompts import align_prompt_cache
    from captioning.object_reward import make_targets
    start=time.monotonic();out=Path(cfg['output'])
    if out.exists():raise FileExistsError('Preserve old output; use a fresh session/path')
    if cfg['policy']!=POLICY or cfg['arm'] not in ('C','G'):raise ValueError('Changed locked recipe')
    for name,expected in [('checkpoint',CHECKPOINT_SHA),('dataset_json',DATA_SHA),('legacy_prompt',LEGACY_SHA),('detections',DETECTION_SHA),('metadata',META_SHA)]:
        print('Hashing pinned input:',name,flush=True)
        if sha(cfg[name])!=expected:raise ValueError('Pinned input differs: '+name)
    if sha(Path(cfg['chair'])/'chair.pkl')!=CHAIR_SHA:raise ValueError('Wrong CHAIR cache')
    data=json.loads(Path(cfg['dataset_json']).read_bytes());old=json.loads(gzip.decompress(Path(cfg['metadata']).read_bytes()))
    cohort=json.loads(gzip.decompress((HERE/'TRAIN_PRESENCE.json.gz').read_bytes()))
    train,dev=cohort['train_rows'],old['groups']['val'];validate_rows(train,dev)
    rules=json.loads((ROOT/'experiments/coco_grounding_feasibility/MAPPING_POLICY_V11.json').read_bytes())
    lex=lexicon(old['word2idx'],rules);order=order_indices().flatten();selected=[train[int(i)] for i in order]
    rows=[]
    for position,row in enumerate(selected+dev):
        original=data['images'][row['image_id']]
        if original['filename']!=row['filename'] or original['cocoid']!=row['coco_id']:raise ValueError('COCO/Karpathy identity mismatch')
        if position<4000 and original['split'] not in ('train','restval'):raise ValueError('Heldout training leakage')
        rows.append(dict(row,references=[s['raw'] for s in original['sentences'][:5]]))
        if len(rows[-1]['references'])!=5:raise ValueError('Need five references')
    detectors=json.loads(Path(cfg['detections']).read_bytes());det={}
    for filename,entry in detectors.items():
        name=Path(filename.replace('\\','/')).name
        if name in det:raise ValueError('Duplicate detector filename')
        det[name]=entry
    reward=make_targets(dict(train=rows[:4000],val=dev),data,Path(cfg['chair'])/'chair.pkl',detectors,'aliases_v2','unique_count_v2')
    reward_support={r['image_id']:set(r['supported']) for r in reward['data']}
    targets=[]
    for row in rows[:4000]:
        gt=set(cohort['gt_classes'][str(row['coco_id'])]);hints,safe=reference_support(row['references'],lex)
        entry=det.get(row['filename']);objects=entry.get('objects') if isinstance(entry,dict) else None
        eligible=isinstance(objects,list) and len(objects)>0
        veto=set()
        if objects is not None:
            for obj in objects:
                c=obj.get('label');conf=obj.get('conf')
                if c not in COCO_LABELS or not isinstance(conf,(int,float)) or not np.isfinite(conf) or not 0<=conf<=1:raise ValueError('Bad detector record')
                if conf>=POLICY['detector_veto']:veto.add(c)
        targets.append(dict(image_id=row['image_id'],coco_id=row['coco_id'],supported=sorted(gt|hints|reward_support[row['image_id']]),
            unknown=sorted(veto-(gt|hints|reward_support[row['image_id']])),positive_anchor_classes=sorted(gt&safe),negative_eligible=eligible,gt_classes=sorted(gt),reference_hints=sorted(hints),detector_veto_hints=sorted(veto),
            detector_record_available=objects is not None))
    static_possible=sum(sum(bool(lex['events'][c]) and c not in t['supported'] and c not in t['unknown'] for c in COCO_LABELS) for t in targets if t['negative_eligible'])
    if static_possible==0:raise RuntimeError('No statically eligible weak-negative classes; do not run either arm')
    if not any(t['positive_anchor_classes'] for t in targets):raise RuntimeError('No supported preservation anchors')
    print('Static eligible image/class weak negatives:',static_possible,flush=True)
    out.mkdir(parents=True);save(out/'protocol.json',cfg)
    save(out/'object_targets.json',reward)
    meta=dict(schema='concept-mass-data-v1',rows=rows,word2idx=old['word2idx'],baseline=old['baseline'],
        train_indices=list(range(4000)),dev_indices=list(range(4000,5000)),targets=targets,lexicon=lex,
        schedule_cohort_positions=order.reshape(500,8).tolist(),source_train_images=10000,reference_rule='step modulo5, same C/G')
    save(out/'metadata.json',meta)
    if shutil.disk_usage(out).free<4*1024**3:raise RuntimeError('Need4GiB free output disk')
    bundle=torch.load(cfg['legacy_prompt'],map_location='cpu',weights_only=False)
    prompts=align_prompt_cache(bundle['data'],[r['filename'] for r in rows]);del bundle
    if len(prompts)!=5000:raise ValueError('Incomplete original prompt cache')
    visual=VisualCache([cfg['visual_cache']]);visual.require_ids([r['coco_id'] for r in rows],'locked TRAIN4000+dev1000')
    expected=dict(dataset_json_sha256=DATA_SHA,image_size=[224,224],preprocessing='bilinear',model_id='openai/clip-vit-base-patch16',
        output='vision_model.last_hidden_state',mean=[.48145466,.4578275,.40821073],std=[.26862954,.26130258,.27577711],antialias=True,compute_dtype='float32',autocast=False)
    from recover_stage import Previous,row_sha
    previous=Previous(cfg.get('previous_prepared'),cfg,meta,reward)
    grouped=defaultdict(list)
    for i,r in enumerate(rows):
        file_id,position=visual.index[r['coco_id']];grouped[file_id].append((position,i))
    try:
        with h5py.File(out/'inputs.h5','x') as target:
            target.attrs['complete']=False
            for k,shape,dtype in [('visual',(197,768),'f2'),('prompt',(20,512),'f4'),('mask',(20,),'u1')]:target.create_dataset(k,(5000,*shape),dtype=dtype)
            target.create_dataset('coco_ids',data=[r['coco_id'] for r in rows]);done=0
            journal=target.create_dataset('row_sha256',(5000,),dtype='S64')
            handles={k:target[k] for k in ('visual','prompt','mask')}
            for file_id,positions in sorted(grouped.items()):
                with h5py.File(visual.paths[file_id],'r') as source:
                    info=json.loads(source.attrs['metadata_json']);features=source['features']
                    if str(features.dtype)!='float16' or any(info.get(k)!=v for k,v in expected.items()):raise ValueError('Visual source preprocessing mismatch')
                    for position,i in sorted(positions):
                        r=rows[i];p=prompts[r['filename']];tokens=p['tokens'].float().numpy();mask=p['mask'].numpy()
                        cached=previous.verified(i,tokens,mask)
                        if cached is None:
                            cached=previous.source_checked(i,dict(visual=features[position],prompt=tokens,mask=mask))
                        for key,value in cached.items():handles[key][i]=value
                        journal[i]=row_sha(cached).encode('ascii');done+=1
                        if done==1 or done%250==0:
                            target.flush()
                            print(f'Staged verified caches {done}/5000; {time.monotonic()-start:.1f}s; recovery={previous.stats}',flush=True)
                        if time.monotonic()-start>POLICY['prepare_seconds']:raise TimeoutError('Preparation cap; incomplete')
            target.attrs['complete']=True
    finally:visual.close();previous.close()
    content=hashlib.sha256()
    with h5py.File(out/'inputs.h5','r') as f:
        for key in ('coco_ids','visual','prompt','mask'):
            content.update(str((key,str(f[key].dtype),tuple(f[key].shape))).encode())
            for first in range(0,5000,64):content.update(f[key][first:first+64].tobytes())
    receipt=dict(status='PREPARATION_COMPLETE',train_population=10000,scheduled_train=4000,dev=1000,neural_forward=False,feature_extraction=False,
        static_eligible_negative_image_classes=static_possible,policy=POLICY,tensor_content_sha256=content.hexdigest(),input_hashes=dict(checkpoint=CHECKPOINT_SHA,legacy=LEGACY_SHA,data=DATA_SHA,detector=DETECTION_SHA),
        payload_sha256={n:sha(out/n) for n in ('metadata.json','inputs.h5','object_targets.json')},
        recovery=dict(source=cfg.get('previous_prepared'),counts=previous.stats,legacy_requires_source_revalidation=True),
        detector_upstream_provenance='Unknown; original extractor not available',seconds=time.monotonic()-start)
    save(out/'preparation.json',receipt);print('Preparation COMPLETE; next user GPU preflight.',flush=True)
