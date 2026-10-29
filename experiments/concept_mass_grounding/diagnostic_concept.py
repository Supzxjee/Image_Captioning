"""Post-training neutral-prefix diagnostic; dev labels are never training targets."""
import gzip
import hashlib
import json
from pathlib import Path
from collections import defaultdict
from concept_data import HERE,POLICY,reference_support

def selection(meta,gt,detectors,reference_annotation_support=None):
    lex=meta['lexicon'];result=[]
    for category,paths in lex['events'].items():
        if not paths:continue
        pools={'supported':[],'weak_negative':[]}
        for i in meta['dev_indices']:
            row=meta['rows'][i];present=set(gt[str(row['coco_id'])]);hints,_=reference_support(row['references'],lex)
            if reference_annotation_support is not None:hints|=set(reference_annotation_support.get(row['coco_id'],[]))
            entry=detectors.get(row['filename']);objects=entry.get('objects') if isinstance(entry,dict) else None
            if category in present:pools['supported'].append(i)
            elif category not in hints and isinstance(objects,list) and objects and not any(o['label']==category and o['conf']>=POLICY['detector_veto'] for o in objects):pools['weak_negative'].append(i)
        if min(map(len,pools.values()))<POLICY['diagnostic_classes_minimum']:continue
        for role,indices in pools.items():
            # Fixed cap per class/state, selected without scores; all eligible classes retained.
            chosen=sorted(indices,key=lambda i:hashlib.sha256(('concept-neutral-v1:'+category+':'+str(meta['rows'][i]['coco_id'])).encode()).hexdigest())[:32]
            result.extend(dict(index=i,category=category,state=role) for i in chosen)
    return result

def diagnostic(model,data,tok,device,cfg,rank):
    import torch
    gt=json.loads(gzip.decompress((HERE/'DEV_DIAGNOSTIC_GT.json.gz').read_bytes()))['classes']
    detectors={Path(k.replace('\\','/')).name:v for k,v in json.loads(Path(cfg['detections']).read_bytes()).items()}
    from captioning.object_reward import _AnnotationReader
    with (Path(cfg['chair'])/'chair.pkl').open('rb') as stream:annotation_support=_AnnotationReader(stream).load().imid_to_objects
    selected=selection(data.meta,gt,detectors,annotation_support);by_image=defaultdict(list)
    for row in selected:by_image[row['index']].append(row)
    indices=sorted(by_image)[rank::2];results=[];prefix=[tok.bos_idx]+[tok.word2idx[w] for w in POLICY['diagnostic_prefix'].split()]
    with torch.no_grad():
        for first in range(0,len(indices),8):
            ix=indices[first:first+8];x=model.inputs(data.batch(ix,device));ids=torch.tensor([prefix]*len(ix),device=device);events=[];lookup=[]
            for b,i in enumerate(ix):
                for row in by_image[i]:events.append(dict(batch=b,position=len(prefix)-1,category=row['category']));lookup.append(row)
            a=model.class_logmass(ids,x,events,data.meta['lexicon'],model.logits(ids,x,teacher=True),teacher=True)
            g=model.class_logmass(ids,x,events,data.meta['lexicon'],model.logits(ids,x))
            for row,qa,qg in zip(lookup,a.exp().tolist(),g.exp().tolist()):results.append(dict(row,coco_id=data.meta['rows'][row['index']]['coco_id'],A_mass=qa,end_mass=qg))
            if first%128==0:print(f'Neutral-prefix diagnostic {min(first+8,len(indices))}/{len(indices)} local',flush=True)
    return dict(prefix=POLICY['diagnostic_prefix'],selection_rule='All classes with>=5 states, at most32 hash-first per class/state, fixed before scores',
        records=results,weak_labels=True,causal_grounding_proven=False,unknown_detector_provenance=True)
