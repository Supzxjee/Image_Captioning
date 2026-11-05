"""Data-only contracts and reference-free selectors for fixed VALID candidates."""
import io
import json
import math
from pathlib import Path
import numpy as np
from candidate_data import baseline,check_record
from real_bank_data import read,save,sha
from captioning.multiscore_reranking import rerank_multiscore
from captioning.object_consistency import index_detections

INPUT_SHA={
 'candidates_rank0.jsonl':'9010f88223114d0a0681519094f1c1e26e7d799317163250380573f06ee54bf4',
 'candidates_rank1.jsonl':'7b3504629110403f01149815cd179d103d1aa6179871fc26ae1e546517dff8ec'}
DATA_SHA='2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'
DETECTION_SHA='498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c'
MODELS={
 'clip':dict(id='openai/clip-vit-base-patch16',revision='57c216476eefef5ab752ec549e440a49ae4ae5f3'),
 'blip':dict(id='Salesforce/blip-itm-base-coco',revision='bed8ad38cb2d04a5a4bdf2d071b3c3c0a4aa724c')}
POLICIES=('decoder_rank0','clip_object_fixed','blip_itm')


def candidates(root):
    root=Path(root);saved=baseline();expected={r['coco_id']:r for r in saved['predictions']};rows=[]
    for rank,(name,h) in enumerate(INPUT_SHA.items()):
        matches=[p for p in root.rglob(name) if sha(p)==h]
        if len(matches)!=1:raise ValueError('Expected unique exact163 '+name)
        part=[json.loads(line) for line in matches[0].read_text(encoding='utf-8').splitlines()]
        if [r['coco_id'] for r in part]!=saved['order'][rank::2]:raise ValueError('Wrong163 worker coverage/order')
        for row in part:check_record(row,expected[row['coco_id']],saved['vocabulary'])
        rows.extend(part)
    if len(rows)!=5000 or len({r['coco_id'] for r in rows})!=5000:raise ValueError('Missing/duplicate candidates')
    return rows


def image_paths(root, records):
    root=Path(root).resolve();found={};missing=[]
    for r in records:
        name=r['filename'];relative=Path(r['filepath'])
        paths=[root/relative/name,root/'images'/relative/name,root/'images'/name,root/name]
        available=[p.resolve() for p in paths if p.is_file()]
        if available:found[name]=available[0]
        else:missing.append(name)
    if missing:
        wanted=set(missing);by_name={}
        for p in root.rglob('*.jpg'):
            if p.name in wanted:by_name.setdefault(p.name,[]).append(p.resolve())
        for name in missing:
            matches=set(by_name.get(name,[]))
            if len(matches)!=1:raise FileNotFoundError(f'Need one raw image {name} under {root}; found {len(matches)}')
            found[name]=matches.pop()
    if any(not p.is_relative_to(root) for p in found.values()):raise ValueError('Image path escapes configured root')
    return found


def prepare(cfg):
    from PIL import Image
    out=Path(cfg['output']);root=Path(cfg['coco_root'])
    for key,h in (('dataset_json',DATA_SHA),('detections',DETECTION_SHA)):
        if sha(cfg[key])!=h:raise ValueError('Changed pinned input '+key)
    rows=candidates(cfg['candidate_root']);saved=baseline();refs={r['coco_id']:r['references'] for r in saved['references']}
    dataset=read(cfg['dataset_json'])['images'];lookup={}
    for r in dataset:
        if r['split']=='val':
            coco=int(Path(r['filename']).stem.split('_')[-1])
            if coco in lookup:raise ValueError('Duplicate COCO ID')
            lookup[coco]=r
    if set(lookup)!=set(refs):raise ValueError('Not identical Karpathy VALID5000')
    if any([s['raw'] for s in lookup[i]['sentences']]!=refs[i] for i in refs):raise ValueError('Reference provenance differs')
    paths=image_paths(root,list(lookup.values()));detected=index_detections(read(cfg['detections']))
    payload=[]
    for i,row in enumerate(rows):
        name=lookup[row['coco_id']]['filename'];p=paths[name]
        if name not in detected:raise ValueError('Missing detector entry, not an empty detection: '+name)
        d=detected[name]
        if any(not math.isfinite(v) or not .5<=v<=1 for v in d.values()):raise ValueError('Invalid accepted detector confidence')
        raw=p.read_bytes()
        import hashlib
        with Image.open(io.BytesIO(raw)) as image:
            width,height=image.size;image.verify()
        if min(width,height)<=0:raise ValueError('Invalid raw image dimensions')
        payload.append(dict(**row,filename=name,image_path=str(p),image_sha256=hashlib.sha256(raw).hexdigest(),
            width=width,height=height,detected=d))
        if (i+1)%500==0:print('Prepared raw images/detector/candidates',i+1,'/5000',flush=True)
    save(out/'inference_inputs.json',payload) # No references, annotations, CIDEr or oracle labels.
    save(out/'validation_references.json',saved['references'])
    save(out/'prepared.json',dict(status='RERANK_VALID_PREPARED',images=5000,candidate_files_sha256=INPUT_SHA,
        inference_inputs_sha256=sha(out/'inference_inputs.json'),references_sha256=sha(out/'validation_references.json'),
        dataset_sha256=DATA_SHA,detections_sha256=DETECTION_SHA,detector_threshold=.5,models=MODELS,test_used=False))


def select(row):
    candidates=row['candidates'];clip=row['clipscore'];logits=np.asarray(row['itm_logits'],dtype=float)
    if len(candidates)!=5 or len(clip)!=5 or logits.shape!=(5,2) or not np.isfinite(logits).all():raise ValueError('Wrong scoring width/nonfinite ITM')
    if any(not math.isfinite(float(v)) or not 0<=v<=2.50001 for v in clip):raise ValueError('Invalid CLIPScore')
    if any(not isinstance(c['caption'],str) or not math.isfinite(c['logprob']) for c in candidates):raise ValueError('Invalid candidate')
    if [c['logprob'] for c in candidates]!=sorted((c['logprob'] for c in candidates),reverse=True):raise ValueError('Unsorted beams')
    enriched=[dict(c,clipscore=float(s)) for c,s in zip(candidates,clip)]
    fixed,_=rerank_multiscore(dict(candidates=enriched),row['detected'],.4,.3)
    margin=logits[:,1]-logits[:,0]
    return dict(decoder_rank0=0,clip_object_fixed=fixed['original_rank'],blip_itm=int(margin.argmax()))
