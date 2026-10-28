"""User-run CPU cache copy only. Never instantiate an encoder, detector or captioner."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import time

MANIFEST_SHA='6ca6af23d0a31043fecdf33398f5b60098be07cbeea9fb91cbfec4b314c2cfa7'
LEGACY_SHA='d7910e9bf87e3aad86d806c77ec336b34793e3bef68a3f662fdcb2e9456ca38d'
DETECTOR_SHA='498825daf8f8773a1c92f4111dc4ccf1f4296343f952b486e0aa56caf82ec04c'
DATA_SHA='2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def records(manifest):
    rows={}
    for split in ('train','heldout'):
        for pair in manifest[split]:
            for role in ('positive','negative'):
                r=pair[role];iid=r['image_id'];image=r['image']
                if iid in rows and rows[iid]['filename']!=image['file_name']:raise ValueError('Conflicting image identity')
                rows[iid]=dict(coco_id=iid,filename=image['file_name'],width=image['width'],height=image['height'])
    return [rows[i] for i in sorted(rows)]


def valid_prompt(tokens,mask):
    import numpy as np
    return (tokens.shape==(20,512) and mask.shape==(20,) and np.isfinite(tokens).all()
            and np.isin(mask,[0,1]).all() and mask.any())


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--visual',type=Path,required=True);p.add_argument('--legacy',type=Path,required=True)
    p.add_argument('--detections',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();start=time.monotonic()
    if args.output.exists():raise FileExistsError('Use a fresh output directory; existing results are preserved')
    for name,path,expected in [('manifest',args.manifest,MANIFEST_SHA),('legacy embedding cache',args.legacy,LEGACY_SHA),('detector cache',args.detections,DETECTOR_SHA)]:
        print('Hashing pinned '+name,flush=True)
        if digest(path)!=expected:raise ValueError('Wrong input: '+name)
    manifest=json.loads(gzip.decompress(args.manifest.read_bytes()));rows=records(manifest)
    if len(rows)!=612:raise ValueError('Wrong v2 cohort')
    import h5py,numpy as np,torch
    from visual_cache import VisualCache
    from semantic_prompts import align_prompt_cache
    print('Opening original caches on CPU; no model download or forward.',flush=True)
    cache=VisualCache([args.visual]);cache.require_ids([r['coco_id'] for r in rows],'V2')
    source_metadata=[]
    for path in cache.paths:
        with h5py.File(path,'r') as f:
            info=json.loads(f.attrs['metadata_json'])
            expected=dict(dataset_json_sha256=DATA_SHA,image_size=[224,224],preprocessing='bilinear',
                model_id='openai/clip-vit-base-patch16',output='vision_model.last_hidden_state',
                mean=[.48145466,.4578275,.40821073],std=[.26862954,.26130258,.27577711],
                antialias=True,compute_dtype='float32',autocast=False)
            if any(info.get(k)!=v for k,v in expected.items()) or str(f['features'].dtype)!='float16':raise ValueError('Visual metadata differs from baseline')
            source_metadata.append(dict(filename=Path(path).name,metadata=info))
    bundle=torch.load(args.legacy,map_location='cpu',weights_only=False)
    names=[r['filename'] for r in rows];legacy=align_prompt_cache(bundle['data'],names);del bundle
    missing=sorted(set(names)-set(legacy))
    print(f'Original prompt coverage: {len(legacy)}/{len(rows)}',flush=True)
    if missing:
        raise ValueError('Original prompt missing for '+str(len(missing))+' images; examples '+repr(missing[:8])+'. Do not substitute GT names or VSR text.')
    detections={}
    wanted={r['coco_id'] for r in rows}
    dimensions={r['coco_id']:(r['width'],r['height']) for r in rows}
    for filename,value in json.loads(args.detections.read_bytes()).items():
        iid=int(Path(filename).stem.rsplit('_',1)[-1])
        if iid in wanted:
            if iid in detections:raise ValueError('Duplicate detector image')
            objects=sorted(enumerate(value['objects']),key=lambda p:(-float(p[1]['conf']),p[0]))[:12]
            boxes=[];width,height=dimensions[iid]
            for index,o in objects:
                x1,y1,x2,y2=map(float,o['bbox'])
                b=[max(0.,x1),max(0.,y1),min(float(width),x2),min(float(height),y2)]
                if not np.isfinite(b).all() or b[2]<=b[0] or b[3]<=b[1] or not 0<=float(o['conf'])<=1:
                    raise ValueError('Invalid top12 predicted box')
                boxes.append(dict(original_index=index,label=o['label'],bbox_xyxy=b,confidence=o['conf']))
            detections[str(iid)]=boxes
    if len(detections)!=612:raise ValueError('Detector coverage differs')
    for split in ('train','heldout'):
        for pair in manifest[split]:
            for role in ('positive','negative'):
                r=pair[role];bank={b['original_index']:b for b in detections[str(r['image_id'])]}
                for entity in ('subject','object'):
                    expected=r['predicted'][entity];actual=bank[expected['index']]
                    if actual['label']!=expected['label'] or actual['bbox_xyxy']!=expected['bbox_xyxy']:
                        raise ValueError('Selected target pair differs from full predicted bank')
    args.output.mkdir(parents=True)
    if shutil.disk_usage(args.output).free<1024**3:raise RuntimeError('Need1GiB free for prepared output')
    target=args.output/'inputs.h5'
    try:
        with h5py.File(target,'x') as f:
            f.attrs['complete']=False
            for name,shape,dtype in [('visual',(197,768),'f2'),('prompt',(20,512),'f4'),('mask',(20,),'u1')]:
                f.create_dataset(name,(612,*shape),dtype=dtype)
            f.create_dataset('coco_ids',data=[r['coco_id'] for r in rows])
            for count,index in enumerate(sorted(range(612),key=lambda i:cache.index[rows[i]['coco_id']]),1):
                r=rows[index];entry=legacy[r['filename']]
                tokens=entry['tokens'].float().numpy();mask=entry['mask'].numpy()
                if not valid_prompt(tokens,mask):raise ValueError('Invalid original prompt')
                visual=cache.read(r['coco_id'])
                f['visual'][index]=visual;f['prompt'][index]=tokens;f['mask'][index]=mask
                if not np.array_equal(f['visual'][index].astype('f4'),visual) or not np.array_equal(f['prompt'][index],tokens) or not np.array_equal(f['mask'][index],mask):raise ValueError('Saved cache value differs')
                if count%25==0 or count==612:print(f'Copied and verified {count}/612 | {time.monotonic()-start:.1f}s',flush=True)
                if time.monotonic()-start>1800:raise RuntimeError('CPU preparation cap1800s exceeded; no retries or model fallback')
            f.attrs['complete']=True
    finally:cache.close()
    (args.output/'relation_frames_manifest.bin').write_bytes(args.manifest.read_bytes())
    for name,value in [('images.json',rows),('detector_boxes.json',detections)]:
        (args.output/name).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    receipt=dict(status='CACHE_COPY_COMPLETE_PENDING_LABEL_AUDIT',manifest_sha256=MANIFEST_SHA,
        images=612,legacy_prompt_coverage=612,inputs=dict(legacy_sha256=LEGACY_SHA,detector_sha256=DETECTOR_SHA),
        visual_sources=source_metadata,feature_extraction=False,caption_training=False,neural_forward=False,
        preparation_source_sha256=digest(__file__),
        human_label_audit_approved=False,training_approved=False,seconds=time.monotonic()-start,
        payload_sha256={p.name:digest(p) for p in args.output.iterdir() if p.is_file()})
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print('KEEP FOLDER:',args.output,flush=True)
    print('CACHE COPY COMPLETE; human label audit and model preflight still pending.',flush=True)


if __name__=='__main__':main()
