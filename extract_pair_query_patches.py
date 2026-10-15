"""Frozen CPU CLIP patch extraction on the existing detector train/dev rows.

Downloads only named official COCO images; archives/documents are never executed.
"""
import argparse
import hashlib
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from captioning.interaction_regions import boxes_in_crop, region_weights
from captioning.relation_candidates import union_crop_box

REVISION = '57c216476eefef5ab752ec549e440a49ae4ae5f3'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''): digest.update(block)
    return digest.hexdigest()


def run(args):
    from transformers import CLIPVisionModel
    torch.set_num_threads(args.threads)
    output, package, model_dir = map(Path, (args.output, args.package, args.model))
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output/'images'; image_dir.mkdir(exist_ok=True)
    paths = [Path('audit')/f'interaction_region_{s}_pairs.json' for s in ('train', 'val')]
    rows_by_split = [json.loads(p.read_text()) for p in paths]
    rows = rows_by_split[0] + rows_by_split[1]
    provenance = json.loads((package/'image_provenance.json').read_text())
    known = {p['filename']: p for p in provenance}
    if model_dir.name != REVISION: raise ValueError('Expected pinned CLIP snapshot.')
    protocol = {'variant': 'pair_query_patch_extraction_v1', 'revision': REVISION,
        'weights_sha256': sha(model_dir/'pytorch_model.bin'),
        'config_sha256': sha(model_dir/'config.json'),
        'source_pair_sha256': {s: sha(p) for s, p in zip(('train', 'val'), paths)},
        'review_provenance_sha256': sha(package/'image_provenance.json'),
        'feature': '197 final vision tokens; post-layernorm; float16 cache; 196 patches plus CLS',
        'preprocessing': 'RGB; padded union .1; resize entire crop to 224 bicubic; /255; CLIP mean/std',
        'training': False, 'device': 'cpu', 'splits': [len(x) for x in rows_by_split]}
    pp = output/'extraction_protocol.json'
    if pp.exists() and json.loads(pp.read_text()) != protocol: raise ValueError('Use a fresh output for changed protocol.')
    pp.write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    unique = {r['filename']: r for r in rows}

    def obtain(item):
        filename, row = item
        if Path(filename).name != filename or row['filepath'] not in ('train2014', 'val2014'):
            raise ValueError('Unexpected image path.')
        dest = image_dir/filename
        url = f"http://images.cocodataset.org/{row['filepath']}/{filename}"
        if not dest.exists():
            local = package/'images'/filename
            if local.is_file():
                data = local.read_bytes()
            else:
                with urllib.request.urlopen(url, timeout=45) as response: data = response.read()
            temporary = dest.with_suffix('.partial'); temporary.write_bytes(data)
            with Image.open(temporary) as image: image.verify()
            temporary.replace(dest)
        digest = sha(dest)
        if filename in known and digest != known[filename]['sha256']: raise ValueError('Reviewed image bytes differ.')
        with Image.open(dest) as image: size = list(image.size)
        return {'filename': filename, 'source_url': url, 'sha256': digest, 'image_size': size}

    images = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(obtain, item) for item in unique.items()]
        for future in as_completed(futures):
            images.append(future.result())
            if len(images) % 50 == 0: print(f'Images ready {len(images)}/{len(unique)}', flush=True)
    image_manifest = output/'image_provenance.json'
    image_manifest.write_text(json.dumps(sorted(images, key=lambda x:x['filename']), indent=2), encoding='utf-8')
    model = CLIPVisionModel.from_pretrained(model_dir, local_files_only=True).eval().requires_grad_(False)
    cfg = model.config
    if (cfg.image_size, cfg.patch_size, cfg.hidden_size) != (224, 16, 768): raise ValueError('Unexpected CLIP dimensions.')
    mean = torch.tensor([.48145466, .4578275, .40821073])[:, None, None]
    std = torch.tensor([.26862954, .26130258, .27577711])[:, None, None]
    completed=[]
    for path in sorted(output.glob('patches_*.pt')):
        cached=torch.load(path,weights_only=True,map_location='cpu')
        for record in cached:
            i=record['index']
            if i!=len(completed) or i>=len(rows) or record['tokens'].shape!=(197,768) or not torch.isfinite(record['tokens']).all():
                raise ValueError('Existing shard is not a contiguous finite prefix.')
            w,h=record['image_size'];row=rows[i]
            crop=union_crop_box(row['subject_box'],row['object_box'],w,h,padding=.1)
            boxes=boxes_in_crop(row['subject_box'],row['object_box'],crop,w,h)
            if list(crop)!=record['crop_pixels'] or not torch.equal(boxes,record['boxes_in_crop']):
                raise ValueError('Cached crop changed.')
            completed.append(i)
    resumed=len(completed)
    print(f'Resume verified prefix: {resumed}/{len(rows)}',flush=True)
    records = []; started = time.monotonic()
    for start in range(resumed,len(rows),args.batch_size):
        batch=[];metadata=[]
        for i in range(start,min(start+args.batch_size,len(rows))):
            row=rows[i];split='train' if i<len(rows_by_split[0]) else 'val'
            with Image.open(image_dir/row['filename']) as source:
                image=source.convert('RGB');width,height=image.size
                crop=union_crop_box(row['subject_box'],row['object_box'],width,height,padding=.1)
                boxes=boxes_in_crop(row['subject_box'],row['object_box'],crop,width,height)
                image=image.crop(crop).resize((224,224),Image.Resampling.BICUBIC)
                pixels=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1).float()/255
            batch.append((pixels-mean)/std)
            metadata.append({'index':i,'split':split,'boxes_in_crop':boxes,'region_support':region_weights(boxes),
                'crop_pixels':list(crop),'image_size':[width,height]})
        with torch.inference_mode():
            tokens=model.vision_model.post_layernorm(model(pixel_values=torch.stack(batch)).last_hidden_state)
        if tokens.shape[1:]!=(197,768) or not torch.isfinite(tokens).all():raise ValueError('Invalid patch grid.')
        for record,token in zip(metadata,tokens):
            i=record['index'];records.append({**record,'tokens':token.half().clone()})
            if (i+1)%50==0 or i==len(rows)-1:
                print(f'CPU patches {i+1}/{len(rows)} | {time.monotonic()-started:.1f}s since resume',flush=True)
            if len(records)==100 or i==len(rows)-1:
                torch.save(records,output/f'patches_{i+1-len(records):04d}.pt');records=[]
    shards = sorted(output.glob('patches_*.pt'))
    summary = {'variant': protocol['variant'], 'protocol_sha256': sha(pp), 'pairs': len(rows),
        'images': len(images), 'image_manifest_sha256': sha(image_manifest),
        'feature_sha256': {p.name: sha(p) for p in shards}, 'seconds': time.monotonic()-started,
        'training_performed': False, 'torch': torch.__version__,
        'runtime':{'batch_size':args.batch_size,'threads':args.threads,'resumed_verified_pairs':resumed}}
    (output/'extraction_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('model', 'package', 'output'): parser.add_argument('--'+name, required=True)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--threads',type=int,default=4)
    run(parser.parse_args())
