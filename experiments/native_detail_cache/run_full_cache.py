"""User-run full H/L cache. Completed 500-image shards are immutable resume units."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
from tile_data import make_tiles,resize_global,tile_centers,save,REVISION,ARCHIVE_SHA
from run_benchmark import prepare as benchmark_prepare,archive,digest
from cache_data import plan,spec_for,validate_spec,check_shard,verify_receipt,identity,EXPECTED_MODELS


def load_plan(cfg):
    if digest(cfg['groups_archive'])!=ARCHIVE_SHA:raise ValueError('Locked group archive differs.')
    with zipfile.ZipFile(cfg['groups_archive']) as z:groups=json.loads(z.read('groups.json'))
    data=json.loads(Path(cfg['dataset_json']).read_text(encoding='utf-8'))['images']
    return plan(groups,data)


def import_resume(source,out,shards,spec):
    validate_spec(source,spec)
    for shard in shards:
        name=shard['name'];path=Path(source)/(name+'.h5');receipt=Path(source)/(name+'.json')
        if not path.exists():continue
        if not receipt.exists():
            print('Resume source has no committed receipt; skip:',name,flush=True);continue
        record=json.loads(receipt.read_text(encoding='utf-8'));verify_receipt(path,record)
        if record['spec_sha256']!=identity(spec) or record['rows_sha256']!=identity(shard['rows']):raise ValueError('Resume receipt identity differs.')
        target=out/path.name
        if target.exists():
            verify_receipt(target,record)
        else:
            if shutil.disk_usage(out).free < path.stat().st_size+500_000_000:raise RuntimeError('Insufficient disk to import resume shard.')
            temporary=out/(name+'.importing');shutil.copyfile(path,temporary);verify_receipt_as(temporary,record,path.name);temporary.replace(target)
        if not (out/receipt.name).exists():save(out/receipt.name,record)
        print('Imported verified resume shard:',name,flush=True)


def verify_receipt_as(path,record,name):
    if path.stat().st_size!=record['bytes'] or digest(path)!=record['sha256'] or record['file']!=name:raise ValueError('Imported payload hash mismatch.')


def worker(cfg,rank):
    import h5py
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image,ImageFile
    from transformers import CLIPVisionModel
    from captioning.data import build_image_transform
    from captioning.visual_cache import VisualCache
    from captioning.scst_runtime import configure_strict_runtime,model_digest,runtime_manifest
    from tile_data import MODEL_ID
    out=Path(cfg['output']);attempt=Path(cfg['attempt']);spec=json.loads((out/'cache_spec.json').read_text(encoding='utf-8'))
    shards=json.loads((out/'manifest.json').read_text(encoding='utf-8'));mine=[s for s in shards if s['rank']==rank]
    torch.set_num_threads(2);torch.cuda.set_device(rank);configure_strict_runtime(True);device=torch.device('cuda',rank)
    ImageFile.LOAD_TRUNCATED_IMAGES=True
    model=CLIPVisionModel.from_pretrained(MODEL_ID,revision=REVISION,local_files_only=True,attn_implementation='eager').eval().to(device)
    state=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)['model_state_dict'];projection=torch.nn.Linear(768,512)
    projection.load_state_dict({k:state['base.encoder.vis_projection.'+k] for k in ('weight','bias')},strict=True);del state
    projection=projection.eval().to(device)
    for module in (model,projection):module.requires_grad_(False)
    models=[model_digest(m) for m in (model,projection)]
    if models!=EXPECTED_MODELS:raise ValueError('Frozen encoder/projection differ from the passed benchmark.')
    transform=build_image_transform('bilinear')
    runtime=runtime_manifest()
    from importlib.metadata import version
    runtime['pillow']=version('Pillow')
    runtime_sha=identity(runtime)
    save(attempt/f'runtime_rank{rank}.json',runtime)

    def image(row):
        with Image.open(Path(cfg['images'])/row['filepath']/row['filename']) as im:return im.convert('RGB')

    def encode(pixels):
        raw=model(pixel_values=pixels).last_hidden_state[:,1:]
        pooled=F.avg_pool2d(raw.reshape(-1,14,14,768).permute(0,3,1,2),2).permute(0,2,3,1).reshape(-1,49,768)
        return torch.relu(projection(pooled))

    with torch.inference_mode():
        probe=torch.arange(196*768,device=device,dtype=torch.float32).reshape(1,14,14,768)
        pooled=F.avg_pool2d(probe.permute(0,3,1,2),2).permute(0,2,3,1)
        torch.testing.assert_close(pooled[0,0,0],probe[0,:2,:2].mean((0,1)),rtol=0,atol=0)
        # Reuse the passed benchmark's eight fixed global compatibility examples, not val captions.
        sample=json.loads((attempt/'sample.json').read_text(encoding='utf-8'));cache=VisualCache([cfg['visual_cache']]);compat=[]
        for row in sample['warmup'][rank::2][:4]:
            with Image.open(row['source_path']) as im:im=im.convert('RGB')
            torch.testing.assert_close(transform(im),transform(resize_global(im)),rtol=0,atol=0)
            fresh=model(pixel_values=transform(im).unsqueeze(0).to(device)).last_hidden_state[0].cpu()
            old=torch.from_numpy(cache.read(row['coco_id']));torch.testing.assert_close(fresh,old,atol=.01,rtol=.005)
            compat.append(dict(coco_id=row['coco_id'],max_absolute_error=float((fresh-old).abs().max())))
        cache.close()
        save(attempt/f'preflight_rank{rank}.json',dict(status='PASS',model_sha256=models,compatibility=compat,atol=.01,rtol=.005))
        new=skipped=0;started=time.monotonic();torch.cuda.reset_peak_memory_stats()
        for shard in mine:
            name=shard['name'];target=out/(name+'.h5');receipt=out/(name+'.json');rows=shard['rows']
            if target.exists():
                record=check_shard(target,shard,spec,models,runtime_sha)
                if receipt.exists() and json.loads(receipt.read_text(encoding='utf-8'))!=record:raise ValueError('Existing shard receipt differs.')
                if not receipt.exists():save(receipt,record)
                skipped+=len(rows);print(f'GPU {rank}: verified/skip {name}; {new+skipped}/7500 images',flush=True);continue
            if receipt.exists():raise ValueError('Receipt exists without shard: '+name)
            temporary=out/(name+'.partial.h5')
            # Only this worker's uncommitted, deterministic shard file may be rewritten after interruption.
            with h5py.File(temporary,'w') as f:
                f.attrs.update(complete=False,benchmark_only=False,spec_sha256=identity(spec),rows_sha256=identity(rows),model_sha256=json.dumps(models),runtime_sha256=runtime_sha)
                datasets={c:f.create_dataset(c,(len(rows),196,512),dtype='f2') for c in ('H','L')}
                f.create_dataset('coco_ids',data=np.asarray([r['coco_id'] for r in rows],dtype='i8'))
                f.create_dataset('image_ids',data=np.asarray([r['image_id'] for r in rows],dtype='i8'));f.create_dataset('xy',data=tile_centers())
                for offset in range(0,len(rows),4):
                    batch_rows=rows[offset:offset+4];pixels=[]
                    for row in batch_rows:
                        im=image(row);pixels.extend(transform(t) for c in ('H','L') for t in make_tiles(im,c))
                    fp32=encode(torch.stack(pixels).to(device)).reshape(len(batch_rows),2,196,512);fp16=fp32.half()
                    if not torch.isfinite(fp32).all() or not torch.isfinite(fp16).all():raise ValueError('Nonfinite features.')
                    torch.testing.assert_close(fp16.float(),fp32,atol=.01,rtol=.005);array=fp16.cpu().numpy()
                    for j,c in enumerate(('H','L')):datasets[c][offset:offset+len(batch_rows)]=array[:,j]
                    f.flush()
                    save(attempt/f'progress_rank{rank}.json',dict(shard=name,shard_rows=offset+len(batch_rows),committed_images=new+skipped,total=7500))
                    if (offset+len(batch_rows))%100==0:print(f'GPU {rank}: {name} {offset+len(batch_rows)}/500; committed {new+skipped}/7500',flush=True)
                f.attrs['complete']=True;f.flush()
            record=check_shard(temporary,shard,spec,models,runtime_sha)
            with h5py.File(temporary,'r') as f:
                for j,c in enumerate(('H','L')):np.testing.assert_array_equal(f[c][-len(batch_rows):],array[:,j])
            temporary.replace(target);record['file']=target.name;save(receipt,record)
            new+=len(rows);print(f'GPU {rank}: committed {name}; {new+skipped}/7500 images',flush=True)
        if [model_digest(m) for m in (model,projection)]!=models:raise ValueError('Frozen model changed.')
        save(attempt/f'complete_rank{rank}.json',dict(status='COMPLETE',new_images=new,reused_images=skipped,total_images=new+skipped,
            model_sha256=models,runtime_sha256=runtime_sha,seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved()))


def run(cfg):
    out=Path(cfg['output']);shards=load_plan(cfg)
    spec=spec_for(shards,cfg['implementation_sha256'],ARCHIVE_SHA,REVISION)
    if out.exists() and any(out.iterdir()):validate_spec(out,spec)
    out.mkdir(parents=True,exist_ok=True)
    if not (out/'cache_spec.json').exists():save(out/'cache_spec.json',spec)
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text(encoding='utf-8'))!=shards:raise ValueError('Existing manifest differs.')
    save(out/'manifest.json',shards)
    attempt=out/'attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True)
    cfg=dict(cfg,attempt=str(attempt));save(attempt/'protocol.json',cfg)
    processes=[];streams=[];started=time.monotonic()
    try:
        if cfg.get('resume_from'):import_resume(Path(cfg['resume_from']),out,shards,spec)
        missing=sum(not (out/(s['name']+'.h5')).exists() for s in shards)
        if shutil.disk_usage(out).free < missing*500*2*196*512*2+500_000_000:raise RuntimeError('Insufficient free disk for remaining cache plus safety margin.')
        benchmark_prepare(cfg,attempt)
        for rank in range(2):
            stream=(attempt/f'worker_rank{rank}.log').open('w',encoding='utf-8');streams.append(stream)
            processes.append(subprocess.Popen([sys.executable,'-u',__file__,'worker','--config',str(attempt/'protocol.json'),'--rank',str(rank)],stdout=stream,stderr=subprocess.STDOUT))
        last=0
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('GPU worker failed; preserve completed shards and return review ZIP.')
            if time.monotonic()-last>20:
                for rank in range(2):
                    p=attempt/f'progress_rank{rank}.json';print(f'GPU {rank}: '+(p.read_text(encoding='utf-8') if p.exists() else 'setup / preflight / resume verification'),flush=True)
                last=time.monotonic()
            time.sleep(1)
        if any(p.returncode!=0 for p in processes):raise RuntimeError('GPU worker failed.')
        workers=[json.loads((attempt/f'complete_rank{r}.json').read_text(encoding='utf-8')) for r in range(2)]
        if any(w['total_images']!=7500 or w['status']!='COMPLETE' for w in workers) or workers[0]['model_sha256']!=workers[1]['model_sha256'] or workers[0]['runtime_sha256']!=workers[1]['runtime_sha256']:raise ValueError('Worker completion/model mismatch.')
        records=[]
        for shard in shards:
            path=out/(shard['name']+'.h5');record=json.loads((out/(shard['name']+'.json')).read_text(encoding='utf-8'));verify_receipt(path,record)
            if record['spec_sha256']!=identity(spec) or record['rows_sha256']!=identity(shard['rows']) or record['model_sha256']!=workers[0]['model_sha256'] or record['runtime_sha256']!=workers[0]['runtime_sha256']:raise ValueError('Final shard identity differs.')
            records.append(record)
        if sum(r['images'] for r in records)!=15000:raise ValueError('Full cache coverage incomplete.')
        save(out/'cache_receipt.json',dict(status='CACHE_COMPLETE',schema=spec['schema'],images=15000,train_images=10000,val_images=5000,
            shards=records,spec_sha256=identity(spec),manifest_sha256=identity(shards),model_sha256=workers[0]['model_sha256'],runtime_sha256=workers[0]['runtime_sha256'],benchmark_only=False,trained=False,caption_evaluated=False))
        save(attempt/'run_summary.json',dict(status='COMPLETE',seconds=time.monotonic()-started,workers=workers))
        print('CACHE COMPLETE. KEEP ENTIRE FOLDER AS DATASET:',out,flush=True)
    except BaseException:
        save(attempt/'run_summary.json',dict(status='FAILED',traceback=traceback.format_exc()));raise
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=20)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for stream in streams:stream.close()
        for rank in range(2):
            p=attempt/f'worker_rank{rank}.log'
            if p.exists():print(p.name+' tail:\n'+p.read_text(encoding='utf-8',errors='replace')[-3500:],flush=True)
        archive(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','worker','archive']);parser.add_argument('--config',required=True);parser.add_argument('--rank',type=int,choices=[0,1])
    args=parser.parse_args();cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.mode=='run':run(cfg)
    elif args.mode=='worker':worker(cfg,args.rank)
    else:archive(Path(cfg['output']))
