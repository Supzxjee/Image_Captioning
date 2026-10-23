"""Two independent GPU workers benchmark H/L cache extraction. No training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

from tile_data import *

A_SHA='630a4b6d4bcef84515bbd44a52b6474811377bda53caa9722c5806795fa7f901'
DATA_SHA='2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d'


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):result.update(block)
    return result.hexdigest()


def prepare(cfg,out):
    start=time.monotonic()
    for key,expected in [('checkpoint',A_SHA),('dataset_json',DATA_SHA),('groups_archive',ARCHIVE_SHA)]:
        print('Hashing pinned input:',key,flush=True)
        if digest(cfg[key])!=expected:raise ValueError('Pinned input mismatch: '+key)
    with zipfile.ZipFile(cfg['groups_archive']) as archive:groups=json.loads(archive.read('groups.json'))
    data=json.loads(Path(cfg['dataset_json']).read_text(encoding='utf-8'))['images']
    print('Locked group keys:',list(groups),flush=True)
    train=groups['train'];val=groups['val']
    if len(val)!=5000 or len({r['coco_id'] for r in val})!=5000:raise ValueError('Expected val5000.')
    if {r['coco_id'] for r in train}&{r['coco_id'] for r in val}:raise ValueError('Split overlap.')
    images=Path(cfg['images']).resolve()
    for i,row in enumerate(train+val):
        source=data[row['image_id']]
        if source['filename']!=row['filename'] or int(source['filename'].rsplit('_',1)[1].split('.')[0])!=row['coco_id']:raise ValueError('Image row mismatch.')
        path=(images/source['filepath']/row['filename']).resolve()
        if not path.is_relative_to(images) or not path.is_file():raise FileNotFoundError(path)
        row['source_path']=str(path)
        if (i+1)%5000==0:print('Source files present:',i+1,'/15000',flush=True)
    sample=select_sample(train)
    save(out/'sample.json',sample)
    from captioning.visual_cache import VisualCache
    cache=VisualCache([cfg['visual_cache']]);cache.require_ids([r['coco_id'] for r in sample['warmup']],'global compatibility')
    import h5py
    metadata=[]
    for path in cache.paths:
        with h5py.File(path,'r') as handle:
            metadata.append(dict(path=path,attrs={k:str(v) for k,v in handle.attrs.items()}))
    cache.close()
    save(out/'old_visual_cache_metadata.json',metadata)
    from transformers import CLIPVisionModel
    print('Prefetch pinned CLIP backbone on CPU (setup, outside measured loop).',flush=True)
    t=time.monotonic();model=CLIPVisionModel.from_pretrained(MODEL_ID,revision=REVISION,attn_implementation='eager');del model
    save(out/'preparation.json',dict(status='COMPLETE',source_files_checked=15000,checkpoint_sha256=A_SHA,
        dataset_sha256=DATA_SHA,groups_archive_sha256=ARCHIVE_SHA,sample_sha256=digest(out/'sample.json'),
        preparation_seconds=time.monotonic()-start,prefetch_seconds=time.monotonic()-t,
        old_cache_revision='Metadata may omit revision; compatibility is checked on actual features in each worker.'))


def worker(cfg,rank):
    import h5py
    import torch
    import torch.nn.functional as F
    from PIL import ImageFile
    from transformers import CLIPVisionModel
    from captioning.data import build_image_transform
    from captioning.visual_cache import VisualCache
    from captioning.scst_runtime import configure_strict_runtime,model_digest,runtime_manifest
    out=Path(cfg['output']);torch.set_num_threads(2);torch.cuda.set_device(rank)
    configure_strict_runtime(True);device=torch.device('cuda',rank);ImageFile.LOAD_TRUNCATED_IMAGES=True
    start=time.monotonic();sample=json.loads((out/'sample.json').read_text(encoding='utf-8'))
    rows=sample['measure'][rank::2];warmup=sample['warmup'][rank::2]
    model=CLIPVisionModel.from_pretrained(MODEL_ID,revision=REVISION,local_files_only=True,attn_implementation='eager').eval().to(device)
    checkpoint=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=False)['model_state_dict']
    projection=torch.nn.Linear(768,512)
    projection.load_state_dict({k:checkpoint['base.encoder.vis_projection.'+k] for k in ('weight','bias')},strict=True)
    del checkpoint
    projection=projection.eval().to(device)
    for module in (model,projection):module.requires_grad_(False)
    before=[model_digest(m) for m in (model,projection)]
    transform=build_image_transform('bilinear')
    save(out/f'runtime_rank{rank}.json',runtime_manifest())

    def read(row):
        with Image.open(row['source_path']) as opened:return opened.convert('RGB')

    def encode(batch):
        raw=model(pixel_values=batch).last_hidden_state
        patches=raw[:,1:].reshape(-1,14,14,768).permute(0,3,1,2)
        pooled=F.avg_pool2d(patches,2).permute(0,2,3,1).reshape(-1,49,768)
        return torch.relu(projection(pooled))

    with torch.inference_mode():
        # Explicitly test spatial pooling order before the nonlinear projection.
        probe=torch.arange(196*768,device=device,dtype=torch.float32).reshape(1,14,14,768)
        pooled=F.avg_pool2d(probe.permute(0,3,1,2),2).permute(0,2,3,1)
        torch.testing.assert_close(pooled[0,0,0],probe[0,:2,:2].mean((0,1)),rtol=0,atol=0)
        cache=VisualCache([cfg['visual_cache']]);compat=[]
        for row in warmup[:4]:
            image=read(row)
            torch.testing.assert_close(transform(image),transform(resize_global(image)),rtol=0,atol=0)
            fresh=model(pixel_values=transform(image).unsqueeze(0).to(device)).last_hidden_state[0].cpu()
            stored=torch.from_numpy(cache.read(row['coco_id']))
            torch.testing.assert_close(fresh,stored,atol=.01,rtol=.005)
            compat.append(dict(coco_id=row['coco_id'],max_absolute_error=float((fresh-stored).abs().max())))
        cache.close()
        for row in warmup:
            image=read(row);batch=torch.stack([transform(t) for c in ('H','L') for t in make_tiles(image,c)]).to(device)
            features=encode(batch)
            if not torch.isfinite(features).all():raise ValueError('Nonfinite warmup features.')
        torch.cuda.synchronize()
        save(out/f'ready_rank{rank}.json',dict(preflight='PASS',compatibility=compat,setup_and_warmup_seconds=time.monotonic()-start,
            model_sha256=before,compatibility_tolerance=dict(atol=.01,rtol=.005),weight_revision=REVISION))
        deadline=time.monotonic()+1800
        while not (out/'release.json').is_file():
            if time.monotonic()>deadline:raise TimeoutError('Parent did not release workers.')
            time.sleep(.2)
        torch.cuda.reset_peak_memory_stats();timings=[];resolutions=[];source_bytes=0
        path=out/f'features_rank{rank}.h5'
        with h5py.File(path,'w') as handle:
            handle.attrs.update(complete=False,benchmark_only=True,model_revision=REVISION,projection_source_sha256=A_SHA)
            datasets={c:handle.create_dataset(c,(128,196,512),dtype='f2') for c in ('H','L')}
            handle.create_dataset('coco_ids',data=np.asarray([r['coco_id'] for r in rows],dtype=np.int64))
            handle.create_dataset('xy',data=tile_centers());handle.flush()
            begin=time.monotonic_ns()
            for offset in range(0,128,4):
                t=time.monotonic();batch=[]
                for row in rows[offset:offset+4]:
                    image=read(row);resolutions.append(list(image.size));source_bytes+=Path(row['source_path']).stat().st_size
                    batch.extend(transform(tile) for c in ('H','L') for tile in make_tiles(image,c))
                pixels=torch.stack(batch);cpu=time.monotonic()-t
                t=time.monotonic();pixels=pixels.to(device)
                fp32=encode(pixels).reshape(4,2,196,512)
                if not torch.isfinite(fp32).all():raise ValueError('Nonfinite measured features.')
                fp16=fp32.half()
                if not torch.isfinite(fp16).all():raise ValueError('FP16 overflow.')
                torch.testing.assert_close(fp16.float(),fp32,atol=.01,rtol=.005)
                array=fp16.cpu().numpy();torch.cuda.synchronize();gpu=time.monotonic()-t
                t=time.monotonic()
                for j,c in enumerate(('H','L')):datasets[c][offset:offset+4]=array[:,j]
                handle.flush();write=time.monotonic()-t
                timings.append(dict(cpu_decode_preprocess=cpu,transfer_encode_pool_project_copy=gpu,hdf5_write_flush=write))
                save(out/f'progress_rank{rank}.json',dict(images=offset+4,total=128))
                print(f'GPU {rank}: {offset+4}/128 measured images',flush=True)
            handle.attrs['complete']=True;handle.flush()
        end=time.monotonic_ns()
        with h5py.File(path,'r') as handle:
            assert bool(handle.attrs['complete']) and len(handle['coco_ids'])==128
            np.testing.assert_array_equal(handle['xy'][:],tile_centers())
            for j,c in enumerate(('H','L')):np.testing.assert_array_equal(handle[c][-4:],array[:,j])
        after=[model_digest(m) for m in (model,projection)]
        if before!=after:raise ValueError('Frozen model changed.')
        totals={k:sum(x[k] for x in timings) for k in timings[0]}
        batch_times=[sum(x.values()) for x in timings]
        save(out/f'complete_rank{rank}.json',dict(status='COMPLETE',measured_images=128,measure_start_ns=begin,measure_end_ns=end,
            phase_seconds=totals,batch_wall_p50_seconds=float(np.median(batch_times)),batch_wall_p90_seconds=float(np.quantile(batch_times,.9)),
            source_bytes=source_bytes,resolutions=resolutions,hdf5_bytes=path.stat().st_size,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),model_sha256=after))


def launch(cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    save(out/'protocol.json',cfg);start=time.monotonic();processes=[];streams=[]
    try:
        prepare(cfg,out)
        for rank in range(2):
            stream=(out/f'worker_rank{rank}.log').open('w',encoding='utf-8');streams.append(stream)
            processes.append(subprocess.Popen([sys.executable,'-u',__file__,'worker','--config',str(out/'protocol.json'),'--rank',str(rank)],stdout=stream,stderr=subprocess.STDOUT))
        deadline=time.monotonic()+1800;last=0
        while any(p.poll() is None for p in processes):
            if any(p.poll() not in (None,0) for p in processes):raise RuntimeError('GPU worker failed; see worker logs.')
            ready=all((out/f'ready_rank{r}.json').is_file() for r in range(2))
            if ready and not (out/'release.json').exists():save(out/'release.json',dict(status='START',monotonic_ns=time.monotonic_ns()))
            if not ready and time.monotonic()>deadline:raise TimeoutError('GPU setup/preflight exceeded 30 minutes.')
            if time.monotonic()-last>15:
                for r in range(2):
                    p=out/f'progress_rank{r}.json'
                    print(f'GPU {r}: '+(p.read_text(encoding='utf-8') if p.exists() else 'setup / compatibility / warmup'),flush=True)
                last=time.monotonic()
            time.sleep(1)
        if any(p.returncode!=0 for p in processes):raise RuntimeError('GPU worker failed.')
        workers=[json.loads((out/f'complete_rank{r}.json').read_text(encoding='utf-8')) for r in range(2)]
        if workers[0]['model_sha256']!=workers[1]['model_sha256']:raise ValueError('GPU workers have different model weights.')
        report=estimates(workers);report.update(status='COMPLETE',total_run_seconds=time.monotonic()-start,
            observed_hdf5_bytes=sum(w['hdf5_bytes'] for w in workers),free_disk_bytes=shutil.disk_usage(out).free,
            next_step='Review measured cost with remaining GPU quota before preparing full cache/training. No automatic go decision.',
            benchmark_only=True,trained=False,caption_evaluated=False)
        save(out/'benchmark_summary.json',report);print(json.dumps(report,indent=2),flush=True)
    except BaseException:
        save(out/'benchmark_summary.json',dict(status='FAILED',traceback=traceback.format_exc(),trained=False,caption_evaluated=False))
        raise
    finally:
        for process in processes:
            if process.poll() is None:process.terminate()
        for process in processes:
            try:process.wait(timeout=20)
            except subprocess.TimeoutExpired:process.kill();process.wait()
        for stream in streams:stream.close()
        for rank in range(2):
            path=out/f'worker_rank{rank}.log'
            if path.exists():print(path.name+' tail:\n'+path.read_text(encoding='utf-8',errors='replace')[-5000:],flush=True)
        archive(out)


def archive(out):
    with zipfile.ZipFile(str(out)+'_review.zip','w',zipfile.ZIP_DEFLATED) as target:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.suffix in ('.json','.log','.py'):target.write(path,path.relative_to(out))
    print('SEND SMALL REVIEW ZIP:',str(out)+'_review.zip',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['run','worker','archive']);parser.add_argument('--config',required=True);parser.add_argument('--rank',type=int,choices=[0,1])
    args=parser.parse_args();config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    if args.mode=='worker':worker(config,args.rank)
    elif args.mode=='archive':archive(Path(config['output']))
    else:launch(config)
