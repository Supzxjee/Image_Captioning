"""USER-RUN CPU-only mounted/local feature IO probe. No neural imports."""
import argparse
from collections import OrderedDict
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import queue
import shutil
import time
import traceback
import numpy as np
from full_data import prepare,load_stage,FeatureReader,order_for_epoch,TRAIN_IMAGES
from real_bank_data import read,save,sha

BATCHES=64

def tasks(events):
    events=np.asarray(events,dtype=np.int64)
    if events.shape!=(BATCHES*32,) or len(set(events.tolist()))!=len(events):raise ValueError('Expected first64 global batches')
    return [[events[b*32+r*16:b*32+(r+1)*16].tolist() for b in range(w,BATCHES,2)] for r in (0,1) for w in (0,1)]

class LocalReader:
    def __init__(self,root):
        self.array=np.load(Path(root)/'patches.npy',mmap_mode='r',allow_pickle=False)
        self.ids=np.load(Path(root)/'image_indices.npy',allow_pickle=False)
        if self.array.shape!=(len(self.ids),196,768) or self.array.dtype!=np.float16 or len(set(self.ids.tolist()))!=len(self.ids):raise ValueError('Local shape/IDs differ')
        self.positions={int(x):i for i,x in enumerate(self.ids)};self.cache=OrderedDict()
    def get(self,image):
        image=int(image)
        if image in self.cache:self.cache.move_to_end(image);return self.cache[image]
        value=self.array[self.positions[image]]
        if not np.isfinite(value).all():raise ValueError('Nonfinite local row')
        self.cache[image]=value
        if len(self.cache)>256:self.cache.popitem(last=False)
        return value
    def close(self):self.array._mmap.close()

def run_worker(worker,mode,stage,local,batches,q,go,deadline):
    reader=None
    try:
        data,_,_=load_stage(stage,verify=False);reader=FeatureReader(stage) if mode=='mounted' else LocalReader(local)
        q.put(dict(kind='ready',worker=worker));go.wait(max(0,deadline-time.monotonic()));started=time.monotonic();times=[];digest=hashlib.sha256();events=0
        for batch in batches:
            if time.monotonic()>deadline:raise TimeoutError('CPU IO worker deadline')
            t=time.monotonic();images=data['caption_image'][batch]
            if (images>=TRAIN_IMAGES).any():raise ValueError('NonTRAIN feature access')
            x=np.stack([reader.get(i) for i in images]).astype(np.float32);times.append(time.monotonic()-t)
            digest.update(x.tobytes());events+=len(batch)
        q.put(dict(kind='done',worker=worker,mode=mode,events=events,batches=len(batches),read_stack_seconds=sum(times),batch_seconds=times,task_seconds=time.monotonic()-started,feature_stream_sha256=digest.hexdigest()))
    except BaseException:q.put(dict(kind='error',worker=worker,error=traceback.format_exc()))
    finally:
        if reader is not None:reader.close()

def measure(mode,stage,local,assignment,deadline,out,label):
    ctx=mp.get_context('spawn');q=ctx.Queue();go=ctx.Event();started=time.monotonic()
    workers=[ctx.Process(target=run_worker,args=(i,mode,str(stage),str(local),b,q,go,deadline)) for i,b in enumerate(assignment)]
    try:
        for p in workers:p.start()
        def get():
            remaining=deadline-time.monotonic()
            if remaining<=0:raise TimeoutError('CPU IO phase deadline')
            try:msg=q.get(timeout=min(30,remaining))
            except queue.Empty:
                if any(p.exitcode is not None and p.exitcode!=0 for p in workers):raise RuntimeError('IO worker failed before result')
                print('CPU IO waiting',label,round(time.monotonic()-started),'s',flush=True);return get()
            if msg['kind']=='error':raise RuntimeError(msg['error'])
            return msg
        ready=[get() for _ in workers]
        if {m['worker'] for m in ready}!={0,1,2,3} or any(m['kind']!='ready' for m in ready):raise ValueError('Reader readiness differs')
        startup=time.monotonic()-started;phase=time.monotonic();go.set();reports=[get() for _ in workers];wall=time.monotonic()-phase
        if {r['worker'] for r in reports}!={0,1,2,3} or any(r['kind']!='done' or r['events']!=512 for r in reports):raise ValueError('Incomplete reader result')
        for p in workers:p.join(timeout=max(0,min(30,deadline-time.monotonic())))
        if any(p.exitcode!=0 for p in workers):raise RuntimeError('Reader failed to exit')
        result=dict(label=label,mode=mode,events=2048,global_batches=BATCHES,startup_seconds=startup,concurrent_phase_wall_seconds=wall,
            CPU_delivery_wall_seconds_per_global_batch=wall/BATCHES,workers=sorted(reports,key=lambda r:r['worker']),not_GPU_step_time=True)
        save(Path(out)/(label+'.json'),result);print('CPU IO',label,'delivery seconds/globalbatch',round(wall/BATCHES,5),flush=True);return result
    finally:
        go.set()
        for p in workers:
            if p.is_alive():p.terminate()
        for p in workers:
            if p.pid is not None:p.join(timeout=10)
        q.close()

def stage_sample(stage,local,events,deadline):
    import h5py
    local=Path(local);local.mkdir(parents=True,exist_ok=False);data,_,meta=load_stage(stage)
    image_indices=np.unique(data['caption_image'][events])
    if (image_indices>=TRAIN_IMAGES).any():raise ValueError('Stage selected heldout images')
    image_indices=sorted(map(int,image_indices),key=lambda i:tuple(map(int,data['locations'][i])))
    needed=len(image_indices)*196*768*2
    if shutil.disk_usage(local).free<needed+2*1024**3:raise RuntimeError('Insufficient local scratch for sample')
    dest=np.lib.format.open_memmap(local/'patches.npy',mode='w+',dtype=np.float16,shape=(len(image_indices),196,768));started=time.monotonic();row_hashes={};handles={}
    try:
        for offset in range(0,len(image_indices),128):
            if time.monotonic()>deadline:raise TimeoutError('Sample stage deadline')
            group=image_indices[offset:offset+128]
            # Physical sorted row groups, one file per h5py selection; never choose TEST rows.
            by_file={}
            for i in group:
                file,row=map(int,data['locations'][i]);by_file.setdefault(file,[]).append((i,row))
            values={}
            for file,pairs in by_file.items():
                source=meta['sources'][file];path=Path(source['path'])
                if path.stat().st_size!=source['bytes']:raise ValueError('Changed raw source')
                if file not in handles:handles[file]=h5py.File(path,'r',rdcc_nbytes=2*1024**2)
                raw=handles[file]['features'][[r for _,r in pairs]]
                if not np.isfinite(raw).all() or raw.shape!=(len(pairs),197,768):raise ValueError('Invalid sampled features')
                for (i,_),v in zip(pairs,raw):values[i]=v[1:]
            for j,i in enumerate(group):
                v=values[i];dest[offset+j]=v;row_hashes[str(i)]=hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
            print('Local sample staged',min(offset+128,len(image_indices)),'/',len(image_indices),flush=True)
        dest.flush();stage_seconds=time.monotonic()-started
        for j,i in enumerate(image_indices):
            if hashlib.sha256(np.ascontiguousarray(dest[j]).tobytes()).hexdigest()!=row_hashes[str(i)]:raise ValueError('Local feature bytes differ')
    finally:
        dest._mmap.close()
        for h in handles.values():h.close()
    np.save(local/'image_indices.npy',np.asarray(image_indices,np.int64))
    info=dict(unique_TRAIN_images=len(image_indices),patch_bytes=needed,stage_seconds=stage_seconds,all_row_sha256=row_hashes,storage='float16 exact source patches, no projection/quantization',array_sha256=sha(local/'patches.npy'),image_index_sha256=sha(local/'image_indices.npy'))
    save(local/'receipt.json',info);return info

def resource_info(directory):
    disk=shutil.disk_usage(directory);ram={}
    mem=Path('/proc/meminfo')
    if mem.is_file():
        for line in mem.read_text().splitlines():
            key=line.split(':')[0]
            if key in ('MemTotal','MemAvailable'):ram[key+'_bytes']=int(line.split()[1])*1024
    return dict(path=str(directory),disk_free_bytes=disk.free,filesystem_device=Path(directory).stat().st_dev,ram=ram)

def run(cfg):
    started=time.monotonic();deadline=started+cfg.get('wall_seconds',1800);out=Path(cfg['output']);local=Path(cfg['local']);stage=Path(cfg['stage'])
    prepare(cfg);data,_,meta=load_stage(stage);events=order_for_epoch(data,1)[:BATCHES*32];assignment=tasks(events)
    save(out/'scope.json',dict(status='CPU_IO_PROBE_RUNNING',train_only=True,test_features_read=False,models_executed=False,events=events.tolist(),assignment=assignment,local_resources=resource_info(local.parent),source_resources=resource_info(cfg['visual_cache']),full_TRAIN_VALID_patch_bytes=(113287+5000)*196*768*2,full_staging_performed=False))
    reports=[measure('mounted',stage,local,assignment,deadline,out,'mounted_pass1'),measure('mounted',stage,local,assignment,deadline,out,'mounted_pass2')]
    sample=stage_sample(stage,local,events,deadline);save(out/'sample_stage_receipt.json',sample)
    reports += [measure('local',stage,local,assignment,deadline,out,'local_pass1'),measure('local',stage,local,assignment,deadline,out,'local_pass2')]
    original=[r['feature_stream_sha256'] for r in reports[0]['workers']]
    if any([r['feature_stream_sha256'] for r in item['workers']]!=original for item in reports):raise ValueError('Mounted/local feature streams differ')
    save(out/'result.json',dict(status='CPU_PATCH_IO_PROBE_COMPLETE',feature_streams_equal=True,models_executed=False,test_features_read=False,full_staging_performed=False,
        reports=reports,sample_stage_seconds=sample['stage_seconds'],sample_unique_images=sample['unique_TRAIN_images'],seconds=time.monotonic()-started,
        limits=['Sampled CPU read+stack delivery, not full training or GPU/H2D/DDP timing.','OS page cache not controlled; mounted passes precede local staging, local pages recently written.','Local sample fits RAM; full34GiB-scale staging may not. No full-corpus speedup or ETA established.'],next='Inspect capacity and timing before choosing a full loader path. Do not automatically rerun GPU full10.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);run(read(p.parse_args().config))
