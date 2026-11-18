"""Lossless sequential H5 -> local patch memmap, user-run preparation only."""
import hashlib
import os
from pathlib import Path
import shutil
import time
import numpy as np
from real_bank_data import read,save,sha
from full_data import load_stage,TRAIN_IMAGES,VALID_IMAGES


def stage_local(cfg,deadline):
    import h5py
    started=time.monotonic();root=Path(cfg['stage']);out=Path(cfg['output'])
    meta=read(root/'manifest.json');count=TRAIN_IMAGES+VALID_IMAGES
    if meta.get('feature_arrays_staged'):raise ValueError('Expected fresh complete metadata stage')
    needed=count*196*768*2
    if shutil.disk_usage(root).free<needed+4*1024**3:raise RuntimeError('Need full local FP16 payload plus4GiB scratch; do not load into RAM')
    patch_path=root/'local_patches.npy';map_path=root/'local_rows.npy'
    if patch_path.exists() or map_path.exists():raise RuntimeError('Preserve old local stage; fresh stage required')
    data,_,meta=load_stage(root)
    if len(data['ids'])!=count:raise ValueError('Wrong local-stage image count')
    dest=np.lib.format.open_memmap(patch_path,mode='w+',dtype=np.float16,shape=(count,196,768))
    locations=np.asarray(data['locations']);local_rows=np.full(count,-1,np.int64);cursor=0;golden_seen=set()
    golden=read(Path(__file__).with_name('SMOKE_SOURCE.json'))['raw_selected_row_sha256']
    progress=dict(status='LOCAL_PATCH_STAGE_RUNNING',images=0,expected_images=count,patch_bytes=needed,test_features_read=False)
    def time_guard():
        if time.monotonic()>deadline:raise TimeoutError('Full local stage deadline; preserve log, not complete training data')
    try:
        for file,source in enumerate(meta['sources']):
            time_guard();path=Path(source['path'])
            if path.stat().st_size!=source['bytes']:raise ValueError('Changed raw H5 size')
            selected=np.flatnonzero(locations[:,0]==file)
            selected=selected[np.argsort(locations[selected,1],kind='stable')]
            rows=locations[selected,1]
            if len(np.unique(rows))!=len(rows):raise ValueError('Duplicate physical feature rows')
            with h5py.File(path,'r',rdcc_nbytes=2*1024**2) as f:
                features=f['features']
                if list(features.shape)!=source['shape'] or features.dtype!=np.dtype('float16'):raise ValueError('Changed H5 feature shape/dtype')
                for start in range(0,len(selected),256):
                    time_guard();image=selected[start:start+256];physical=rows[start:start+256]
                    # Read selected rows only, in increasing physical order. TEST rows never selected.
                    raw=features[physical]
                    if raw.shape!=(len(image),197,768) or not np.isfinite(raw).all():raise ValueError('Invalid full-stage feature block')
                    actual_ids=np.asarray(f['imgids'][physical],np.int64)
                    if not np.array_equal(actual_ids,data['ids'][image]):raise ValueError('Source COCO mapping changed')
                    for j,i in enumerate(image):
                        coco=str(int(data['ids'][i]))
                        if coco in golden:
                            if hashlib.sha256(np.ascontiguousarray(raw[j]).tobytes()).hexdigest()!=golden[coco]:raise ValueError('Smoke raw replay differs '+coco)
                            golden_seen.add(coco)
                    values=np.ascontiguousarray(raw[:,1:]);end=cursor+len(image)
                    dest[cursor:end]=values
                    if not np.array_equal(dest[cursor:end].view(np.uint16),values.view(np.uint16)):raise ValueError('Local FP16 bytes changed')
                    if (local_rows[image]!=-1).any():raise ValueError('Repeated logical image')
                    local_rows[image]=np.arange(cursor,end);cursor=end
                    progress.update(images=cursor,seconds=time.monotonic()-started);save(out/'local_stage_progress.json',progress)
                    if start%1024==0 or start+256>=len(selected):print('Local full patches',cursor,'/',count,round(time.monotonic()-started),'s',flush=True)
        if cursor!=count or not np.array_equal(np.sort(local_rows),np.arange(count)):raise ValueError('Incomplete/nonpermutation local rows')
        expected_golden=set(golden)&{str(int(x)) for x in data['ids']}
        if golden_seen!=expected_golden:raise ValueError('Incomplete smoke row replay')
        dest.flush()
    except BaseException:
        progress.update(status='LOCAL_PATCH_STAGE_INCOMPLETE',seconds=time.monotonic()-started);save(out/'local_stage_progress.json',progress);raise
    finally:
        dest._mmap.close()
        for value in data.values():value._mmap.close()
    np.save(map_path,local_rows);time_guard()
    print('Sealing full local patch SHA; sequential read, no model.',flush=True)
    patches_sha=sha(patch_path);mapping_sha=sha(map_path);time_guard()
    meta['files_sha256'].update({'local_patches.npy':patches_sha,'local_rows.npy':mapping_sha})
    meta.update(feature_arrays_staged=True,feature_storage='local_fp16_memmap_v2',local_features=dict(file='local_patches.npy',rows='local_rows.npy',shape=[count,196,768],dtype='float16'),
        local_stage=dict(images=count,patch_bytes=needed,seconds=time.monotonic()-started,smoke_raw_rows_replayed=len(golden_seen),all_copy_blocks_byte_equal=True,test_features_read=False))
    temp=root/'manifest.tmp';save(temp,meta);os.replace(temp,root/'manifest.json')
    save(out/'stage_summary.json',meta);save(out/'local_stage_progress.json',dict(status='LOCAL_PATCH_STAGE_COMPLETE',**meta['local_stage']))
    print('Local full stage COMPLETE:',count,'images,',round(meta['local_stage']['seconds'],1),'s',flush=True)
