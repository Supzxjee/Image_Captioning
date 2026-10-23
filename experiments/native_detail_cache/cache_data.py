"""Locked full-cache layout, identities and resumable shard verification; no model imports."""
import hashlib
import json
from pathlib import Path
import numpy as np
from tile_data import tile_centers
from run_benchmark import digest,A_SHA,DATA_SHA

# Frozen tensor identities observed in the passed256-image benchmark.
EXPECTED_MODELS=['5afb7ecf2d5e841f41cbc470d64c15630c66fea5c9d336c260f8a8b5dd697335',
                 '93e3e39d86c54e5b6d976ec81da6c67a54f5a655518a641708a0c8dfcbe17adb']


def identity(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def plan(groups,data):
    shards=[];seen=set()
    for group,count in [('train',10000),('val',5000)]:
        rows=groups[group]
        if len(rows)!=count:raise ValueError('Wrong locked group size: '+group)
        canonical=[]
        for row in rows:
            source=data[row['image_id']];coco_id=int(source['filename'].rsplit('_',1)[1].split('.')[0])
            if row['coco_id']!=coco_id or row['filename']!=source['filename'] or coco_id in seen:raise ValueError('Image ID/order/split mismatch.')
            seen.add(coco_id)
            canonical.append(dict(coco_id=coco_id,image_id=row['image_id'],filename=source['filename'],filepath=source['filepath'],group=group))
        for offset in range(0,count,500):
            shards.append(dict(name=f'{group}_{offset//500:03d}',rank=len(shards)%2,rows=canonical[offset:offset+500]))
    return shards


def spec_for(shards,sources,archive_sha,revision):
    return dict(schema='native_detail_HL_15k_v1',checkpoint_sha256=A_SHA,dataset_sha256=DATA_SHA,
        groups_archive_sha256=archive_sha,manifest_sha256=identity(shards),clip_revision=revision,
        implementation_sha256=sources,images=15000,conditions=['H','L'],shape_per_condition=[196,512],storage_dtype='float16',
        model_sha256=EXPECTED_MODELS,compute_dtype='float32',batch_images=4,shard_images=500,preprocessing='bilinear RGB; fractional quadrants; H=native; L=whole224 first',
        pooling='CLIP discardCLS,14x14 avgpool2x2 before frozen A projection768to512/ReLU',coordinate_order='TL,TR,BL,BR;7x7 rowmajor')


def validate_spec(directory,spec):
    path=Path(directory)/'cache_spec.json'
    if not path.is_file() or json.loads(path.read_text(encoding='utf-8'))!=spec:raise ValueError('Resume cache protocol differs; preserve it and use matching notebook/inputs.')


def check_shard(path,shard,spec,expected_models=None,expected_runtime=None):
    import h5py
    path=Path(path)
    with h5py.File(path,'r') as f:
        if not bool(f.attrs.get('complete',False)) or bool(f.attrs.get('benchmark_only',True)):raise ValueError('Shard incomplete/benchmark-only: '+path.name)
        if f.attrs['spec_sha256']!=identity(spec) or f.attrs['rows_sha256']!=identity(shard['rows']):raise ValueError('Shard identity mismatch.')
        models=json.loads(f.attrs['model_sha256']);runtime=f.attrs['runtime_sha256']
        if expected_runtime is not None and runtime!=expected_runtime:raise ValueError('Resume compute runtime differs; do not mix encoded features.')
        if expected_models is not None and models!=expected_models:raise ValueError('Frozen model identity differs.')
        np.testing.assert_array_equal(f['coco_ids'][:],[r['coco_id'] for r in shard['rows']])
        np.testing.assert_array_equal(f['image_ids'][:],[r['image_id'] for r in shard['rows']])
        np.testing.assert_array_equal(f['xy'][:],tile_centers())
        for c in ('H','L'):
            ds=f[c]
            if ds.shape!=(len(shard['rows']),196,512) or ds.dtype!=np.dtype('float16'):raise ValueError('Shard feature layout differs.')
            for start in range(0,len(ds),32):
                if not np.isfinite(ds[start:start+32]).all():raise ValueError('Nonfinite persisted features.')
    return dict(file=path.name,bytes=path.stat().st_size,sha256=digest(path),images=len(shard['rows']),
        spec_sha256=identity(spec),rows_sha256=identity(shard['rows']),model_sha256=models,runtime_sha256=runtime)


def verify_receipt(path,receipt):
    path=Path(path)
    if path.name!=receipt['file'] or path.stat().st_size!=receipt['bytes'] or digest(path)!=receipt['sha256']:raise ValueError('Shard payload hash mismatch: '+path.name)
