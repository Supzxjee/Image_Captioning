"""Read HDF5 headers and ID arrays only. No feature extraction, torch or training."""
import argparse
import hashlib
import json
from pathlib import Path


def inspect(root):
    import h5py
    import numpy as np
    shards=sorted(Path(root).rglob('*.h5'))+sorted(Path(root).rglob('*.hdf5'))
    assert shards,'No HDF5 shards under specified VISUAL_CACHE'
    rows=[];all_ids=set();overlap=[]
    def scalar(value):
        if isinstance(value,bytes):return value.decode('utf-8',errors='replace')
        if isinstance(value,(str,int,float,bool)) or value is None:return value
        if isinstance(value,np.generic):return value.item()
        if hasattr(value,'tolist'):return value.tolist()
        return str(value)
    for path in shards:
        with h5py.File(path,'r') as f:
            attrs={k:scalar(v) for k,v in f.attrs.items()};meta=attrs.get('metadata_json')
            if isinstance(meta,str):
                try:meta=json.loads(meta)
                except json.JSONDecodeError:meta={'unparsed_metadata_json':meta}
            datasets={k:dict(shape=list(v.shape),dtype=str(v.dtype),attributes={a:scalar(b) for a,b in v.attrs.items()}) for k,v in f.items() if isinstance(v,h5py.Dataset)}
            idname=next((k for k in ('imgids','coco_ids','image_ids','ids') if k in f),None)
            ids=f[idname][...] if idname else None
            if ids is not None:
                assert ids.ndim==1 and np.issubdtype(ids.dtype,np.integer),'ID mapping needs manual inspection'
                ints=list(map(int,ids));assert len(ints)==len(set(ints)),f'Duplicate IDs in {path}'
                duplicate=set(ints)&all_ids;overlap.extend(sorted(duplicate));all_ids.update(ints)
                idsummary=dict(key=idname,count=len(ints),first=ints[:5],last=ints[-5:],sha256=hashlib.sha256(ids.tobytes()).hexdigest())
            else:idsummary={'status':'MISSING_EXPECTED_ID_KEY'}
            rows.append(dict(filename=path.name,bytes=path.stat().st_size,attributes=attrs,metadata=meta,datasets=datasets,id_summary=idsummary))
    return dict(status='METADATA_ONLY_NO_MODEL_NO_TRAINING',shards=rows,unique_ids=len(all_ids),overlapping_ids=sorted(set(overlap)),
        feature_values_read=False,encoder_revision_verified=False,feature_compatibility_verified=False,
        next='Read metadata; later pinned re-encoding must verify feature compatibility before any training. Missing metadata is unknown, not automatically a failure.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    result=inspect(a.root);Path(a.output).write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8');print(json.dumps(result,indent=2,ensure_ascii=False))
