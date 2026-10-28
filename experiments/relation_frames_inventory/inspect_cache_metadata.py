"""Inspect HDF5 headers/IDs only. No feature tensors, torch or neural execution."""
import argparse
import gzip
import json
from pathlib import Path
import sys

from inventory import HERE,sha,CAPTION_SHA


def main():
    p=argparse.ArgumentParser();p.add_argument('--visual',type=Path,required=True);p.add_argument('--h5py-path',type=Path)
    args=p.parse_args()
    if args.h5py_path:sys.path.insert(0,str(args.h5py_path))
    import h5py
    raw=(HERE/'V2_MANIFEST.json.gz').read_bytes();manifest=json.loads(gzip.decompress(raw))
    wanted={p[k]['image_id'] for s in ('train','heldout') for p in manifest[s] for k in ('positive','negative')}
    found=set();seen=set();shards=[]
    for path in sorted(args.visual.glob('*.h5')):
        with h5py.File(path,'r') as f:
            ids=[int(x) for x in f['imgids'][:]];metadata=json.loads(f.attrs['metadata_json'])
            if not f.attrs['complete'] or f['features'].shape!=(len(ids),197,768) or str(f['features'].dtype)!='float16':raise ValueError('Wrong feature schema')
            if metadata['dataset_json_sha256']!=CAPTION_SHA or metadata['preprocessing']!='bilinear':raise ValueError('Wrong feature source')
            if len(ids)!=len(set(ids)) or seen&set(ids):raise ValueError('Duplicate IDs')
            seen.update(ids);hits=set(ids)&wanted;found.update(hits)
            shards.append(dict(filename=path.name,complete=True,shape=list(f['features'].shape),dtype=str(f['features'].dtype),
                wanted_ids_found=len(hits),id_list_sha256=sha(json.dumps(ids,separators=(',',':')).encode()),metadata=metadata))
    result=dict(status='ALL_V2_IDS_PRESENT_HEADERS_ONLY' if found==wanted else 'CACHE_IDS_INCOMPLETE',manifest_sha256=sha(raw),
                wanted_images=len(wanted),found_images=len(found),missing=sorted(wanted-found),shards=shards,
                feature_values_read=False,cache_file_bytes_fully_hashed=False,local_neural_execution=False,
                legacy_prompt_coverage='UNVERIFIED_LOCALLY; CPU Kaggle preparation must check pinned original embedding cache')
    (HERE/'CACHE_COVERAGE.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='shards'},indent=2))


if __name__=='__main__':main()
