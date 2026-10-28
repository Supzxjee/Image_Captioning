"""One-time image metadata extraction only; no image/features/model execution."""
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
VAL_SHA='e1b4d1bc54906827b1a029981b1c77f2088bd4462569b7615db7929070c3ab47'


def main():
    archive=ROOT/'experiments/coco_grounding_feasibility/review/full_annotations/annotations_trainval2014.zip'
    with zipfile.ZipFile(archive) as z:raw=z.read('annotations/instances_val2014.json')
    if hashlib.sha256(raw).hexdigest()!=VAL_SHA:raise ValueError('Changed official COCO val annotation member')
    images={r['id']:r for r in json.loads(raw)['images']};del raw
    meta=json.loads(gzip.decompress((ROOT/'experiments/semantic_grounding_method/data/gain_metadata.json.gz').read_bytes()))
    rows=[]
    for r in meta['groups']['val']:
        original=images[r['coco_id']]
        if original['file_name']!=r['filename']:raise ValueError('Image identity differs')
        rows.append({**r,'width':original['width'],'height':original['height']})
    if len(rows)!=1000 or len({r['coco_id'] for r in rows})!=1000:raise ValueError('Wrong fixed dev cohort')
    obj=dict(schema='fixed_dev1000_dimensions_v1',source_member='annotations/instances_val2014.json',
        source_member_sha256=VAL_SHA,source_url='https://images.cocodataset.org/annotations/annotations_trainval2014.zip',
        purpose='Original image dimensions only, for normalizing predicted boxes; no GT boxes used as model inputs',rows=rows)
    target=HERE/'DEV_DIMENSIONS.json';target.write_bytes((json.dumps(obj,indent=2)+'\n').encode())
    print('Verified1000 image dimensions; SHA',hashlib.sha256(target.read_bytes()).hexdigest())


if __name__=='__main__':main()
