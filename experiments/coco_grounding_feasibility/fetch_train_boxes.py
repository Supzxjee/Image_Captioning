"""Fetch official COCO metadata, stream only locked TRAIN10k instances; no image pixels."""
import hashlib
import json
from pathlib import Path
import sys
import time
from urllib.request import urlopen
import zipfile

HERE=Path(__file__).resolve().parent
URL='https://s3.amazonaws.com/images.cocodataset.org/annotations/annotations_trainval2014.zip'
OUT=HERE/'review/full_annotations'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    sys.path.insert(0,str(HERE.parents[2]/'.test-deps/coco-grounding'))
    import ijson
    OUT.mkdir(parents=True,exist_ok=True)
    archive=OUT/'annotations_trainval2014.zip'
    if not archive.exists():
        partial=archive.with_suffix('.zip.partial');last=time.monotonic();total=0
        print('Downloading official COCO instance/caption metadata archive; no images:',URL,flush=True)
        with urlopen(URL,timeout=90) as response,partial.open('wb') as f:
            length=int(response.headers['Content-Length']);etag=response.headers.get('ETag')
            while block:=response.read(4*1024**2):
                f.write(block);total+=len(block)
                if time.monotonic()-last>=8:
                    print(f'COCO annotation archive {total/1024**2:.1f}/{length/1024**2:.1f} MiB',flush=True);last=time.monotonic()
        assert total==length
        partial.replace(archive)
        (OUT/'download_receipt.json').write_text(json.dumps(dict(url=URL,bytes=length,etag=etag,sha256=digest(archive)),indent=2)+'\n',encoding='utf-8')
    receipt=json.loads((OUT/'download_receipt.json').read_text(encoding='utf-8'))
    assert receipt['url']==URL and archive.stat().st_size==receipt['bytes'] and digest(archive)==receipt['sha256']
    artifact=HERE.parent/'vg_region_caption/review/loaded_A.json'
    groups=json.loads(artifact.read_text(encoding='utf-8'))['json']['groups.json']['train']
    wanted={r['coco_id']:r for r in groups};assert len(wanted)==10000
    images=[];annotations=[];categories=None;members=[]
    with zipfile.ZipFile(archive) as z:
        assert len(z.namelist())==len(set(z.namelist()))
        for split in ('train','val'):
            name=f'annotations/instances_{split}2014.json';info=z.getinfo(name)
            print('Verifying uncompressed member hash and CRC:',name,flush=True)
            h=hashlib.sha256()
            with z.open(name) as f:
                for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
            member_sha=h.hexdigest()
            if split=='train':
                assert member_sha=='777ef504abfeba349930485bb32ecd5b6529129a3e010f7852e1437cb049b732','Official train member differs from previously verified provenance.'
            members.append(dict(member=name,crc32=f'{info.CRC:08x}',compressed_bytes=info.compress_size,
                uncompressed_bytes=info.file_size,sha256=member_sha))
            for section in ('categories','images','annotations'):
                print('Streaming',split,section,'for locked TRAIN IDs only',flush=True)
                with z.open(name) as f:
                    items=ijson.items(f,section+'.item',use_float=True)
                    if section=='categories':
                        current=list(items)
                        if categories is None: categories=current
                        else: assert categories==current
                    elif section=='images':
                        for r in items:
                            if r['id'] in wanted:
                                assert r['file_name']==wanted[r['id']]['filename']
                                images.append({k:r[k] for k in ('id','file_name','width','height')})
                    else:
                        for r in items:
                            if r['image_id'] in wanted:
                                annotations.append({k:r[k] for k in ('id','image_id','category_id','bbox','iscrowd')})
    assert len(images)==10000 and {r['id'] for r in images}==set(wanted)
    assert len({r['id'] for r in annotations})==len(annotations)
    target=OUT/'train10k_instances.json'
    payload=dict(images=sorted(images,key=lambda r:r['id']),annotations=sorted(annotations,key=lambda r:r['id']),categories=categories)
    target.write_text(json.dumps(payload,separators=(',',':'))+'\n',encoding='utf-8')
    provenance=dict(source=receipt,members=members,ijson_version=ijson.__version__,
        image_count=len(images),annotation_count=len(annotations),category_count=len(categories),
        split_policy='Only exact A train10k Karpathy train/restval IDs; no held-out supervision retained.',
        cohort_sha256=hashlib.sha256(json.dumps(groups,sort_keys=True).encode()).hexdigest(),
        filtered_file_sha256=digest(target),image_pixels_downloaded=False,model_run=False)
    (OUT/'train10k_instances.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print('TRAIN10k annotation READY:',len(images),'images',len(annotations),'instances',flush=True)


if __name__=='__main__':main()
