"""Collect official COCO images for an offline annotation artifact; never run a model."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
from urllib.request import urlopen
from PIL import Image


def collect(output,local_root):
    output=Path(output);manifest=json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    events={e['filename']:e for g in manifest['groups'] for e in g['events']}
    directory=output/'images';directory.mkdir(exist_ok=True)
    listing=subprocess.run(['rg','--files',str(local_root),'-g','*.jpg'],capture_output=True,text=True,encoding='utf-8',check=True).stdout
    local={}
    for name in listing.splitlines():
        p=Path(name)
        if p.name in events:local.setdefault(p.name,p)
    def fetch(item):
        name,event=item;target=directory/name
        def valid(raw):
            with Image.open(io.BytesIO(raw)) as im:
                assert im.format=='JPEG' and im.size==(event['width'],event['height']),name
                im.verify()
        if target.exists():raw=target.read_bytes();valid(raw);source='existing_offline_copy'
        elif name in local:
            raw=local[name].read_bytes();valid(raw);shutil.copy2(local[name],target);source='existing_local_COCO_copy'
        else:
            error=None
            for scheme in ('https','http'):
                url=f'{scheme}://images.cocodataset.org/{event["filepath"]}/{name}'
                try:
                    with urlopen(url,timeout=35) as response:raw=response.read()
                    valid(raw);target.write_bytes(raw);source=url;break
                except Exception as exc:error=exc
            else:raise RuntimeError(f'Could not collect {name}: {error}')
        return dict(filename=name,source=source,sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),size=[event['width'],event['height']])
    records=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for record in pool.map(fetch,sorted(events.items())):
            records.append(record);print(f'Images {len(records)}/{len(events)} {record["filename"]}',flush=True)
    (output/'image_receipt.json').write_text(json.dumps(records,indent=2),encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--local-root',type=Path,required=True)
    a=p.parse_args();collect(a.output,a.local_root)
