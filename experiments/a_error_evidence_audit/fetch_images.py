"""Retrieve only the selected COCO review pixels; no model execution."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import textwrap
import urllib.request
from PIL import Image,ImageDraw,ImageOps

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OUT=HERE/'review';IMAGES=OUT/'images'

def main():
    rows=json.loads((OUT/'candidate30.json').read_text(encoding='utf-8'));IMAGES.mkdir(exist_ok=True)
    wanted={r['filename'] for r in rows};local={}
    for p in ROOT.glob('experiments/**/review/**/*.jpg'):
        if p.name in wanted and IMAGES not in p.parents:local.setdefault(p.name,p)
    def fetch(r):
        # COCO annotation coco_url uses HTTP; CDN HTTPS certificate does not match
        # this hostname. Preserve the public source URL rather than disable TLS.
        dest=IMAGES/r['filename'];url=f'http://images.cocodataset.org/val2014/{r["filename"]}'
        if dest.exists():data=dest.read_bytes();method='existing audit copy'
        elif r['filename'] in local:data=local[r['filename']].read_bytes();method='existing local review pixels'
        else:
            with urllib.request.urlopen(url,timeout=45) as response:data=response.read()
            method='official COCO image CDN'
        import io
        with Image.open(io.BytesIO(data)) as im:im.verify()
        dest.write_bytes(data)
        return dict(coco_id=r['coco_id'],filename=r['filename'],url=url,method=method,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    with ThreadPoolExecutor(max_workers=5) as pool:receipts=list(pool.map(fetch,rows))
    (OUT/'image_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n',encoding='utf-8')
    for page in range(3):
        canvas=Image.new('RGB',(1600,2200),'white');d=ImageDraw.Draw(canvas)
        for j,r in enumerate(rows[page*10:(page+1)*10]):
            col=j%2;line=j//2;x=col*800;y=line*440
            with Image.open(IMAGES/r['filename']) as raw:im=ImageOps.contain(raw.convert('RGB'),(780,340))
            canvas.paste(im,(x+(800-im.width)//2,y))
            d.text((x+8,y+343),f'{page*10+j+1:02d} | COCO {r["coco_id"]}',fill='black',font_size=21)
            for k,line in enumerate(textwrap.wrap('A: '+r['caption_A'],width=64)[:3]):d.text((x+8,y+370+22*k),line,fill='black',font_size=20)
        canvas.save(OUT/f'contact_{page+1}.jpg',quality=93)
    print('30 image receipts; 3 contact sheets. No model execution.')

if __name__=='__main__':main()
