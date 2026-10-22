"""Collect at most16 public COCO images for the fixed mapping audit; no ML inference."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
from urllib.request import urlopen
from PIL import Image,ImageDraw,ImageOps

HERE=Path(__file__).resolve().parent
OUT=HERE/'review/noun_box_train10k_v1'


def main():
    rows=json.loads((OUT/'manual16.json').read_text(encoding='utf-8'));assert len(rows)==16
    directory=OUT/'manual_images';directory.mkdir(exist_ok=True)
    listing=subprocess.run(['rg','--files','--hidden','--no-ignore',str(HERE.parent),'-g','*.jpg'],capture_output=True,text=True,encoding='utf-8',check=True).stdout
    local={Path(p).name:Path(p) for p in listing.splitlines()}
    images={r['filename']:r for r in rows}
    def collect(item):
        filename,row=item;target=directory/filename;source='existing_review_image'
        if not target.exists():
            if filename in local:shutil.copy2(local[filename],target);source='existing_local_copy'
            else:
                split=filename.split('_')[1]
                source=f'https://s3.amazonaws.com/images.cocodataset.org/{split}/{filename}'
                with urlopen(source,timeout=45) as response:raw=response.read()
                with Image.open(io.BytesIO(raw)) as im:im.verify()
                target.write_bytes(raw)
        with Image.open(target) as im:
            assert list(im.size)==row['box']['image_size'];im.verify()
        return dict(filename=filename,source=source,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    with ThreadPoolExecutor(max_workers=4) as pool:receipts=list(pool.map(collect,sorted(images.items())))
    (OUT/'manual_image_receipt.json').write_text(json.dumps(receipts,indent=2)+'\n',encoding='utf-8')
    for page in range(2):
        canvas=Image.new('RGB',(1536,840),'white');draw=ImageDraw.Draw(canvas)
        for j,row in enumerate(rows[8*page:8*(page+1)]):
            x=j%4*384;y=j//4*420
            with Image.open(directory/row['filename']) as im:im=im.convert('RGB')
            overlay=im.copy();annotate=ImageDraw.Draw(overlay)
            bx,by,bw,bh=row['box']['bbox_xywh'];annotate.rectangle((bx,by,bx+bw,by+bh),outline='#00e070',width=3)
            overlay=ImageOps.contain(overlay,(378,280));canvas.paste(overlay,(x,y))
            text=[f"{page*8+j+1}. COCO{row['coco_id']} [{row['category']}]",
                  f"span={row['raw_span']} logits={row['decoder_logit_positions']}"]
            line=''
            for word in row['caption'].split():
                if len(line+' '+word)>56:text.append(line);line=word
                else:line=(line+' '+word).strip()
            text.append(line)
            text.append(f"area={row['box']['image_fraction']:.3f} / annotation={row['box']['annotation_id']}")
            draw.multiline_text((x+3,y+285),'\n'.join(text),fill='black',spacing=5)
        canvas.save(OUT/f'manual_contact_{page+1}.jpg')
    print('Fixed16 mapping images collected:',len(images),'unique images;no model run.')
    for i,r in enumerate(rows,1):print(i,r['coco_id'],r['category'],r['raw_span'],r['caption'])


if __name__=='__main__':main()
