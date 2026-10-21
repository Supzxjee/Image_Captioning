"""Lock <=24 stored-caption cases before viewing; collect images, never infer."""
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import subprocess
from urllib.request import urlopen
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageOps, ImageDraw

HERE = Path(__file__).resolve().parent
SHA = '5f8eead3f4e960189345b1a9bf492763fe52ec71f70115b73244ffe9d03d2e75'


def main():
    source = HERE.parent / 'vg_region_caption/review/loaded_A.json'
    data = json.loads(source.read_text(encoding='utf-8'))['json']
    assert data['protocol.json']['arm'] == 'A'
    assert data['protocol.json']['source_checkpoint_sha256'] == SHA
    groups = {r['image_id']: r for r in data['groups.json']['val']}
    predictions = data['origin_predictions.json']
    assert len(predictions) == len(groups) == 1000
    assert {r['image_id'] for r in predictions} == set(groups)
    nouns = r'\b(surfboard|surfboards|skateboard|skateboards|phone|phones)\b'
    verbs = r'\b(riding|ride|rides|holding|hold|holds|carrying|carry|carries|standing|sitting|talking|looking|ear)\b'
    selected = [r for r in predictions if re.search(nouns, r['caption'].lower()) and re.search(verbs, r['caption'].lower())]
    selected.sort(key=lambda r: hashlib.sha256(('drsa-screen-v1:'+str(r['image_id'])).encode()).hexdigest())
    rows = [dict(order=i+1, **groups[r['image_id']], original_caption=r['caption']) for i, r in enumerate(selected[:24])]
    assert len(rows) == 24 and len({r['coco_id'] for r in rows}) == 24
    assert not any(r['split'] == 'test' for r in rows)
    out = HERE / 'SCREEN_24_SELECTION.json'
    payload = dict(status='SELECTION_LOCKED_BEFORE_IMAGE_REVIEW', checkpoint_sha256=SHA,
        source_artifact='vg_region_caption/review/loaded_A.json',
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        captions_sha256=hashlib.sha256(json.dumps(predictions,sort_keys=True).encode()).hexdigest(),
        noun_regex=nouns, relation_regex=verbs, order='sha256(UTF8(drsa-screen-v1: + decimal Karpathy image_id)), ascending',
        qualifying_captions=len(selected), maximum_images=24, rows=rows,
        selection_uses_reference=False, selection_uses_cider=False, model_run=False)
    if out.exists():
        assert json.loads(out.read_text(encoding='utf-8')) == payload, 'Do not change a locked selection.'
    else:
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    image_dir = HERE / 'review/screen24/images'
    image_dir.mkdir(parents=True, exist_ok=True)
    files = subprocess.run(['rg','--files','--hidden','--no-ignore',str(HERE.parent),'-g','*.jpg'],capture_output=True,text=True,encoding='utf-8',check=True).stdout.splitlines()
    local = {Path(p).name:Path(p) for p in files}

    def collect(row):
        name=row['filename']; target=image_dir/name
        if not target.exists():
            if name in local:
                shutil.copy2(local[name],target)
            else:
                split=name.split('_')[1]
                # COCO's public image host does not present a matching HTTPS certificate.
                with urlopen(f'http://images.cocodataset.org/{split}/{name}',timeout=35) as response:
                    raw=response.read()
                with Image.open(io.BytesIO(raw)) as im: im.verify()
                target.write_bytes(raw)
        with Image.open(target) as im: im.verify()
        return dict(coco_id=row['coco_id'],filename=name,sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    with ThreadPoolExecutor(max_workers=4) as pool:
        receipts=list(pool.map(collect,rows))
    (image_dir.parent/'image_receipt.json').write_text(json.dumps(receipts,indent=2)+'\n',encoding='utf-8')
    for page in range(3):
        canvas=Image.new('RGB',(1200,740),'white'); draw=ImageDraw.Draw(canvas)
        for j,row in enumerate(rows[page*8:(page+1)*8]):
            x=j%4*300;y=j//4*370
            with Image.open(image_dir/row['filename']) as im:
                canvas.paste(ImageOps.contain(im.convert('RGB'),(296,290)),(x,y))
            draw.text((x+3,y+293),f"{row['order']} COCO {row['coco_id']} / imgid {row['image_id']}",fill='black')
            words=row['original_caption'].split(); lines=[]; line=''
            for word in words:
                if len(line+' '+word)>43: lines.append(line);line=word
                else: line=(line+' '+word).strip()
            lines.append(line)
            draw.multiline_text((x+3,y+311),'\n'.join(lines),fill='black',spacing=3)
        canvas.save(image_dir.parent/f'contact_{page+1}.jpg')
    print('Locked and collected',len(rows),'images from',len(selected),'qualifying captions; no model run.')


if __name__ == '__main__':
    main()
