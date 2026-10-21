"""Fetch only audit-sample COCO JPEGs and render phrase/box evidence, no models."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import html
import io
import json
from pathlib import Path
import textwrap
import urllib.request
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).parent / 'review'


def fetch(row):
    path = ROOT / 'images' / row['filename']
    path.parent.mkdir(exist_ok=True)
    url = f"https://images.cocodataset.org/{row['filepath']}/{row['filename']}"
    if not path.exists():
        try:
            raw = urllib.request.urlopen(url, timeout=30).read()
        except Exception:
            # Public original COCO image server also supports HTTP; no credentials.
            url = url.replace('https://', 'http://', 1)
            raw = urllib.request.urlopen(url, timeout=30).read()
        with Image.open(io.BytesIO(raw)) as im:
            im.verify()
        path.write_bytes(raw)
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as im:
        if list(im.size) != row['original_wh']:
            raise ValueError('Image/cache dimensions differ: ' + row['filename'])
    return dict(coco_id=row['coco_id'], url=url, sha256=hashlib.sha256(raw).hexdigest(),
                byte_identity_matches_cache=hashlib.sha256(raw).hexdigest() == row['cache_image_sha256'])


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size)


def make():
    rows = json.loads((ROOT / 'sample30.json').read_text(encoding='utf-8'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        sources = list(pool.map(fetch, rows))
    (ROOT / 'sample_image_provenance.json').write_text(json.dumps(sources, indent=2) + '\n', encoding='utf-8')
    cards = []
    for index, row in enumerate(rows):
        image = Image.open(ROOT / 'images' / row['filename']).convert('RGB')
        image.thumbnail((480, 330))
        draw = ImageDraw.Draw(image)
        for kind, boxes in [('VG', row['vg_boxes_normalized']), ('UPT', row['predicted_boxes_normalized'])]:
            for endpoint, box in zip(('S', 'O'), boxes):
                xy = [box[0]*image.width, box[1]*image.height, box[2]*image.width, box[3]*image.height]
                color = ('#159b44' if endpoint == 'S' else '#e26f00') if kind == 'UPT' else '#0575ed'
                draw.rectangle(xy, outline=color, width=3 if kind == 'UPT' else 1)
                draw.text((xy[0]+2, max(0, xy[1]-(17 if kind == 'UPT' else 32))), kind+endpoint, font=font(14), fill=color, stroke_width=1, stroke_fill='white')
        card = Image.new('RGB', (500, 435), 'white'); card.paste(image, ((500-image.width)//2, 55))
        d = ImageDraw.Draw(card)
        d.text((8, 2), f"#{index+1} COCO {row['coco_id']} | {row['predicate']}", font=font(18), fill='black')
        for line, text in enumerate(textwrap.wrap(row['phrase'], 48)[:2]):
            d.text((8, 23+line*17), text, font=font(15), fill='black')
        d.text((8, 390), 'UPT: S green, O orange. VG: blue thin.', font=font(14), fill='black')
        d.text((8, 410), 'IoU S/O: ' + '/'.join(f'{v:.2f}' for v in row['endpoint_iou']), font=font(14), fill='black')
        card.save(ROOT / f'card_{index+1:02d}.jpg', quality=92)
        cards.append(card)
    for page in range((len(cards)+5)//6):
        sheet = Image.new('RGB', (1000, 1305), '#dddddd')
        for offset, card in enumerate(cards[page*6:page*6+6]):
            sheet.paste(card, ((offset % 2)*500, (offset//2)*435))
        sheet.save(ROOT / f'contact_{page+1:02d}.jpg', quality=94)
    parts = ['<!doctype html><meta charset="utf-8"><title>VG phrase–pair audit 30</title>',
             '<style>body{font:18px Arial;background:#eef2f6;max-width:1100px;margin:30px auto}article{background:white;padding:20px;margin:20px 0}img{max-width:100%}pre{white-space:pre-wrap}</style>',
             '<h1>30 phrase–pair candidates — training images only</h1>',
             '<p>Rule-filtered candidates, not human-verified labels. UPT subject green/object orange; VG annotation thin blue. Read full phrase and check both instances; annotation agreement is not proof of image truth.</p>']
    for index, row in enumerate(rows):
        parts += [f'<article><h2>#{index+1}: {html.escape(row["phrase"])}</h2>',
                  f'<img src="card_{index+1:02d}.jpg"><details><summary>Raw original</summary><img src="images/{row["filename"]}"></details>',
                  '<pre>'+html.escape(json.dumps(row, ensure_ascii=False, indent=2))+'</pre></article>']
    (ROOT / 'index.html').write_text('\n'.join(parts), encoding='utf-8')
    print('Review ready:', ROOT / 'index.html')
    print('Original JPEG byte identity matches cache:', sum(r['byte_identity_matches_cache'] for r in sources), '/', len(sources))


if __name__ == '__main__':
    make()
