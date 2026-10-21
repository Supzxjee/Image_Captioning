"""Display the locked region24 sample as original image plus original region crop."""
from concurrent.futures import ThreadPoolExecutor
import html
import hashlib
import json
import math
from pathlib import Path
import textwrap
from PIL import Image, ImageDraw
from audit_regions import ROOT, OUTPUT
from make_review import fetch, font


def make():
    rows = json.loads((ROOT / 'REGION24_MANIFEST.json').read_text(encoding='utf-8'))
    with ThreadPoolExecutor(max_workers=4) as pool:
        provenance = list(pool.map(fetch, rows))
    assert all(r['byte_identity_matches_cache'] for r in provenance), 'Image bytes differ; stop review.'
    (ROOT / 'REGION24_IMAGE_PROVENANCE.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    cards = []
    review_file=ROOT/'REGION24_ASSISTANT_REVIEW.json'
    review=json.loads(review_file.read_text(encoding='utf-8')) if review_file.exists() else None
    if review is not None:
        assert review['sample_sha256']==hashlib.sha256((ROOT/'REGION24_MANIFEST.json').read_bytes()).hexdigest()
    for number, row in enumerate(rows,1):
        original = Image.open(ROOT / 'review/images' / row['filename']).convert('RGB')
        b=row['region_normalized']; w,h=original.size
        xy=[b[0]*w,b[1]*h,b[2]*w,b[3]*h]
        # Integer rounding for display only; annotation coordinates stay unchanged.
        crop=original.crop((math.floor(xy[0]),math.floor(xy[1]),math.ceil(xy[2]),math.ceil(xy[3])))
        full=original.copy();d=ImageDraw.Draw(full);d.rectangle(xy, outline='#00bc4d',width=3)
        full.thumbnail((238,300));crop.thumbnail((238,300))
        card=Image.new('RGB',(500,450),'white');d=ImageDraw.Draw(card)
        d.text((6,3),f"#{number} COCO {row['coco_id']} | {row['diagnostic_group']}",font=font(14),fill='black')
        for i,line in enumerate(textwrap.wrap(row['phrase'],58)[:3]):
            d.text((6,23+i*17),line,font=font(14),fill='black')
        d.text((6,80),'Full image + original region',font=font(13),fill='black')
        d.text((260,80),'Original region crop',font=font(13),fill='black')
        card.paste(full,((246-full.width)//2,102+(300-full.height)//2))
        card.paste(crop,(254+(246-crop.width)//2,102+(300-crop.height)//2))
        d.text((6,411),f"Area {100*row['region_area_fraction']:.1f}% | UPT pairs {row['selected_UPT_pairs']}",font=font(14),fill='black')
        card.save(OUTPUT/f'card_{number:02d}.jpg',quality=94);cards.append(card)
    for page in range(6):
        image=Image.new('RGB',(1000,900),'#dddddd')
        for j,card in enumerate(cards[page*4:page*4+4]):
            image.paste(card,((j%2)*500,(j//2)*450))
        image.save(OUTPUT/f'contact_{page+1:02d}.jpg',quality=95)
    parts=['<!doctype html><meta charset="utf-8"><title>Phrase–region 24</title>',
           '<style>body{font:18px Arial;background:#eef2f6;max-width:1100px;margin:30px auto}article{padding:20px;background:white;margin:20px 0}img{max-width:100%}pre{white-space:pre-wrap}</style>',
           '<h1>24 phrase–region candidates — original annotations</h1>',
           '<p>Eight human–object, eight spatial/other, eight no-selected-UPT-pair images. All train/restval. Sample locked before viewing; previous30 images excluded. No edits to phrase or region.</p>',
           '<p>Check separately: whole phrase supported? original region sufficient? correct instance? uncertainty allowed. This is a diagnostic sample, not dataset precision or human-confirmed GT.</p>']
    for number,row in enumerate(rows,1):
        parts += [f'<article><h2>#{number}: {html.escape(row["phrase"])}</h2>',
                  f'<img src="card_{number:02d}.jpg"><details><summary>Full original JPEG</summary><img src="../images/{row["filename"]}"></details>']
        if review is not None:
            assessment=review['rows'][number-1]
            parts += ['<p><strong>Assistant provisional, not human GT</strong></p><pre>'+html.escape(json.dumps(assessment,ensure_ascii=False,indent=2))+'</pre>']
        parts += ['<details><summary>Original annotation/mapping</summary><pre>'+html.escape(json.dumps(row,ensure_ascii=False,indent=2))+'</pre></details></article>']
    (OUTPUT/'index.html').write_text('\n'.join(parts),encoding='utf-8')
    print('Region24 gallery ready:',OUTPUT/'index.html',flush=True)


if __name__=='__main__':
    make()
