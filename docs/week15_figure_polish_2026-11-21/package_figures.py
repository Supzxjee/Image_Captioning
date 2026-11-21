"""Package the verified figure exports and a contact sheet for the user."""
from pathlib import Path
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from zipfile import ZipFile, ZIP_DEFLATED
from PIL import Image, ImageOps, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
OUT=Path('C:/Users/leduc/Downloads/Bo_hinh_tuan15_dep_v1')
NAMES=['01_full_pipeline','02_M1_semantic_bank','03_M2_bank_visual_alignment','04_M3_prompt_visual_alignment']
SIZES=[(3680,2320),(2300,1620),(3520,2320),(3280,1640)]


def main():
    OUT.mkdir(exist_ok=True)
    for name,size in zip(NAMES,SIZES):
        with Image.open(ROOT/(name+'.png')) as im:
            assert im.size==size,(name,im.size,size)
            assert ImageOps.grayscale(im).getextrema()[0]<80,'Empty raster'
        for suffix in ('png','svg','drawio'):
            dest=OUT/suffix.upper();dest.mkdir(exist_ok=True)
            shutil.copy2(ROOT/(name+'.'+suffix),dest/(name+'.'+suffix))
        for suffix in ('svg','drawio'):ET.parse(ROOT/(name+'.'+suffix))
    combined=ROOT/'Tat_ca_4_hinh.drawio'
    assert len(ET.parse(combined).getroot().findall('diagram'))==4
    shutil.copy2(combined,OUT/combined.name)
    # Contact sheet only; the four individual full-resolution exports are primary.
    gallery=Image.new('RGB',(1900,1390),'#EDF1F5')
    draw=ImageDraw.Draw(gallery)
    font=ImageFont.truetype('C:/Windows/Fonts/timesbd.ttf',25)
    for idx,(name,label) in enumerate(zip(NAMES,['Pipeline tổng thể','M1 · Xây dựng bank','M2 · Căn chỉnh bank–ảnh','M3 · Căn chỉnh prompt–ảnh'])):
        x=25+(idx%2)*950;y=20+(idx//2)*690
        draw.rectangle((x,y,x+925,y+665),fill='white',outline='#CBD5DE',width=1)
        draw.text((x+20,y+12),label,font=font,fill='#243747')
        with Image.open(ROOT/(name+'.png')) as im:
            preview=ImageOps.contain(im.convert('RGB'),(895,600),Image.Resampling.LANCZOS)
            gallery.paste(preview,(x+(925-preview.width)//2,y+55+(600-preview.height)//2))
    gallery.save(ROOT/'Xem_truoc.png')
    gallery.save(OUT/'Xem_truoc.png')
    readme='''BỘ HÌNH BÁO CÁO TUẦN 15 — 21/11/2026

PNG: ảnh nền trắng độ phân giải cao, dùng trong Word/PowerPoint.
SVG: vector, phóng lớn không vỡ; chữ và các thành phần vẫn là vector.
DRAWIO: từng hình chỉnh sửa được.
Tat_ca_4_hinh.drawio: mở cả bộ bốn trang trong diagrams.net.
Xem_truoc.png: xem nhanh; dùng ảnh riêng trong PNG khi đưa vào báo cáo.

01: Pipeline tổng thể. 02: M1. 03: M2. 04: M3.
Các đường nối thẳng/gấp vuông; phân biệt điểm rẽ với đường đi ngang.
Giữ bank danh từ/động từ và các memory đối tượng/quan hệ tách riêng.
Nội dung dựa trên bốn hình trong báo cáo người dùng cung cấp.
Không thay đổi file Word, mô hình hoặc kết quả thực nghiệm.
'''
    (OUT/'README.txt').write_text(readme,encoding='utf-8')
    checks=json.loads((ROOT/'CHECKS.json').read_text(encoding='utf-8'))
    checks.update(status='GEOMETRY_AND_VISUAL_REVIEW_PASS',visual_review=dict(all_four_renders_inspected=True,refinements=['Restored consistent object/relation tint in M1','Removed unnecessary aggregation elbows in M2','Preserved plus-circles and separate M3 output memories'],remaining_aesthetic_acceptance='User review pending'),raster_sizes={n:s for n,s in zip(NAMES,SIZES)},output_folder=str(OUT))
    report=Path('C:/Users/leduc/Downloads/Bao_cao_nghien_cuu_tuan_15.docx')
    assert hashlib.sha256(report.read_bytes()).hexdigest()==checks['report_sha256']
    # Cross-check every delivered original export, not just ZIP CRC.
    manifest={str(p.relative_to(OUT)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file() and p.name!='MANIFEST.json'}
    for name in NAMES:
        for suffix in ('png','svg','drawio'):
            key=suffix.upper()+'/'+name+'.'+suffix
            assert manifest[key]==hashlib.sha256((ROOT/(name+'.'+suffix)).read_bytes()).hexdigest()
    (OUT/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    archive=OUT.with_suffix('.zip')
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(OUT))
    with ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len([p for p in OUT.rglob('*') if p.is_file()])
    checks.update(zip_crc_pass=True,copied_sha256_pass=True,combined_drawio_pages=4)
    (ROOT/'CHECKS.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(folder=str(OUT),zip=str(archive),files=len(manifest)+1,report_unchanged=True)))


if __name__=='__main__':main()
