"""Audit and package the six drawing artifacts; no neural or scoring dependencies."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import ast
import hashlib
import html
import json
import re
import shutil
import struct
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
FIGURES=ROOT/'figures'
BUNDLE=Path('C:/Users/leduc/Downloads/Bo_hinh_pipeline_M1_M2_M3')
ARCHIVE=BUNDLE.parent/'Bo_hinh_pipeline_M1_M2_M3.zip'
NAMES=[('01_M1_semantic_bank','M1 — Xây dựng semantic bank'),
       ('02_M2_bank_visual_alignment','M2 — Căn chỉnh bank–thị giác'),
       ('03_M3_prompt_visual_alignment','M3 — Căn chỉnh prompt–thị giác'),
       ('04_baseline','Baseline'),
       ('05_full_pipeline','Pipeline đầy đủ — mở M1, M2, M3'),
       ('06_modular_pipeline','Pipeline tổng quan — dùng tên M1, M2, M3')]
NS='{http://www.w3.org/2000/svg}'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def audit():
    ast.parse((ROOT/'build_suite.py').read_text(encoding='utf-8'))
    result={}
    for stem,title in NAMES:
        svg=ET.parse(FIGURES/(stem+'.svg'))
        root=svg.getroot()
        rects=[]
        for r in svg.findall('.//'+NS+'rect'):
            if 'stroke' in r.attrib and 'stroke-dasharray' not in r.attrib:
                x,y,w,h=[float(r.get(k,0)) for k in ('x','y','width','height')]
                rects.append((x,y,x+w,y+h))
        edges=svg.findall('.//'+NS+'path[@class="edge"]')
        for e in edges:
            pts=[tuple(map(float,p.split(','))) for p in re.findall(r'[ML]([0-9.,]+)',e.get('d'))]
            assert len(pts)>1,(stem,e.get('d'))
            assert all(abs(a[0]-b[0])<1e-8 or abs(a[1]-b[1])<1e-8 for a,b in zip(pts,pts[1:])),(stem,e.get('d'))
            for a,b in zip(pts,pts[1:]):
                for x0,y0,x1,y1 in rects:
                    cross=(x0<a[0]<x1 and max(min(a[1],b[1]),y0)<min(max(a[1],b[1]),y1)) if a[0]==b[0] else (y0<a[1]<y1 and max(min(a[0],b[0]),x0)<min(max(a[0],b[0]),x1))
                    assert not cross or 'marker-end:none' in e.get('style',''),(stem,a,b,(x0,y0,x1,y1))
        labels=[''.join(t.itertext()) for t in svg.findall('.//'+NS+'text')]
        assert not any('CSA' in text for text in labels)
        native=ET.parse(FIGURES/(stem+'.drawio'))
        assert len(native.findall('.//diagram'))==1
        assert len(native.findall('.//mxCell[@vertex="1"]'))>15
        png=FIGURES/(stem+'.png')
        assert png.is_file() and png.stat().st_size>5000
        dims=struct.unpack('>II',png.read_bytes()[16:24])
        result[stem]={'title':title,'canvas':[root.get('width'),root.get('height')],
                      'png_dimensions':dims,'orthogonal_connectors':len(edges),
                      'unrelated_box_crossings':0,'sha256':{ext:sha(FIGURES/(stem+ext)) for ext in ('.png','.svg','.drawio')}}
        text=' '.join(labels)
        if stem.startswith('01_'):
            assert 'spaCy en_core_web_sm' in text and '5269' in text and '1594' in text
            assert 'a photo of a {lemma}' in text
            assert 'a photo contains the action of {lemma}' in text
        if stem.startswith('04_'):
            assert '4 layers' in text and 'Final LayerNorm' in text
            assert sum(t=='LayerNorm' or t=='Final LayerNorm' for t in labels)==2
            assert 'reranking' not in text
        if stem.startswith('05_'):
            assert 'spaCy en_core_web_sm' in text and 'L2' in text
            assert 'Object alignment 2' in text and 'Slot MLP' in text and '3 layers' in text
        if stem.startswith('06_'):
            assert all(token in labels for token in ('M1','M2','M3'))
            assert 'Slot MLP' not in text and 'Gated fusion' not in text
    combined=ET.parse(FIGURES/'ALL_6_FIGURES.drawio')
    assert len(combined.findall('.//diagram'))==6
    report={'figures':result,'combined_drawio_pages':6,'models_executed':False,
            'validation':'XML/AST, geometry, baseline/full identities, six renders visually inspected'}
    (ROOT/'METADATA_CHECKS.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    return report


def prepare():
    report=audit()
    for category in ('PNG','SVG','DRAWIO'):(BUNDLE/category).mkdir(parents=True,exist_ok=True)
    for stem,title in NAMES:
        for category,ext in [('PNG','.png'),('SVG','.svg'),('DRAWIO','.drawio')]:
            target=BUNDLE/category/(stem+ext)
            shutil.copyfile(FIGURES/(stem+ext),target)
            assert sha(target)==report['figures'][stem]['sha256'][ext]
    shutil.copyfile(FIGURES/'ALL_6_FIGURES.drawio',BUNDLE/'Tat_ca_6_hinh.drawio')
    (BUNDLE/'MANIFEST.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    readme='''BỘ 6 HÌNH PIPELINE

PNG/: ảnh để chèn báo cáo.
SVG/: vector để phóng to hoặc chỉnh sửa.
DRAWIO/: file chỉnh sửa riêng từng hình.
Tat_ca_6_hinh.drawio: cả 6 hình trong 6 trang của cùng một file.
XEM_TRUOC.html: xem nhanh và mở từng hình.

01: M1 — Xây dựng semantic bank.
02: M2 — Căn chỉnh bank với thị giác.
03: M3 — Căn chỉnh prompt với thị giác.
04: Baseline CLIP ViT-B/16 + Transformer Decoder.
05: Pipeline đầy đủ, mở cơ chế M1/M2/M3.
06: Pipeline đầy đủ, dùng ô M1/M2/M3 thay phần bên trong ba module.

M3 riêng nhận prompt embeddings/ROI links và memory từ M2; quá trình tạo các đầu vào này nằm trong pipeline đầy đủ.
Baseline có 4 decoder layers và LayerNorm sau projection/cuối decoder; pipeline cuối có 3 layers với hai nhánh cross-attention. Hình ghi đúng hai kiến trúc hiện có.
Các ví dụ chữ trong bank/prompt chỉ minh họa cách biểu diễn, không phải đầu ra mới của mô hình cho ảnh này.
Không chạy mô hình hay chấm điểm, không sửa báo cáo Word trong tác vụ tạo hình này.
'''
    (BUNDLE/'README.txt').write_text(readme,encoding='utf-8')
    cards=[]
    for stem,title in NAMES:
        cards.append(f'<article><h2>{html.escape(title)}</h2><a href="PNG/{stem}.png"><img src="SVG/{stem}.svg" alt="{html.escape(title)}"></a><p><a href="PNG/{stem}.png">PNG</a><a href="SVG/{stem}.svg">SVG</a><a href="DRAWIO/{stem}.drawio">draw.io</a></p></article>')
    page='''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Bộ 6 hình pipeline</title>
<style>*{box-sizing:border-box}body{margin:0;padding:24px;background:#fff;color:#24323d;font-family:Segoe UI,Arial,sans-serif}header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}h1{font-size:25px;font-weight:600;margin:0}header a{font-size:14px}main{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}article{border:1px solid #d5dfe5;border-radius:7px;padding:12px;background:#fff}h2{font-size:16px;font-weight:600;margin:0 0 8px}img{display:block;width:100%;height:285px;object-fit:contain;background:white}p{display:flex;gap:15px;margin:8px 0 0}a{color:#205b7a;text-decoration:none}a:hover{text-decoration:underline}footer{margin-top:14px;font-size:13px;color:#5a6b76}</style>
<header><h1>Bộ hình pipeline</h1><a href="Tat_ca_6_hinh.drawio">Mở file draw.io có 6 trang</a></header><main>'''+''.join(cards)+'''</main><footer>PNG để chèn báo cáo · SVG để phóng to · draw.io để chỉnh sửa. Các ví dụ chữ là minh họa.</footer></html>'''
    (BUNDLE/'XEM_TRUOC.html').write_text(page,encoding='utf-8')
    (ROOT/'README.md').write_text('# Bộ sáu hình\n\n'+readme+'\n\nNguồn và phạm vi tại DESIGN.md; metadata/SHA tại METADATA_CHECKS.json. Artifacts trong figures/.\n',encoding='utf-8')
    print('Audited six SVG/draw.io/PNG sets; copied to',BUNDLE)


def archive():
    assert (BUNDLE/'XEM_TRUOC.html').is_file()
    files=sorted(p for p in BUNDLE.rglob('*') if p.is_file())
    with ZipFile(ARCHIVE,'w',ZIP_DEFLATED,compresslevel=9) as z:
        for p in files:z.write(p,Path(BUNDLE.name)/p.relative_to(BUNDLE))
    with ZipFile(ARCHIVE) as z:
        assert z.testzip() is None
        for p in files:
            member=(Path(BUNDLE.name)/p.relative_to(BUNDLE)).as_posix()
            assert hashlib.sha256(z.read(member)).hexdigest()==sha(p)
        assert len([n for n in z.namelist() if '/PNG/' in n and n.endswith('.png')])==6
    print('Verified ZIP:',ARCHIVE,'files:',len(files),'bytes:',ARCHIVE.stat().st_size)


if __name__=='__main__':
    if '--zip' in sys.argv:archive()
    else:prepare()
