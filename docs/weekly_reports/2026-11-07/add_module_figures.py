"""Add the three approved module figures before the retained overall pipeline."""
from pathlib import Path
import hashlib
from docx import Document
from docx.shared import Inches,Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
path=HERE/'Bao_cao_nghien_cuu_07_11_2026.docx'
assets=ROOT/'docs/captioning_pipeline_2026-11-07/modules'
backup=ROOT.parent/'.review_cache/report_modules_2026-11-08'
backup.mkdir(parents=True,exist_ok=True)
raw=path.read_bytes();saved=backup/('before_'+hashlib.sha256(raw).hexdigest()[:12]+'.docx')
if not saved.exists():saved.write_bytes(raw)
doc=Document(path);table_xml=[t._tbl.xml for t in doc.tables]
assert len(doc.inline_shapes)==1,'Preserve existing module figures if this step was already applied.'
anchor=next(p for p in doc.paragraphs if p.text=='1 Pipeline đã hoàn thành')
# Existing sections retain their content; numbering accommodates the new module section.
for p in doc.paragraphs:
 if p.style.name=='Heading 1' and p.text[:1].isdigit():
  number,rest=p.text.split(' ',1);p.text=str(int(number)+1)+' '+rest
 if p.text=='Hình 1  Pipeline tổng thể với ba module M1 M2 và M3':p.text='Hình 4  Pipeline tổng thể với ba module M1 M2 và M3'
heading=anchor.insert_paragraph_before('1 Các module của mô hình',style='Heading 1')
width=doc.sections[0].page_width-doc.sections[0].left_margin-doc.sections[0].right_margin
items=[('M1 Xây dựng bank ngữ nghĩa','01-bank','Hình 1  Module M1 xây dựng bank danh từ và động từ từ caption TRAIN.'),('M2 Căn chỉnh bank với vùng ảnh','02-bank-region','Hình 2  Module M2. Phần trên là luồng qua các tầng; phần dưới mở chi tiết một tầng căn chỉnh.'),('M3 Căn chỉnh prompt với vùng ảnh','03-prompt-region','Hình 3  Module M3. Prompt được ghép với ROI rồi dùng để cập nhật riêng hai nhánh đối tượng và quan hệ.')]
for i,(title,name,caption) in enumerate(items):
 p=anchor.insert_paragraph_before(title,style='Heading 2')
 p.paragraph_format.keep_with_next=True
 if i==2:p.paragraph_format.page_break_before=True
 picture=anchor.insert_paragraph_before()
 picture.alignment=WD_ALIGN_PARAGRAPH.CENTER
 picture.paragraph_format.space_before=Pt(0);picture.paragraph_format.space_after=Pt(3)
 picture.paragraph_format.keep_with_next=True
 shape=picture.add_run().add_picture(str(assets/(name+'.png')),width=width)
 shape._inline.docPr.set('descr',title)
 p=anchor.insert_paragraph_before(caption)
 p.alignment=WD_ALIGN_PARAGRAPH.CENTER
 p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(8)
 p.paragraph_format.keep_together=True
 for run in p.runs:run.font.name='Times New Roman';run.font.size=Pt(11)
anchor.paragraph_format.page_break_before=True
assert [t._tbl.xml for t in doc.tables]==table_xml
assert len(doc.inline_shapes)==4
assert all(s.page_width<s.page_height for s in doc.sections)
doc.save(path)
print('Added M1/M2/M3 before overall pipeline; all four tables preserved exactly.')
