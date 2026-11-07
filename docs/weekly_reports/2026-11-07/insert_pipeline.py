"""Replace only the overview graphic and module references; preserve all results."""
from pathlib import Path
import shutil,hashlib
from docx import Document
from docx.oxml.ns import qn
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
path=HERE/'Bao_cao_nghien_cuu_07_11_2026.docx'
image=ROOT/'docs/captioning_pipeline_2026-11-07/modules/00-overall.png'
backup=ROOT.parent/'.review_cache/report_pipeline_2026-11-07'
backup.mkdir(parents=True,exist_ok=True)
original=path.read_bytes()
saved=backup/('before_'+hashlib.sha256(original).hexdigest()[:12]+'.docx')
if not saved.exists():saved.write_bytes(original)
doc=Document(path)
before_tables=[t._tbl.xml for t in doc.tables]
before_sections=[s._sectPr.xml for s in doc.sections]
assert len(doc.inline_shapes)==1
width=doc.inline_shapes[0].width
p=next(p for p in doc.paragraphs if p._p.xpath('.//a:blip'))
for child in list(p._p):
 if child.tag!=qn('w:pPr'):p._p.remove(child)
shape=p.add_run().add_picture(str(image),width=width)
shape._inline.docPr.set('descr','Pipeline tổng thể với module M1 xây dựng bank, M2 căn chỉnh bank với vùng ảnh và M3 căn chỉnh prompt với vùng ảnh.')
for p in doc.paragraphs:
 if p.text=='Hình 1  Luồng sinh caption của mô hình hiện tại':p.text='Hình 1  Pipeline tổng thể với ba module M1 M2 và M3'
 elif p.text.startswith('Ảnh được mã hóa bằng CLIP;'):
  p.text=p.text.replace('CSA căn chỉnh','Module M2 căn chỉnh')
 elif p.text.startswith('Adapter kết hợp embedding prompt'):
  p.text=p.text.replace('Adapter kết hợp','Module M3 kết hợp').replace('Bank từ vẫn được giữ trong pipeline.','Bank từ được xây dựng bởi module M1 và giữ trong pipeline.')
assert [t._tbl.xml for t in doc.tables]==before_tables
assert [s._sectPr.xml for s in doc.sections]==before_sections
doc.save(path)
check=Document(path);assert len(check.inline_shapes)==1 and len(check.tables)==4 and len(check.sections)==3
print('Report updated; result tables and section settings unchanged.')
