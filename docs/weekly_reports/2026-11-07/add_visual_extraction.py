"""Insert detail of the existing visual extraction block before M2."""
from pathlib import Path
import hashlib
from copy import deepcopy
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
path=HERE/'Bao_cao_nghien_cuu_07_11_2026.docx'
backup=ROOT.parent/'.review_cache/report_visual_2026-11-08'
backup.mkdir(parents=True,exist_ok=True)
raw=path.read_bytes();saved=backup/('before_'+hashlib.sha256(raw).hexdigest()[:12]+'.docx')
if not saved.exists():saved.write_bytes(raw)
doc=Document(path);tables=[t._tbl.xml for t in doc.tables]
assert len(doc.inline_shapes)==4
anchor=next(p for p in doc.paragraphs if p.text=='M2 Căn chỉnh bank với vùng ảnh')
def rename_caption(p,old,new):
 formatting=deepcopy(p.runs[0]._r.rPr) if p.runs and p.runs[0]._r.rPr is not None else None
 p.text=p.text.replace(old,new,1)
 if formatting is not None:p.runs[0]._r.insert(0,formatting)
for p in doc.paragraphs:
 if p.text.startswith('Hình 2  Module M2'):rename_caption(p,'Hình 2','Hình 3')
 elif p.text.startswith('Hình 3  Module M3'):rename_caption(p,'Hình 3','Hình 4')
 elif p.text.startswith('Hình 4  Pipeline tổng thể'):rename_caption(p,'Hình 4','Hình 5')
heading=anchor.insert_paragraph_before('Trích xuất đặc trưng vùng ảnh',style='Heading 2')
heading.paragraph_format.keep_with_next=True
picture=anchor.insert_paragraph_before();picture.alignment=WD_ALIGN_PARAGRAPH.CENTER
picture.paragraph_format.space_before=Pt(0);picture.paragraph_format.space_after=Pt(3);picture.paragraph_format.keep_with_next=True
width=doc.sections[0].page_width-doc.sections[0].left_margin-doc.sections[0].right_margin
shape=picture.add_run().add_picture(str(ROOT/'docs/captioning_pipeline_2026-11-07/modules/04-visual-extraction.png'),width=width)
shape._inline.docPr.set('descr','Ảnh qua CLIP và YOLO; ROI pooling và lớp chiếu tạo V cho M2; box và nhãn đi sang M3.')
p=anchor.insert_paragraph_before('Hình 2  Khối trích xuất đặc trưng vùng tạo V cho M2 và cung cấp box cùng nhãn đối tượng cho M3.')
p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(8)
for run in p.runs:run.font.name='Times New Roman';run.font.size=Pt(11)
for text in [
 'Ảnh I đi song song vào bộ mã hóa thị giác OpenAI CLIP ViT-B/16 và bộ phát hiện YOLO. CLIP tạo lưới 14 × 14 patch, mỗi patch có 768 chiều; YOLO cung cấp box và nhãn đối tượng. ROI pooling dùng tọa độ các box để tổng hợp đặc trưng patch thành một vector 768 chiều cho mỗi vùng.',
 'Các vector vùng tiếp tục qua lớp chiếu tuyến tính được học để giảm từ 768 xuống 512 chiều, tạo tập đặc trưng V. N trên hình là số vùng được biểu diễn. V đi sang M2 để căn chỉnh với bank; box B và nhãn YOLO đồng thời đi sang M3 để nối với các đối tượng do UPT phát hiện.'
]:
 p=anchor.insert_paragraph_before(text);p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
 p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(6);p.paragraph_format.line_spacing=1.08
 p.paragraph_format.widow_control=True
 for run in p.runs:run.font.name='Times New Roman';run.font.size=Pt(12)
assert [t._tbl.xml for t in doc.tables]==tables
assert len(doc.inline_shapes)==5 and all(s.page_width<s.page_height for s in doc.sections)
doc.save(path)
print('Visual extraction figure and connected prose added before M2; results untouched.')
