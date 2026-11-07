"""Apply the six user annotations to the retained report and M3 image."""
from pathlib import Path
import hashlib
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
path=HERE/'Bao_cao_nghien_cuu_07_11_2026.docx'
backup=ROOT.parent/'.review_cache/report_week13_comments_2026-11-08'
backup.mkdir(parents=True,exist_ok=True)
raw=path.read_bytes();saved=backup/('before_'+hashlib.sha256(raw).hexdigest()[:12]+'.docx')
if not saved.exists():saved.write_bytes(raw)
doc=Document(path);tables=[t._tbl.xml for t in doc.tables]
doc.paragraphs[0].text='Báo cáo nghiên cứu tuần 13'
remove=[p for p in doc.paragraphs if p.text.startswith('Cập nhật ngày ') or p.text.startswith('Nội dung trọng tâm là hoàn thiện mô hình')]
assert len(remove)==2
for p in remove:p._p.getparent().remove(p._p)
for p in doc.paragraphs:
 if p.text=='1 Các module của mô hình':p.text='1 Các khối chức năng của mô hình'
anchor=next(p for p in doc.paragraphs if p.text.startswith('M2 nhận đặc trưng các vùng ảnh V'))
explanations=[
 'V là tập đặc trưng vùng lấy từ ảnh đầu vào. CLIP ViT-B/16 tạo patch features; YOLO cung cấp box; ROI pooling tổng hợp đặc trưng theo từng box. Lớp chiếu tuyến tính đưa mỗi vector từ 768 về 512 chiều để tạo V. Mỗi hàng của V biểu diễn một vùng ảnh; đây là đầu ra của khối Visual region extraction trong pipeline tổng thể.',
 'X trong phần mở chi tiết là ký hiệu chung cho đầu vào của một tầng căn chỉnh. Ở tầng đối tượng thứ nhất, X là V; ở tầng đối tượng thứ hai, X là V₁; ở tầng quan hệ, X là Vo. X không phải một đặc trưng được trích xuất thêm từ ảnh.'
]
for text in explanations:
 p=anchor.insert_paragraph_before(text);p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
 p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(6);p.paragraph_format.line_spacing=1.08
 for run in p.runs:run.font.name='Times New Roman';run.font.size=Pt(12)
# Same-size image replacement retains the approved geometry and paragraph placement.
shape=doc.inline_shapes[2]
rel=shape._inline.xpath('.//a:blip')[0].get(qn('r:embed'))
part=doc.part.related_parts[rel];assert part.content_type=='image/png'
part._blob=(ROOT/'docs/captioning_pipeline_2026-11-07/modules/03-prompt-region.png').read_bytes()
for section in doc.sections:
 for p in section.footer.paragraphs:
  for run in p.runs:run.text=run.text.replace('Báo cáo nghiên cứu ngày 07 tháng 11 năm 2026','Báo cáo nghiên cứu tuần 13')
assert [t._tbl.xml for t in doc.tables]==tables
assert len(doc.inline_shapes)==4
assert all(s.page_width<s.page_height for s in doc.sections)
doc.save(path)
print('Applied all six annotations; data tables and four-figure layout retained.')
