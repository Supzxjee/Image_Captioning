"""Align experiment definitions in section3 with the two comparisons in section4."""
from pathlib import Path
import hashlib
from docx import Document
from docx.shared import Pt
HERE=Path(__file__).resolve().parent
path=HERE/'Bao_cao_nghien_cuu_07_11_2026.docx'
backup=HERE.parents[2].parent/'.review_cache/report_groups_2026-11-08'
backup.mkdir(parents=True,exist_ok=True)
raw=path.read_bytes();saved=backup/('before_'+hashlib.sha256(raw).hexdigest()[:12]+'.docx')
if not saved.exists():saved.write_bytes(raw)
doc=Document(path);tables=[t._tbl.xml for t in doc.tables]
heading=next(p for p in doc.paragraphs if p.text=='3 Cách huấn luyện và chọn câu')
heading.text='3 Thiết kế thực nghiệm và huấn luyện'
first=next(p for p in doc.paragraphs if p.text.startswith('Cả CONTROL và PROMPT được chạy'))
scst=next(p for p in doc.paragraphs if p.text.startswith('Sau đó, mỗi bản học tiếp'))
rerank=next(p for p in doc.paragraphs if p.text.startswith('Khi suy luận, beam search sinh 5 caption;'))
first.text='Em thực hiện hai nhóm thực nghiệm. Nhóm thứ nhất so sánh cấu hình không dùng và có dùng căn chỉnh bank, cùng được huấn luyện 10 epoch XE, nhằm đánh giá vai trò của căn chỉnh bank–vùng ảnh. Nhóm thứ hai đánh giá việc bổ sung prompt đối tượng–quan hệ riêng của ảnh.'
definition=scst.insert_paragraph_before('Trong nhóm thứ hai, em chạy trên hai cấu hình gồm CONTROL và PROMPT. CONTROL sử dụng đặc trưng vùng ảnh, bank danh từ–động từ, khối căn chỉnh M2 và Transformer Decoder. PROMPT giữ các thành phần này và bổ sung M3 để kết hợp prompt riêng ảnh với ROI trước khi sinh caption.')
xe=scst.insert_paragraph_before('Cả CONTROL và PROMPT được huấn luyện 10 epoch XE trên 113.287 ảnh TRAIN của MS COCO Karpathy; khối prompt của PROMPT được học ngay từ đầu. Em chọn checkpoint theo CIDEr trên VALID5000: CONTROL ở epoch 10 và PROMPT ở epoch 9.')
scst.text='Từ các checkpoint đã chọn của nhóm thứ hai, em cho mỗi cấu hình học tiếp một lượt toàn bộ TRAIN bằng SCST kết hợp CE, gồm 14.161 update với batch toàn cục 8 ảnh. Adam sử dụng learning rate 0,000005. Loss gồm CE cộng 0,05 lần loss SCST. SCST dùng chênh lệch thưởng CIDEr giữa câu lấy mẫu và câu greedy; CE giữ mô hình bám caption tham chiếu TRAIN. Lượt này không thêm thưởng đối tượng vào loss.'
rerank.text='Khi suy luận, cả CONTROL và PROMPT dùng beam search để sinh 5 caption ứng viên. Em chọn câu cuối bằng reranking với 0,3 điểm decoder, 0,4 điểm CLIP và 0,3 điểm nhất quán đối tượng. Các trọng số được cố định trước TEST; caption tham chiếu không được dùng để chọn câu.'
for p in [first,definition,xe,scst,rerank]:
 p.paragraph_format.line_spacing=1
 p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(4)
 for run in p.runs:run.font.name='Times New Roman';run.font.size=Pt(12)
for p in doc.paragraphs:
 if p.text=='Đối chứng căn chỉnh bank sau 10 epoch XE':p.text='Nhóm 1 Đánh giá căn chỉnh bank sau 10 epoch XE'
 elif p.text=='Đối chứng prompt sau khi học tiếp bằng SCST và CE':p.text='Nhóm 2 Đánh giá prompt sau SCST và CE'
 elif p.text.startswith('Bản đối chứng dùng ba lớp Transformer thị giác'):
  p.text='Ở nhóm 1, bản không căn chỉnh bank dùng ba lớp Transformer thị giác và decoder; bản có căn chỉnh bank dùng M2. Hai bản cùng ROI, 10 epoch XE và cách giải mã, nhưng khác số tham số. Đây là hai lượt riêng, không phải checkpoint CONTROL/PROMPT của nhóm 2.'
  for run in p.runs:run.font.name='Times New Roman';run.font.size=Pt(12)
for p in doc.paragraphs:
 if p.text.startswith(('Các bảng dùng cùng 5.000 ảnh','Ở nhóm 1, bản không căn chỉnh bank','Thêm căn chỉnh bank tăng','Trước reranking, PROMPT','CONTROL và PROMPT đều có căn chỉnh bank')):
  p.paragraph_format.line_spacing=1
  p.paragraph_format.space_after=Pt(4)
assert [t._tbl.xml for t in doc.tables]==tables
assert len(doc.inline_shapes)==5 and all(s.page_width<s.page_height for s in doc.sections)
doc.save(path)
print('Section3 introduces two groups and CONTROL/PROMPT definitions; section4 group headings match; numeric tables untouched.')
