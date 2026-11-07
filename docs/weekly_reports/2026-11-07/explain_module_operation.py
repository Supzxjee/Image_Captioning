"""Add implementation-checked module explanations without changing results."""
from pathlib import Path
import hashlib
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
path=HERE/'Bao_cao_nghien_cuu_07_11_2026.docx'
backup=ROOT.parent/'.review_cache/report_operation_2026-11-08'
backup.mkdir(parents=True,exist_ok=True)
raw=path.read_bytes();saved=backup/('before_'+hashlib.sha256(raw).hexdigest()[:12]+'.docx')
if not saved.exists():saved.write_bytes(raw)
doc=Document(path);tables=[t._tbl.xml for t in doc.tables]
assert not any(p.text.startswith('Trước hết, M1 đọc') for p in doc.paragraphs),'Explanations already inserted.'
texts={
 'Hình 1  Module M1':[
 'Trước hết, M1 đọc các caption của tập TRAIN, dùng spaCy để xác định từ loại rồi giữ lại danh từ và động từ. Các từ được đưa về dạng gốc để tránh lưu nhiều biến thể cùng nghĩa, ví dụ “riding” được chuẩn hóa thành “ride”. Kết quả được tách thành hai danh sách: danh từ biểu diễn đối tượng và động từ biểu diễn hành động.',
 'Mỗi từ được đặt vào một mẫu câu rồi đưa qua CLIP text encoder để nhận vector ngữ nghĩa 512 chiều. Các vector danh từ tạo thành noun bank; các vector động từ tạo thành verb bank. Hai bank được chuẩn bị trước và giữ cố định, dùng chung cho mọi ảnh. M1 không chọn từ riêng cho từng ảnh; phần lựa chọn ngữ nghĩa liên quan được thực hiện ở M2.'
 ],
 'Hình 2  Module M2':[
 'M2 nhận đặc trưng các vùng ảnh V và hai bank từ M1. Khối Semantic aggregation học cách tổng hợp mỗi bank thành 150 vector ngữ nghĩa đại diện, ký hiệu Pn và Pv. Các vector này được so sánh với đặc trưng vùng bằng độ tương đồng cosine. Relevance weighting dùng kết quả so sánh để đặt trọng số cao hơn cho ngữ nghĩa liên quan đến ảnh; Selected T là các vector sau bước đặt trọng số này.',
 'Trên nhánh thị giác, self-attention giúp mỗi vùng ảnh tham khảo các vùng khác để bổ sung bối cảnh. Sau đó, cross-attention cho đặc trưng vùng truy vấn các vector Selected T: vùng ảnh đóng vai trò Query, còn ngữ nghĩa đóng vai trò Key và Value. Nhờ đó, đặc trưng vùng được bổ sung thông tin từ bank. Các khối Add & Norm chuẩn hóa kết quả có cộng tắt; FFN tiếp tục biến đổi đặc trưng. Gated fusion học tỷ lệ kết hợp giữa nhánh thị giác và nhánh đã đọc thông tin ngữ nghĩa.',
 'Nhánh đối tượng thực hiện hai tầng liên tiếp. Tầng đầu trả ra đặc trưng V₁ và ngữ nghĩa đã chọn T₁; tầng thứ hai nhận cả hai để tạo Vo. Nhánh quan hệ nhận Vo cùng các vector từ bank động từ để tạo Vr. Vo và Vr được chuyển sang M3. Trong M2, bank động từ cung cấp ngữ nghĩa hành động chung; thông tin đối tượng–quan hệ riêng của ảnh được bổ sung bằng UPT ở M3.'
 ],
 'Hình 3  Module M3':[
 'M3 dùng UPT để phát hiện các đối tượng và tương tác giữa người với vật trong ảnh, từ đó tạo các cụm mô tả, chẳng hạn “a person holding a cell phone”. CLIP text encoder mã hóa các cụm này thành vector văn bản. Vì box UPT và box YOLO có thể khác nhau, ROI matching ghép chúng theo lớp đối tượng và IoU từ 0,5 trở lên. IoU đo mức chồng lấp giữa hai box. Kết quả cho biết mỗi prompt gắn với ROI nào hoặc cặp ROI nào.',
 'Slot MLP tạo một vector tổng hợp, gọi là slot, từ chữ trong prompt, đặc trưng vùng và tọa độ box. Slot đối tượng dùng một ROI; slot quan hệ dùng hai ROI có thứ tự. Cả hai đều lấy đặc trưng từ Vo gốc do M2 trả ra. Như vậy, mỗi slot vừa mang nội dung cụm mô tả vừa chứa thông tin về vùng ảnh liên quan.',
 'Masked cross-attention cho từng vùng chỉ đọc các slot đã liên kết với vùng đó. Nhánh đối tượng dùng Vo làm Query; nhánh quan hệ dùng Vr làm Query; các slot tương ứng làm Key và Value. Mask chặn các liên kết không phù hợp. Kết quả attention đi qua Projection rồi được cộng vào đặc trưng ban đầu, tạo Vo′ và Vr′ để đưa vào decoder. Vùng không có prompt được ghép hợp lệ sẽ giữ nguyên đặc trưng. Cơ chế này giới hạn nơi prompt có thể tác động, nhưng vẫn phụ thuộc độ chính xác của UPT và bước ghép box.'
 ]}
# Insert each explanation immediately after its figure caption and before the next heading.
for prefix,paragraphs in texts.items():
 caption=next(p for p in doc.paragraphs if p.text.startswith(prefix))
 next_element=caption._p.getnext()
 from docx.text.paragraph import Paragraph
 anchor=Paragraph(next_element,caption._parent)
 for text in paragraphs:
  p=anchor.insert_paragraph_before(text)
  p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
  p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(6)
  p.paragraph_format.line_spacing=1.08
  p.paragraph_format.widow_control=True
  for run in p.runs:run.font.name='Times New Roman';run.font.size=Pt(12)
# Keep each detailed module with its full explanation rather than splitting a module across pages.
next(p for p in doc.paragraphs if p.text=='M2 Căn chỉnh bank với vùng ảnh').paragraph_format.page_break_before=True
assert [t._tbl.xml for t in doc.tables]==tables
assert len(doc.inline_shapes)==4 and all(s.page_width<s.page_height for s in doc.sections)
doc.save(path)
print('Added 8 source-checked explanatory paragraphs for M1/M2/M3; tables unchanged.')
