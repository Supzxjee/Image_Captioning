"""Rewrite module/overview prose in diagram traversal order; preserve results and figures."""
from pathlib import Path
import hashlib
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.text.paragraph import Paragraph
HERE=Path(__file__).resolve().parent
path=HERE/'Bao_cao_nghien_cuu_07_11_2026.docx'
backup=HERE.parents[2].parent/'.review_cache/report_flow_2026-11-08'
backup.mkdir(parents=True,exist_ok=True)
raw=path.read_bytes();p=backup/('before_'+hashlib.sha256(raw).hexdigest()[:12]+'.docx')
if not p.exists():p.write_bytes(raw)
doc=Document(path);tables=[t._tbl.xml for t in doc.tables]
sections={
 'Hình 1  Module M1':[
 ('Bước 1  Từ caption đến danh sách từ gốc. ', 'Bắt đầu ở ô TRAIN captions bên trái. Caption đi qua POS tagging để xác định danh từ và động từ, rồi qua Lemmatization để đưa chúng về dạng gốc, ví dụ “riding” thành “ride”. Đầu ra là hai danh sách noun lemmas và verb lemmas.'),
 ('Bước 2  Từ danh sách từ đến bank. ', 'Mỗi từ gốc được đặt vào một mẫu câu rồi đưa qua CLIP text encoder để nhận vector 512 chiều. Nhánh danh từ tạo Noun bank; nhánh động từ tạo Verb bank. Hai đầu ra này được giữ cố định, dùng chung cho các ảnh và chuyển sang M2. Việc tạo bank chỉ dùng caption TRAIN; khi xử lý ảnh mới không cần caption tham chiếu của ảnh đó.')
 ],
 'Hình 2  Module M2':[
 ('Bước 1  Chuẩn bị ngữ nghĩa từ hai bank ở hàng trên. ', 'Noun bank đi qua Semantic aggregation để tạo 150 vector đại diện Pn; Verb bank đi qua một khối tổng hợp riêng để tạo Pv. Pn đi xuống tầng Object alignment stage 1, còn Pv đi xuống tầng Relation alignment.'),
 ('Bước 2  Đưa đặc trưng vùng vào tầng đối tượng thứ nhất. ', 'Ô Regions V nhận đầu ra Visual region extraction: ảnh đi qua CLIP ViT-B/16 để lấy patch features, YOLO cung cấp box, ROI pooling tổng hợp đặc trưng theo box và lớp chiếu đưa mỗi vector từ 768 về 512 chiều. Mỗi hàng của V biểu diễn một vùng. Tầng đối tượng thứ nhất nhận V cùng Pn, rồi trả ra đặc trưng V₁ và ngữ nghĩa đã chọn T₁.'),
 ('Bước 3  Đi tiếp qua tầng đối tượng thứ hai. ', 'V₁ và T₁ cùng đi theo mũi tên vào Object alignment stage 2 để tạo Vo. Từ đây, Vo rẽ hai hướng: một nhánh được giữ làm đầu ra đối tượng gửi sang M3, nhánh còn lại đi tiếp vào Relation alignment.'),
 ('Bước 4  Tạo đầu ra quan hệ. ', 'Relation alignment kết hợp Vo với Pv để tạo Vr. Kết thúc M2 có hai đầu ra Vo và Vr, cùng được chuyển sang M3. Bank động từ cung cấp ngữ nghĩa hành động chung; thông tin tương tác riêng của ảnh sẽ được bổ sung ở M3.'),
 ('Đường đi bên trong một tầng ở phần hình dưới. ', 'Regions X là đầu vào của tầng đang xét: X lần lượt là V, V₁ hoặc Vo ở ba tầng phía trên. Nhánh trên đưa X qua Self-attention, Add & Norm và FFN để tạo đặc trưng có bối cảnh H. Nhánh dưới so sánh X với Semantics P bằng Cosine similarity, rồi qua Relevance weighting để tạo Selected T. Hai nhánh gặp nhau tại Cross-attention: H là Query, T là Key và Value. Kết quả đi qua Add & Norm và FFN, sau đó gặp lại H tại Gated fusion. Gate học cách phối hợp hai nhánh để tạo đầu ra X′ của tầng.')
 ],
 'Hình 3  Module M3':[
 ('Bước 1  Từ ảnh tạo hai nhánh thông tin. ', 'Bắt đầu ở Image I, ảnh đi qua UPT detector. Đầu ra rẽ thành Object / relation phrases, chứa các cụm mô tả đối tượng–tương tác, và UPT boxes / classes, chứa vị trí cùng lớp đối tượng. Ví dụ một cụm có thể là “a person holding a cell phone”.'),
 ('Bước 2  Ghép chữ với vùng ảnh. ', 'Nhánh cụm mô tả đi qua CLIP text encoder để tạo các vector to và tr. Nhánh box UPT được ghép với YOLO boxes B tại Class + IoU matching; chỉ nhận cặp cùng lớp và có IoU từ 0,5 trở lên. IoU đo mức chồng lấp giữa hai box. Hai nhánh được gom vào Prompt package, gồm embedding chữ, chỉ số ROI đã nối và tọa độ box, rồi chuyển xuống cả hai nhánh cập nhật.'),
 ('Bước 3  Đi qua Slot MLP ở từng nhánh. ', 'Ở nhánh đối tượng, to được ghép với đặc trưng Vo của một ROI và tọa độ box. Ở nhánh quan hệ, tr được ghép với đặc trưng Vo của hai ROI có thứ tự và tọa độ hai box. Cả hai dùng Vo gốc từ M2. Slot MLP biến mỗi nhóm đầu vào thành một vector tổng hợp: slot Zo cho đối tượng hoặc slot Zr cho quan hệ.'),
 ('Bước 4  Đọc slot rồi cập nhật đặc trưng vùng. ', 'Slot đi sang Masked cross-attention làm Key và Value; Vo ở nhánh trên hoặc Vr ở nhánh dưới đi vào làm Query. Mask chỉ cho mỗi vùng đọc các slot đã nối với vùng đó. Kết quả đi qua Projection rồi tới nút cộng để cộng với đặc trưng ban đầu, tạo Vo′ và Vr′. Hai đầu ra được chuyển sang decoder. Vùng không có liên kết hợp lệ được giữ nguyên; chất lượng liên kết vẫn phụ thuộc UPT và bước ghép box.')
 ],
 'Hình 4  Pipeline tổng thể':[
 ('Chuẩn bị bank ở nhánh phía trên. ', 'Caption TRAIN đi vào M1 để tạo hai bank danh từ và động từ. Hai bank được giữ lại làm đầu vào dùng chung cho M2; bước này không cần lặp lại cho từng ảnh mới.'),
 ('Đi theo hàng chính từ trái sang phải. ', 'Image I đi qua Visual region extraction để tạo V. V cùng hai bank đi vào M2, tạo Vo và Vr. Tiếp đó, M3 nhận Vo, Vr cùng chính ảnh I và box YOLO để tạo prompt, nối ROI và cập nhật thành Vo′, Vr′. Ô Image I + YOLO boxes B trên hình là dữ liệu bổ sung của cùng ảnh, không phải ảnh khác.'),
 ('Từ đặc trưng đã căn chỉnh đến caption. ', 'Vo′ và Vr′ đi vào Transformer decoder. Decoder sinh lần lượt từng từ dựa trên các từ đã có và hai nguồn đặc trưng này. Luồng đi xuống Beam search với K bằng 5 để tạo năm caption ứng viên, rồi đi sang trái qua Caption reranking. Bộ chọn kết hợp điểm decoder, độ phù hợp ảnh–caption từ CLIP và mức nhất quán đối tượng với nhãn YOLO; câu có điểm kết hợp cao nhất trở thành Final caption. Caption tham chiếu không được dùng để chọn câu khi suy luận.')
 ]}
for prefix,items in sections.items():
 cap=next(p for p in doc.paragraphs if p.text.startswith(prefix))
 cursor=cap._p.getnext();delete=[]
 while cursor is not None:
  assert cursor.tag==qn('w:p'),'Unexpected table or section inside explanation.'
  p=Paragraph(cursor,cap._parent)
  if p.style.name.startswith('Heading'):break
  delete.append(cursor);cursor=cursor.getnext()
 assert cursor is not None
 anchor=Paragraph(cursor,cap._parent)
 for e in delete:e.getparent().remove(e)
 for lead,body in items:
  p=anchor.insert_paragraph_before();p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
  p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(6);p.paragraph_format.line_spacing=1.08
  p.paragraph_format.widow_control=True
  p.add_run(lead).bold=True;p.add_run(body)
  for run in p.runs:run.font.name='Times New Roman';run.font.size=Pt(12)
assert [t._tbl.xml for t in doc.tables]==tables
assert len(doc.inline_shapes)==4 and all(s.page_width<s.page_height for s in doc.sections)
doc.save(path)
print('Rewrote explanations following each diagram: M1 2 steps, M2 4 steps plus stage detail, M3 4 steps, overview 3 stages.')
