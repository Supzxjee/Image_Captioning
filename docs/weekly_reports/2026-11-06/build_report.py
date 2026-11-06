"""Create the user report from audited stored metrics; no models or scorers."""
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];DATA=ROOT/'experiments/csa_paper'
test=json.loads((DATA/'PROMPT_TEST_RESULT.json').read_text(encoding='utf-8'))
val=json.loads((DATA/'PROMPT_SCST_RESULT.json').read_text(encoding='utf-8'))
img=Image.new('RGB',(1800,650),'white');draw=ImageDraw.Draw(img);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',39)
blue='#1B4266'
def box(cx,cy,label,fill='#EFF4F8'):
 rect=(cx-175,cy-65,cx+175,cy+65);draw.rounded_rectangle(rect,radius=12,fill=fill,outline=blue,width=3)
 draw.multiline_text((cx,cy),label,font=font,fill='#172B3A',anchor='mm',align='center',spacing=5)
def arrow(points):
 draw.line(points,fill=blue,width=4);x,y=points[-1];a,b=points[-2];dx,dy=x-a,y-b
 import math
 t=math.atan2(dy,dx);draw.polygon([(x,y),(x-19*math.cos(t-.45),y-19*math.sin(t-.45)),(x-19*math.cos(t+.45),y-19*math.sin(t+.45))],fill=blue)
for x,y,label in [(200,70,'Ảnh và\nbox YOLO'),(650,70,'Bank danh từ\nvà động từ'),(200,245,'Đặc trưng ROI\ntừ CLIP'),(650,245,'Căn chỉnh\nCSA'),(1100,245,'Adapter\nprompt và ROI'),(1550,245,'Decoder\nsinh 5 caption'),(200,470,'UPT\nđối tượng quan hệ'),(650,470,'Prompt qua CLIP\nembedding chữ'),(1550,470,'Reranking\nchọn câu cuối')]:box(x,y,label)
for points in [[(200,135),(200,180)],[(650,135),(650,180)],[(375,245),(475,245)],[(825,245),(925,245)],[(1275,245),(1375,245)],[(375,470),(475,470)],[(825,470),(1100,470),(1100,310)],[(1550,310),(1550,405)]]:arrow(points)
# The UPT branch also receives the image; the line is an input branch, not CLIP ROI -> UPT.

# Clear the ROI-to-UPT link; input branch comes directly from the image.
arrow([(25,70),(10,70),(10,470),(25,470)])
arrow([(1550,535),(1550,565)])
draw.text((1550,600),'Caption cuối cùng',font=font,fill=blue,anchor='mm')
img.save(HERE/'pipeline.png')
doc=Document();sec=doc.sections[0];sec.page_width=Inches(8.5);sec.page_height=Inches(11);sec.top_margin=Inches(.65);sec.bottom_margin=Inches(.65);sec.left_margin=Inches(.7);sec.right_margin=Inches(.7)
normal=doc.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(12);normal.paragraph_format.space_after=Pt(6);normal.paragraph_format.line_spacing=1.12
for style in doc.styles:
 if style.type==1:
  pr=style.element.find(qn('w:pPr'))
  if pr is not None:
   for border in list(pr.findall(qn('w:pBdr'))):pr.remove(border)
for name in ('Title','Heading 1','Heading 2'):
 st=doc.styles[name];st.font.name='Times New Roman';st.font.color.rgb=RGBColor(0,0,0);st.font.size=Pt(18 if name=='Title' else 14 if name=='Heading 1' else 12);st.paragraph_format.space_after=Pt(6)
 st.paragraph_format.space_before=Pt(8)
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT;r=footer.add_run('Báo cáo nghiên cứu ngày 06 tháng 11 năm 2026   |   Trang ');r.font.size=Pt(9)
fld=OxmlElement('w:fldSimple');fld.set(qn('w:instr'),'PAGE');footer._p.append(fld)
def para(text,bold=False):
 p=doc.add_paragraph();r=p.add_run(text);r.bold=bold
 if p._p.getprevious() is not None and p._p.getprevious().tag==qn('w:tbl'):p.paragraph_format.space_before=Pt(6)
 return p
def head(text):doc.add_heading(text,1)
def fmt(n):return f'{n:.2f}'.replace('.',',')
def table(headers,rows,widths):
 t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
 for j,w in enumerate(widths):t.columns[j].width=Inches(w)
 for j,label in enumerate(headers):t.rows[0].cells[j].text=label
 for row in rows:
  cells=t.add_row().cells
  for j,text in enumerate(row):cells[j].text=str(text)
 for i,row in enumerate(t.rows):
  pr=row._tr.get_or_add_trPr();tag=OxmlElement('w:cantSplit');pr.append(tag)
  if i==0:repeat=OxmlElement('w:tblHeader');pr.append(repeat)
  for j,c in enumerate(row.cells):
   c.width=Inches(widths[j]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER;tc=c._tc.get_or_add_tcPr();borders=OxmlElement('w:tcBorders')
   for edge in ('top','left','bottom','right'):
    e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
   tc.append(borders);m=OxmlElement('w:tcMar')
   for edge in ('top','left','bottom','right'):
    e=OxmlElement('w:'+edge);e.set(qn('w:w'),'70');e.set(qn('w:type'),'dxa');m.append(e)
   tc.append(m);sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'1B4266' if i==0 else 'F1F5F8' if i%2==0 else 'FFFFFF');tc.append(sh)
   for p in c.paragraphs:
    p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.space_after=Pt(1);p.paragraph_format.space_before=Pt(1);p.paragraph_format.line_spacing=1.0
    for r in p.runs:r.font.size=Pt(11);r.bold=i==0;r.font.color.rgb=RGBColor.from_string('FFFFFF' if i==0 else '000000')
 return t

doc.add_paragraph('Báo cáo kết quả nghiên cứu',style='Title');para('Cập nhật ngày 06 tháng 11 năm 2026')
para('Nội dung trọng tâm là hoàn thiện mô hình chú thích ảnh có prompt đối tượng–quan hệ, học tiếp bằng SCST kết hợp CE và chọn caption bằng reranking. Bản có prompt đạt CIDEr 115,35 và CHAIRs 3,20% trên TEST5000; đối chứng đạt 114,37 và 4,54%.')
head('1 Pipeline đã hoàn thành')
doc.add_picture(str(HERE/'pipeline.png'),width=Inches(7.05))
para('Hình 1  Luồng sinh caption của mô hình hiện tại',False).alignment=WD_ALIGN_PARAGRAPH.CENTER
para('Ảnh được mã hóa bằng CLIP; box YOLO dùng để lấy đặc trưng từng vùng bằng ROI pooling. CSA căn chỉnh các vùng ảnh với bank danh từ và động từ. Song song, UPT cung cấp thông tin đối tượng–quan hệ để xây dựng prompt riêng cho ảnh; CLIP mã hóa các cụm này thành embedding văn bản.')
para('Adapter kết hợp embedding prompt với vùng ảnh tương ứng trước khi decoder sinh câu. Prompt đối tượng đi với một ROI; prompt quan hệ đi với cặp ROI có thứ tự. Hai hệ box được nối theo nhãn và IoU từ 0,5 trở lên; trường hợp không nối được không được tự gán vào toàn ảnh. Bank từ vẫn được giữ trong pipeline.')
head('2 Cách huấn luyện và chọn câu')
para('Cả CONTROL và PROMPT được chạy 10 epoch XE trên 113.287 ảnh TRAIN của MS COCO Karpathy. CONTROL không dùng adapter prompt; PROMPT có adapter được học ngay từ đầu. Checkpoint được chọn bằng CIDEr trên VALID5000: CONTROL ở epoch 10, PROMPT ở epoch 9.')
para('Sau đó, mỗi bản học tiếp một lượt toàn bộ TRAIN bằng SCST và CE: 14.161 update, batch toàn cục 8 ảnh, Adam với learning rate 0,000005. Loss gồm CE cộng 0,05 lần loss SCST. SCST dùng chênh lệch thưởng CIDEr giữa câu lấy mẫu và câu greedy của chính mô hình; CE giữ mô hình bám caption tham chiếu TRAIN. Lượt này không thêm thưởng đối tượng vào loss.')
para('Khi suy luận, beam search sinh 5 caption; reranking chọn câu bằng 0,3 điểm decoder, 0,4 điểm CLIP và 0,3 điểm nhất quán đối tượng. Các trọng số được giữ cố định trước TEST. Không dùng caption tham chiếu để chọn câu.')

doc.add_page_break();head('3 Kết quả trên tập TEST5000')
para('Hai checkpoint được giữ nguyên khi đánh giá. CONTROL là nhánh đối chứng không dùng prompt riêng ảnh; cả hai nhánh vẫn có CSA và bank từ. Bảng dưới dùng cùng 5.000 ảnh TEST, cùng bộ tham chiếu và cách chấm. CIDEr được trình bày theo thang nhân 100.')
rows=[]
for arm,policy,label in [('CONTROL','normal','CONTROL'),('PROMPT','normal','PROMPT'),('CONTROL','fixed_rerank','CONTROL + rerank'),('PROMPT','fixed_rerank','PROMPT + rerank')]:
 m=test['arms'][arm]['metrics'][policy];rows.append([label]+[fmt(100*m[k]) for k in ('CIDEr','CHAIRs','CHAIRi','Recall')]+[m['counts']['hallucinated_captions']])
table(['Cấu hình','CIDEr ↑','CHAIRs % ↓','CHAIRi % ↓','Recall % ↑','Caption ảo giác ↓'],rows,[2.1,.85,.95,.95,.95,1.25])
para('CHAIRs là tỷ lệ caption bị phát hiện nhắc đến đối tượng không được dữ liệu ảnh hỗ trợ. CHAIRi tính tỷ lệ lượt nhắc đối tượng bị đánh dấu. Recall phản ánh mức bao phủ các lớp đối tượng đúng theo bộ chấm; không phải recall của detector.')
para('Trước reranking, PROMPT tăng 1,80 điểm CIDEr và giảm 93 caption bị đánh dấu so với CONTROL. Sau reranking, PROMPT tăng 0,99 điểm CIDEr và giảm 67 caption bị đánh dấu, từ 227 xuống 160, tương đương giảm 29,5%. Recall tăng từ 44,06% lên 45,18%, cho thấy mô hình không chỉ giảm ảo giác bằng cách né nhắc đối tượng.')
para('BLEU-4 và ROUGE-L sau reranking của CONTROL lần lượt là 36,11 và 56,35; PROMPT là 36,17 và 56,64, cùng thang nhân 100. NLL không được đo trong lượt TEST chỉ sinh câu này.')
head('4 Mức cải thiện qua các bước')
para('Trên cùng VALID5000, quá trình phát triển của nhánh PROMPT có kết quả như sau. Các số VALID được tách riêng với TEST, không dùng chênh lệch giữa hai tập để kết luận mức tăng.')
vr=[]
for phase,policy,label in [('original','decoder_rank0','Sau XE và chọn checkpoint'),('end','decoder_rank0','Sau SCST và CE'),('end','fixed_rerank','Thêm reranking')]:
 m=val['arms']['PROMPT']['metrics'][phase][policy];vr.append([label,fmt(100*m['CIDEr']),fmt(100*m['CHAIRs']),m['counts']['hallucinated_captions']])
table(['Nhánh PROMPT trên VALID5000','CIDEr ↑','CHAIRs % ↓','Caption ảo giác ↓'],vr,[3.4,1.0,1.15,1.5])
para('Từ mốc XE đến pipeline hoàn chỉnh, CIDEr tăng 14,96 điểm và số caption bị đánh dấu giảm 69, từ 220 xuống 151. Mức tăng này gồm cả tác động của SCST và reranking, không phải phần cải thiện chỉ do prompt.')
para('Khoảng tin cậy bootstrap ghép cặp trên TEST cho chênh lệch CIDEr PROMPT trừ CONTROL là [0,51; 3,10] trước reranking và [−0,39; 2,43] sau reranking. Do đó, chưa kết luận chắc chắn về lợi ích CIDEr riêng sau bước chọn câu. Khoảng chênh lệch CHAIRs đều âm trong phép kiểm tra này, phù hợp với lợi ích giảm ảo giác.')

doc.add_page_break();head('5 Kiểm tra vai trò của nội dung prompt')
para('Để kiểm tra mô hình có thực sự dùng phần chữ hay không, em giữ nguyên checkpoint PROMPT, đặc trưng ảnh, adapter, liên kết ROI, hình học và mask. Chỉ thay embedding chữ theo ba chế độ: đúng ảnh, đặt bằng 0, hoặc lấy từ ảnh khác. Phép kiểm tra này dùng caption trước reranking.')
ir=[]
for policy,label in [('normal','Prompt đúng ảnh'),('zero','Embedding chữ bằng 0'),('shuffle','Chữ lấy từ ảnh khác')]:
 m=test['arms']['PROMPT']['metrics'][policy];ir.append([label,fmt(100*m['CIDEr']),fmt(100*m['CHAIRs']),fmt(100*m['Recall']),m['counts']['hallucinated_captions']])
table(['Đầu vào chữ của PROMPT','CIDEr ↑','CHAIRs % ↓','Recall % ↑','Caption ảo giác ↓'],ir,[2.65,.95,1.15,1.15,1.2])
para('Khi bỏ nội dung chữ, CIDEr giảm từ 107,36 xuống 89,37. Khi tráo chữ, CIDEr còn 89,62 và số caption bị đánh dấu tăng từ 188 lên 539. Kết quả cho thấy mô hình phụ thuộc vào nội dung ngữ nghĩa của prompt, và prompt phù hợp với ảnh có ích trong phép kiểm tra suy luận này.')
para('Chữ được tráo giữa các slot cùng loại đối tượng hoặc quan hệ, luôn lấy từ ảnh khác. Danh sách tráo được khóa trước khi chấm. Trong 16.821 slot, 2.105 slot vẫn có cùng chuỗi chữ dù lấy từ ảnh khác; không loại các trường hợp này để làm đẹp kết quả.')
head('6 Kết luận và giới hạn')
para('Kết quả hiện tại ủng hộ việc giữ pipeline PROMPT kết hợp SCST, CE và reranking để trình bày trong bài báo. Điểm mạnh nhất là giảm ảo giác đối tượng trên cả VALID và TEST; các phép can thiệp phần chữ bổ sung bằng chứng rằng prompt thực sự tham gia vào quá trình sinh caption.')
para('Chưa thể quy toàn bộ cải thiện cho riêng nội dung chữ: hai nhánh bắt đầu SCST từ checkpoint có số epoch XE khác nhau, và adapter làm thay đổi số tham số được học. Đặt chữ bằng 0 hoặc tráo chữ còn có thể đưa mô hình vào đầu vào khác phân phối lúc học; đây chưa phải đối chứng huấn luyện khớp hoàn toàn.')
para('Các kết quả mới có một seed; TEST đã được sử dụng ở các lượt phát triển trước. CHAIR đánh giá ảo giác đối tượng, chưa xác nhận mọi thuộc tính hay quan hệ đều đúng. Bản CSA cũ có CIDEr TEST cao hơn là 117,06 nhưng CHAIRs 4,40%; bản PROMPT hiện tại đạt 115,35 và 3,20%. Vì khác lịch sử huấn luyện, chỉ dùng mốc cũ làm tham khảo.')
head('7 Công việc tiếp theo')
para('Em giữ nguyên checkpoint và quy tắc chọn câu hiện tại, hoàn thiện phần mô tả phương pháp và bảng thực nghiệm. Nếu cần tách riêng đóng góp của nội dung chữ mạnh hơn, sẽ bổ sung đối chứng huấn luyện cùng adapter và cùng ngân sách nhưng không có nội dung chữ; chưa triển khai lượt huấn luyện mới.')
p=para('Nguồn số liệu  Báo cáo TEST và kiểm tra 139 mục đã lưu trên GitHub. Trọng số CONTROL ở version 180, PROMPT ở version 60; output TEST tương ứng ở version 184 và 61.');p.runs[0].font.size=Pt(10)
p=para('https://github.com/Supzxjee/Image_Captioning/blob/14500f9/experiments/csa_paper/PROMPT_TEST_RESULT.md');p.runs[0].font.size=Pt(9)
path=HERE/'Bao_cao_nghien_cuu_06_11_2026.docx';doc.save(path);print(path.name)
