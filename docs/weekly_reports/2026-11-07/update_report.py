"""Update the retained report using audited JSON only; no model/scorer work."""
import json
from pathlib import Path
from docx import Document
from docx.enum.section import WD_SECTION_START, WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DATA=ROOT/'experiments/csa_paper'
read=lambda n:json.loads((DATA/n).read_text(encoding='utf-8'))
base=read('PLAIN_CSA10_EXTRA_RESULT.json')
extra=read('PROMPT_TEST_EXTRA_RESULT.json')
original=read('PROMPT_TEST_RESULT.json')
valid=read('PROMPT_SCST_RESULT.json')
doc=Document(HERE.parent/'2026-11-06/Bao_cao_nghien_cuu_06_11_2026.docx')

# Preserve the existing overview, diagram, typography and training description.
cut=next(p for p in doc.paragraphs if p.text=='3 Kết quả trên tập TEST5000')
body=doc.element.body;start=body.index(cut._p)
if start and 'w:type="page"' in body[start-1].xml:start-=1
for e in list(body)[start:]:
    if e.tag!=qn('w:sectPr'):body.remove(e)
doc.paragraphs[0].text='Báo cáo nghiên cứu chú thích ảnh'
for p in doc.paragraphs:
    if p.text=='Cập nhật ngày 06 tháng 11 năm 2026':p.text='Cập nhật ngày 07 tháng 11 năm 2026'
for r in doc.sections[0].footer.paragraphs[0].runs:
    r.text=r.text.replace('06 tháng 11','07 tháng 11')


def para(text,size=None):
    p=doc.add_paragraph(text)
    if p._p.getprevious() is not None and p._p.getprevious().tag==qn('w:tbl'):p.paragraph_format.space_before=Pt(6)
    if size:
        for r in p.runs:r.font.size=Pt(size)
    return p


def heading(text,level=1):doc.add_heading(text,level)


def section(landscape=False):
    s=doc.add_section(WD_SECTION_START.NEW_PAGE)
    s.orientation=WD_ORIENT.LANDSCAPE if landscape else WD_ORIENT.PORTRAIT
    s.page_width=Inches(11 if landscape else 8.5)
    s.page_height=Inches(8.5 if landscape else 11)
    s.top_margin=s.bottom_margin=Inches(.65)
    s.left_margin=s.right_margin=Inches(.7)
    return s


def table(headers,rows,widths):
    t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    for j,w in enumerate(widths):t.columns[j].width=Inches(w)
    for j,x in enumerate(headers):t.rows[0].cells[j].text=x
    for values in rows:
        cells=t.add_row().cells
        for j,x in enumerate(values):cells[j].text=str(x)
    for i,row in enumerate(t.rows):
        tr=row._tr.get_or_add_trPr();tr.append(OxmlElement('w:cantSplit'))
        if i==0:tr.append(OxmlElement('w:tblHeader'))
        for j,c in enumerate(row.cells):
            c.width=Inches(widths[j]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pr=c._tc.get_or_add_tcPr();b=OxmlElement('w:tcBorders')
            for edge in ('top','left','bottom','right'):
                e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');b.append(e)
            pr.append(b);m=OxmlElement('w:tcMar')
            for edge in ('top','left','bottom','right'):
                e=OxmlElement('w:'+edge);e.set(qn('w:w'),'70');e.set(qn('w:type'),'dxa');m.append(e)
            pr.append(m);shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'1B4266' if i==0 else 'F1F5F8' if i%2==0 else 'FFFFFF');pr.append(shade)
            for p in c.paragraphs:
                p.paragraph_format.space_before=p.paragraph_format.space_after=Pt(1)
                p.paragraph_format.line_spacing=1
                for r in p.runs:
                    r.font.name='Times New Roman';r.font.size=Pt(11);r.bold=i==0
                    r.font.color.rgb=RGBColor.from_string('FFFFFF' if i==0 else '000000')
    return t


fmt=lambda n:f'{n:.2f}'.replace('.',',')
keys=('Bleu_1','Bleu_4','METEOR','ROUGE_L','CIDEr','SPICE','CHAIRs','CHAIRi')
headers=['Cấu hình','BLEU-1 ↑','BLEU-4 ↑','METEOR ↑','ROUGE-L ↑','CIDEr ↑','SPICE ↑','CHAIRs % ↓','CHAIRi % ↓']

section(True)
heading('3 Kết quả đầy đủ trên tập TEST5000')
para('Các bảng dùng cùng 5.000 ảnh MS COCO Karpathy TEST và cùng bộ tham chiếu. Tất cả điểm chất lượng được nhân 100; CHAIRs và CHAIRi là tỷ lệ phần trăm. Dấu ↑ là càng cao càng tốt, dấu ↓ là càng thấp càng tốt.')
heading('Đối chứng không CSA và có CSA sau 10 epoch XE',2)
para('Baseline không CSA dùng ba lớp Transformer thị giác và decoder chung, không dùng bank ngữ nghĩa. Bản CSA thay encoder bằng các khối căn chỉnh với bank danh từ và động từ. Hai bản cùng ROI, 10 epoch XE và cách giải mã, nhưng khác số tham số.')
table(headers,[[label]+[fmt(base['display_x100'][policy][k]) for k in keys]
              for policy,label in [('NoCSA10','Không CSA 10 epoch'),('CSA10','CSA 10 epoch')]], [2.2]+[.92]*8)
para('CSA tăng 45,00 điểm CIDEr và tăng cả BLEU, METEOR, ROUGE-L, SPICE, nhưng CHAIRs tăng từ 3,92% lên 6,18%. NoCSA có 1.853 câu không quá hai từ, còn CSA có 388 câu. Vì vậy, CHAIRs thấp hơn của NoCSA phải được xem cùng việc mô tả còn ngắn và thiếu thông tin.')
heading('Đối chứng prompt sau khi học tiếp bằng SCST và CE',2)
table(headers,[[label]+[fmt(extra['arms'][arm]['display_x100'][policy][k]) for k in keys]
              for arm,policy,label in [('CONTROL','normal','CONTROL'),('PROMPT','normal','PROMPT'),('CONTROL','fixed_rerank','CONTROL và rerank'),('PROMPT','fixed_rerank','PROMPT và rerank')]], [2.2]+[.92]*8)
para('Trước reranking, PROMPT tăng 1,80 điểm CIDEr và giảm 93 caption bị CHAIR đánh dấu. Sau reranking, PROMPT tăng 0,99 điểm CIDEr và giảm 227 xuống 160 caption bị đánh dấu, tương đương 29,5%. Cả sáu chỉ số chất lượng đều nhỉnh hơn CONTROL, nhưng mức tăng nhỏ; lợi ích chính là giảm ảo giác đối tượng.')
para('CONTROL và PROMPT đều có CSA và bank từ; CONTROL không dùng adapter prompt. Các dòng trước reranking lấy câu hạng đầu của beam search, không phải greedy decoding. CHAIRs đo tỷ lệ câu bị đánh dấu, còn CHAIRi đo tỷ lệ lượt nhắc đối tượng bị đánh dấu. Các điểm CHAIR được giữ từ kết quả đã xác minh đúng caption; BLEU-1, METEOR và SPICE được chấm bổ sung trên nguyên các câu đó.',11)

section()
heading('4 Mức cải thiện qua các bước trên VALID')
para('Giữ riêng VALID5000 với TEST5000. Trên VALID, nhánh PROMPT có diễn biến sau; bảng này chỉ trình bày những chỉ số đã được chấm ở từng mốc.')
vr=[]
for phase,policy,label in [('original','decoder_rank0','Sau XE và chọn checkpoint'),('end','decoder_rank0','Sau SCST và CE'),('end','fixed_rerank','Thêm reranking')]:
    m=valid['arms']['PROMPT']['metrics'][phase][policy]
    vr.append([label,fmt(100*m['CIDEr']),fmt(100*m['CHAIRs']),m['counts']['hallucinated_captions']])
table(['Nhánh PROMPT trên VALID5000','CIDEr ↑','CHAIRs % ↓','Caption bị đánh dấu ↓'],vr,[3.3,1,1.1,1.7])
para('Từ mốc XE đến pipeline hoàn chỉnh, CIDEr tăng 14,96 điểm và số caption bị đánh dấu giảm 220 xuống 151. Mức tăng này gồm tác động của cả SCST và reranking, không phải cải thiện riêng do prompt.')
heading('5 Kiểm tra mô hình có sử dụng nội dung prompt')
para('Giữ nguyên checkpoint PROMPT, đặc trưng ảnh, adapter, ROI và mask; chỉ thay embedding chữ. Kết quả dưới đây dùng caption TEST trước reranking.')
probe=[]
for policy,label in [('normal','Prompt đúng ảnh'),('zero','Embedding chữ bằng 0'),('shuffle','Chữ lấy từ ảnh khác')]:
    m=original['arms']['PROMPT']['metrics'][policy]
    probe.append([label,fmt(100*m['CIDEr']),fmt(100*m['CHAIRs']),m['counts']['hallucinated_captions']])
table(['Đầu vào chữ','CIDEr ↑','CHAIRs % ↓','Caption bị đánh dấu ↓'],probe,[3.3,1,1.1,1.7])
para('Bỏ hoặc tráo chữ làm CIDEr giảm; tráo chữ làm số caption bị đánh dấu tăng từ 188 lên 539. Điều này cho thấy mô hình sử dụng nội dung prompt. Tuy nhiên, đầu vào can thiệp khác phân phối lúc học nên chưa thay thế đối chứng huấn luyện khớp hoàn toàn.')
heading('6 Kết luận và phần còn thiếu')
para('Pipeline hiện tại đã có căn chỉnh CSA, prompt đối tượng–quan hệ gắn với ROI, SCST kết hợp CE và reranking. Kết quả ủng hộ việc giữ bản PROMPT để viết bài: chất lượng caption được duy trì và nhỉnh nhẹ so với CONTROL, đồng thời giảm ảo giác đối tượng.')
para('Các kết quả chỉ có một seed; hai nhánh bắt đầu SCST từ checkpoint XE khác epoch và khác số tham số được học. Khoảng tin cậy 95% của chênh lệch CIDEr sau reranking là [−0,39; 2,43], còn chứa 0. TEST đã được sử dụng ở các vòng phát triển trước. CHAIR không xác nhận mọi thuộc tính hoặc quan hệ đều đúng.')
para('Mốc CSA15 cùng SCST và reranking trước đây đạt CIDEr 117,06, CHAIRs 4,40%, nhưng chưa bổ sung BLEU-1, METEOR và SPICE. Đây là mốc tham khảo có lịch huấn luyện khác; không gọi PROMPT hiện tại là bản có CIDEr cao nhất. Tiếp theo là hoàn thiện phần mô tả phương pháp và bảng bài báo; chưa mở thêm huấn luyện.')
para('Nguồn đã kiểm tra: NoCSA10/CSA10 chấm đủ 8 chỉ số ở version 186; CONTROL ở 185, PROMPT ở 62. Caption CONTROL/PROMPT từ 184/61, trọng số từ 180/60. Kết quả được lưu tại GitHub và Sheet Kết quả mới.',9)
para('https://github.com/Supzxjee/Image_Captioning/tree/main/experiments/csa_paper',9)

path=HERE/'Bao_cao_nghien_cuu_07_11_2026.docx'
doc.save(path)
assert len(doc.tables)==4 and len(doc.sections)==3
assert len(doc.tables[0].rows)==3 and len(doc.tables[1].rows)==5
assert all(len(t.columns)==9 for t in doc.tables[:2])
print(path.name)
