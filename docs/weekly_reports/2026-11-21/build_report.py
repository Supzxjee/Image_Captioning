"""Author the weekly report from reviewed figures and audited saved JSON only."""
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT/'experiments/clip_patch_baseline'
OUT = HERE/'Bao_cao_ket_qua_nghien_cuu_tuan_15.docx'
ASSETS = HERE/'review_assets'


def read(name):
    return json.loads((DATA/name).read_text(encoding='utf8'))


xe = read('CSA_PER_CAPTION_XE10_THREE_SEEDS_TEST_RESULT.json')
rerank = read('CSA_PER_CAPTION_XE10_THREE_SEEDS_RERANK_TEST_RESULT.json')
scst = read('CSA_PER_CAPTION_THREE_SEEDS_SCST1500_TEST_RESULT.json')
kl = read('CSA_SEED43_SCST_KL500_VALID5000_RESULT.json')
assert xe['seeds'] == rerank['seeds'] == scst['seeds'] == [42,43,44]
assert not kl['result']['test_used'] and kl['result']['images'] == 5000
assert kl['source_version'] == 'ducanh3/notebook7bc83c4c76/186'

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.left_margin = sec.right_margin = Inches(.7)
sec.top_margin = sec.bottom_margin = Inches(.62)
sec.header_distance = sec.footer_distance = Inches(.26)
for name in ('Normal','Title','Subtitle','Heading 1','Heading 2','Caption'):
    style = doc.styles[name]
    style.font.name = 'Times New Roman'
    style.font.color.rgb = RGBColor(0,0,0)
    style.element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Times New Roman')
    ppr = style.element.find(qn('w:pPr'))
    if ppr is not None:
        for border in list(ppr.findall(qn('w:pBdr'))):ppr.remove(border)
normal=doc.styles['Normal'];normal.font.size=Pt(12)
normal.paragraph_format.line_spacing=1.08
normal.paragraph_format.space_after=Pt(6)
normal.paragraph_format.widow_control=True
for name,size in [('Title',19),('Heading 1',14),('Heading 2',12),('Caption',10.5)]:
    doc.styles[name].font.size=Pt(size)
    doc.styles[name].font.bold=name!='Caption'
    doc.styles[name].paragraph_format.space_before=Pt(8 if name!='Title' else 0)
    doc.styles[name].paragraph_format.space_after=Pt(7)
    doc.styles[name].paragraph_format.keep_with_next=name!='Caption'

header=sec.header.paragraphs[0]
header.text='CHÚ THÍCH ẢNH CÓ CĂN CHỈNH NGỮ NGHĨA'
header.alignment=WD_ALIGN_PARAGRAPH.RIGHT
for r in header.runs:r.font.name='Times New Roman';r.font.size=Pt(8)
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
footer.add_run('Trang ')
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)
for r in footer.runs:r.font.size=Pt(9)


def para(text, bold=False):
    p=doc.add_paragraph()
    p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    p.add_run(text).bold=bold
    return p


def heading(text):
    return doc.add_paragraph(text,'Heading 1')


def newpage():
    doc.add_page_break()


def figure(filename,width,caption,alt):
    p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after=Pt(3);p.paragraph_format.keep_with_next=True
    pic=p.add_run().add_picture(str(ASSETS/filename),width=Inches(width))
    pic._inline.docPr.set('descr',alt)
    cap=doc.add_paragraph(caption,'Caption');cap.alignment=WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after=Pt(9)


def metric_table(headers,labels,columns,sd=False):
    t=doc.add_table(rows=1, cols=4);t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    widths=[1.12,1.86,2.03,2.09]
    for cell,w in zip(t.rows[0].cells,widths):cell.width=Inches(w)
    for cell,text in zip(t.rows[0].cells,headers):cell.text=text
    repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
    for label,key in labels:
        row=t.add_row();row.cells[0].text=label
        vals=[col[key]['mean'] if sd else col[key] for col in columns]
        low=key in ('CHAIRs','CHAIRi','NLL');best=min(vals) if low else max(vals)
        for j,(cell,value,col) in enumerate(zip(row.cells[1:],vals,columns)):
            text=f'{value:.2f}' if key!='NLL' else f'{value:.4f}'
            if sd:text+=f" ± {col[key]['sample_std']:.2f}"
            cell.text=text.replace('.',',')
            cell.paragraphs[0].runs[0].bold=abs(value-best)<1e-10
    for i,row in enumerate(t.rows):
        trpr=row._tr.get_or_add_trPr()
        no_split=OxmlElement('w:cantSplit');trpr.append(no_split)
        for j,cell in enumerate(row.cells):
            cell.width=Inches(widths[j]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pr=cell._tc.get_or_add_tcPr()
            borders=OxmlElement('w:tcBorders')
            for side in ('top','left','bottom','right','insideH','insideV'):
                edge=OxmlElement('w:'+side);edge.set(qn('w:val'),'single');edge.set(qn('w:sz'),'4');edge.set(qn('w:color'),'D9D9D9');borders.append(edge)
            pr.append(borders)
            margins=OxmlElement('w:tcMar')
            for side,amount in [('top','100'),('bottom','100'),('left','90'),('right','90')]:
                el=OxmlElement('w:'+side);el.set(qn('w:w'),amount);el.set(qn('w:type'),'dxa');margins.append(el)
            pr.append(margins)
            if i==0:
                shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'EDEDED');pr.append(shade)
            for p in cell.paragraphs:
                p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 and i>0 else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.04
                for r in p.runs:
                    r.font.name='Times New Roman';r.font.size=Pt(10.5);r.font.color.rgb=RGBColor(0,0,0)
                    if i==0:r.bold=True
    doc.add_paragraph().paragraph_format.space_after=Pt(0)
    return t


title=doc.add_paragraph('Báo cáo kết quả nghiên cứu tuần 15','Title')
title.alignment=WD_ALIGN_PARAGRAPH.CENTER
p=doc.add_paragraph('Ngày 21 tháng 11 năm 2026');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
para('Trong tuần, em hoàn tất đối chiếu kết quả trên ba lần huấn luyện với seed khác nhau, kiểm tra thử nghiệm học tiếp có bổ sung KL và rà soát sơ đồ của mô hình. Kết quả cho thấy reranking cải thiện đồng thời CIDEr và các chỉ số CHAIR ở cả ba seed. SCST tăng thêm CIDEr nhưng làm CHAIR xấu hơn; thử nghiệm KL hiện chưa khắc phục được sự đánh đổi này.')
heading('1 Tổng quan mô hình')
figure('Tat_ca_6_hinh-Page-7.drawio.png',7.02,'Hình 1. Pipeline chú thích ảnh.','Ảnh và bank ngữ nghĩa đi qua M2; prompt riêng ảnh và ROI đi vào M3; hai bộ nhớ cập nhật đi vào decoder song song, beam search và reranking.')
para('Mô hình kết hợp đặc trưng ảnh với hai nguồn ngữ nghĩa: bank danh từ–động từ xây dựng từ caption TRAIN và prompt đối tượng–quan hệ riêng của từng ảnh. M1 xây dựng bank; M2 căn chỉnh bank với đặc trưng ảnh; M3 bổ sung thông tin prompt đã liên kết với các vùng ảnh. Transformer Decoder sử dụng hai bộ nhớ đã cập nhật để sinh các caption ứng viên, sau đó reranking chọn câu cuối cùng.')

newpage();heading('2 Xây dựng bank ngữ nghĩa trong M1')
figure('Tat_ca_6_hinh-01_M1_semantic_bank.drawio.png',5.45,'Hình 2. Xây dựng bank danh từ và động từ.','Caption TRAIN qua spaCy POS tagging và lemmatization, lọc tần suất, tạo prompt, CLIP text encoder và chuẩn hóa L2 thành hai bank.')
para('Em sử dụng spaCy với mô hình en_core_web_sm để gán từ loại và đưa từ về dạng gốc. Dạng gốc giúp các biến thể như “riding” và “rides” được quy về “ride” trong bank đã hiệu chỉnh. Danh từ và động từ được thống kê trên caption TRAIN, lọc theo ngưỡng xuất hiện tối thiểu 5 lần và loại trùng; caption VALID và TEST không dùng để xây dựng bank.')
para('Mỗi từ được điền vào mẫu văn bản cố định. Ví dụ, danh từ “bicycle” tạo thành “a photo of a bicycle”; động từ “ride” được điền vào mẫu “a photo contains the action of {lemma}”. Đây là mẫu thực tế của cache đã dùng trong thí nghiệm. Phần mã hóa văn bản của CLIP ViT-B/16 chuyển mỗi prompt thành vector 512 chiều, sau đó vector được chuẩn hóa L2.')
para('Kết quả M1 gồm bank danh từ Bₙ có 5.269 vector và bank động từ Bᵥ có 1.594 vector. Các vector bank được giữ cố định khi huấn luyện mô hình chú thích ảnh. Chúng cung cấp vốn ngữ nghĩa chung của tập TRAIN; thông tin này không có nghĩa mọi đối tượng hoặc hành động trong bank đều xuất hiện ở ảnh đang xét.')

newpage();heading('3 Căn chỉnh bank với đặc trưng ảnh trong M2')
figure('Tat_ca_6_hinh-M2.drawio.png',7.05,'Hình 3. Hai tầng đối tượng và một tầng quan hệ trong M2.','Ba tầng cùng dùng cosine reweighting, self-attention, cross-attention, residual FFN và gated fusion; T1 cùng V1 đi sang tầng đối tượng thứ hai.')
para('Ảnh đầu vào đi qua phần mã hóa thị giác đã đóng băng của CLIP ViT-B/16. Mô hình lấy 196 patch token kích thước 768 chiều, bỏ token CLS đại diện toàn ảnh. Phép chiếu tuyến tính đưa chúng về bộ nhớ thị giác V có kích thước 196 × 512. Semantic Aggregation dùng phép chiếu tuyến tính, MLP và softmax theo từ để tổng hợp mỗi bank thành 150 vector đại diện Pₙ hoặc Pᵥ.')
para('Trong tầng đối tượng thứ nhất, cosine reweighting dùng V để tái trọng số Pₙ, tạo T₁. Nhánh thị giác đi qua self-attention và FFN với cộng dư và chuẩn hóa; cross-attention sau đó dùng đặc trưng thị giác làm query, T₁ làm key và value. Gated fusion kết hợp nhánh thị giác với nhánh đã nhận ngữ nghĩa để tạo V₁. Tầng đối tượng thứ hai nhận V₁ và T₁, tái trọng số T₁ thành T₂ rồi tiếp tục căn chỉnh để tạo bộ nhớ đối tượng Vₒ. Tầng quan hệ nhận Vₒ cùng Pᵥ để tạo Vᵣ. Hai bộ nhớ Vₒ và Vᵣ đều giữ 196 × 512 và được chuyển riêng sang M3. Tầng quan hệ ở đây sử dụng ngữ nghĩa động từ; nó không trực tiếp dựng bộ ba quan hệ từ ảnh.')

newpage();heading('4 Căn chỉnh prompt riêng ảnh trong M3')
figure('Tat_ca_6_hinh-Page-8.drawio.png',7.05,'Hình 4. Hai nhánh cập nhật bộ nhớ trong M3.','Cả object pooling và subject–object pooling lấy đặc trưng từ Vo; query và residual nhánh quan hệ lấy từ Vr. Các ROI links điều khiển vùng hỗ trợ và attention mask.')
para('UPT cung cấp dự đoán tương tác cùng các box liên quan. Em ghép box UPT với box YOLO khi cùng lớp đối tượng và có IoU từ 0,5 trở lên. IoU đo mức giao trên hợp của hai box. Phép ghép này giúp nối thông tin đối tượng–quan hệ của UPT với hệ vùng YOLO đã dùng trong cache. Các dự đoán không đủ điều kiện không được coi là liên kết ROI hợp lệ.')
para('Các cụm đối tượng và quan hệ được mã hóa bằng phần văn bản của CLIP ViT-B/16 đã đóng băng, tạo tₒ và tᵣ là các vector ngữ nghĩa 512 chiều, chuẩn hóa L2. Tọa độ box, hình học cặp, liên kết ROI và cờ hợp lệ được lưu riêng; chúng không đi qua CLIP Text Encoder. Vì vậy, embedding của prompt phải được phân biệt với đặc trưng ảnh lấy từ vùng tương ứng.')
para('Ở nhánh đối tượng, mô hình pooling từ Vₒ theo trọng số giao diện tích giữa box YOLO đã ghép và lưới 14 × 14 patch. Vector vùng được nối với tₒ và thông tin box, rồi đi qua Slot MLP để tạo một biểu diễn đối tượng. Slot MLP gồm hai lớp tuyến tính với ReLU ở giữa. Masked cross-attention dùng Vₒ làm query và các slot làm key, value; mask giới hạn slot mà mỗi patch được phép chú ý. Phần cập nhật được cộng dư vào Vₒ để tạo Vₒ′.')
para('Ở nhánh quan hệ, cả vùng chủ thể và vùng đối tượng đều được pooling từ Vₒ gốc. Hai vector này được nối với tᵣ và hình học cặp trước khi tạo slot quan hệ. Query và nhánh cộng dư lại lấy từ Vᵣ, cho đầu ra Vᵣ′. Với patch không có slot hỗ trợ hợp lệ, phần cập nhật bị đặt bằng 0 để giữ bộ nhớ gốc. Hai đầu ra Vₒ′ và Vᵣ′ đi vào hai nhánh cross-attention song song của decoder; chúng không được hợp nhất thành một bộ nhớ duy nhất trước decoder.')

newpage();heading('5 Huấn luyện sinh caption và kết quả TEST')
para('Mô hình có căn chỉnh được huấn luyện bằng cross-entropy, tức học dự đoán từ tiếp theo theo caption tham chiếu TRAIN. Em chọn checkpoint theo CIDEr trên VALID, rồi đánh giá trên TEST. Từ checkpoint của mỗi seed, em cho mô hình học tiếp bằng SCST thuần tại cùng mốc 1.500 update; lượt này không trộn CE và không thêm thưởng đối tượng. SCST sử dụng chênh lệch thưởng CIDEr giữa caption lấy mẫu và caption greedy của chính mô hình.')
para('Transformer Decoder gồm ba tầng với attention đối tượng và quan hệ song song. Beam search sinh 5 caption mỗi ảnh. Reranking kết hợp 0,3 điểm decoder, 0,4 điểm phù hợp ảnh–văn bản từ CLIP và 0,3 điểm nhất quán đối tượng với nhãn YOLO. Quy tắc này được cố định trước các lượt TEST, không dùng caption tham chiếu để chọn câu.')
cols=[xe['statistics'],rerank['statistics']['clip_object_fixed'],scst['statistics']['clip_object_fixed']]
labels=[('BLEU-1','BLEU_1'),('BLEU-4','BLEU_4'),('METEOR','METEOR'),('ROUGE-L','ROUGE_L'),('CIDEr','CIDEr'),('SPICE','SPICE'),('CHAIRs (%)','CHAIRs'),('CHAIRi (%)','CHAIRi')]
metric_table(['Thang đo','Căn chỉnh','Căn chỉnh\n+ rerank','Căn chỉnh\n+ SCST + rerank'],labels,cols,True)
cap=doc.add_paragraph('Bảng 1. Kết quả trên TEST5000.','Caption');cap.alignment=WD_ALIGN_PARAGRAPH.CENTER
para('Số liệu là trung bình ± độ lệch chuẩn mẫu của ba seed 42, 43 và 44. Sáu chỉ số chất lượng caption được trình bày trên thang nhân 100; CHAIRs và CHAIRi là phần trăm và càng thấp càng tốt. Các giá trị tốt nhất trong từng hàng được tô đậm. Độ lệch chuẩn giữa seed không phải khoảng tin cậy.')
para('Reranking tăng trung bình 3,12 điểm CIDEr, giảm CHAIRs 0,54 điểm phần trăm và CHAIRi 0,46 điểm phần trăm; chiều cải thiện này xuất hiện ở cả ba seed. Học tiếp bằng SCST tăng thêm 0,89 điểm CIDEr sau reranking, nhưng CHAIRs tăng 0,51 điểm phần trăm và CHAIRi tăng 0,25 điểm phần trăm so với bản chưa SCST. Cả ba seed đều thể hiện sự đánh đổi này; không phải mọi thang đo đều tăng.')
para('Baseline lịch sử đạt CIDEr 120,51, CHAIRs 5,16% và CHAIRi 3,60%. Tuy nhiên, baseline dùng decoder 4 tầng, dropout 0,1, còn mô hình có căn chỉnh dùng decoder song song 3 tầng, dropout 0,2; baseline mới có một seed. Vì vậy, so sánh này phản ánh hai hệ thống hoàn chỉnh, chưa cho phép quy riêng mức tăng cho prompt hoặc căn chỉnh ngữ nghĩa.')

newpage();heading('6 Thử nghiệm KL và kiểm tra lỗi')
para('Để thử hạn chế sự lệch phân phối khi tối ưu CIDEr, em bổ sung KL so với một bản sao đóng băng của chính mô hình trước SCST. Thử nghiệm dùng seed 43 và VALID5000, so sánh ba bản: trước SCST, SCST thường 500 update và SCST có KL 500 update. Hai nhánh học tiếp khởi đầu từ cùng trọng số, dùng cùng dữ liệu và lịch cập nhật; cả ba dùng cùng reranking.')
raw=kl['result']['arms']
cols=[]
for name in ('xe','scst500','kl500'):
    m=raw[name]['metrics']['clip_object_fixed']
    cols.append({k:v if k=='NLL' else v*100 for k,v in m.items() if isinstance(v,(int,float))})
metric_table(['Thang đo','Trước SCST','SCST500','SCST + KL500'],[('CIDEr','CIDEr'),('CHAIRs (%)','CHAIRs'),('CHAIRi (%)','CHAIRi'),('NLL','NLL')],cols)
cap=doc.add_paragraph('Bảng 2. Pilot trên VALID5000.','Caption');cap.alignment=WD_ALIGN_PARAGRAPH.CENTER
para('KL tăng 1,40 điểm CIDEr so với SCST thường và giữ NLL, tức loss dự đoán token tham chiếu, gần bản trước SCST. Tuy nhiên, so với bản trước SCST, CHAIRs vẫn tăng 0,18 điểm phần trăm và CHAIRi tăng khoảng 0,02 điểm phần trăm. Mức tăng CIDEr 0,67 điểm có khoảng tin cậy bootstrap 95% từ −0,12 đến +1,46, vẫn chứa 0. Các khoảng tin cậy chênh lệch CHAIR cũng chứa 0. Pilot chưa đạt tiêu chí đồng thời tăng CIDEr và không làm CHAIR xấu hơn; KL chưa được đánh giá trên TEST.')
para('Em cũng đối chiếu caption trước và sau SCST 1.500 update trên VALID của seed 43. Số caption bị CHAIR đánh dấu tăng từ 209 lên 273, gồm 103 ca mới và 39 ca hết bị đánh dấu. Trong mẫu cố định 20 ca mới đã xem ảnh, nhận xét trợ lý ghi nhận 7 ca có bằng chứng lỗi thị giác, 3 ca mơ hồ về lớp hoặc nhãn, 8 ca chưa chắc chắn và 2 lỗi phân tích từ của thang đo. Đây là kiểm tra hỗ trợ, không mù và chưa được xem là nhãn người độc lập.')
heading('7 Rà soát hình và hướng tiếp theo')
para('Em đã kiểm tra năm hình dựa trên mã mô hình. Báo cáo dùng sơ đồ tổng quan theo M1–M2–M3 và ba hình module riêng để giữ luồng đọc rõ ràng; sơ đồ mở rộng toàn bộ được giữ làm bản tham khảo vì quá dày khi thu nhỏ. Các đường cộng dư và đầu ra M3 vẫn đầy đủ trên nền trắng. Những hình nền trong suốt được đặt lên nền trắng khi đưa vào Word, không thay đổi cấu trúc của bản vẽ gốc.')
para('Hiện em ưu tiên mô hình căn chỉnh kết hợp reranking khi cần cân bằng CIDEr với hạn chế ảo giác; bản học SCST được báo cáo như cấu hình thiên về CIDEr. Em chưa mở rộng pilot KL hiện tại. Công việc tiếp theo là chuẩn hóa đối chứng có cùng decoder để tách đóng góp của prompt, và kiểm tra thêm các ca lỗi dựa trên ảnh trước khi thay đổi cơ chế. Điểm nhất quán đối tượng chỉ là tín hiệu hỗ trợ từ detector; detector bỏ sót có thể phạt caption đúng, còn CHAIR không xác nhận đầy đủ lỗi thuộc tính, hành động hoặc quan hệ.')

for p in doc.paragraphs:
    for run in p.runs:
        run.font.name='Times New Roman'
        fonts=run._element.get_or_add_rPr().get_or_add_rFonts()
        for theme in ('asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme'):
            if qn('w:'+theme) in fonts.attrib:del fonts.attrib[qn('w:'+theme)]
        if run.font.color.type is not None:run.font.color.rgb=RGBColor(0,0,0)
doc.core_properties.title='Báo cáo kết quả nghiên cứu tuần 15'
doc.core_properties.subject='Chú thích ảnh có căn chỉnh ngữ nghĩa và kiểm soát ảo giác đối tượng'
doc.core_properties.author='Lê Đức Anh'
doc.save(OUT)
assert len(doc.inline_shapes)==4 and len(doc.tables)==2 and all(s.page_width<s.page_height for s in doc.sections)
text='\n'.join(p.text for p in doc.paragraphs)
assert 'CSA' not in text and 'learning rate' not in text
assert 'torch' not in sys.modules
checks=dict(report=str(OUT),input_figures=5,embedded_figures=4,
    chosen_overview='Tat_ca_6_hinh-Page-7.drawio.png',expanded_full='Reviewed, not embedded at report scale',
    original_figures_edited=False,source_metrics=[str(DATA/n) for n in (
        'CSA_PER_CAPTION_XE10_THREE_SEEDS_TEST_RESULT.json','CSA_PER_CAPTION_XE10_THREE_SEEDS_RERANK_TEST_RESULT.json',
        'CSA_PER_CAPTION_THREE_SEEDS_SCST1500_TEST_RESULT.json','CSA_SEED43_SCST_KL500_VALID5000_RESULT.json')],
    tests=dict(metric_tables_from_audited_json=True,TEST_VALID_separate=True,portrait=True,
               black_headings=True,models_executed=False),
    report_sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),render_pending=True)
(HERE/'CHECKS.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(OUT)
