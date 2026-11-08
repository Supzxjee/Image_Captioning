"""Extend the user's edited week13 report; preserve existing tables and figures."""
from pathlib import Path
import hashlib
import json
import shutil
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DATA=ROOT/'experiments/csa_paper'
SOURCE=Path('C:/Users/leduc/Downloads/Bao_cao_nghien_cuu_tuan_13_hai_nhom_thuc_nghiem.docx')
OUTPUT=HERE/'Bao_cao_nghien_cuu_tuan_13_bo_sung_SPICE.docx'


def load(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    doc=Document(SOURCE)
    old_tables=[t._tbl.xml for t in doc.tables]
    backup=ROOT.parent/'.review_cache/report_spice_2026-11-08'
    backup.mkdir(parents=True,exist_ok=True)
    shutil.copy2(SOURCE,backup/('user_source_'+hashlib.sha256(SOURCE.read_bytes()).hexdigest()[:12]+'.docx'))
    anchor=next(p for p in doc.paragraphs if p.text=='6 Kết luận')
    anchor.text='8 Kết luận và hướng tiếp theo'
    anchor.paragraph_format.page_break_before=False
    for p in doc.paragraphs:
        if p.text.startswith('Cả CONTROL và PROMPT được huấn luyện 10 epoch trên'):
            p.text='Cả CONTROL và PROMPT được huấn luyện 10 epoch XE trên 113.287 ảnh TRAIN của MS COCO Karpathy. Em chọn checkpoint theo CIDEr trên VALID5000: CONTROL ở epoch 10 và PROMPT ở epoch 9.'
        elif p.text.startswith('Từ các checkpoint đã chọn của nhóm thứ hai'):
            p.text='Từ các checkpoint đã chọn, mỗi cấu hình học tiếp một lượt toàn bộ TRAIN bằng SCST kết hợp CE: 14.161 update, batch toàn cục 8 ảnh, Adam với learning rate 0,000005. Loss bằng CE cộng 0,05 lần SCST; phần thưởng SCST là CIDEr, không thêm thưởng đối tượng.'
        elif p.text.startswith('Trong tuần tới, em sẽ tập trung kiểm tra và cải thiện'):
            p.text='Tiếp theo, em hoàn thiện mô tả phương pháp và phân tích lỗi cho bài báo, đồng thời kiểm tra lại các ví dụ với người hướng dẫn. Chưa mở thêm huấn luyện hoặc thay backbone trước khi chốt nội dung thực nghiệm hiện có.'
    def para(text,style='Normal',page=False,size=12):
        p=anchor.insert_paragraph_before(text,style)
        p.paragraph_format.page_break_before=page
        p.paragraph_format.space_after=Pt(5)
        p.paragraph_format.line_spacing=1.05
        if style=='Normal':
            for r in p.runs:r.font.name='Times New Roman';r.font.size=Pt(size)
        return p
    def table(headers,values,widths):
        t=doc.add_table(rows=1,cols=len(headers));t.autofit=False
        for i,h in enumerate(headers):t.rows[0].cells[i].text=h
        for row in values:
            for c,v in zip(t.add_row().cells,row):c.text=str(v)
        for row in t.rows:
            for i,c in enumerate(row.cells):
                c.width=Cm(widths[i]);c.vertical_alignment=1
                pr=c._tc.get_or_add_tcPr();margin=OxmlElement('w:tcMar')
                for edge in ('top','bottom','left','right'):
                    e=OxmlElement('w:'+edge);e.set(qn('w:w'),'55');e.set(qn('w:type'),'dxa');margin.append(e)
                pr.append(margin)
                for p in c.paragraphs:
                    p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1
                    for run in p.runs:run.font.name='Times New Roman';run.font.size=Pt(11)
        pr=t._tbl.tblPr;b=OxmlElement('w:tblBorders')
        for edge in ('top','left','bottom','right','insideH','insideV'):
            e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');b.append(e)
        pr.append(b)
        for c in t.rows[0].cells:
            shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'1F4E78');c._tc.get_or_add_tcPr().append(shade)
            for run in c.paragraphs[0].runs:run.bold=True;run.font.color.rgb=RGBColor(255,255,255)
        for n,row in enumerate(t.rows[1:]):
            if n%2:
                for c in row.cells:
                    e=OxmlElement('w:shd');e.set(qn('w:fill'),'F1F5F8');c._tc.get_or_add_tcPr().append(e)
        anchor._p.addprevious(t._tbl)
        return t
    para('6 Phân tích ngữ nghĩa và chi phí','Heading 1',True)
    para('SPICE được tách thành đối tượng, quan hệ và thuộc tính bằng cách so scene graph của caption sinh ra với caption tham chiếu. Bảng dưới dùng đúng caption TEST5000 đã chấm trước đây; không huấn luyện, sinh câu hoặc chọn câu lại. Điểm được nhân 100; “trước rerank” là câu hạng đầu beam search, không phải greedy.')
    result=load(DATA/'SPICE_SUBSETS_RESULT.json')
    values=[]
    for label,key in [('SPICE tổng','All'),('Đối tượng','Objects'),('Quan hệ','Relations'),('Thuộc tính','Attributes')]:
        row=[label]
        for policy,arm in [('normal','CONTROL'),('normal','PROMPT'),('fixed_rerank','CONTROL'),('fixed_rerank','PROMPT')]:
            row.append(f"{result['arms'][arm]['subsets'][policy][key]['f_x100']:.2f}".replace('.',','))
        values.append(row)
    t=table(['Thành phần','CONTROL\ntrước rerank','PROMPT\ntrước rerank','CONTROL\nsau rerank','PROMPT\nsau rerank'],values,[3.1,3.2,3.2,3.2,3.2])
    for row in t.rows[1:]:
        for a,b in [(1,2),(3,4)]:
            numbers=[float(row.cells[i].text.replace(',','.')) for i in (a,b)];win=max(numbers)
            for i,value in zip((a,b),numbers):
                if value==win:
                    for run in row.cells[i].paragraphs[0].runs:run.bold=True
    para('Mỗi cấu hình có 5.000 ảnh hợp lệ cho điểm tổng và đối tượng, 4.998 ảnh cho quan hệ, 4.989 ảnh cho thuộc tính. Hai cấu hình dùng cùng tập ảnh hợp lệ trong từng nhóm; trường hợp không xác định bị loại khỏi trung bình, không thay bằng 0.')
    para('Sau reranking, PROMPT tăng 0,31 điểm đối tượng và 0,37 điểm quan hệ, nhưng giảm 0,05 điểm thuộc tính. Chênh lệch quan hệ có khoảng tin cậy bootstrap 95% [0,09; 0,64]. Đây là phân tích trên caption cố định, chưa hiệu chỉnh nhiều phép so sánh và không phản ánh biến động giữa các seed huấn luyện. SPICE không kiểm tra trực tiếp việc định vị từ vào đúng box.')
    para('Số tham số và thời gian xử lý','Heading 2')
    costs=load(DATA/'paper_analysis/costs.json')
    table(['Chỉ tiêu','CONTROL','PROMPT'],[
        ['Tham số caption được học','49.239.288','53.708.024'],
        ['Worker huấn luyện 10 epoch XE','5,91 giờ','5,26 giờ'],
        ['Worker học tiếp SCST và CE','1,61 giờ','1,80 giờ']], [7.3,4.3,4.3])
    para('PROMPT thêm 4.468.736 tham số được học, khoảng 9,08%; chưa tính các mô hình đóng băng. Trên TEST5000 với hai T4, pha beam search có thời gian rank chậm nhất 216,30 giây ở CONTROL và 205,66 giây ở PROMPT; pha reranking tương ứng 98,95 và 88,50 giây. Đây là log các phiên riêng, chưa gồm trích xuất ảnh/prompt, tải mô hình và đồng bộ; không dùng để kết luận PROMPT nhanh hơn hoặc làm độ trễ toàn hệ thống.',size=11)
    para('Giới hạn: một seed; checkpoint khởi đầu SCST khác epoch (CONTROL 10, PROMPT 9); số tham số khác; TEST đã được xem trong các vòng phát triển. Kết quả đánh giá toàn bộ module prompt, chưa tách riêng ảnh hưởng của nội dung chữ khỏi dung lượng adapter.',size=11)
    para('7 Ví dụ caption và giới hạn của CHAIR','Heading 1',True)
    para('Bốn ví dụ được chọn sau khi xem 24 ảnh trong danh sách phân tầng theo thay đổi cờ CHAIR. Đây là minh họa có chủ đích, không đại diện phân phối dữ liệu. Nhận xét ảnh do trợ lý kiểm tra, chưa phải một nghiên cứu chấm nhãn độc lập của con người.',size=11)
    review=load(DATA/'paper_analysis/visual_review_24.json')
    selection=[359270,559364,321014,359465]
    labels={359270:'Bỏ đối tượng sai nhưng chưa mô tả đầy đủ hành động',559364:'Nhận diện hình dạng đối tượng phù hợp hơn',321014:'PROMPT phát sinh lỗi đối tượng',359465:'Bỏ cờ CHAIR chưa đồng nghĩa caption tốt hơn'}
    for j,i in enumerate(selection):
        row=next(r for r in review['items'] if r['coco_id']==i)
        para(f'{j+1}. {labels[i]}','Heading 2',page=(j==2))
        p=para('');p.alignment=1
        photo=HERE/'qualitative_assets'/f'{i}.jpg'
        with Image.open(photo) as im:w,h=im.size
        scale=min(7/w,4/h)
        p.add_run().add_picture(str(photo),width=Cm(w*scale),height=Cm(h*scale))
        p.paragraph_format.keep_with_next=True
        para(f"COCO {i} — {'Trước' if row['policy']=='normal' else 'Sau'} reranking.",size=10)
        para('CONTROL: '+row['CONTROL']['caption'],size=11)
        para('PROMPT: '+row['PROMPT']['caption'],size=11)
        para(row['assistant_visual_review']['note'],size=11)
    assert [t._tbl.xml for t in doc.tables[:3]]==old_tables
    assert len(doc.inline_shapes)==9
    doc.save(OUTPUT)
    print(OUTPUT.name)


if __name__=='__main__':main()
