"""Insert approved figures and apply the five read-only-review recommendations."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
from PIL import Image
from docx import Document
from docx.oxml.ns import qn
from docx.shared import Inches, Mm

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
SOURCE=Path('C:/Users/leduc/Downloads/Bao_cao_nghien_cuu_tuan_15.docx')
FIGURES=REPO/'docs/week15_figure_polish_2026-11-21'
OUTDIR=HERE/'revised'
OUT=OUTDIR/'Bao_cao_nghien_cuu_tuan_15_cap_nhat.docx'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def table_snapshot(doc):
    return [[[dict(text=c.text,bold=[r.bold for p in c.paragraphs for r in p.runs]) for c in row.cells] for row in table.rows] for table in doc.tables]


def main():
    OUTDIR.mkdir(exist_ok=True)
    before=sha(SOURCE)
    doc=Document(SOURCE)
    original_tables=table_snapshot(doc)
    assert len(doc.inline_shapes)==4 and len(doc.tables)==2
    edits={
      'Mô hình có căn chỉnh được huấn luyện bằng cross-entropy':
      'Em thực nghiệm trên MS COCO theo cách chia Karpathy, gồm 113.287 ảnh TRAIN, 5.000 ảnh VALID và 5.000 ảnh TEST. Mô hình có căn chỉnh được huấn luyện 10 epoch bằng cross-entropy (CE), tức học dự đoán từ tiếp theo theo caption tham chiếu TRAIN. Em chọn checkpoint có CIDEr cao nhất trên VALID rồi đánh giá trên TEST. Từ checkpoint của mỗi seed, mô hình học tiếp bằng SCST thuần tại cùng mốc 1.500 update, không trộn CE hoặc thêm thưởng đối tượng. Thưởng SCST là chênh lệch CIDEr giữa caption lấy mẫu và caption greedy của chính mô hình.',
      'Transformer Decoder gồm ba tầng với attention đối tượng':
      'Transformer Decoder gồm ba tầng với attention đối tượng và quan hệ song song. Beam search sinh 5 caption mỗi ảnh. Reranking kết hợp 0,3 điểm decoder, 0,4 điểm CLIP và 0,3 điểm nhất quán đối tượng với nhãn YOLO. Trong mỗi nhóm 5 ứng viên của một ảnh, điểm decoder và CLIP được chuẩn hóa min–max; điểm đối tượng được ánh xạ về [0, 1] theo thang điểm cố định. Các trọng số và quy tắc chấm được cố định trước TEST, không dùng caption tham chiếu để chọn câu.',
      'Để thử hạn chế sự lệch phân phối khi tối ưu CIDEr':
      'KL (Kullback–Leibler divergence) đo độ lệch giữa hai phân phối xác suất. Em bổ sung hạng KL để hạn chế phân phối dự đoán token của mô hình đang học tiếp lệch quá xa so với bản sao đóng băng trước SCST. Pilot dùng seed 43 và VALID5000, so sánh ba bản: trước SCST, SCST thường 500 update và SCST có KL 500 update. Hai nhánh học tiếp khởi đầu từ cùng trọng số, dùng cùng dữ liệu và lịch cập nhật; cả ba dùng cùng reranking.',
      '7 Rà soát hình và hướng tiếp theo':
      '7 Hoàn thiện sơ đồ và hướng tiếp theo',
      'Em đã kiểm tra năm hình dựa trên mã mô hình':
      'Em đã hoàn thiện bốn hình trong báo cáo: pipeline tổng thể và ba sơ đồ M1, M2, M3. Các hình tách riêng bank danh từ và động từ, hai bộ nhớ đối tượng và quan hệ; đồng thời thể hiện đường liên kết ROI, attention và cộng dư.',
    }
    changed=[]
    for prefix,replacement in edits.items():
        matches=[p for p in doc.paragraphs if p.text.startswith(prefix)]
        assert len(matches)==1,(prefix,len(matches))
        matches[0].text=replacement
        for run in matches[0].runs:
            run.font.name='Times New Roman'
            run._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Times New Roman')
        changed.append(prefix)
    figures=[('01_full_pipeline.png',6.94),('02_M1_semantic_bank.png',5.75),('03_M2_bank_visual_alignment.png',6.94),('04_M3_prompt_visual_alignment.png',6.94)]
    embedded=[]
    for shape,(name,width) in zip(doc.inline_shapes,figures):
        path=FIGURES/name
        rid=shape._inline.graphic.graphicData.pic.blipFill.blip.embed
        part=doc.part.related_parts[rid]
        part._blob=path.read_bytes()
        with Image.open(path) as im:w,h=im.size
        shape.width=Inches(width);shape.height=Inches(width*h/w)
        shape._inline.docPr.set('name',name)
        embedded.append(dict(name=name,sha256=sha(path),width_inches=width))
    for sec in doc.sections:
        sec.page_width=Mm(210);sec.page_height=Mm(297)
        sec.left_margin=sec.right_margin=Inches(.65)
    widths=[1.12,1.86,2.03,2.09]
    widths=[w*6.94/sum(widths) for w in widths]
    for table in doc.tables:
        table.autofit=False
        for col,width in zip(table.columns,widths):col.width=Inches(width)
        for row in table.rows:
            for cell,width in zip(row.cells,widths):cell.width=Inches(width)
    assert table_snapshot(doc)==original_tables,'Metric values or best-value bold changed'
    doc.save(OUT)
    reread=Document(OUT)
    assert table_snapshot(reread)==original_tables
    for shape,item in zip(reread.inline_shapes,embedded):
        rid=shape._inline.graphic.graphicData.pic.blipFill.blip.embed
        assert hashlib.sha256(reread.part.related_parts[rid].blob).hexdigest()==item['sha256']
    assert sha(SOURCE)==before
    checks=dict(source=str(SOURCE),source_sha256=before,source_preserved=True,output=str(OUT),output_sha256=sha(OUT),embedded_figures=embedded,paragraphs_updated=len(changed),table_values_and_bold_unchanged=True,page_size='A4 portrait',models_executed=False,render_pending=True)
    (OUTDIR/'CHECKS.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(output=str(OUT),embedded=4,paragraphs_updated=len(changed),tables_preserved=True)))


def finalize():
    checks=json.loads((OUTDIR/'CHECKS.json').read_text(encoding='utf-8'))
    assert checks['output_sha256']==sha(OUT)
    assert checks['source_sha256']==sha(SOURCE)
    pages=list((OUTDIR/'render').glob('page-*.png'))
    assert len(pages)==6
    texts='\n'.join(p.read_text(encoding='utf-8') for p in sorted((OUTDIR/'render').glob('page-*.txt')))
    for expected in ['113.287','Karpathy','10 epoch','min–max','Kullback','bốn hình']:
        assert expected in texts,expected
    assert 'Em đã kiểm tra năm hình' not in texts
    delivery=Path('C:/Users/leduc/Downloads')/OUT.name
    shutil.copy2(OUT,delivery)
    assert sha(delivery)==sha(OUT)
    checks.update(render_pending=False,pages=6,renderer='Canonical render_docx attempted: LibreOffice missing. Hidden Word COM exported a valid six-page A4 PDF; bundled PDFium rendered every page. COM Quit returned RPC-unavailable after export; PDF parsed and all page rasters verified.',visual_review=dict(all_pages_inspected=[1,2,3,4,5,6],refinement='Explicit Times New Roman applied to revised heading7 and paragraphs; final pages5/6 re-rendered and inspected',clipping_or_overlap_observed=False),pdf_sha256=sha(OUTDIR/'render/report.pdf'),page_png_sha256={p.name:sha(p) for p in sorted(pages)},delivery=str(delivery),delivery_sha256=sha(delivery))
    (OUTDIR/'CHECKS.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(delivery=str(delivery),pages=6,source_preserved=True)))


if __name__=='__main__':
    finalize() if '--finalize' in sys.argv else main()
