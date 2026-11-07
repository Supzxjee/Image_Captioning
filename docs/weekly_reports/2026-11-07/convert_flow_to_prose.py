"""Convert diagram-order explanation to connected prose without step labels."""
from pathlib import Path
import hashlib
from docx import Document
from docx.shared import Pt
HERE=Path(__file__).resolve().parent
path=HERE/'Bao_cao_nghien_cuu_07_11_2026.docx'
backup=HERE.parents[2].parent/'.review_cache/report_prose_2026-11-08'
backup.mkdir(parents=True,exist_ok=True)
raw=path.read_bytes();saved=backup/('before_'+hashlib.sha256(raw).hexdigest()[:12]+'.docx')
if not saved.exists():saved.write_bytes(raw)
doc=Document(path);tables=[t._tbl.xml for t in doc.tables]
changed=[]
for p in doc.paragraphs:
 if not p.runs or not p.runs[0].bold:continue
 lead=p.runs[0].text
 if not (lead.startswith('Bước ') or lead.startswith('Đường đi bên trong') or lead.startswith('Chuẩn bị bank ở nhánh') or lead.startswith('Đi theo hàng chính') or lead.startswith('Từ đặc trưng đã căn chỉnh')):continue
 text=p.text[len(lead):]
 if 'Từ caption đến danh sách từ gốc.' in lead:
  text=text.replace('Bắt đầu ở ô TRAIN captions bên trái. Caption đi qua POS tagging','M1 nhận các caption của tập TRAIN và đưa qua POS tagging',1)
 elif 'Từ danh sách từ đến bank.' in lead:
  text='Từ hai danh sách này, '+text[0].lower()+text[1:]
 elif 'Chuẩn bị ngữ nghĩa từ hai bank' in lead:
  text='M2 nhận hai bank từ M1. '+text
 elif 'Đưa đặc trưng vùng vào tầng' in lead:
  text=text.replace('Ô Regions V nhận đầu ra Visual region extraction:','Song song với nhánh bank, đặc trưng vùng V được lấy từ khối Visual region extraction:',1)
 elif 'Đi tiếp qua tầng đối tượng thứ hai.' in lead:
  text='Tiếp đó, '+text
 elif 'Tạo đầu ra quan hệ.' in lead:
  text='Ở nhánh quan hệ, '+text
 elif lead.startswith('Đường đi bên trong'):
  text=text.replace('Regions X là đầu vào của tầng đang xét: X lần lượt là V, V₁ hoặc Vo ở ba tầng phía trên.','Bên trong mỗi tầng căn chỉnh, X là ký hiệu chung cho đầu vào: V ở tầng đối tượng thứ nhất, V₁ ở tầng thứ hai và Vo ở tầng quan hệ.',1)
 elif 'Từ ảnh tạo hai nhánh thông tin.' in lead:
  text=text.replace('Bắt đầu ở Image I, ảnh đi qua UPT detector.','M3 nhận ảnh I và đưa qua UPT detector.',1)
 elif 'Ghép chữ với vùng ảnh.' in lead:
  text=text.replace('Nhánh cụm mô tả đi qua CLIP text encoder','Các cụm mô tả tiếp tục đi qua CLIP text encoder',1)
 elif 'Đi qua Slot MLP ở từng nhánh.' in lead:
  text=text.replace('Ở nhánh đối tượng, to được ghép','Từ gói thông tin đã ghép ROI, nhánh đối tượng kết hợp to',1)
 elif 'Đọc slot rồi cập nhật đặc trưng vùng.' in lead:
  text=text.replace('Slot đi sang Masked cross-attention','Các slot sau đó đi vào Masked cross-attention',1)
 elif lead.startswith('Chuẩn bị bank ở nhánh'):
  text=text.replace('Caption TRAIN đi vào M1','Trước khi xử lý ảnh, caption TRAIN đi vào M1',1)
 elif lead.startswith('Đi theo hàng chính'):
  text=text.replace('Image I đi qua Visual region extraction để tạo V.','Khi nhận một ảnh I, mô hình dùng khối Visual region extraction để tạo đặc trưng vùng V.',1)
 elif lead.startswith('Từ đặc trưng đã căn chỉnh'):
  text=text.replace('Vo′ và Vr′ đi vào Transformer decoder.','Hai đầu ra Vo′ và Vr′ từ M3 được chuyển sang Transformer decoder.',1)
  text=text.replace('Luồng đi xuống Beam search với K bằng 5 để tạo năm caption ứng viên, rồi đi sang trái qua Caption reranking.','Beam search với K bằng 5 tạo năm caption ứng viên để chuyển sang Caption reranking.',1)
 p.text=text
 for run in p.runs:run.bold=False;run.font.name='Times New Roman';run.font.size=Pt(12)
 changed.append(lead.strip())
assert len(changed)==14,len(changed)
assert not any(p.text.startswith('Bước ') for p in doc.paragraphs)
assert [t._tbl.xml for t in doc.tables]==tables
assert len(doc.inline_shapes)==4
doc.save(path)
print('Converted 14 explanation paragraphs to connected prose; tables and figures unchanged.')
