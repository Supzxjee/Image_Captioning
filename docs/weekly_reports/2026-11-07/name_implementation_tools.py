"""Name verified tools/models while retaining the prose and diagram flow."""
from pathlib import Path
import hashlib
from copy import deepcopy
from docx import Document
HERE=Path(__file__).resolve().parent
path=HERE/'Bao_cao_nghien_cuu_07_11_2026.docx'
backup=HERE.parents[2].parent/'.review_cache/report_tools_2026-11-08'
backup.mkdir(parents=True,exist_ok=True)
raw=path.read_bytes();saved=backup/('before_'+hashlib.sha256(raw).hexdigest()[:12]+'.docx')
if not saved.exists():saved.write_bytes(raw)
doc=Document(path);tables=[t._tbl.xml for t in doc.tables];changes=[]
for p in doc.paragraphs:
 text=p.text;new=text
 if text.startswith('M1 nhận các caption của tập TRAIN'):
  new=text.replace('M1 nhận các caption của tập TRAIN và đưa qua POS tagging để xác định danh từ và động từ, rồi qua Lemmatization để đưa chúng về dạng gốc,','M1 nhận các caption của tập TRAIN và sử dụng thư viện spaCy với mô hình tiếng Anh en_core_web_sm để gán nhãn từ loại (POS tagging) và đưa từ về dạng gốc (Lemmatization),',1)
  new=new.replace('Đầu ra là hai danh sách noun lemmas và verb lemmas.','Sau khi giữ lại các danh từ và động từ, đầu ra là hai danh sách noun lemmas và verb lemmas.')
 elif text.startswith('Từ hai danh sách này, mỗi từ gốc'):
  new=text.replace('đưa qua CLIP text encoder để nhận vector 512 chiều','đưa qua bộ mã hóa văn bản của OpenAI CLIP ViT-B/16, được nạp bằng thư viện Transformers, để nhận vector 512 chiều',1)
 elif text.startswith('M2 nhận hai bank từ M1.'):
  new=text.replace('M2 nhận hai bank từ M1.','M2 được cài đặt bằng thư viện PyTorch và nhận hai bank từ M1.',1)
 elif text.startswith('Song song với nhánh bank, đặc trưng vùng V'):
  new=text.replace('ảnh đi qua CLIP ViT-B/16','ảnh đi qua bộ mã hóa thị giác OpenAI CLIP ViT-B/16',1)
 elif text.startswith('M3 nhận ảnh I và đưa qua UPT detector.'):
  new=text.replace('M3 nhận ảnh I và đưa qua UPT detector.','M3 nhận ảnh I và dùng mô hình UPT để phát hiện đối tượng và tương tác giữa người với vật.',1)
 elif text.startswith('Các cụm mô tả tiếp tục đi qua CLIP text encoder'):
  new=text.replace('Các cụm mô tả tiếp tục đi qua CLIP text encoder','Các cụm mô tả tiếp tục đi qua bộ mã hóa văn bản của OpenAI CLIP ViT-B/16',1)
 if new!=text:
  formatting=deepcopy(p.runs[0]._r.rPr) if p.runs and p.runs[0]._r.rPr is not None else None
  p.text=new
  if formatting is not None:p.runs[0]._r.insert(0,formatting)
  changes.append(text[:60])
assert len(changes)==6,len(changes)
assert any('spaCy' in p.text and 'en_core_web_sm' in p.text for p in doc.paragraphs)
assert [t._tbl.xml for t in doc.tables]==tables
assert len(doc.inline_shapes)==4 and not any(p.text.startswith('Bước ') for p in doc.paragraphs)
doc.save(path)
print('Named spaCy/en_core_web_sm, OpenAI CLIP ViT-B/16, Transformers, PyTorch and UPT; YOLO retained without guessing a version.')
