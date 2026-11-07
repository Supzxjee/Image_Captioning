"""Mark within-table best results, remove direction arrows, use descriptive method names."""
from pathlib import Path
from decimal import Decimal
import hashlib,json,re
from copy import deepcopy
from docx import Document
HERE=Path(__file__).resolve().parent
path=HERE/'Bao_cao_nghien_cuu_07_11_2026.docx'
backup=HERE.parents[2].parent/'.review_cache/report_best_2026-11-08'
backup.mkdir(parents=True,exist_ok=True)
raw=path.read_bytes();original=backup/('before_'+hashlib.sha256(raw).hexdigest()[:12]+'.docx')
if not original.exists():original.write_bytes(raw)
doc=Document(path)
paragraph_formats=[deepcopy(p.runs[0]._r.rPr) if p.runs and p.runs[0]._r.rPr is not None else None for p in doc.paragraphs]
num=lambda s:Decimal(s.strip().replace(',','.'))
number_pattern=re.compile(r'^\d+(?:,\d+)?$')
before_numbers=[[c.text for row in t.rows for c in row.cells if number_pattern.fullmatch(c.text)] for t in doc.tables]
legend='Dấu ↑ là càng cao càng tốt, dấu ↓ là càng thấp càng tốt.'
for p in doc.paragraphs:
 if legend in p.text:p.text=p.text.replace(legend,'In đậm giá trị tốt nhất; CHAIRs và CHAIRi càng thấp càng tốt.')
 if p.text=='Đối chứng không CSA và có CSA sau 10 epoch XE':p.text='Đối chứng căn chỉnh bank sau 10 epoch XE'
 elif p.text.startswith('Baseline không CSA dùng'):
  p.text=p.text.replace('Baseline không CSA dùng','Bản đối chứng dùng').replace('Bản CSA thay encoder','Bản có căn chỉnh bank thay encoder')
 elif p.text.startswith('CSA tăng 45,00'):
  p.text=p.text.replace('CSA tăng','Thêm căn chỉnh bank tăng').replace('NoCSA có','Bản không bank có').replace('còn CSA có','còn bản có bank có').replace('của NoCSA','của bản không bank')
 else:
  replacements=[('có CSA và bank từ','có căn chỉnh bank'),('căn chỉnh CSA','căn chỉnh bank–vùng ảnh'),('Mốc CSA15','Mốc 15 epoch XE với căn chỉnh bank'),('NoCSA10/CSA10','Đối chứng căn chỉnh bank')]
  for a,b in replacements:
   if a in p.text:p.text=p.text.replace(a,b)
 if 'https://github.com/Supzxjee/Image_Captioning/tree/main/experiments/csa_paper' in p.text:
  p.text=p.text.replace('/experiments/csa_paper','/experiments')
for p,formatting in zip(doc.paragraphs,paragraph_formats):
 if formatting is not None:
  for run in p.runs:
   if run._r.rPr is not None:run._r.remove(run._r.rPr)
   run._r.insert(0,deepcopy(formatting))
for t in doc.tables:
 for row in t.rows:
  for cell in row.cells:
   for p in cell.paragraphs:
    for run in p.runs:
     run.text=run.text.replace('Không CSA 10 epoch','Không căn chỉnh bank 10 epoch').replace('CSA 10 epoch','Có căn chỉnh bank 10 epoch').replace(' ↑','').replace(' ↓','').replace('↑','').replace('↓','')
# Preserve header emphasis; only numeric body cells are compared.
winners=[]
for ti,t in enumerate(doc.tables):
 if ti<2:
  for ri,row in enumerate(t.rows[1:],1):
   values=[num(c.text) for c in row.cells[1:]]
   lower='chair' in row.cells[0].text.lower();best=min(values) if lower else max(values)
   for ci,(cell,v) in enumerate(zip(row.cells[1:],values),1):
    for p in cell.paragraphs:
     for run in p.runs:run.bold=(v==best)
    if v==best:winners.append([ti,ri,ci,cell.text])
 else:
  for ci,header in enumerate(t.rows[0].cells[1:],1):
   values=[num(row.cells[ci].text) for row in t.rows[1:]]
   lower='chair' in header.text.lower() or 'bị đánh dấu' in header.text.lower();best=min(values) if lower else max(values)
   for ri,(row,v) in enumerate(zip(t.rows[1:],values),1):
    for p in row.cells[ci].paragraphs:
     for run in p.runs:run.bold=(v==best)
    if v==best:winners.append([ti,ri,ci,row.cells[ci].text])
after_numbers=[[c.text for row in t.rows for c in row.cells if number_pattern.fullmatch(c.text)] for t in doc.tables]
assert before_numbers==after_numbers
all_text=' '.join(p.text for p in doc.paragraphs)+' '+' '.join(c.text for t in doc.tables for row in t.rows for c in row.cells)
assert not any(x in all_text.lower() for x in ['csa','↑','↓']),all_text
assert len(winners)==22
assert len(doc.inline_shapes)==1 and len(doc.tables)==4
assert all(s.page_width<s.page_height for s in doc.sections)
doc.save(path)
(HERE/'best_results_qa.json').write_text(json.dumps({'numeric_cells_unchanged':True,'best_cells':winners,'arrows_removed':True,'method_names_descriptive_no_csa':True,'selection_scope':'within each table; higher quality scores, lower CHAIR/counts; all ties bold'},indent=2,ensure_ascii=False),encoding='utf-8')
print('22 best numeric cells bold; arrow symbols removed; method names updated; all numbers preserved.')
