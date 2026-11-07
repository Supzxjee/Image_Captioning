"""Remove only the two source-note paragraphs requested by the user."""
from pathlib import Path
from docx import Document
import hashlib
HERE=Path(__file__).resolve().parent
path=HERE/'Bao_cao_nghien_cuu_07_11_2026.docx'
backup=HERE.parents[2].parent/'.review_cache/report_no_sources_2026-11-08'
backup.mkdir(parents=True,exist_ok=True)
raw=path.read_bytes();saved=backup/('before_'+hashlib.sha256(raw).hexdigest()[:12]+'.docx')
if not saved.exists():saved.write_bytes(raw)
doc=Document(path);tables=[t._tbl.xml for t in doc.tables]
targets=[p for p in doc.paragraphs if p.text.startswith('Nguồn đã kiểm tra:') or p.text.strip()=='https://github.com/Supzxjee/Image_Captioning/tree/main/experiments']
assert len(targets)==2
removed=[p._p for p in targets]
expected=[p._p.xml for p in doc.paragraphs if p._p not in removed]
for p in targets:p._p.getparent().remove(p._p)
assert [p._p.xml for p in doc.paragraphs]==expected
assert [t._tbl.xml for t in doc.tables]==tables
doc.save(path)
print('Removed exactly two source-note paragraphs; all other paragraphs and tables unchanged.')
