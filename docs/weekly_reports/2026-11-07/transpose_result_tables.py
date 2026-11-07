"""Transpose result tables for portrait reading; preserve every metric string."""
from pathlib import Path
import hashlib,json
from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches,Pt,RGBColor
HERE=Path(__file__).resolve().parent
path=HERE/'Bao_cao_nghien_cuu_07_11_2026.docx'
backup=HERE.parents[2].parent/'.review_cache/report_portrait_2026-11-07'
backup.mkdir(parents=True,exist_ok=True)
raw=path.read_bytes();p=backup/('before_'+hashlib.sha256(raw).hexdigest()[:12]+'.docx')
if not p.exists():p.write_bytes(raw)
doc=Document(path);original_paragraphs=[p.text for p in doc.paragraphs]
original_other=[t._tbl.xml for t in doc.tables[2:]]
qa=[]
for table_index,widths in [(0,[1.6,2.75,2.75]),(1,[1.4,1.2,1.2,1.525,1.775])]:
 old=doc.tables[table_index]
 if len(old.columns)!=9:
  assert len(old.rows)==9
  continue
 source=[[c.text for c in row.cells] for row in old.rows]
 result=[list(row) for row in zip(*source)];result[0][0]='Thang đo'
 table=doc.add_table(rows=len(result),cols=len(result[0]));table.alignment=WD_TABLE_ALIGNMENT.CENTER;table.autofit=False
 for j,w in enumerate(widths):table.columns[j].width=Inches(w)
 for i,row in enumerate(table.rows):
  props=row._tr.get_or_add_trPr();props.append(OxmlElement('w:cantSplit'))
  if i==0:props.append(OxmlElement('w:tblHeader'))
  for j,cell in enumerate(row.cells):
   cell.text=result[i][j];cell.width=Inches(widths[j]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   pr=cell._tc.get_or_add_tcPr()
   borders=OxmlElement('w:tcBorders')
   for edge in ('top','left','bottom','right'):
    e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
   pr.append(borders)
   margins=OxmlElement('w:tcMar')
   for edge,v in [('top','30'),('bottom','30'),('left','70'),('right','70')]:
    e=OxmlElement('w:'+edge);e.set(qn('w:w'),v);e.set(qn('w:type'),'dxa');margins.append(e)
   pr.append(margins)
   shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'1B4266' if i==0 else 'F1F5F8' if i%2==0 else 'FFFFFF');pr.append(shade)
   for p in cell.paragraphs:
    p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before=p.paragraph_format.space_after=Pt(0)
    p.paragraph_format.line_spacing=1
    for run in p.runs:
     run.font.name='Times New Roman';run.font.size=Pt(11);run.bold=i==0
     run.font.color.rgb=RGBColor.from_string('FFFFFF' if i==0 else '000000')
 old._tbl.addprevious(table._tbl);old._tbl.getparent().remove(old._tbl)
 actual=[[c.text for c in row.cells] for row in table.rows]
 assert actual==result
 qa.append({'table':table_index,'metrics_rows':8,'config_columns':len(result[0])-1,'all_numeric_strings_exact':True})
for section in doc.sections:
 section.orientation=WD_ORIENT.PORTRAIT;section.page_width=Inches(8.5);section.page_height=Inches(11)
assert [p.text for p in doc.paragraphs]==original_paragraphs
assert [t._tbl.xml for t in doc.tables[2:]]==original_other
assert len(doc.inline_shapes)==1
assert all(len(t.rows)==9 for t in doc.tables[:2])
doc.save(path)
(HERE/'portrait_tables_qa.json').write_text(json.dumps({'transpose':qa,'all_prose_and_other_tables_unchanged':True,'all_sections_portrait':True},indent=2),encoding='utf-8')
print('Portrait layout; both result tables transposed; every numeric string preserved.')
