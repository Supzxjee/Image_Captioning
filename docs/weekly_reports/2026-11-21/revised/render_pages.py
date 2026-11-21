"""Rasterize Word's PDF fallback using the bundled PDFium runtime for visual QA."""
from pathlib import Path
import json
import pypdfium2 as pdfium

HERE=Path(__file__).resolve().parent
pdf=pdfium.PdfDocument(HERE/'render/report.pdf')
assert len(pdf)==6,len(pdf)
for i,page in enumerate(pdf):
    width,height=page.get_size()
    assert abs(width-595.3)<1 and abs(height-841.9)<1,(width,height)
    page.render(scale=1.5).to_pil().save(HERE/f'render/page-{i+1}.png')
    (HERE/f'render/page-{i+1}.txt').write_text(page.get_textpage().get_text_range(),encoding='utf-8')
print(json.dumps(dict(pages=len(pdf),size='A4',rasterized=True)))
