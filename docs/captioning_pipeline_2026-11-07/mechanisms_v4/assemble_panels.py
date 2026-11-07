"""Assemble existing vector PDF pages for convenient reading; no rasterization."""
from pathlib import Path
import fitz
ROOT=Path(__file__).resolve().parent
PAGES=[('panel-overview-v4','Overall architecture'),('panel-roi-pooling-v4','Box-weighted ROI pooling'),('panel-a-banks-v4','A — Banks and aggregation'),('panel-b-fusion-v4','B — Semantic fusion'),('panel-c-adapters-v4','C — Image-specific prompts and adapters'),('panel-d-decoder-v4','D — Caption generation'),('panel-e-reranking-v4','E — Caption reranking'),('panel-f-training-v4','F — Training only')]
doc=fitz.open()
for stem,_ in PAGES:
    with fitz.open(ROOT/(stem+'.pdf')) as source:
        assert len(source)==1
        doc.insert_pdf(source)
doc.set_toc([[1,title,i+1] for i,(_,title) in enumerate(PAGES)])
doc.set_metadata({'title':'Image captioning mechanisms V4 — standalone panels','subject':'User-requested architecture diagram; graphics only'})
out=ROOT/'captioning-mechanisms-v4-panels.pdf'
doc.save(out,garbage=4,deflate=True);doc.close()
print(out.name,'8 vector pages')
