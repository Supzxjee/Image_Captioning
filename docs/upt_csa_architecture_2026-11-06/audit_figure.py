"""Check rendered typography, vector PDFs, and all orthogonal connector routes."""
from pathlib import Path
from itertools import combinations
import json,hashlib,subprocess
import fitz
ROOT=Path(__file__).resolve().parent
G=json.loads((ROOT/'diagram_geometry.json').read_text())
issues=[]
def inside_seg_box(a,b,n):
 x,y,w,h=n['x'],n['y'],n['w'],n['h']
 if a[0]==b[0]: return x+.1<a[0]<x+w-.1 and min(max(a[1],b[1]),y+h-.1)>max(min(a[1],b[1]),y+.1)
 return y+.1<a[1]<y+h-.1 and min(max(a[0],b[0]),x+w-.1)>max(min(a[0],b[0]),x+.1)
def cross(a,b,c,d):
 if a[1]==b[1] and c[0]==d[0]:
  if min(a[0],b[0])<c[0]<max(a[0],b[0]) and min(c[1],d[1])<a[1]<max(c[1],d[1]): return [c[0],a[1]]
 if a[0]==b[0] and c[1]==d[1]: return cross(c,d,a,b)
 return None
def overlap(a,b,c,d):
 if a[0]==b[0]==c[0]==d[0]: return min(max(a[1],b[1]),max(c[1],d[1]))>max(min(a[1],b[1]),min(c[1],d[1]))
 if a[1]==b[1]==c[1]==d[1]: return min(max(a[0],b[0]),max(c[0],d[0]))>max(min(a[0],b[0]),min(c[0],d[0]))
 return False
for r in G['routes']:
 for a,b in zip(r['points'],r['points'][1:]):
  assert a[0]==b[0] or a[1]==b[1]
  for n in G['nodes']:
   if n['id'] not in [r['src'],r['dst']] and inside_seg_box(a,b,n):issues.append(['transit',r['id'],n['id']])
for ra,rb in combinations(G['routes'],2):
 for a,b in zip(ra['points'],ra['points'][1:]):
  for c,d in zip(rb['points'],rb['points'][1:]):
   pt=cross(a,b,c,d)
   if pt:issues.append(['crossing',ra['id'],rb['id'],pt])
   if overlap(a,b,c,d):issues.append(['overlap',ra['id'],rb['id']])
for a,b in combinations(G['nodes'],2):
 if min(a['x']+a['w'],b['x']+b['w'])>max(a['x'],b['x']) and min(a['y']+a['h'],b['y']+b['h'])>max(a['y'],b['y']):issues.append(['node-overlap',a['id'],b['id']])
render=json.loads((ROOT/'render_audit.json').read_text())
for slug,r in render.items():
 if r['overflow']:issues.append(['text-overflow',slug,r['overflow']])
pdfs={}
for path in ROOT.glob('*.pdf'):
 with fitz.open(path) as doc:
  texts=''.join(p.get_text() for p in doc)
  images=sum(len(p.get_images()) for p in doc)
  drawings=sum(len(p.get_drawings()) for p in doc)
  pdfs[path.name]={'pages':len(doc),'text_characters':len(texts),'raster_images':images,'vector_drawings':drawings}
  if len(doc)!=1 or len(texts)<200 or images!=0 or drawings<20:issues.append(['pdf-not-vector-single-page',path.name])
for html in ROOT.glob('*.html'):
 p=subprocess.run(['python','/home/agent/.codex/plugins/cache/diagram-design/diagram-design/2.6.142/skills/diagram-design/scripts/self_check.py',str(html)],capture_output=True,text=True)
 if p.returncode:issues.append(['skill-self-check',html.name,p.stdout+p.stderr])
report={'status':'PASS' if not issues else 'FAIL','routes_checked':len(G['routes']),'nodes_checked':len(G['nodes']),'issues':issues,'pdfs':pdfs,'render_report':'render_audit.json','source_brief_sha256':hashlib.sha256((ROOT/'USER_BRIEF.txt').read_bytes()).hexdigest(),'limits':'Figure follows user-supplied architecture, not an implementation audit or a model/metric result. Unrounded axis-aligned route envelopes are checked conservatively; actual SVG corners are rounded.'}
(ROOT/'figure_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps(report,indent=2,ensure_ascii=False))
raise SystemExit(bool(issues))
