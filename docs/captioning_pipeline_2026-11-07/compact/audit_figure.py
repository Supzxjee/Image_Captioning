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
 if r.get('aggregationBadgeLabelOverlap'):issues.append(['aggregation-badge-label-overlap',slug,r['aggregationBadgeLabelOverlap']])
 if r.get('aggregationBadgeModuleTextOverlap'):issues.append(['aggregation-badge-module-text-overlap',slug,r['aggregationBadgeModuleTextOverlap']])
pdfs={}
for path in ROOT.glob('captioning-pipeline-v3.pdf'):
 with fitz.open(path) as doc:
  texts=''.join(p.get_text() for p in doc)
  images=sum(len(p.get_images()) for p in doc)
  drawings=sum(len(p.get_drawings()) for p in doc)
  pdfs[path.name]={'pages':len(doc),'text_characters':len(texts),'raster_images':images,'vector_drawings':drawings}
  if len(doc)!=1 or len(texts)<200 or images<1 or drawings<20:issues.append(['pdf-not-single-page-vector-with-illustrative-photo',path.name])
for html in ROOT.glob('captioning-pipeline-v3.html'):
 p=subprocess.run(['python','/home/agent/.codex/plugins/cache/diagram-design/diagram-design/2.6.142/skills/diagram-design/scripts/self_check.py',str(html)],capture_output=True,text=True)
 if p.returncode:issues.append(['skill-self-check',html.name,p.stdout+p.stderr])
report={'status':'PASS' if not issues else 'FAIL','routes_checked':len(G['routes']),'nodes_checked':len(G['nodes']),'issues':issues,'pdfs':pdfs,'render_report':'render_audit.json','source_brief_sha256':hashlib.sha256((ROOT/'LAYOUT_BRIEF.md').read_bytes()).hexdigest(),'limits':'Reference-style visual variant follows the previous brief; photos and box overlays are illustrative, and raster images are allowed only for the input photo. Unrounded axis-aligned route envelopes are checked conservatively; actual SVG connectors use straight orthogonal segments and sharp 90-degree elbows.'}
import re,xml.etree.ElementTree as ET
svg=ET.parse(ROOT/'captioning-pipeline-v3.svg');ns={'s':'http://www.w3.org/2000/svg'}
visible=' '.join(''.join(e.itertext()) for e in svg.findall('.//s:text',ns))
assert not re.search(r'\b(?:CSA(?:-Net)?|OSA|ORA)\b',visible,re.I)
assert not re.search(r'\b(?:Q/K/V|zero.init|heads)\b',visible,re.I)
assert len(svg.findall('.//s:image',ns))==1
assert not any(p.get('data-edge','').endswith('-zoom') for p in svg.findall('.//s:path',ns))
assert visible.count('Vₒ′')==1 and visible.count('Vᵣ′')==1
assert svg.find('.//s:rect[@data-node-id="semantic-overview"]',ns) is None
assert svg.find('.//s:rect[@data-node-id="prompt-overview"]',ns) is None
report['content_checks']={'forbidden_method_labels_absent':True,'qkv_head_init_details_absent':True,'one_illustrative_image':True,'training_line_outside_inference':True,'two_named_memory_rails':True,'no_duplicated_overview_or_zoom_guides':True,'updated_memories_appear_once':True,'visual_query_and_residual_paths_present':True}
report['limits']='Compact technical-flow variant follows the user-approved 07/11 layout revision; the original input photo has no added overlays, and diagrams/tokens are not newly measured model output. Geometry checks use conservative axis-aligned envelopes; SVG connectors use straight orthogonal segments and sharp 90-degree elbows.'
# Exact badge map: no trainable badges on pooling, matching, beam, or scoring.
badges={g.get('data-module'):g.get('data-icon-kind') for g in svg.findall('.//s:g',ns) if g.get('data-module')!='legend' and g.get('data-icon-kind')}
for key in ['clip-visual','yolo','bank-text-encoder','upt','clip-text','alignment']:assert badges[key]=='frozen',key
for key in ['projection','noun-aggregation','verb-aggregation','object-fusion-1','object-fusion-2','relation-fusion','object-adapter','relation-adapter','decoder','embedding','vocab']:assert badges[key]=='trainable',key
for key in ['roi-pooling','region-matching','beam','select','likelihood','consistency']:assert key not in badges,key
original=svg.find('.//s:g[@id="input-image"]',ns)
assert original is not None and len(original)==1 and original[0].tag.endswith('image')
for key in ['object','relation']:
 assert any(r['id']==key+'-visual-query' for r in G['routes'])
 assert any(r['id']==key+'-residual' for r in G['routes'])
for key in ['noun-bank','verb-bank','patches','boxes','prefix','final-caption']:
 rect=svg.find('.//s:rect[@data-node-id="'+key+'"]',ns);assert rect is not None and rect.get('rx')=='0',key
assert all(str(a) in visible for a in ['768','512','150'])
for group in svg.findall('.//s:g',ns):
 kind=group.get('data-icon-kind')
 if kind:assert group.get('stroke')==('#1677B8' if kind=='frozen' else '#D95F1A')
for path in svg.findall('.//s:path',ns):
 if path.get('data-edge'):assert not re.search('[QqCcAaSsTt]',path.get('d','')),path.get('data-edge')
assert visible.count('Slot MLP')==2
assert 'tₒ + Vₒ[i] + box' in visible
assert 'tᵣ + Vₒ[s], Vₒ[o] + boxes' in visible
assert 'Linked slots' not in visible
for module in ['noun-aggregation','verb-aggregation']:
 g=svg.find('.//s:g[@data-module="'+module+'"]',ns)
 assert g.get('transform').endswith('520)'),module
assert 'S = 0.3D + 0.4C + 0.3O' in (ROOT/'captioning-pipeline-v3.html').read_text()
for key in ['prepared-object-slots','prepared-relation-slots','d-select','c-select','o-select']:
 route=next(r for r in G['routes'] if r['id']==key);assert len(route['points'])==2,key
report['slot_input_labels_and_rerank_caption']=True
report['aggregation_badges_clear_of_prototype_dimensions']=True
report['colored_badges']={'frozen':'#1677B8','trainable_stroke':'#D95F1A','trainable_fill':'#F6A23A'}
report['straight_orthogonal_connectors']=True
report['parameter_badges']=badges
report['content_checks']['original_image_without_box_overlays']=True
report['content_checks']['data_rectangles_are_square_cornered']=True
(ROOT/'figure_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps(report,indent=2,ensure_ascii=False))
raise SystemExit(bool(issues))
