"""Editable paper figure from the user brief; no model/training code is executed."""
from pathlib import Path
import html, json, math
ROOT=Path(__file__).resolve().parent
C={'paper':'#ffffff','ink':'#26333e','line':'#53616c','muted':'#52616e','border':'#82909b','visual':'#eaf3fb','semantic':'#edf5e9','decoder':'#f1edf8','rank':'#fcf0df','plain':'#ffffff','rule':'#c7d0d6'}
class Canvas:
 def __init__(self): self.back=[];self.edges=[];self.labels=[];self.nodes=[];self.meta=[];self.routes=[]
 def text(self,x,y,s,size=26,weight=400,anchor='middle',color=None,owner=None,italic=False):
  a=f' data-owner="{owner}"' if owner else ''
  return f'<text x="{x}" y="{y}" font-family="Arial, Liberation Sans, sans-serif" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color or C["ink"]}"'+a+(' font-style="italic"' if italic else '')+'>'+html.escape(s).replace('y_g','y<tspan baseline-shift="sub" font-size="0.75em">g</tspan>')+'</text>'
 def box(self,id,x,y,w,h,title,subs=(),fill='plain',tag=None,fs=30,ss=24,stack=False):
  p=[]
  if stack:
   for dx in [8,4]: p.append(f'<rect x="{x+dx}" y="{y-dx}" width="{w}" height="{h}" rx="6" fill="{C[fill]}" stroke="{C["border"]}" stroke-width="1.2"/>')
  p.append(f'<rect data-role="node" data-node-id="{id}" x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{C[fill]}" stroke="{C["border"]}" stroke-width="1.4"/>')
  lines=[(title,fs,600)]+[(s,ss,400) for s in subs]
  step=ss+8; total=(len(lines)-1)*step
  cy=y+(h-total)/2+fs*.32
  for j,(s,f,weight) in enumerate(lines): p.append(self.text(x+w/2,cy+j*step,s,f,weight,owner=id))
  if tag:
   p.append(f'<rect x="{x+8}" y="{y-12}" width="22" height="22" rx="2" fill="#fff" stroke="{C["border"]}" stroke-width="1"/>')
   p.append(self.text(x+19,y+5,tag,17,600))
  self.nodes.append('<g id="'+id+'">'+''.join(p)+'</g>')
  self.meta.append(dict(id=id,x=x,y=y,w=w,h=h,kind='node'))
 def route(self,id,points,src=None,dst=None,dash=False,color=None):
  # Rounded, axis-aligned elbows; retain unrounded routes for geometry checks.
  p=[tuple(v) for v in points]
  for a,b in zip(p,p[1:]): assert a[0]==b[0] or a[1]==b[1],(id,a,b)
  d=f'M {p[0][0]},{p[0][1]}'
  for i in range(1,len(p)-1):
   a,b,c=p[i-1],p[i],p[i+1]; r=min(8,math.dist(a,b)/2,math.dist(b,c)/2)
   u=((b[0]-a[0])/math.dist(a,b),(b[1]-a[1])/math.dist(a,b));v=((c[0]-b[0])/math.dist(b,c),(c[1]-b[1])/math.dist(b,c))
   d+=f' L {b[0]-u[0]*r},{b[1]-u[1]*r} Q {b[0]},{b[1]} {b[0]+v[0]*r},{b[1]+v[1]*r}'
  d+=f' L {p[-1][0]},{p[-1][1]}'
  ds=' stroke-dasharray="7 5"' if dash else ''
  self.edges.append(f'<path data-edge="{id}" d="{d}" fill="none" stroke="{color or C["line"]}" stroke-width="1.7"{ds} marker-end="url(#arrow)"/>')
  self.routes.append(dict(id=id,points=p,src=src,dst=dst))
 def label(self,x,y,s,size=24):
  w=len(s)*size*.59+16
  self.labels.append(f'<rect x="{x-w/2}" y="{y-size+2}" width="{w}" height="{size+5}" fill="#fff"/>'+self.text(x,y,s,size))
 def plain(self,x,y,s,size=24,weight=400,anchor='middle'):
  self.nodes.append(self.text(x,y,s,size,weight,anchor))
 def plus(self,id,x,y):
  self.nodes.append(f'<g id="{id}"><circle cx="{x}" cy="{y}" r="18" fill="#fff" stroke="{C["line"]}" stroke-width="1.7"/>'+self.text(x,y+10,'+',32)+'</g>')
  self.meta.append(dict(id=id,x=x-18,y=y-18,w=36,h=36,kind='sum'))
 def panel(self,x,y,w,h,label):
  self.back.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="0" fill="#fff" stroke="{C["rule"]}" stroke-width="1"/>')
  self.back.append(self.text(x+24,y+42,label,32,600,'start'))
 def content(self): return ''.join(self.back+self.edges+self.labels+self.nodes)
 def svg(self,slug,view=(0,0,2400,1840)):
  x,y,w,h=view
  return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{x} {y} {w} {h}" role="img" aria-labelledby="{slug}-title {slug}-desc"><title id="{slug}-title">Region-linked prompts for image captioning</title><desc id="{slug}-desc">Frozen visual and semantic encoders feed cascade alignment, region-linked prompt adapters, a parallel decoder and caption reranking, with a separate training objective.</desc><defs><marker id="arrow" viewBox="0 0 10 8" refX="9" refY="4" markerWidth="8" markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,4 L0,8 Z" fill="{C["line"]}"/></marker></defs><rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#fff"/>'+self.content()+'</svg>'

def write_html(name,canvas,view):
 slug=name;w=view[2]
 svg=canvas.svg(slug,view)
 captions={
 'upt-csa-architecture':'Kiến trúc mô hình captioning với căn chỉnh ngữ nghĩa theo cascade và prompt gắn vùng. (a) CLIP/YOLO cung cấp đặc trưng ROI, bank toàn cục đi qua CSA; prompt UPT được nối với ROI và đưa qua hai adapter trước decoder. Beam search sinh năm ứng viên, chọn theo điểm kết hợp decoder, CLIPScore và độ nhất quán đối tượng. (b) Adapter đọc slot qua mask và cập nhật residual; slot quan hệ dùng Vₒ gốc và cặp (s,o) có thứ tự. (c) Huấn luyện XE rồi XE + 0,05 SCST, với reward CIDEr từ reference TRAIN. O chỉ dùng để reranking; CHAIR là metric hậu kiểm.',
 'panel-a-overall':'Luồng suy luận và bank ngữ nghĩa được dựng từ TRAIN. Các cổng cùng tên là cùng một tensor: Pᵥ từ bank prototype; O-links từ Linked prompts; YOLO boxes và I + YOLO labels từ nhánh thị giác. F: đóng băng; L: học được.',
 'panel-b-adapters':'Hai adapter có hidden dimension 512 và 8 attention heads. Object mask chỉ cho ROI i đọc slot tương ứng; relation mask chỉ cho hai đầu s hoặc o đọc slot. Relation slot dùng Vₒ gốc, không dùng memory sau object adapter. Wout khởi tạo 0; không có slot hợp lệ thì cập nhật bằng 0.',
 'panel-c-training':'Mục tiêu huấn luyện dùng reference TRAIN. SCST so CIDEr của câu sampling với baseline greedy từ cùng mô hình và ảnh; advantage được stop-gradient. Loss tổng là LCE + 0,05 LSCST; không có object reward.'}
 caption=html.escape(captions[name])
 text=f'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Region-linked prompts for image captioning</title><style>body{{margin:0;background:#fff;color:#26333e;font-family:Arial,"Liberation Sans",sans-serif}}main{{min-width:0;max-width:100%}}.diagram-container{{width:100%;overflow-x:auto}}svg{{width:100%;min-width:{w}px;display:block}}figcaption{{font-size:16px;line-height:1.6;max-width:1000px;margin:20px auto;padding:0 24px 24px}}@media print{{.diagram-container{{overflow:visible}}svg{{min-width:0}}figcaption{{display:none}}}}</style></head><body><main><div class="diagram-container">{svg}</div><figcaption lang="vi">{caption}</figcaption></main></body></html>'''
 (ROOT/(name+'.html')).write_text(text)

A=Canvas();A.panel(24,24,2352,936,'(a) Overall architecture')
# Global TRAIN-derived semantic bank, distinct from per-image prompts.
A.box('train-captions',200,88,224,104,'TRAIN captions',('Nouns + verbs',),fill='semantic',fs=28)
A.box('global-bank',472,88,432,104,'Lemma → CLIP text',('Frozen noun / verb banks',),fill='semantic',tag='F',fs=30)
A.box('semantic-aggregation',952,88,248,104,'Aggregation ×2',('150 / bank',),fill='semantic',tag='L',fs=28)
A.box('prototypes',1248,88,384,104,'Pₙ  /  Pᵥ',('Global semantic prototypes',),fill='semantic',stack=True,fs=30)
A.route('caption-bank',[(424,140),(472,140)],'train-captions','global-bank')
A.route('bank-aggregate',[(904,140),(952,140)],'global-bank','semantic-aggregation')
A.route('aggregate-prototypes',[(1200,140),(1248,140)],'semantic-aggregation','prototypes')
A.plain(184,236,'VISUAL EVIDENCE',22,600,'start')
# An inline image glyph, avoiding external assets and any implied data result.
A.box('image',48,312,104,104,'I',(),fill='visual',fs=34)
A.box('clip-visual',200,256,224,104,'CLIP visual',('Frozen encoder',),fill='visual',tag='F')
A.box('patches',472,256,200,104,'Patches',('14 × 14 × 768','No CLS'),fill='visual',fs=30,ss=22,stack=True)
A.box('roi-pool',720,256,184,104,'ROI pool',('N × 768',),fill='visual',fs=28)
A.box('visual-projection',952,256,200,104,'Linear',('768 → 512','V: N × 512'),fill='visual',tag='L',fs=30,ss=22)
A.box('yolo',200,448,224,104,'YOLO',('Detections',),fill='visual',tag='F')
A.box('yolo-boxes',472,448,200,104,'Boxes',('+ class labels',),fill='visual',fs=30,ss=22)
A.route('image-clip',[(152,336),(176,336),(176,308),(200,308)],'image','clip-visual')
A.route('image-yolo',[(152,376),(184,376),(184,500),(200,500)],'image','yolo')
A.route('clip-patches',[(424,308),(472,308)],'clip-visual','patches')
A.route('patches-pool',[(672,308),(720,308)],'patches','roi-pool')
A.route('yolo-boxes',[(424,500),(472,500)],'yolo','yolo-boxes')
A.route('boxes-pool',[(572,448),(572,408),(812,408),(812,360)],'yolo-boxes','roi-pool')
A.route('pool-project',[(904,308),(952,308)],'roi-pool','visual-projection')
# CSA order and two distinct memories.
A.box('object-fusion-1',1200,256,224,104,'Object Fusion 1',('V, Pₙ → V₁, T₁',),fill='visual',tag='L',fs=26,ss=24)
A.box('object-fusion-2',1200,408,224,104,'Object Fusion 2',('V₁, T₁ → Vₒ',),fill='visual',tag='L',fs=26,ss=24)
A.box('relation-fusion',1200,560,224,104,'Relation Fusion',('Vₒ, Pᵥ → Vᵣ',),fill='visual',tag='L',fs=26,ss=24)
A.box('verb-prototype-port',1080,568,88,80,'Pᵥ',(),fill='semantic',fs=32)
A.route('visual-object1',[(1152,308),(1200,308)],'visual-projection','object-fusion-1')
A.route('noun-prototypes-object1',[(1320,192),(1320,256)],'prototypes','object-fusion-1')
A.label(1356,232,'Pₙ')
A.route('object1-object2',[(1320,360),(1320,408)],'object-fusion-1','object-fusion-2')
A.label(1380,388,'V₁, T₁',22)
A.route('object2-relation',[(1320,512),(1320,560)],'object-fusion-2','relation-fusion')
A.label(1368,540,'Vₒ',24)
A.route('verb-relation',[(1168,608),(1200,608)],'verb-prototype-port','relation-fusion')
# Per-image UPT prompts and class/IoU matching.
A.plain(184,636,'IMAGE-SPECIFIC PROMPTS',22,600,'start')
A.box('upt',200,656,224,104,'UPT',('Boxes · object labels','Actions · scores'),fill='semantic',tag='F',fs=30,ss=20)
A.box('prompt-construction',472,656,256,104,'Filter + construct',('Object / relation phrases',),fill='semantic',fs=27,ss=21)
A.box('prompt-text',776,656,288,104,'CLIP text encoder',('tₒ, tᵣ',),fill='semantic',tag='F',fs=28)
A.box('roi-matching',776,808,288,96,'Class + IoU ≥ 0.5',('i; ordered pair (s, o)',),fill='semantic',fs=27,ss=23)
A.box('linked-slots',1200,752,240,104,'Linked prompts',('Text + ROI + boxes','Valid O-links / R-links'),fill='semantic',fs=27,ss=22)
A.route('image-upt',[(100,416),(100,708),(200,708)],'image','upt')
A.route('upt-construction',[(424,696),(472,696)],'upt','prompt-construction')
A.route('construction-text',[(728,708),(776,708)],'prompt-construction','prompt-text')
A.route('upt-match',[(312,760),(312,856),(776,856)],'upt','roi-matching')
A.box('yolo-boxes-port',472,784,240,48,'YOLO boxes',(),fill='visual',fs=25)
A.route('yolo-match',[(712,808),(744,808),(744,832),(776,832)],'yolo-boxes-port','roi-matching')
A.route('text-linked',[(1064,708),(1136,708),(1136,784),(1200,784)],'prompt-text','linked-slots')
A.route('match-linked',[(1064,856),(1120,856),(1120,824),(1200,824)],'roi-matching','linked-slots')
# Adapters, explicitly on memory paths, detailed in panel b.
A.plain(1200,236,'CSA',22,600,'start')
A.box('object-links-port',1496,256,232,104,'O-links',('tₒ, i, bᵢ',),fill='semantic',fs=30)
A.box('object-adapter',1496,408,232,104,'Object Adapter',('Panel (b)',),fill='semantic',tag='L',fs=28)
A.box('relation-adapter',1496,560,232,104,'Relation Adapter',('Panel (b)',),fill='semantic',tag='L',fs=27)
A.route('vo-object-adapter',[(1424,448),(1496,448)],'object-fusion-2','object-adapter');A.label(1460,436,'Vₒ',24)
A.route('vr-relation-adapter',[(1424,600),(1496,600)],'relation-fusion','relation-adapter');A.label(1460,588,'Vᵣ',24)
A.route('slots-object-adapter',[(1612,360),(1612,408)],'object-links-port','object-adapter')
A.route('slots-relation-adapter',[(1440,824),(1476,824),(1476,628),(1496,628)],'linked-slots','relation-adapter')
A.box('prefix',1816,256,264,104,'Token prefix',('Word emb. e + position',),fill='decoder',tag='L',fs=28,ss=21)
A.box('decoder',1816,408,264,208,'Transformer ×3',('Masked SA → H','Object CA ∥ Relation CA','Q=H; K,V=Vₒ′ / Vᵣ′','gₒ(cₒ,e), gᵣ(cᵣ,e)','gₒ⊙cₒ+gᵣ⊙cᵣ → FFN'),fill='decoder',tag='L',fs=28,ss=21)
A.box('head-beam',2128,408,232,160,'Head → beam 5',('Next-token p(yₜ)','≤20 tokens','5 captions: y₁,…,y₅'),fill='decoder',tag='L',fs=27,ss=22)
A.route('prefix-decoder',[(1948,360),(1948,408)],'prefix','decoder')
A.route('object-adapter-decoder',[(1728,448),(1816,448)],'object-adapter','decoder');A.label(1772,436,'Vₒ′',24)
A.route('relation-adapter-decoder',[(1728,588),(1816,588)],'relation-adapter','decoder');A.label(1772,576,'Vᵣ′',24)
A.route('decoder-head',[(2080,488),(2128,488)],'decoder','head-beam')
A.box('rerank',2016,640,344,160,'Reranking',('D: Σ log p · C: CLIPScore','O: object consistency','S = 0.3D + 0.4C + 0.3O'),fill='rank',fs=32,ss=22)
A.box('rerank-evidence',1752,704,216,64,'I + YOLO labels',(),fill='visual',fs=25)
A.box('selected-caption',2016,848,344,64,'Caption: argmax S',(),fill='rank',fs=30)
A.route('beam-rerank',[(2244,568),(2244,640)],'head-beam','rerank')
A.route('evidence-rerank',[(1968,736),(2016,736)],'rerank-evidence','rerank')
A.route('rerank-caption',[(2188,800),(2188,848)],'rerank','selected-caption')
A.plain(1752,944,'D, C: min–max over 5 beams · O: normalized · ties: earlier beam',22,400)
# Panel b: the exact slot inputs, masks, and residual bypasses.
B=Canvas();B.panel(24,976,1456,776,'(b) Region-linked prompt adapters')
for kind,y,mem,out,inputlines in [('object',1120,'Vₒ','Vₒ′',('[tₒ; Vₒ[i]; bᵢ]','bᵢ ∈ ℝ⁴')),('relation',1440,'Vᵣ','Vᵣ′',('[tᵣ; Vₒ[s]; Vₒ[o]; bₛ; bₒ]','[bₛ; bₒ] ∈ ℝ⁸'))]:
 title='Object' if kind=='object' else 'Relation'
 B.plain(56,y-48,title+' adapter',28,600,'start')
 B.box(kind+'-slot-input',56,y,352,104,title+' slot',inputlines,fill='semantic',fs=30,ss=23)
 B.box(kind+'-mlp',456,y,128,104,'MLP',(),fill='semantic',tag='L',fs=30)
 B.box(kind+'-slots',632,y,176,104,'Slots',('K, V',),fill='semantic',stack=True,fs=30)
 B.box(kind+'-mha',856,y,216,104,'Masked MHA',('Q = '+mem,),fill='semantic',tag='L',fs=27)
 B.box(kind+'-wout',1120,y,128,104,'Wout',('Zero-init',),fill='semantic',tag='L',fs=28,ss=22)
 B.box(kind+'-memory',872,y-88,184,56,mem,(),fill='visual',fs=32)
 B.plus(kind+'-sum',1320,y+52)
 B.plain(1412,y+62,out,34,600)
 for src,dst,pts in [
  ('slot-input','mlp',[(408,y+52),(456,y+52)]),
  ('mlp','slots',[(584,y+52),(632,y+52)]),
  ('slots','mha',[(808,y+52),(856,y+52)]),
  ('mha','wout',[(1072,y+52),(1120,y+52)]),
  ('wout','sum',[(1248,y+52),(1302,y+52)]),
  ('memory','mha',[(916,y-32),(916,y)]),
  ('memory','sum',[(1012,y-32),(1012,y-16),(1320,y-16),(1320,y+34)]),
  ('sum',None,[(1338,y+52),(1380,y+52)])]:
  B.route(kind+'-'+src+'-'+str(dst),pts,kind+'-'+src,(kind+'-'+dst) if dst else None)
 mask='Mₒ[n,k] = 1 ⇔ n = iₖ' if kind=='object' else 'Mᵣ[n,k] = 1 ⇔ n ∈ {sₖ,oₖ}'
 B.box(kind+'-mask',736,y+144,416,48,mask,(),fill='plain',fs=24)
 B.route(kind+'-mask-mha',[(964,y+144),(964,y+104)],kind+'-mask',kind+'-mha',dash=True)
B.plain(56,1644,'Ordered (s,o) · relation slots use original Vₒ',24,400,'start')
B.plain(752,1716,'d = 512 · 8 heads · Wout = 0 at init · No valid slot ⇒ ΔV = 0',25)
# Panel c: TRAIN-only computation; reranking is absent from the reward.
D=Canvas();D.panel(1512,976,864,776,'(c) Training objective')
D.plain(1544,1060,'Stage 1 · Cross-entropy',29,600,'start')
D.box('train-prefix',1544,1096,320,88,'Shifted TRAIN caption',('Teacher forcing',),fill='decoder',fs=27,ss=24)
D.box('xe-model',1912,1096,432,88,'Same caption model',('pθ(y*ₜ | I, y*<ₜ)',),fill='decoder',tag='L',fs=30,ss=25)
D.route('teacher-forcing',[(1864,1140),(1912,1140)],'train-prefix','xe-model')
D.plain(1944,1232,'LCE = −mean log pθ(y*ₜ | I, y*<ₜ)',28)
D.plain(1544,1300,'Stage 2 · Self-critical training',29,600,'start')
D.box('scst-model',1544,1360,192,104,'Same model',('Same image I',),fill='decoder',fs=27,ss=23)
D.box('sampling',1784,1336,224,72,'Sampling → yₛ',(),fill='decoder',fs=27)
D.box('greedy',1784,1456,224,72,'Greedy → y_g',(),fill='decoder',fs=27)
D.box('advantage',2072,1376,272,128,'CIDEr advantage',('A = r(yₛ) − r(y_g)',),fill='plain',fs=27,ss=25)
D.box('train-refs',2072,1544,272,56,'Ytrain references',(),fill='semantic',fs=25)
D.route('model-sample',[(1736,1384),(1760,1384),(1760,1372),(1784,1372)],'scst-model','sampling')
D.route('model-greedy',[(1736,1440),(1760,1440),(1760,1492),(1784,1492)],'scst-model','greedy')
D.route('sample-advantage',[(2008,1372),(2040,1372),(2040,1412),(2072,1412)],'sampling','advantage')
D.route('greedy-advantage',[(2008,1492),(2056,1492),(2056,1468),(2072,1468)],'greedy','advantage')
D.route('refs-advantage',[(2208,1544),(2208,1504)],'train-refs','advantage')
D.plain(1544,1560,'r(y) = CIDEr(y, Ytrain)',25,400,'start')
D.plain(1944,1640,'LSCST = −mean_batch [stopgrad(A)',27)
D.plain(1944,1676,'× Σₜ log pθ(yₛ,ₜ | I, yₛ,<ₜ)]',27)
# This formula sits below both stages, without introducing an inference edge.
D.back.append('<rect x="1544" y="1696" width="800" height="48" rx="6" fill="#f1edf8" stroke="#82909b" stroke-width="1.2"/>')
D.plain(1944,1730,'L = LCE + 0.05 LSCST',30,600)
# Combined canvas and a bottom legend; separate panels retain original coordinates.
F=Canvas()
for cv in [A,B,D]:
 for key in ['back','edges','labels','nodes','meta','routes']: getattr(F,key).extend(getattr(cv,key))
for x,fill,label in [(64,'visual','Visual'),(344,'semantic','Semantic / prompts'),(824,'decoder','Decoder / training'),(1320,'rank','Reranking')]:
 F.nodes.append(f'<rect x="{x}" y="1784" width="28" height="24" rx="3" fill="{C[fill]}" stroke="{C["border"]}" stroke-width="1"/>')
 F.plain(x+44,1804,label,25,400,'start')
F.plain(1960,1804,'F  Frozen     L  Learned     ⊕  Residual sum',25)
write_html('upt-csa-architecture',F,(0,0,2400,1840))
write_html('panel-a-overall',A,(0,0,2400,968))
write_html('panel-b-adapters',B,(0,952,1504,824))
write_html('panel-c-training',D,(1488,952,912,824))
(ROOT/'diagram_geometry.json').write_text(json.dumps({'canvas':[2400,1840],'nodes':F.meta,'routes':F.routes,'palette':C},indent=2))
(ROOT/'style-profile.json').write_text(json.dumps({'profile':'paper-cv-pastel','source':'User-specified white scientific-figure style, four pastel categories.','font_source':'system','font_family':'Arial, Liberation Sans, sans-serif','palette':C,'instruction_priority':'Explicit user styling overrides the default skin and per-diagram node cap; three requested panels and separately exported panels preserve detail.'},indent=2))
print('Built 3-panel HTML figure and separate panel sources.')
