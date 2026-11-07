"""Functional-name overview and two linked mechanism insets; no model execution."""
from pathlib import Path
import html,base64,json
ROOT=Path(__file__).resolve().parent
PRIMITIVES=ROOT.parents[1]/'upt_csa_architecture_2026-11-06/build_figure.py'
exec(compile(PRIMITIVES.read_text().split('\ndef write_html')[0].replace('step=ss+8','step=ss+4'),'figure_primitives','exec'),globals())
# User revision: straight orthogonal connectors, without rounded elbows.
def straight_route(self,id,points,src=None,dst=None,dash=False,color=None):
 p=[tuple(v) for v in points]
 for a,b in zip(p,p[1:]):assert a[0]==b[0] or a[1]==b[1],(id,a,b)
 d='M '+str(p[0][0])+','+str(p[0][1])+''.join(' L '+str(x)+','+str(y) for x,y in p[1:])
 ds=' stroke-dasharray="7 5"' if dash else ''
 self.edges.append(f'<path data-edge="{id}" d="{d}" fill="none" stroke="{color or C["line"]}" stroke-width="1.7"{ds} marker-end="url(#arrow)"/>')
 self.routes.append(dict(id=id,points=p,src=src,dst=dst))
Canvas.route=straight_route
C.update({'visual':'#EDF3F8','semantic':'#FCEBDD','decoder':'#EDF3F8','rank':'#EDF3F8','relation':'#E2F1F5','border':'#8a969e','line':'#56616B','ink':'#263238'})
c=Canvas();ORANGE='#FCEBDD';BLUE='#E2F1F5'
def frame(id,x,y,w,h,title,color='#a8b4bc',fs=28,dashed=False):
 ds=' stroke-dasharray="10 7"' if dashed else ''
 c.back.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="#fff" stroke="{color}" stroke-width="1.4"{ds}/>')
 c.back.append(c.text(x+24,y+42,title,fs,600,'start'))
def token(id,x,y,label,color,label_side='above',n=3):
 for j in range(n):c.nodes.append(f'<rect x="{x}" y="{y+j*12}" width="20" height="8" fill="{color}" stroke="#65727b" stroke-width=".7"/>')
 if label_side=='above':c.plain(x+10,y-12,label,25,600)
 elif label_side=='right':c.plain(x+36,y+24,label,25,600,'start')
 else:c.plain(x+10,y+60,label,24,600)
 c.meta.append(dict(id=id,x=x,y=y,w=20,h=n*12-4,kind='tokens'))
def edge(id,points,src=None,dst=None,dash=False):c.route(id,points,src,dst,dash)
def leader(id,pts):
 # Zoom leader, not a data-flow arrow.
 edge(id,pts,dash=True)
 c.edges[-1]=c.edges[-1].replace(' marker-end="url(#arrow)"','').replace('stroke-width="1.7"','stroke-width="1.2"')
# Left: one image only, parallel visual feature and detection paths.
frame('visual-zone',24,160,456,1184,'Visual region extraction',fs=29)
c.box('clip-visual',48,688,200,80,'CLIP ViT-B/16',('Frozen visual encoder',),fill='visual',fs=24,ss=19)
c.box('yolo',280,688,176,80,'YOLO',('Detector',),fill='visual',fs=29,ss=23)
c.box('patches',48,552,200,80,'Patch features',('14 × 14 × 768',),fill='visual',fs=25,ss=22,stack=True)
c.box('boxes',280,552,176,80,'Bounding',('boxes B',),fill='visual',fs=25,ss=24)
c.box('roi-pooling',160,416,256,80,'ROI pooling',(),fill='visual',fs=30)
c.box('projection',160,280,256,80,'Linear projection',('768 → 512',),fill='visual',fs=27,ss=25)
photo=ROOT.parents[1]/'csa_pipeline_2026-11-02/v3/assets/COCO_train2014_000000263974.jpg'
data=base64.b64encode(photo.read_bytes()).decode()
c.nodes.append(f'<g id="input-image"><image x="80" y="1016" width="264" height="198" href="data:image/jpeg;base64,{data}"/>')
c.nodes.append('</g>');c.meta.append(dict(id='input-image',x=80,y=1016,w=264,h=198,kind='image'))
c.plain(212,1256,'Input image I',30,600)
edge('image-clip',[(148,1016),(148,768)],'input-image','clip-visual')
edge('image-yolo',[(316,1016),(316,984),(368,984),(368,768)],'input-image','yolo')
edge('clip-patches',[(148,688),(148,632)],'clip-visual','patches')
edge('yolo-boxes',[(368,688),(368,632)],'yolo','boxes')
edge('patches-pool',[(148,552),(148,520),(224,520),(224,496)],'patches','roi-pooling')
edge('boxes-pool',[(368,552),(368,496)],'boxes','roi-pooling')
edge('pool-projection',[(288,416),(288,360)],'roi-pooling','projection')
token('v-main',504,224,'V',C['visual'])
c.plain(544,276,'N × 512',16)
edge('projection-v',[(416,320),(514,320),(514,256)],'projection','v-main')
# Main chain: functional names only, with explicit separate memory rails.
c.box('semantic-overview',576,180,320,112,'Cascaded semantic',('fusion',),fill='semantic',fs=32,ss=31)
c.box('prompt-overview',1104,180,336,112,'Region-linked prompt',('conditioning',),fill='semantic',fs=29,ss=31)
edge('v-semantic',[(524,236),(576,236)],'v-main','semantic-overview')
for key,y,color,label in [('vo',212,ORANGE,'Vₒ'),('vr',260,BLUE,'Vᵣ')]:
 token(key+'-main',968,y-16,'',color)
 c.plain(1032,y-12,label,25,600)
 edge(key+'-from-fusion',[(896,y),(968,y)],'semantic-overview',key+'-main')
 edge(key+'-into-conditioning',[(988,y),(1104,y)],key+'-main','prompt-overview')
# Two mechanism insets, linked by dashed guides rather than data arrows.
frame('semantic-inset',552,416,520,928,'Semantic bank–region alignment',color='#aab7c0',fs=25,dashed=True)
frame('prompt-inset',1112,416,520,928,'Image-specific prompt–region alignment',color='#aab7c0',fs=24,dashed=True)
leader('semantic-zoom',[(736,292),(736,352),(812,352),(812,416)])
leader('prompt-zoom',[(1272,292),(1272,352),(1384,352),(1384,416)])
# Inset 1: TRAIN-only bank preparation and two distinct memories.
c.box('train-lemmas',576,488,472,40,'TRAIN captions → noun / verb lemmas',(),fill='plain',fs=23)
c.box('bank-text-encoder',672,552,280,56,'CLIP text encoder',(),fill='visual',fs=27)
edge('lemmas-encoder',[(812,528),(812,552)],'train-lemmas','bank-text-encoder')
c.box('noun-bank',576,656,208,64,'TRAIN noun bank',('Fixed embeddings',),fill='semantic',fs=23,ss=20)
c.box('verb-bank',824,656,224,64,'TRAIN verb bank',('Fixed embeddings',),fill='relation',fs=23,ss=20)
edge('encoder-noun-bank',[(752,608),(752,632),(680,632),(680,656)],'bank-text-encoder','noun-bank')
edge('encoder-verb-bank',[(872,608),(872,632),(936,632),(936,656)],'bank-text-encoder','verb-bank')
for key,x in [('noun',576),('verb',824)]:
 w=208 if key=='noun' else 224
 c.box(key+'-aggregation',x,760,w,64,'Learned semantic',('aggregation',),fill='semantic' if key=='noun' else 'relation',fs=23,ss=23)
 centre=x+w/2
 edge(key+'-bank-aggregate',[(centre,720),(centre,760)],key+'-bank',key+'-aggregation')
 token(key+'-prototypes',centre-10,864,'Pₙ' if key=='noun' else 'Pᵥ',ORANGE if key=='noun' else BLUE,'right')
 edge(key+'-aggregate-prototypes',[(centre,824),(centre,864)],key+'-aggregation',key+'-prototypes')
c.plain(736,928,'150 × 512',18)
c.plain(936,952,'150 × 512',18)
c.box('object-fusion-1',624,944,208,64,'Object fusion 1',(),fill='semantic',fs=27)
c.box('object-fusion-2',624,1056,208,64,'Object fusion 2',(),fill='semantic',fs=27)
c.box('relation-fusion',824,1152,224,64,'Relation fusion',(),fill='relation',fs=28)
token('v-inset',576,960,'V',C['visual'])
edge('v-object1',[(596,976),(624,976)],'v-inset','object-fusion-1')
edge('noun-object1',[(680,896),(680,944)],'noun-prototypes','object-fusion-1')
edge('object1-object2',[(728,1008),(728,1056)],'object-fusion-1','object-fusion-2')
c.label(784,1040,'V₁, T₁',21)
token('vo-inset',864,1072,'Vₒ',ORANGE,'right')
edge('object2-vo',[(832,1088),(864,1088)],'object-fusion-2','vo-inset')
edge('vo-relation',[(728,1120),(728,1184),(824,1184)],'object-fusion-2','relation-fusion')
c.label(776,1172,'Vₒ',22)
edge('verb-relation',[(936,896),(936,912),(1000,912),(1000,1152)],'verb-prototypes','relation-fusion')
token('vr-inset',926,1256,'Vᵣ',BLUE,'right')
edge('relation-vr',[(936,1216),(936,1256)],'relation-fusion','vr-inset')
c.plain(812,1320,'Vₒ, Vᵣ : N × 512',21)
# Inset 2: image-specific text and links are separate, then jointly form valid slots.
c.box('upt',1136,504,224,64,'Image I → UPT',('Interaction detector',),fill='plain',fs=25,ss=22)
c.box('region-matching',1416,520,176,64,'Region matching',('Valid ROI links',),fill='plain',fs=20,ss=20)
c.plain(1504,488,'YOLO boxes',21)
edge('yolo-region-matching',[(1504,496),(1504,520)],None,'region-matching')
edge('upt-matching',[(1360,536),(1416,536)],'upt','region-matching')
c.box('object-phrases',1136,616,104,64,'Object',('phrases',),fill='semantic',fs=21,ss=21)
c.box('relation-phrases',1256,616,104,64,'Relation',('phrases',),fill='relation',fs=20,ss=21)
c.box('clip-text',1136,728,224,64,'CLIP text encoder',('Frozen',),fill='visual',fs=24,ss=23)
edge('upt-object-phrases',[(1216,568),(1216,592),(1188,592),(1188,616)],'upt','object-phrases')
edge('upt-relation-phrases',[(1280,568),(1280,592),(1308,592),(1308,616)],'upt','relation-phrases')
edge('object-phrases-clip',[(1188,680),(1188,704),(1216,704),(1216,728)],'object-phrases','clip-text')
edge('relation-phrases-clip',[(1308,680),(1308,704),(1280,704),(1280,728)],'relation-phrases','clip-text')
# A short two-color embedding row is a shared encoder output, not an attention map.
token('relation-text',1214,824,'',BLUE,'right');token('object-text',1250,824,'',ORANGE,'right')
edge('clip-embeddings',[(1248,792),(1248,812)],'clip-text',None)
c.plain(1192,844,'tᵣ',21)
c.plain(1294,844,'tₒ',21)
c.plain(1428,848,'Prompt embeddings',21)
for key,y,mem,out,color in [('object',900,'Vₒ','Vₒ′',ORANGE),('relation',1060,'Vᵣ','Vᵣ′',BLUE)]:
 title='Object prompt adapter' if key=='object' else 'Relation prompt adapter'
 c.back.append(f'<rect x="1128" y="{y}" width="480" height="128" rx="6" fill="#ffffff" stroke="#c3c8ca" stroke-width="1"/>')
 c.nodes.append(c.text(1144,y+24,title,20,600,'start'))
 c.box(key+'-slots',1136,y+44,144,64,'Linked slots',(),fill='semantic' if key=='object' else 'relation',fs=22)
 c.box(key+'-attention',1320,y+44,128,64,'Masked',('attention',),fill='semantic' if key=='object' else 'relation',fs=23,ss=22)
 c.plus(key+'-sum',1504,y+76)
 edge(key+'-slots-attention',[(1280,y+76),(1320,y+76)],key+'-slots',key+'-attention')
 edge(key+'-attention-residual',[(1448,y+76),(1486,y+76)],key+'-attention',key+'-sum')
 token(key+'-updated',1568,y+60,out,color,'above')
 edge(key+'-updated-memory',[(1522,y+76),(1568,y+76)],key+'-sum',key+'-updated')
 # One memory supplies both the visual query and a separate residual path.
 token(key+'-query-memory',1464,y+8,mem,color)
 edge(key+'-visual-query',[(1464,y+24),(1384,y+24),(1384,y+44)],key+'-query-memory',key+'-attention')
 edge(key+'-residual',[(1484,y+32),(1504,y+32),(1504,y+58)],key+'-query-memory',key+'-sum')
# Link routing uses two separate ports and never passes behind an adapter box.
edge('matching-object-slots',[(1592,552),(1616,552),(1616,1020),(1248,1020),(1248,1008)],'region-matching','object-slots')
edge('matching-relation-slots',[(1592,528),(1624,528),(1624,1200),(1248,1200),(1248,1168)],'region-matching','relation-slots')
edge('object-text-slots',[(1260,856),(1260,872),(1120,872),(1120,972),(1136,972)],'object-text','object-slots')
edge('relation-text-slots',[(1224,856),(1224,864),(1116,864),(1116,1132),(1136,1132)],'relation-text','relation-slots')
c.plain(1248,1048,'Slots: original Vₒ',22)
# Right: actual decoder computation, outside the two mechanism zooms.
c.plain(1856,112,'Caption generation and selection',30,600)
frame('decoder',1680,180,288,336,'Transformer decoder',fs=24)
c.box('gated-ffn',1696,236,256,64,'Gated fusion + FFN',(),fill='decoder',fs=25)
c.box('parallel-attention',1696,332,256,88,'Parallel',('cross-attention',),fill='decoder',fs=28,ss=27)
c.box('masked-self-attention',1696,452,256,48,'Masked self-attention',(),fill='decoder',fs=23)
c.plain(1944,512,'×3',23,600)
c.box('embedding',1696,576,256,64,'Word embedding',('Fixed position',),fill='decoder',fs=24,ss=24)
c.box('prefix',1696,696,256,64,'Token prefix',(),fill='plain',fs=29)
edge('prefix-embedding',[(1824,696),(1824,640)],'prefix','embedding')
edge('embedding-decoder',[(1824,576),(1824,500)],'embedding','masked-self-attention')
edge('self-parallel',[(1824,452),(1824,420)],'masked-self-attention','parallel-attention')
edge('parallel-gated',[(1824,332),(1824,300)],'parallel-attention','gated-ffn')
for key,y,colour,label in [('vo',344,ORANGE,'Vₒ′'),('vr',400,BLUE,'Vᵣ′')]:
 token(key+'-decoder',1592,y-16,'',colour)
 c.plain(1644,y-12,label,25,600)
 pts=[(1440,212),(1504,212),(1504,344),(1592,344)] if key=='vo' else [(1440,260),(1472,260),(1472,400),(1592,400)]
 edge(key+'-conditioned-output',pts,'prompt-overview',key+'-decoder')
 edge(key+'-decoder-input',[(1612,y),(1696,y)],key+'-decoder','parallel-attention')
c.box('vocab',2016,236,224,64,'Vocabulary projection',('+ softmax',),fill='decoder',fs=20,ss=24)
c.box('beam',2304,236,224,64,'Beam search',('K = 5',),fill='decoder',fs=25,ss=23)
edge('decoder-vocab',[(1952,268),(2016,268)],'gated-ffn','vocab')
edge('vocab-beam',[(2240,268),(2304,268)],'vocab','beam')
# Five paper/token glyphs with symbolic candidate labels.
for j in range(5):
 x=2304+j*44
 c.nodes.append(f'<rect x="{x}" y="344" width="36" height="48" rx="3" fill="#fff" stroke="#8c969e" stroke-width="1"/>')
 c.plain(x+18,375,'y'+str(j+1),21)
c.meta.append(dict(id='candidates',x=2304,y=344,w=212,h=48,kind='tokens'))
c.plain(2040,376,'Five candidate captions',22,400,'start')
edge('beam-candidates',[(2416,300),(2416,344)],'beam','candidates')
frame('rerank-frame',2016,440,512,288,'Caption reranking',fs=29)
for id,x,a,b in [('likelihood',2032,'Decoder','likelihood D'),('alignment',2200,'Image–text','alignment C'),('consistency',2368,'Object','consistency O')]:
 c.box(id,x,520,144,88,a,(b,),fill='rank',fs=23,ss=21)
c.box('select',2144,648,256,72,'Weighted scoring',('and selection',),fill='rank',fs=25,ss=24)
edge('likelihood-select',[(2104,608),(2104,624),(2208,624),(2208,648)],'likelihood','select')
edge('alignment-select',[(2272,608),(2272,648)],'alignment','select')
edge('consistency-select',[(2440,608),(2440,624),(2336,624),(2336,648)],'consistency','select')
edge('candidate-rerank',[(2416,392),(2416,440)],'candidates',None)
c.plain(2032,808,'I + YOLO labels',23,600,'start')
edge('evidence-rerank',[(2096,772),(2096,728)],None,None)
c.box('final-caption',2208,824,320,72,'Final caption',(),fill='rank',fs=32)
edge('rerank-final',[(2368,728),(2368,824)],None,'final-caption')
# Training is explicitly outside the inference graph.
c.plain(1280,1392,'Training: cross-entropy pretraining followed by SCST + cross-entropy.',28)
# Icon badges are color-coded vector paths, anchored at the module's upper-right corner.
def icon(kind,x,y,module=None):
 attrs=f' data-icon-kind="{kind}" data-module="{module or "legend"}"'
 if kind=='frozen':
  paths=['M10 0 V20','M1.34 5 L18.66 15','M1.34 15 L18.66 5',
   'M10 3 L7 5 M10 3 L13 5 M10 17 L7 15 M10 17 L13 15',
   'M3.94 6.5 L4.7 10 M3.94 6.5 L7.5 6 M16.06 13.5 L12.5 14 M16.06 13.5 L15.3 10',
   'M3.94 13.5 L7.5 14 M3.94 13.5 L4.7 10 M16.06 6.5 L15.3 10 M16.06 6.5 L12.5 6']
  glyph=''.join(f'<path d="{p}"/>' for p in paths)
 else:
  glyph='<path d="M11 0 C15 7 7 8 13 12 C15 10 16 8 16 6 C22 14 19 23 10 23 C1 23 -1 15 4 9 C4 13 6 14 7 13 C3 7 11 6 11 0 Z"/><path d="M10 14 C15 18 14 21 10 21 C6 21 6 18 10 14 Z"/>'
 stroke='#1677B8' if kind=='frozen' else '#D95F1A'
 fill='none' if kind=='frozen' else '#F6A23A'
 return f'<g{attrs} transform="translate({x} {y})" fill="{fill}" stroke="{stroke}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{glyph}</g>'
frozen=['clip-visual','yolo','bank-text-encoder','upt','clip-text','alignment']
learned=['projection','semantic-overview','prompt-overview','noun-aggregation','verb-aggregation',
 'object-fusion-1','object-fusion-2','relation-fusion','embedding','vocab']
node_map={n['id']:n for n in c.meta}
for key in frozen+learned:
 n=node_map[key]
 c.nodes.append(icon('frozen' if key in frozen else 'trainable',n['x']+n['w']-22,n['y']-22,key))
# Decoder and both complete adapters carry one icon each, not one per internal op.
c.nodes.append(icon('trainable',1946,158,'decoder'))
for key,y in [('object',900),('relation',1060)]:c.nodes.append(icon('trainable',1586,y-22,key+'-adapter'))
for x,kind,label in [(912,'frozen','Frozen parameters'),(1320,'trainable','Trainable parameters')]:
 c.nodes.append(icon(kind,x,1420));c.plain(x+36,1440,label,24,400,'start')
# Shape distinguishes data from processing, independently of color.
import re
for j,item in enumerate(c.nodes):
 for key in ['patches','boxes','noun-bank','verb-bank','train-lemmas','object-phrases','relation-phrases','prefix','final-caption']:
  if item.startswith('<g id="'+key+'">'):item=item.replace('rx="6"','rx="0"')
 # Five candidate data cards also have square corners.
 if 'y="344" width="36" height="48"' in item:item=item.replace('rx="3"','rx="0"')
 c.nodes[j]=item
slug='captioning-pipeline'
svg=c.svg(slug,(0,0,2560,1472))
caption='Kiến trúc chú thích ảnh kết hợp bank ngữ nghĩa TRAIN và prompt UPT riêng ảnh. Hai adapter nhận memory làm visual query và có đường residual riêng; relation slot dùng Vo gốc. Decoder đọc riêng Vo′/Vr′, sinh năm ứng viên rồi reranking. Ảnh I không vẽ bbox; tokens chỉ minh họa. Icon tuyết/lửa mô tả frozen/trainable khi huấn luyện; khi suy luận mọi trọng số giữ nguyên. Lửa ở embedding và vocabulary chỉ áp dụng cho word embedding/projection, không áp dụng cho fixed positional encoding/softmax.'
doc='<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Image captioning pipeline</title><style>body{margin:0;background:white;font-family:Arial,"Liberation Sans",sans-serif}.diagram-container{width:100%;overflow-x:auto}svg{width:100%;min-width:2560px;display:block}figcaption{max-width:1100px;margin:24px auto;padding:0 24px;line-height:1.6;font-size:16px}@media print{.diagram-container{overflow:visible}svg{min-width:0}figcaption{display:none}}</style></head><body><main><div class="diagram-container">'+svg+'</div><figcaption lang="vi">'+html.escape(caption)+'</figcaption></main></body></html>'
(ROOT/(slug+'.html')).write_text(doc)
(ROOT/'diagram_geometry.json').write_text(json.dumps({'canvas':[2560,1472],'nodes':c.meta,'routes':c.routes},indent=2))
print('Built functional-name pipeline with two zoom insets.')
