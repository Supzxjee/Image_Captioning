"""Functional-name overview and two linked mechanism insets; no model execution."""
from pathlib import Path
import html,base64,json
ROOT=Path(__file__).resolve().parent
PRIMITIVES=ROOT.parent/'upt_csa_architecture_2026-11-06/build_figure.py'
exec(compile(PRIMITIVES.read_text().split('\ndef write_html')[0].replace('step=ss+8','step=ss+4'),'figure_primitives','exec'),globals())
C.update({'visual':'#eef4f8','semantic':'#f2f6e9','decoder':'#f0f0f7','rank':'#fbf1df','border':'#697780','line':'#56646c','ink':'#253039'})
c=Canvas();ORANGE='#c88c4d';BLUE='#448fab'
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
frame('visual-zone',24,160,456,1056,'Visual region extraction',fs=29)
c.box('clip-visual',48,688,200,80,'CLIP visual',('Frozen encoder',),fill='visual',fs=27,ss=23)
c.box('yolo',280,688,176,80,'YOLO',('Detector',),fill='visual',fs=29,ss=23)
c.box('patches',48,552,200,80,'Patch features',(),fill='visual',fs=26,stack=True)
c.box('boxes',280,552,176,80,'Boxes',(),fill='visual',fs=29)
c.box('roi-pooling',160,416,256,80,'ROI pooling',(),fill='visual',fs=30)
c.box('projection',160,280,256,80,'Linear projection',(),fill='visual',fs=28)
photo=ROOT.parent/'csa_pipeline_2026-11-02/v3/assets/COCO_train2014_000000263974.jpg'
data=base64.b64encode(photo.read_bytes()).decode()
c.nodes.append(f'<g id="input-image"><image x="80" y="888" width="264" height="198" href="data:image/jpeg;base64,{data}"/>')
for (x,y,w,h),color in [((126,912,62,96),ORANGE),((130,980,90,72),BLUE),((232,926,68,122),'#a96d7b')]:
 c.nodes.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="{color}" stroke-width="2"/>')
c.nodes.append('</g>');c.meta.append(dict(id='input-image',x=80,y=888,w=264,h=198,kind='image'))
c.plain(212,1128,'Input image I',30,600)
edge('image-clip',[(148,888),(148,768)],'input-image','clip-visual')
edge('image-yolo',[(316,888),(368,888),(368,768)],'input-image','yolo')
edge('clip-patches',[(148,688),(148,632)],'clip-visual','patches')
edge('yolo-boxes',[(368,688),(368,632)],'yolo','boxes')
edge('patches-pool',[(148,552),(148,520),(224,520),(224,496)],'patches','roi-pooling')
edge('boxes-pool',[(368,552),(368,496)],'boxes','roi-pooling')
edge('pool-projection',[(288,416),(288,360)],'roi-pooling','projection')
token('v-main',504,224,'V',BLUE)
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
frame('semantic-inset',552,416,520,800,'Semantic bank–region alignment',color='#9cae88',fs=25,dashed=True)
frame('prompt-inset',1112,416,520,800,'Image-specific prompt–region alignment',color='#c1a17a',fs=24,dashed=True)
leader('semantic-zoom',[(736,292),(736,352),(812,352),(812,416)])
leader('prompt-zoom',[(1272,292),(1272,352),(1384,352),(1384,416)])
# Inset 1: fixed TRAIN text banks, two aggregations, three fusions.
c.box('noun-bank',576,504,208,80,'TRAIN noun bank',('CLIP text · frozen',),fill='semantic',fs=23,ss=22)
c.box('verb-bank',824,504,224,80,'TRAIN verb bank',('CLIP text · frozen',),fill='semantic',fs=25,ss=22)
for key,x in [('noun',576),('verb',824)]:
 w=208 if key=='noun' else 224
 c.box(key+'-aggregation',x,632,w,80,'Learned semantic',('aggregation',),fill='semantic',fs=23,ss=25)
 centre=x+w/2
 edge(key+'-bank-aggregate',[(centre,584),(centre,632)],key+'-bank',key+'-aggregation')
 token(key+'-prototypes',centre-10,752,'Pₙ' if key=='noun' else 'Pᵥ',ORANGE if key=='noun' else BLUE,'right')
 edge(key+'-aggregate-prototypes',[(centre,712),(centre,752)],key+'-aggregation',key+'-prototypes')
c.box('object-fusion-1',624,824,208,64,'Object fusion 1',(),fill='semantic',fs=27)
c.box('object-fusion-2',624,928,208,64,'Object fusion 2',(),fill='semantic',fs=27)
c.box('relation-fusion',824,1072,224,64,'Relation fusion',(),fill='semantic',fs=28)
token('v-inset',576,840,'V',BLUE)
edge('v-object1',[(596,856),(624,856)],'v-inset','object-fusion-1')
edge('noun-object1',[(680,784),(680,824)],'noun-prototypes','object-fusion-1')
edge('object1-object2',[(728,888),(728,928)],'object-fusion-1','object-fusion-2')
c.label(784,916,'V₁, T₁',21)
token('vo-inset',864,944,'Vₒ',ORANGE,'right')
edge('object2-vo',[(832,960),(864,960)],'object-fusion-2','vo-inset')
edge('vo-relation',[(874,976),(874,1072)],'vo-inset','relation-fusion')
edge('verb-relation',[(936,784),(936,1072)],'verb-prototypes','relation-fusion')
token('vr-inset',926,1172,'Vᵣ',BLUE,'right')
edge('relation-vr',[(936,1136),(936,1172)],'relation-fusion','vr-inset')
# Inset 2: image-specific text and links are separate, then jointly form valid slots.
c.box('upt',1136,504,224,64,'Image I → UPT',('Frozen detector',),fill='plain',fs=25,ss=22)
c.box('region-matching',1416,520,176,64,'Region matching',('Valid ROI links',),fill='plain',fs=20,ss=20)
c.plain(1504,488,'YOLO boxes',21)
edge('yolo-region-matching',[(1504,496),(1504,520)],None,'region-matching')
edge('upt-matching',[(1360,536),(1416,536)],'upt','region-matching')
c.box('phrases',1136,616,224,64,'Object / relation',('phrases',),fill='semantic',fs=25,ss=24)
c.box('clip-text',1136,728,224,64,'CLIP text encoder',('Frozen',),fill='semantic',fs=24,ss=23)
edge('upt-phrases',[(1248,568),(1248,616)],'upt','phrases')
edge('phrases-clip',[(1248,680),(1248,728)],'phrases','clip-text')
# A short two-color embedding row is a shared encoder output, not an attention map.
token('relation-text',1214,824,'',BLUE,'right');token('object-text',1250,824,'',ORANGE,'right')
edge('clip-embeddings',[(1248,792),(1248,812)],'clip-text',None)
c.plain(1400,848,'Prompt embeddings',21)
for key,y,mem,out,color in [('object',900,'Vₒ','Vₒ′',ORANGE),('relation',1060,'Vᵣ','Vᵣ′',BLUE)]:
 title='Object prompt adapter' if key=='object' else 'Relation prompt adapter'
 c.back.append(f'<rect x="1128" y="{y}" width="480" height="128" rx="6" fill="#fcfcfa" stroke="#c3c8ca" stroke-width="1"/>')
 c.nodes.append(c.text(1144,y+24,title,23,600,'start'))
 c.box(key+'-slots',1136,y+44,144,64,'Linked slots',(),fill='semantic',fs=22)
 c.box(key+'-attention',1320,y+44,128,64,'Masked',('attention',),fill='semantic',fs=23,ss=22)
 c.plus(key+'-sum',1504,y+76)
 edge(key+'-slots-attention',[(1280,y+76),(1320,y+76)],key+'-slots',key+'-attention')
 edge(key+'-attention-residual',[(1448,y+76),(1486,y+76)],key+'-attention',key+'-sum')
 token(key+'-updated',1568,y+60,out,color,'above')
 edge(key+'-updated-memory',[(1522,y+76),(1568,y+76)],key+'-sum',key+'-updated')
 # Original memory bypass into the residual sum, represented by a named tensor.
 c.plain(1504,y+24,mem,24,600)
 edge(key+'-residual',[(1504,y+30),(1504,y+58)],None,key+'-sum')
# Link routing uses two separate ports and never passes behind an adapter box.
edge('matching-object-slots',[(1592,552),(1616,552),(1616,1020),(1248,1020),(1248,1008)],'region-matching','object-slots')
edge('matching-relation-slots',[(1592,528),(1624,528),(1624,1200),(1248,1200),(1248,1168)],'region-matching','relation-slots')
edge('object-text-slots',[(1260,856),(1260,872),(1120,872),(1120,972),(1136,972)],'object-text','object-slots')
edge('relation-text-slots',[(1224,856),(1224,864),(1116,864),(1116,1132),(1136,1132)],'relation-text','relation-slots')
c.plain(1372,1048,'Linked regions use original Vₒ',22)
# Right: actual decoder computation, outside the two mechanism zooms.
c.plain(1856,112,'Caption generation and selection',30,600)
frame('decoder',1680,180,288,336,'Transformer decoder',fs=24)
c.box('gated-ffn',1696,236,256,64,'Gated fusion + FFN',(),fill='decoder',fs=25)
c.box('parallel-attention',1696,332,256,88,'Parallel',('cross-attention',),fill='decoder',fs=28,ss=27)
c.box('masked-self-attention',1696,452,256,48,'Masked self-attention',(),fill='decoder',fs=23)
c.plain(1944,512,'×3',23,600)
c.box('embedding',1696,576,256,64,'Embedding +',('positional encoding',),fill='decoder',fs=24,ss=24)
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
for id,x,a,b in [('likelihood',2032,'Decoder','likelihood'),('alignment',2200,'Image–text','alignment'),('consistency',2368,'Object','consistency')]:
 c.box(id,x,520,144,88,a,(b,),fill='rank',fs=23,ss=23)
c.box('select',2144,648,256,56,'Select highest score',(),fill='rank',fs=25)
edge('candidate-rerank',[(2416,392),(2416,440)],'candidates',None)
c.plain(2032,808,'I + YOLO labels',23,600,'start')
edge('evidence-rerank',[(2096,772),(2096,728)],None,None)
c.box('final-caption',2208,824,320,72,'Final caption',(),fill='rank',fs=32)
edge('rerank-final',[(2368,728),(2368,824)],None,'final-caption')
# Training is explicitly outside the inference graph.
c.plain(1280,1288,'Training: cross-entropy pretraining followed by SCST + cross-entropy.',28)
slug='captioning-pipeline'
svg=c.svg(slug,(0,0,2560,1344))
caption='Pipeline chú thích ảnh với hai nguồn ngữ nghĩa: bank danh từ/động từ toàn cục từ TRAIN và prompt riêng ảnh từ UPT. Fusion theo cascade tạo hai memory; hai prompt adapter cập nhật chúng bằng slot gắn vùng trước khi decoder đọc riêng từng memory. Beam search sinh năm ứng viên và reranking chọn caption bằng ba thành phần điểm. Các cổng memory lặp tên biểu thị cùng tensor. Ảnh và box chỉ minh họa, không phải đầu ra thực nghiệm đã xác minh.'
doc='<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Image captioning pipeline</title><style>body{margin:0;background:white;font-family:Arial,"Liberation Sans",sans-serif}.diagram-container{width:100%;overflow-x:auto}svg{width:100%;min-width:2560px;display:block}figcaption{max-width:1100px;margin:24px auto;padding:0 24px;line-height:1.6;font-size:16px}@media print{.diagram-container{overflow:visible}svg{min-width:0}figcaption{display:none}}</style></head><body><main><div class="diagram-container">'+svg+'</div><figcaption lang="vi">'+html.escape(caption)+'</figcaption></main></body></html>'
(ROOT/(slug+'.html')).write_text(doc)
(ROOT/'diagram_geometry.json').write_text(json.dumps({'canvas':[2560,1344],'nodes':c.meta,'routes':c.routes},indent=2))
print('Built functional-name pipeline with two zoom insets.')
