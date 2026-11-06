"""Reference-inspired visual grammar; exact project modules, no model execution."""
from pathlib import Path
import base64,json,html
ROOT=Path(__file__).resolve().parent
# Reuse drawing primitives without executing the previous figure's build steps.
common=(ROOT.parent/'build_figure.py').read_text().split('\ndef write_html')[0]
exec(compile(common,'shared_drawing_primitives','exec'),globals())
C.update({'visual':'#eaf4f7','semantic':'#fff1e5','decoder':'#edf4f6','rank':'#f9f0de','border':'#51585d','line':'#454d53','ink':'#22282c'})
cv=Canvas()
PHOTO=ROOT.parents[1]/'csa_pipeline_2026-11-02/v3/assets/COCO_train2014_000000263974.jpg'
data=base64.b64encode(PHOTO.read_bytes()).decode()
ORANGE='#d99548'; BLUE='#409dc0'
# Broad dashed semantic zones mirror the supplied reference; the two sources stay distinct.
for x,w,color,title in [(352,728,ORANGE,'Object prompt adapter'),(1112,624,BLUE,'Relation prompt adapter')]:
 cv.back.append(f'<rect x="{x}" y="392" width="{w}" height="656" fill="#fff" stroke="{color}" stroke-width="2" stroke-dasharray="12 7"/>')
 cv.back.append(cv.text(x+24,424,title,26,600,'start'))
cv.back.append('<rect x="520" y="120" width="1200" height="212" fill="#fafafa" stroke="#8e9397" stroke-width="1.5" stroke-dasharray="8 5"/>')
cv.plain(1104,112,'Cascade Semantic Alignment (CSA)',28,600)
# Frozen global banks appear above the encoder, distinct from image-conditioned phrases.
cv.box('noun-bank',536,24,736,64,'TRAIN noun bank → learned aggregation → Pₙ (150)',(),fill='semantic',fs=25)
cv.box('verb-bank',1368,24,320,64,'Verb bank → Agg → Pᵥ (150)',(),fill='visual',fs=22)
cv.plain(328,48,'Frozen CLIP text',21,400)
cv.plain(328,76,'TRAIN lemmas',21,400)
# Inline photos are illustrations, not recorded detector or generated-caption results.
def photo(id,x,y,w,h,boxes=False):
 cv.nodes.append(f'<g id="{id}"><image x="{x}" y="{y}" width="{w}" height="{h}" href="data:image/jpeg;base64,{data}" preserveAspectRatio="xMidYMid slice"/>')
 if boxes:
  # Deliberately illustrative, hand-positioned ROI overlays; provenance in caption.
  for b,color in [((.14,.10,.23,.49),'#df6b57'),((.19,.42,.29,.41),'#d4a52f'),((.46,.18,.24,.51),'#3b95b8')]:
   a,c,ww,hh=b
   cv.nodes.append(f'<rect x="{x+a*w}" y="{y+c*h}" width="{ww*w}" height="{hh*h}" fill="none" stroke="{color}" stroke-width="2.5"/>')
 cv.nodes.append('</g>');cv.meta.append(dict(id=id,x=x,y=y,w=w,h=h,kind='image'))
photo('input-image',48,812,232,174)
photo('roi-example',48,336,232,174,True)
cv.plain(164,1020,'Input image I',27,600)
cv.plain(164,316,'Illustrative ROI links',23,600)
cv.box('clip-visual',48,152,232,120,'CLIP ViT-B/16',('Frozen visual encoder','14×14 patches · no CLS'),fill='visual',fs=26,ss=20)
cv.box('detectors',48,568,232,120,'YOLO + UPT',('Frozen detectors','Boxes · classes · actions'),fill='semantic',fs=28,ss=20)
cv.box('roi-projection',312,160,176,144,'ROI pool',('Linear 768→512','V: N×512'),fill='visual',fs=27,ss=21)
cv.route('image-clip',[(48,900),(24,900),(24,212),(48,212)],'input-image','clip-visual')
cv.route('image-detectors',[(164,812),(164,688)],'input-image','detectors')
cv.route('detectors-rois',[(164,568),(164,510)],'detectors','roi-example')
cv.route('patches-roi',[(280,212),(312,212)],'clip-visual','roi-projection')
cv.route('boxes-roi',[(280,400),(296,400),(296,272),(312,272)],'roi-example','roi-projection')
# Compact vertical operation bars, as in the user's example.
def vbar(id,x,y,w,h,label,fill,owner=None,fs=23):
 cv.nodes.append(f'<g id="{id}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}" stroke="#50575c" stroke-width="1.5"/><text transform="translate({x+w/2},{y+h/2}) rotate(-90)" text-anchor="middle" dominant-baseline="central" font-family="Arial, Liberation Sans, sans-serif" font-size="{fs}" fill="#22282c">{html.escape(label)}</text></g>')
def fusion(id,x,title,align,fill):
 # Frame is one module; vertical bars are a compact expansion of its internals.
 cv.back.append(f'<rect data-node-id="{id}" x="{x}" y="136" width="320" height="172" rx="26" fill="#f6f6f5" stroke="#484f53" stroke-width="1.8"/>')
 cv.nodes.append(cv.text(x+160,170,title,26,600,owner=id))
 cv.meta.append(dict(id=id,x=x,y=136,w=320,h=172,kind='fusion'))
 for j,label in enumerate(['Self-Attn','FFN',align,'FFN','Gate']):
  bx=x+24+j*56
  vbar(id+'-op'+str(j),bx,188,40,96,label,fill,fs=22 if label!='Self-Attn' else 20)
  if j<4:cv.route(id+'-internal'+str(j),[(bx+40,236),(bx+56,236)],id,id)
 cv.plain(x+160,296,'Residual / norm inside',17)
fusion('obj1',536,'Object Fusion 1','OSA',C['semantic'])
fusion('obj2',952,'Object Fusion 2','OSA',C['semantic'])
fusion('rel',1368,'Relation Fusion','ORA',C['visual'])
cv.route('v-obj1',[(488,232),(536,232)],'roi-projection','obj1')
cv.route('obj1-obj2',[(856,232),(952,232)],'obj1','obj2');cv.label(904,219,'V₁, T₁',22)
cv.route('obj2-rel',[(1272,232),(1368,232)],'obj2','rel');cv.label(1320,219,'Vₒ',24)
cv.route('pn-obj1',[(600,88),(600,136)],'noun-bank','obj1')
cv.route('pv-rel',[(1600,88),(1600,136)],'verb-bank','rel')
cv.plain(1648,360,'Vᵣ',24,600)
# Original memories branch to the correct adapter and its residual bypass.
cv.route('vo-object-q',[(1040,308),(1040,344),(820,344),(820,448)],'obj2','object-mha')
cv.route('vo-object-residual',[(1168,308),(1168,368),(1024,368),(1024,484)],'obj2','object-sum')
cv.route('vr-relation-q',[(1472,308),(1472,448)],'rel','relation-mha')
cv.route('vr-relation-residual',[(1632,308),(1632,368),(1676,368),(1676,484)],'rel','relation-sum')
# Attention diagrams use vector mask cells and symbolic region thumbnails.
def mask_grid(x,y,w,h,relation=False):
 color=BLUE if relation else ORANGE
 cv.nodes.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="#fafafa" stroke="#c2c6c9" stroke-width="1"/>')
 cv.plain(x+w/2,y+30,'ROI ↔ slot mask',20 if w<220 else 24,600)
 # Columns are prompt slots, rows are ROIs; two endpoints are permitted for relations.
 cell=30;gx=x+52;gy=y+58
 for r in range(4):
  cv.nodes.append(f'<rect x="{x+14}" y="{gy+r*38}" width="22" height="26" fill="{color}" opacity="{.35+r*.15}"/>')
  for c in range(3):
   active=(r==c) or (relation and r==c+1)
   cv.nodes.append(f'<rect x="{gx+c*38}" y="{gy+r*38}" width="{cell}" height="{cell}" fill="{color if active else "#fff"}" stroke="#b9c0c4" stroke-width="1"/>')
 cv.plain(x+w/2,y+h-18,'ROI i' if not relation else 'ROI s or o',22)
mask_grid(384,456,264,248)
mask_grid(1144,456,188,248,True)
# Cylinders are phrase collections produced from per-image detections, not global banks.
def pool(id,x,y,w,h,title,l1,l2,color):
 cv.nodes.append(f'<g id="{id}"><path d="M{x},{y+16} A{w/2},16 0 0 1 {x+w},{y+16} V{y+h-16} A{w/2},16 0 0 1 {x},{y+h-16} Z" fill="#fff" stroke="{color}" stroke-width="1.6"/><ellipse cx="{x+w/2}" cy="{y+16}" rx="{w/2}" ry="16" fill="#f8f8f8" stroke="{color}" stroke-width="1.6"/>'+cv.text(x+w/2,y-12,title,24,600)+cv.text(x+w/2,y+58,l1,23)+cv.text(x+w/2,y+90,l2,23)+'</g>')
 cv.meta.append(dict(id=id,x=x,y=y,w=w,h=h,kind='pool'))
def text_encoder(id,x,y):
 cv.nodes.append(f'<g id="{id}"><path d="M{x},{y+14} L{x+64},{y} V{y+112} L{x},{y+98} Z" fill="#eef4f6" stroke="#52595e" stroke-width="1.5"/>'+f'<text transform="translate({x+32},{y+56}) rotate(-90)" text-anchor="middle" font-family="Arial, Liberation Sans, sans-serif" font-size="20">CLIP text F</text></g>')
 cv.meta.append(dict(id=id,x=x,y=y,w=64,h=112,kind='encoder'))
def tokens(id,x,y,label,color,arrow_to=None):
 for j in range(4):cv.nodes.append(f'<rect x="{x}" y="{y+j*16}" width="20" height="11" fill="{color}" stroke="#575f64" stroke-width=".8"/>')
 cv.plain(x+10,y-12,label,24,600)
 cv.meta.append(dict(id=id,x=x,y=y,w=20,h=59,kind='tokens'))
# Objects.
cv.box('matching',384,744,304,96,'Class + IoU ≥0.5',('Links ℓ: i, ordered (s,o)',),fill='plain',fs=26,ss=22)
cv.route('detectors-matching',[(280,628),(320,628),(320,792),(384,792)],'detectors','matching')
pool('object-phrases',384,888,200,104,'Object phrases','person · bicycle','pₒ¹, …, pₒᴷ',ORANGE)
cv.route('matching-object-phrases',[(484,840),(484,888)],'matching','object-phrases')
text_encoder('object-text',632,888)
cv.route('object-phrases-clip',[(584,940),(632,940)],'object-phrases','object-text')
cv.box('object-slots',720,624,256,112,'Slot MLP',('[tₒ; Vₒ[i]; bᵢ]','bᵢ ∈ ℝ⁴'),fill='semantic',fs=28,ss=22)
cv.route('object-text-slots',[(696,940),(848,940),(848,736)],'object-text','object-slots');cv.label(884,852,'tₒ',24)
cv.box('object-mha',720,448,200,112,'Masked MHA',('Q = Vₒ','K,V = slots'),fill='semantic',fs=25,ss=22)
vbar('object-wout',944,448,32,112,'Wout',C['semantic'],fs=22)
cv.meta.append(dict(id='object-wout',x=944,y=448,w=32,h=112,kind='projection'))
cv.plus('object-sum',1024,500)
cv.route('object-slots-mha',[(820,624),(820,560)],'object-slots','object-mha');cv.label(860,602,'K,V',21)
cv.route('object-mask',[(648,500),(720,500)],None,'object-mha',dash=True)
cv.route('object-mha-wout',[(920,500),(944,500)],'object-mha','object-wout')
cv.route('object-wout-sum',[(976,500),(1006,500)],'object-wout','object-sum')
cv.route('object-sum-memory',[(1042,500),(1060,500)],'object-sum','vo-updated')
tokens('vo-updated',1060,472,'Vₒ′',ORANGE)
cv.plain(716,1024,'Image-specific · valid linked slots only',22)
# Relations. ℓ is the named output of the same matching operation.
cv.plain(1238,840,'Links ℓ',24,600)
pool('relation-phrases',1144,888,192,104,'Relation phrases','person riding','a bicycle · pᵣᵏ',BLUE)
cv.route('links-relation-phrases',[(1240,852),(1240,888)],None,'relation-phrases')
text_encoder('relation-text',1376,888)
cv.route('relation-phrases-clip',[(1336,940),(1376,940)],'relation-phrases','relation-text')
cv.box('relation-slots',1372,624,256,112,'Slot MLP',('[tᵣ; Vₒ[s]; Vₒ[o];','bₛ; bₒ] ∈ ℝ⁸ geom.'),fill='visual',fs=28,ss=20)
cv.route('relation-text-slots',[(1440,940),(1500,940),(1500,736)],'relation-text','relation-slots');cv.label(1544,852,'tᵣ',24)
cv.box('relation-mha',1372,448,200,112,'Masked MHA',('Q = Vᵣ','K,V = slots'),fill='visual',fs=25,ss=22)
vbar('relation-wout',1596,448,32,112,'Wout',C['visual'],fs=22)
cv.meta.append(dict(id='relation-wout',x=1596,y=448,w=32,h=112,kind='projection'))
cv.plus('relation-sum',1676,500)
cv.route('relation-slots-mha',[(1472,624),(1472,560)],'relation-slots','relation-mha');cv.label(1512,602,'K,V',21)
cv.route('relation-mask',[(1332,500),(1372,500)],None,'relation-mha',dash=True)
cv.route('relation-mha-wout',[(1572,500),(1596,500)],'relation-mha','relation-wout')
cv.route('relation-wout-sum',[(1628,500),(1658,500)],'relation-wout','relation-sum')
cv.route('relation-sum-memory',[(1694,500),(1712,500)],'relation-sum','vr-updated')
tokens('vr-updated',1712,472,'Vᵣ′',BLUE)
cv.plain(1424,1024,'Original Vₒ · ordered (s,o) · 8 heads',22)
# Language generation reads two named memory ports, independently.
cv.plain(1992,88,'Language generator',32,600)
cv.back.append('<rect x="1824" y="440" width="336" height="400" rx="28" fill="#f6f6f5" stroke="#4b5257" stroke-width="1.8"/>')
cv.plain(2144,836,'×3',27,600)
cv.box('prefix',1848,1000,288,64,'Token prefix',(),fill='plain',fs=28)
cv.box('embedding',1848,864,288,64,'Embedding + position',(),fill='decoder',fs=25)
cv.box('causal-sa',1848,760,288,64,'Masked Self-Attn',(),fill='decoder',fs=27)
cv.box('parallel-ca',1848,616,288,104,'Parallel Cross-Attn',('Q=H; K,V=Vₒ′ / Vᵣ′',),fill='decoder',fs=26,ss=22)
cv.box('gated-ffn',1848,472,288,104,'Gated fusion + FFN',('gₒ⊙cₒ + gᵣ⊙cᵣ',),fill='decoder',fs=25,ss=26)
cv.box('vocab',1848,352,288,64,'Vocabulary + softmax',(),fill='decoder',fs=25)
cv.box('beam',1848,248,288,64,'Beam 5 · ≤20 tokens',(),fill='decoder',fs=25)
cv.box('rerank',1848,128,288,80,'Reranking',('0.3D + 0.4C + 0.3O',),fill='rank',fs=28,ss=23)
for id,a,b,src,dst in [('prefix-embedding',1000,928, 'prefix','embedding'),('embedding-sa',864,824,'embedding','causal-sa'),('sa-ca',760,720,'causal-sa','parallel-ca'),('ca-ffn',616,576,'parallel-ca','gated-ffn'),('ffn-vocab',472,416,'gated-ffn','vocab'),('vocab-beam',352,312,'vocab','beam'),('beam-rerank',248,208,'beam','rerank')]:
 cv.route(id,[(1992,a),(1992,b)],src,dst)
tokens('vo-port',1768,612,'Vₒ′',ORANGE);tokens('vr-port',1768,712,'Vᵣ′',BLUE)
cv.route('vo-parallel-ca',[(1788,648),(1848,648)],'vo-port','parallel-ca')
cv.route('vr-parallel-ca',[(1788,736),(1812,736),(1812,688),(1848,688)],'vr-port','parallel-ca')
cv.plain(1976,36,'Selected caption ŷ = argmax S',26,600)
cv.route('rerank-caption',[(2136,176),(2176,176),(2176,52),(2112,52)],'rerank',None)
# Evidence inputs are explicit local ports, as in the preceding figure.
cv.plain(1760,168,'I, YOLO',21,600)
cv.route('image-yolo-rerank',[(1788,176),(1848,176)],None,'rerank')
# Small training strip is outside the inference graph, retaining the previous objective.
cv.plain(1100,1100,'TRAIN only: XE → XE + 0.05 SCST(CIDEr)     |     Wout zero-init; no valid slot ⇒ ΔV = 0',24)
cv.plain(1100,1136,'OSA / ORA: prototype alignment (τ=0.1) + cross-attention     ·     F: frozen     ·     Norm/residual details collapsed',21)
slug='upt-csa-reference-style'
svg=cv.svg(slug,(0,0,2200,1160))
caption='Sơ đồ theo phong cách ảnh tham khảo: cascade encoder ở trên, hai nhánh prompt gắn vùng ở dưới và bộ sinh ngôn ngữ bên phải. Global TRAIN banks tách biệt với prompt riêng ảnh. Các cổng Vₒ′, Vᵣ′ và Links ℓ lặp tên biểu thị cùng tensor; dùng cổng để tránh dây cắt nhau. Ảnh, box màu và cụm từ chỉ minh họa, không phải output detector hoặc caption mới được chạy. Xem bản ba panel để đọc đầy đủ mục tiêu huấn luyện và quy tắc reranking.'
html_doc='<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>UPT CSA reference-style architecture</title><style>body{margin:0;background:#fff;font-family:Arial,"Liberation Sans",sans-serif}.diagram-container{width:100%;overflow-x:auto}svg{width:100%;min-width:2200px;display:block}figcaption{max-width:1000px;margin:20px auto;font-size:16px;line-height:1.6;padding:0 24px}@media print{.diagram-container{overflow:visible}svg{min-width:0}figcaption{display:none}}</style></head><body><main><div class="diagram-container">'+svg+'</div><figcaption lang="vi">'+html.escape(caption)+'</figcaption></main></body></html>'
(ROOT/(slug+'.html')).write_text(html_doc)
(ROOT/'diagram_geometry.json').write_text(json.dumps({'canvas':[2200,1160],'nodes':cv.meta,'routes':cv.routes},indent=2))
print('Built reference-style HTML.')
