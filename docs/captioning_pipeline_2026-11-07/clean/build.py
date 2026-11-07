"""Sparse connected architecture with one mechanism detail; graphics only."""
import base64
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

R = Path(__file__).resolve().parent
W, H = 2220, 1140
INK, LINE = '#29323a', '#59656d'
GRAY, OBJ, REL = '#f0f4f7', '#faecdf', '#e7f2f5'
back, edge, node, lab = [], [], [], []
boxes = {}


def txt(x,y,s,size=25,bold=False,anchor='middle',color=INK):
    rows=s if isinstance(s,list) else [s]
    a=''.join(f'<tspan x="{x}" dy="{0 if i==0 else size*1.18}">{html.escape(t)}</tspan>' for i,t in enumerate(rows))
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{600 if bold else 400}" text-anchor="{anchor}" fill="{color}">{a}</text>'


def badge(x,y,kind):
    if kind=='frozen':
        return f'<path d="M{x} {y-9}V{y+9} M{x-8} {y-4.5}L{x+8} {y+4.5} M{x-8} {y+4.5}L{x+8} {y-4.5}" fill="none" stroke="#4385a7" stroke-width="2.3"/>'
    return f'<path d="M{x} {y-10}C{x+1} {y-1} {x+10} {y} {x+7} {y+8}C{x+2} {y+13} {x-9} {y+8} {x-6} {y+1}Z" fill="#eea365" stroke="#be7841" stroke-width="1.2"/>'


def b(key,x,y,w,h,s,fill=GRAY,data=False,kind=None,size=25):
    rows=s if isinstance(s,list) else [s]
    boxes[key]=dict(x=x,y=y,w=w,h=h,lines=rows,size=size)
    node.append(f'<g id="{key}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{0 if data else 9}" fill="{fill}" stroke="#89959d" stroke-width="1.4"/>')
    node.append(txt(x+w/2,y+h/2-(len(rows)-1)*size*.59+size*.34,rows,size))
    if kind:node.append(badge(x+w-10,y+5,kind))
    node.append('</g>')


def a(key,pts,color=LINE,arrow=True,dashed=False):
    points=' '.join(f'{x},{y}' for x,y in pts)
    attrs=(' marker-end="url(#arrow)"' if arrow else '')+(' stroke-dasharray="6 5"' if dashed else '')
    edge.append(f'<g id="{key}"><polyline points="{points}" fill="none" stroke="white" stroke-width="7"/><polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.8"{attrs}/></g>')


def label(x,y,s,size=23,bold=False,anchor='middle'):
    lab.append(txt(x,y,s,size,bold,anchor))


# One small preparation row; no section boxes or repeated explanatory paragraphs.
label(45,32,'TRAIN semantic preparation',27,True,'start')
b('lemmas',45,60,190,65,['TRAIN noun / verb','lemmas'],data=True,size=21)
b('bank-clip',275,60,200,65,['CLIP text','encoder'],kind='frozen',size=25)
b('banks',515,60,230,65,['Fixed noun / verb','embedding banks'],data=True,size=24)
b('aggregate',785,60,235,65,['Learned semantic','aggregation'],kind='train',size=24)
for key,x1,x2 in [('lemma-text',235,275),('text-bank',475,515),('bank-aggregation',745,785)]:a(key,[(x1,92),(x2,92)])
label(555,158,'Built once',21)
label(920,158,'Pn / Pv: 150 × 512',22)

# Input image branches directly to both visual encoders and UPT.
photo=R.parents[1]/'csa_pipeline_2026-11-02/v3/assets/COCO_train2014_000000263974.jpg'
im=base64.b64encode(photo.read_bytes()).decode()
node.append(f'<image id="image-I" x="45" y="360" width="170" height="127.5" href="data:image/jpeg;base64,{im}"/>')
label(130,522,'Input image I',25,True)
label(270,254,'Visual features',27,True,'start')
b('visual',270,300,190,70,['CLIP ViT-B/16','Visual encoder'],kind='frozen',size=23)
b('yolo',270,490,190,70,'YOLO detector',kind='frozen',size=24)
b('roi',510,300,220,70,['ROI pooling','Linear 768 → 512'],kind='train',size=24)
a('image-visual',[(215,390),(240,390),(240,335),(270,335)])
a('image-yolo',[(215,455),(240,455),(240,525),(270,525)])
a('patch-roi',[(460,335),(510,335)])
label(483,288,'patches',20)
a('yolo-roi',[(460,525),(480,525),(480,420),(620,420),(620,370)])
label(551,451,'Boxes B',22)

# Three cascade instances, not a generic black-box fusion stage.
label(1090,254,'Semantic fusion',27,True,'start')
b('object1',800,300,200,70,'Object fusion 1',OBJ,kind='train',size=24)
b('object2',1090,300,200,70,'Object fusion 2',OBJ,kind='train',size=24)
b('relation',1090,500,200,70,'Relation fusion',REL,kind='train',size=25)
a('roi-object1',[(730,335),(800,335)])
label(764,317,'V',25)
a('object1-object2',[(1000,335),(1090,335)])
label(1045,317,'V1, T1',23)
a('pn-object1',[(1020,82),(1040,82),(1040,220),(900,220),(900,300)],'#ad835f')
label(935,208,'Pn',23)
a('pv-relation',[(1020,105),(1060,105),(1060,455),(1190,455),(1190,500)],'#5c8e9e')
label(1150,446,'Pv',23)

# Two compact prompt update operations: details have one common slot builder below.
label(1370,254,'Prompt conditioning',27,True,'start')
b('object-adapter',1370,300,240,70,['Masked cross-attention','+ residual'],OBJ,kind='train',size=21)
b('relation-adapter',1370,500,240,70,['Masked cross-attention','+ residual'],REL,kind='train',size=21)
a('vo-object',[(1290,335),(1370,335)])
label(1330,316,'Vo',25)
a('vo-relation',[(1320,335),(1320,450),(1240,450),(1240,500)])
a('vr-relation',[(1290,535),(1370,535)])
label(1330,517,'Vr',25)

# Caption generation stays close to the two memory outputs.
label(1700,254,'Caption generation',27,True,'start')
b('prefix',1700,285,230,60,['Token prefix','Embedding + position'],kind='train',size=22)
b('decoder',1700,395,230,100,['Transformer decoder ×3','Parallel cross-attention','+ gated fusion'],kind='train',size=20)
a('prefix-decoder',[(1815,345),(1815,395)])
a('vo-prime-decoder',[(1610,335),(1670,335),(1670,420),(1700,420)])
label(1640,315,'Vo′',24)
a('vr-prime-decoder',[(1610,535),(1670,535),(1670,468),(1700,468)])
label(1640,570,'Vr′',24)
b('beam',1990,395,190,100,['Vocabulary head','+ softmax','Beam search: 5'],kind='train',size=23)
a('decoder-beam',[(1930,445),(1990,445)])
b('candidates',1990,555,190,60,['Five candidate','captions'],data=True,size=23)
a('beam-candidates',[(2085,495),(2085,555)])
b('rerank',1990,685,190,80,['D + C + O','Weighted reranking'],size=20)
a('candidates-rerank',[(2085,615),(2085,685)])
label(2085,800,'0.3D + 0.4C + 0.3O',20)
b('final',1990,850,190,55,'Final caption',data=True,size=25)
a('rerank-final',[(2085,765),(2085,850)])

# Image-specific semantics: a readable left-to-right branch, not multiple upward buses.
label(510,667,'Image-specific prompts',27,True,'start')
b('upt',510,710,180,65,'UPT detector',kind='frozen',size=24)
b('phrases',750,710,220,65,['Object / relation','phrases'],data=True,size=24)
b('prompt-text',1030,710,205,65,['CLIP text encoder','to / tr'],kind='frozen',size=23)
b('matching',750,850,330,65,['UPT–YOLO ROI matching','Class + IoU ≥ 0.5'],size=24)
b('slots',1320,710,290,100,['Object / relation Slot MLPs','Text + Vo region(s)','+ box(es)'],kind='train',size=21)
a('image-upt',[(215,435),(250,435),(250,742),(510,742)])
a('upt-phrases',[(690,742),(750,742)])
a('phrases-text',[(970,742),(1030,742)])
a('text-slots',[(1235,742),(1320,742)])
label(1275,725,'to, tr',22)
a('upt-matching',[(600,775),(600,882),(750,882)])
label(652,873,'UPT boxes',20)
a('yolo-matching',[(460,525),(485,525),(485,825),(915,825),(915,850)])
label(552,815,'Boxes B',20)
a('matching-slots',[(1080,882),(1465,882),(1465,810)])
label(1245,909,'ROI links + attention masks',22)
# Slot/mask bundle goes up in one corridor and branches to the two updates.
a('slots-object',[(1610,760),(1640,760),(1640,280),(1490,280),(1490,300)],'#91785f')
a('slots-relation',[(1640,480),(1490,480),(1490,500)],'#91785f')
label(1540,279,'So',20)
label(1540,480,'Sr',20)
label(1490,850,'Both slot builders use original Vo.',21)

# One mechanism inset only, with minimal text and no repeated overview.
back.append('<rect x="40" y="960" width="1690" height="150" fill="white" stroke="#bbc4ca" stroke-width="1.3" stroke-dasharray="7 6"/>')
label(60,990,'Inside each semantic fusion',25,True,'start')
b('selection',60,1015,260,60,['Semantic selection','Cosine relevance → T'],size=23)
b('self-attn',390,1015,220,60,['Self-attention','Q = K = V = X'],kind='train',size=23)
b('cross-attn',715,1015,260,60,['Cross-attention','Q = H; K = V = T'],kind='train',size=23)
b('gate',1080,1015,255,60,['Learned gate','Mix visual + semantic'],kind='train',size=23)
b('updated',1430,1015,255,60,'Updated region features',data=True,size=22)
a('self-cross',[(610,1045),(715,1045)])
a('cross-gate',[(975,1045),(1080,1045)])
a('gate-output',[(1335,1045),(1430,1045)])
a('selected-kv',[(320,1045),(345,1045),(345,1003),(845,1003),(845,1015)],'#91785f')
a('h-gate',[(660,1045),(660,1093),(1207,1093),(1207,1075)])
label(1880,1000,'D: decoder likelihood',22)
label(1880,1033,'C: frozen CLIP alignment',22)
label(1880,1066,'O: object consistency',22)
lab.append(badge(70,1124,'frozen'));label(90,1131,'Frozen',20,anchor='start')
lab.append(badge(210,1124,'train'));label(230,1131,'Trainable',20,anchor='start')
label(1100,1136,'Training: cross-entropy → SCST + cross-entropy. Norm / FFN details omitted.',21)

svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0L8 4L0 8Z" fill="{LINE}"/></marker></defs><rect width="100%" height="100%" fill="white"/><g font-family="Arial, sans-serif">'+''.join(back+edge+node+lab)+'</g></svg>'
ET.fromstring(svg)
(R/'pipeline-clean.svg').write_text(svg,encoding='utf-8')
(R/'pipeline-clean.html').write_text(f'<!doctype html><meta charset="utf-8"><style>@page{{size:{W}px {H}px;margin:0}}html,body{{margin:0;width:{W}px;height:{H}px}}svg{{display:block}}</style>'+svg,encoding='utf-8')
(R/'geometry.json').write_text(json.dumps(boxes,indent=2),encoding='utf-8')
print('Clean vector architecture written. No model execution.')
