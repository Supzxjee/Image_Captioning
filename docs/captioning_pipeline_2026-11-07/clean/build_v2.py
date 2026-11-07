"""Reference-style architecture: expand a real fusion instance in place."""
import base64
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

R=Path(__file__).resolve().parent
W,H=2340,1270
INK,LINE='#29323a','#59656d'
GRAY,OBJ,REL='#f1f5f7','#faecdf','#e7f2f5'
back,edge,node,lab=[],[],[],[]
boxes={}


def text(x,y,s,size=26,bold=False,anchor='middle'):
    rows=s if isinstance(s,list) else [s]
    ts=''.join(f'<tspan x="{x}" dy="{0 if i==0 else size*1.18}">{html.escape(t)}</tspan>' for i,t in enumerate(rows))
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{600 if bold else 400}" text-anchor="{anchor}" fill="{INK}">{ts}</text>'


def icon(x,y,kind):
    if kind=='frozen':return f'<path d="M{x} {y-9}V{y+9} M{x-8} {y-4.5}L{x+8} {y+4.5} M{x-8} {y+4.5}L{x+8} {y-4.5}" fill="none" stroke="#4385a7" stroke-width="2.3"/>'
    return f'<path d="M{x} {y-10}C{x+1} {y-1} {x+10} {y} {x+7} {y+8}C{x+2} {y+13} {x-9} {y+8} {x-6} {y+1}Z" fill="#eea365" stroke="#be7841" stroke-width="1.2"/>'


def b(key,x,y,w,h,s,fill=GRAY,data=False,kind=None,size=25):
    rows=s if isinstance(s,list) else [s]
    boxes[key]=dict(x=x,y=y,w=w,h=h,lines=rows,size=size)
    node.append(f'<g id="{key}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{0 if data else 9}" fill="{fill}" stroke="#89959d" stroke-width="1.4"/>')
    node.append(text(x+w/2,y+h/2-(len(rows)-1)*size*.59+size*.34,rows,size))
    if kind:node.append(icon(x+w-10,y+5,kind))
    node.append('</g>')


def a(key,pts,color=LINE,arrow=True):
    p=' '.join(f'{x},{y}' for x,y in pts)
    attr=' marker-end="url(#arrow)"' if arrow else ''
    edge.append(f'<g id="{key}"><polyline points="{p}" fill="none" stroke="white" stroke-width="7"/><polyline points="{p}" fill="none" stroke="{color}" stroke-width="1.8"{attr}/></g>')


def label(x,y,s,size=24,bold=False,anchor='middle'):
    lab.append(text(x,y,s,size,bold,anchor))


# Preparation is deliberately a small top branch.
label(400,55,'TRAIN semantic banks',28,True,'start')
b('lemmas',400,90,230,75,['TRAIN noun / verb','lemmas'],data=True,size=25)
b('text-bank',680,90,230,75,['CLIP text encoder','Fixed banks'],kind='frozen',size=25)
b('aggregation',960,90,270,75,['Learned semantic','aggregation'],kind='train',size=25)
a('lemma-bank',[(630,127),(680,127)])
a('bank-aggregation',[(910,127),(960,127)])
a('pn-object1',[(1230,110),(1270,110),(1270,240),(840,240),(840,365)],'#ad835f')
a('pv-relation',[(1230,145),(1290,145),(1290,270),(725,270),(725,820),(1250,820),(1250,765),(1320,765)],'#5c8e9e')
label(875,224,'Pn',25)
label(1275,750,'Pv',25)
label(1120,198,'Pn, Pv: 150 × 512',24)

# Single clean photo and simple visual branch.
photo=R.parents[1]/'csa_pipeline_2026-11-02/v3/assets/COCO_train2014_000000263974.jpg'
im=base64.b64encode(photo.read_bytes()).decode()
node.append(f'<image id="image-I" x="45" y="490" width="170" height="127.5" href="data:image/jpeg;base64,{im}"/>')
label(130,655,'Input image I',26,True)
b('visual',270,375,190,80,['CLIP ViT-B/16','Visual encoder'],kind='frozen',size=24)
b('yolo',270,690,190,75,'YOLO detector',kind='frozen',size=24)
b('roi',510,375,200,80,['ROI pooling','Projection to 512'],kind='train',size=23)
a('image-visual',[(215,520),(240,520),(240,415),(270,415)])
a('image-yolo',[(215,590),(240,590),(240,727),(270,727)])
a('visual-roi',[(460,415),(510,415)])
label(482,360,'patches',21)
a('boxes-roi',[(460,727),(480,727),(480,510),(610,510),(610,455)])
label(550,542,'Boxes B',24)

# The actual second object fusion is expanded, not a duplicate explanatory panel.
b('object1',750,365,180,100,['Object fusion 1'],OBJ,kind='train',size=23)
a('v-object1',[(710,415),(750,415)])
label(700,350,'V',26)
back.append('<rect x="990" y="300" width="600" height="270" rx="10" fill="#fffaf6" stroke="#b89c84" stroke-width="1.4" stroke-dasharray="7 5"/>')
label(1010,336,'Object fusion 2',27,True,'start')
b('selection',1210,350,170,50,['Semantic selection'],OBJ,size=18)
b('self',1020,420,160,65,['Self-attention'],OBJ,kind='train',size=21)
b('cross',1210,420,170,65,['Cross-attention'],OBJ,kind='train',size=20)
b('gate',1420,420,140,65,['Learned gate'],OBJ,kind='train',size=20)
a('v1-self',[(930,415),(950,415),(950,452),(1020,452)])
label(974,440,'V1',23)
a('t1-selection',[(950,415),(950,345),(1295,345),(1295,350)],'#ad835f')
label(1120,365,'T1',22)
# Selection depends on current region features as well as prior prototypes.
a('v1-selection',[(1005,452),(1005,388),(1210,388)])
a('self-cross',[(1180,452),(1210,452)])
a('selected-cross',[(1295,400),(1295,420)],'#ad835f')
label(1345,413,'K,V',19)
label(1194,441,'Q',19)
a('cross-gate',[(1380,452),(1420,452)])
a('self-gate-bypass',[(1195,452),(1195,522),(1490,522),(1490,485)])
label(1310,550,'Residual / Norm / FFN included',21)
label(1600,240,['Fusion stages share this structure;', 'their parameters are separate.'],23)

# Vo and Vr stay genuinely separate throughout prompt conditioning.
a('vo-object',[(1560,452),(1660,452)])
label(1610,430,'Vo',26)
b('relation',1320,720,270,90,'Relation fusion',REL,kind='train',size=26)
a('vo-relation',[(1620,452),(1620,670),(1455,670),(1455,720)])
b('object-update',1660,410,250,85,['Object prompt attention','Masked + residual'],OBJ,kind='train',size=22)
b('relation-update',1660,720,250,90,['Relation prompt attention','Masked + residual'],REL,kind='train',size=21)
a('vr-update',[(1590,765),(1660,765)])
label(1620,747,'Vr',26)

# Prompt sources are now beside the slot builder and both consumers.
label(750,600,'Image-specific prompts',28,True,'start')
b('upt',750,620,180,75,'UPT detector',kind='frozen',size=24)
b('phrases',1000,620,220,75,['Object / relation','phrases'],data=True,size=25)
b('text-prompt',1320,580,270,70,['CLIP text encoder','Text embeddings to / tr'],kind='frozen',size=24)
b('slots',1660,580,250,85,['Object / relation slots','MLPs: text + Vo + boxes'],kind='train',size=21)
a('image-upt',[(215,555),(250,555),(250,610),(700,610),(700,657),(750,657)])
a('upt-phrases',[(930,657),(1000,657)])
a('phrases-text',[(1220,657),(1275,657),(1275,615),(1320,615)])
a('text-slots',[(1590,615),(1660,615)])
a('slots-object',[(1785,580),(1785,495)],'#91785f')
a('slots-relation',[(1785,665),(1785,720)],'#91785f')
label(1820,553,'So',23)
label(1820,704,'Sr',23)
# Metadata and original object features are assembled with text before slot MLPs.
b('matching',1320,880,270,75,['UPT–YOLO ROI matching','Class + IoU ≥ 0.5'],size=21)
a('upt-boxes',[(840,695),(840,917),(1320,917)])
label(1075,902,'UPT boxes',23)
a('yolo-matching',[(460,727),(480,727),(480,990),(1455,990),(1455,955)])
label(880,1015,'YOLO boxes B',23)
a('matching-slots',[(1590,917),(1640,917),(1640,655),(1660,655)])
label(1800,870,'ROI links + masks',23)
a('vo-slots',[(1620,635),(1660,635)])
label(1640,985,['Slots use original Vo.', 'Masks restrict updates to linked regions.'],22)

# Generation and selection are one short end branch.
b('prefix',2010,380,270,70,['Token prefix','Embedding + position'],kind='train',size=24)
b('decoder',2010,520,270,125,['Transformer decoder ×3','Parallel cross-attention','+ gated fusion'],kind='train',size=23)
a('prefix-decoder',[(2145,450),(2145,520)])
a('vo-prime',[(1910,452),(1970,452),(1970,550),(2010,550)])
a('vr-prime',[(1910,765),(1970,765),(1970,605),(2010,605)])
label(1945,432,'Vo′',26)
label(1940,800,'Vr′',26)
b('beam',2010,730,270,80,['Vocabulary head + softmax','Beam search: 5'],kind='train',size=21)
a('decoder-beam',[(2145,645),(2145,730)])
b('candidates',2010,870,270,55,'Five candidate captions',data=True,size=23)
a('beam-candidates',[(2145,810),(2145,870)])
b('rerank',2010,985,270,85,['Weighted reranking','0.3D + 0.4C + 0.3O'],size=24)
a('candidates-rerank',[(2145,925),(2145,985)])
b('final',2010,1150,270,60,'Final caption',data=True,size=27)
a('rerank-final',[(2145,1070),(2145,1150)])

# Only a compact legend/caption remains; no separate mechanism atlas.
label(70,1140,['D: decoder likelihood', 'C: frozen CLIP image–text alignment', 'O: detector-based object consistency'],24,anchor='start')
lab.append(icon(80,1240,'frozen'));label(105,1248,'Frozen',23,anchor='start')
lab.append(icon(260,1240,'train'));label(285,1248,'Trainable',23,anchor='start')
label(1170,1248,'Training: cross-entropy → SCST + cross-entropy',24)

svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0L8 4L0 8Z" fill="{LINE}"/></marker></defs><rect width="100%" height="100%" fill="white"/><g font-family="Arial, sans-serif">'+''.join(back+edge+node+lab)+'</g></svg>'
ET.fromstring(svg)
(R/'pipeline-clean-v2.svg').write_text(svg,encoding='utf-8')
(R/'pipeline-clean-v2.html').write_text(f'<!doctype html><meta charset="utf-8"><style>@page{{size:{W}px {H}px;margin:0}}html,body{{margin:0;width:{W}px;height:{H}px}}svg{{display:block}}</style>'+svg,encoding='utf-8')
(R/'geometry-v2.json').write_text(json.dumps(boxes,indent=2),encoding='utf-8')
print('In-place fusion diagram generated; model unchanged.')
