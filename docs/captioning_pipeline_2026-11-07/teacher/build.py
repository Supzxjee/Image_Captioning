"""Presentation overview: honest model hierarchy, compact native SVG objects."""
from pathlib import Path
import base64
import html
import json
import xml.etree.ElementTree as ET

R=Path(__file__).resolve().parent
W,H=2000,1140
INK,LINE='#25343c','#60727c'
GRAY,OBJ,REL='#eff4f7','#f9eadc','#e2f0f1'
back,edges,nodes,labels=[],[],[],[]
geometry={}


def txt(x,y,s,size=25,bold=False,anchor='middle'):
    rows=s if isinstance(s,list) else [s]
    content=''.join(f'<tspan x="{x}" dy="{0 if i==0 else size*1.17}">{html.escape(t)}</tspan>' for i,t in enumerate(rows))
    return f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{600 if bold else 400}" text-anchor="{anchor}" fill="{INK}">{content}</text>'


def icon(x,y,kind):
    if kind=='frozen':return f'<path d="M{x} {y-8}V{y+8} M{x-7} {y-4}L{x+7} {y+4} M{x-7} {y+4}L{x+7} {y-4}" fill="none" stroke="#438baf" stroke-width="2.2"/>'
    return f'<path d="M{x} {y-9}C{x+1} {y-1} {x+9} {y} {x+6} {y+7}C{x+2} {y+12} {x-8} {y+7} {x-5} {y+1}Z" fill="#eaa270" stroke="#ba7950" stroke-width="1.1"/>'


def label(x,y,s,size=24,bold=False,anchor='middle'):
    labels.append(txt(x,y,s,size,bold,anchor))


def box(key,x,y,w,h,s,fill=GRAY,kind=None,data=False,size=24,stack=False):
    geometry[key]=dict(x=x,y=y,w=w,h=h,lines=s if isinstance(s,list) else [s],size=size)
    if stack:nodes.append(f'<rect x="{x+7}" y="{y-7}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="#a9b8bd" stroke-width="1.2"/>')
    nodes.append(f'<g id="{key}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{0 if data else 9}" fill="{fill}" stroke="#8da0aa" stroke-width="1.4"/>')
    rows=s if isinstance(s,list) else [s]
    nodes.append(txt(x+w/2,y+h/2-(len(rows)-1)*size*.585+size*.34,rows,size))
    if kind:nodes.append(icon(x+w-11,y+6,kind))
    nodes.append('</g>')


def bank(key,x,y,s,fill):
    nodes.append(f'<g id="{key}"><path d="M{x} {y+9}V{y+49}C{x} {y+64} {x+180} {y+64} {x+180} {y+49}V{y+9}Z" fill="{fill}" stroke="#8da0aa" stroke-width="1.3"/><ellipse cx="{x+90}" cy="{y+9}" rx="90" ry="11" fill="{fill}" stroke="#8da0aa" stroke-width="1.3"/>'+txt(x+90,y+42,s,25)+'</g>')


def tokens(key,x,y,s,color):
    nodes.append(f'<g id="{key}">')
    for i,w in enumerate((22,30,25)):
        nodes.append(f'<rect x="{x}" y="{y+i*11}" width="{w}" height="7" fill="{color}" stroke="#8da0aa" stroke-width=".8"/>')
    nodes.append('</g>');label(x+15,y-9,s,24,True)


def a(key,pts,color=LINE,arrow=True):
    p=' '.join(f'{x},{y}' for x,y in pts)
    attr=' marker-end="url(#arrow)"' if arrow else ''
    edges.append(f'<g id="{key}"><polyline points="{p}" fill="none" stroke="white" stroke-width="7" stroke-linejoin="round"/><polyline points="{p}" fill="none" stroke="{color}" stroke-width="1.9" stroke-linejoin="round"{attr}/></g>')


def plus(key,x,y):
    nodes.append(f'<g id="{key}"><circle cx="{x}" cy="{y}" r="13" fill="white" stroke="{LINE}" stroke-width="1.4"/>'+txt(x,y+8,'+',27)+'</g>')


# Compact global preparation, distinct from image-specific UPT prompts.
label(300,32,'TRAIN semantic bank',26,True,'start')
box('train-words',300,75,170,65,['Nouns / verbs','from TRAIN'],data=True,size=23)
box('bank-text',520,90,170,65,['CLIP text','encoder'],kind='frozen',size=24)
bank('noun-bank',750,50,'Noun bank',OBJ)
bank('verb-bank',1000,50,'Verb bank',REL)
box('noun-aggregation',750,200,180,55,['Semantic','aggregation'],OBJ,kind='train',size=21)
box('verb-aggregation',1000,200,180,55,['Semantic','aggregation'],REL,kind='train',size=21)
a('words-text',[(470,107),(520,107)])
a('text-nouns',[(690,122),(720,122),(720,80),(750,80)])
a('text-verbs',[(690,122),(1090,122),(1090,109)])
a('noun-aggregation-input',[(840,111),(840,200)])
a('verb-aggregation-input',[(1090,111),(1090,200)])
a('pn-object',[(930,227),(950,227),(950,400),(840,400),(840,420)],'#b28a62')
a('pv-relation',[(1180,227),(1200,227),(1200,320),(1010,320),(1010,600),(1090,600),(1090,620)],'#528e94')
label(972,300,'Pn',23);label(1120,599,'Pv',23)

# Extraction remains separate operations, while vectors use feature-stack symbols.
label(240,285,'01  Regions',26,True,'start')
photo=R.parent.parent/'csa_pipeline_2026-11-02/v3/assets/COCO_train2014_000000263974.jpg'
im=base64.b64encode(photo.read_bytes()).decode()
nodes.append(f'<image id="input-image" x="40" y="430" width="160" height="120" href="data:image/jpeg;base64,{im}"/>')
label(120,585,'Image I',26,True)
box('clip-visual',240,340,160,65,['CLIP ViT-B/16','Visual encoder'],kind='frozen',size=20)
box('yolo',240,510,160,65,'YOLO detector',kind='frozen',size=21)
box('roi',430,340,160,65,'ROI pooling',size=25)
box('project',430,450,160,65,['Linear','projection'],kind='train',size=24)
box('boxes',420,530,170,50,'Boxes B',data=True,size=25)
a('image-clip',[(200,460),(220,460),(220,372),(240,372)])
a('image-yolo',[(200,500),(220,500),(220,542),(240,542)])
a('patch-roi',[(400,372),(430,372)])
label(415,327,'patches',18)
a('yolo-boxes',[(400,542),(410,542),(410,555),(420,555)])
a('boxes-roi',[(590,555),(610,555),(610,420),(550,420),(550,405)])
a('roi-project',[(510,405),(510,450)])
tokens('v',650,469,'V','#b6cbd8')
a('project-v',[(590,482),(650,482)])
a('v-object',[(680,482),(700,482),(700,460),(750,460)])

# One abstraction level: bank-conditioned stages, with two object stages visible.
label(750,355,'02  Alignment',26,True,'start')
box('object-alignment',750,420,180,80,['Object','alignment'],OBJ,kind='train',size=26,stack=True)
label(840,530,'2 stages',21)
box('relation-alignment',1000,620,180,80,['Relation','alignment'],REL,kind='train',size=26)
a('object-relation',[(930,460),(965,460),(965,660),(1000,660)])
a('vo-attention',[(965,460),(965,375),(1260,375)])
a('vr-attention',[(1180,660),(1210,660),(1210,575),(1260,575)])
label(1095,361,'Vo',25);label(1188,642,'Vr',25)

# Local attention core, inspired by DFCap's grouped mechanism rather than its model.
back.append('<rect x="1240" y="325" width="385" height="320" rx="16" fill="#f5f9f8" stroke="#c7d7d5" stroke-width="1.3"/>')
label(1260,297,'Prompt–region alignment',24,True,'start')
box('object-attention',1260,340,180,70,['Masked cross-','attention'],OBJ,kind='train',size=23)
box('relation-attention',1260,540,180,70,['Masked cross-','attention'],REL,kind='train',size=23)
box('object-output',1470,355,90,40,'Linear',OBJ,kind='train',size=19)
box('relation-output',1470,555,90,40,'Linear',REL,kind='train',size=19)
plus('object-residual',1600,375);plus('relation-residual',1600,575)
a('object-attn-output',[(1440,375),(1470,375)])
a('relation-attn-output',[(1440,575),(1470,575)])
a('object-output-sum',[(1560,375),(1587,375)])
a('relation-output-sum',[(1560,575),(1587,575)])
a('object-bypass',[(1210,375),(1210,310),(1600,310),(1600,362)])
a('relation-bypass',[(1210,575),(1210,630),(1600,630),(1600,588)])
label(1240,363,'Q',19);label(1240,563,'Q',19)

# Source text and ROI links meet only at the slot builder, near its consumers.
label(640,825,'Image-specific prompts',26,True,'start')
box('upt',640,850,160,65,'UPT detector',kind='frozen',size=22)
box('prompt-texts',840,850,180,80,['Object / relation','prompts'],data=True,size=22)
box('image-text',1040,440,180,60,['CLIP text','encoder'],kind='frozen',size=24)
box('slots',1260,440,180,60,'Slot MLPs',kind='train',size=25,stack=True)
tokens('original-vo',1490,456,'Vo','#e8cbb3')
box('matching',1440,660,180,70,['ROI matching','Class + IoU'],size=24)
a('image-upt',[(220,500),(220,882),(640,882)])
a('upt-texts',[(800,882),(840,882)])
a('texts-encoder',[(1020,890),(1230,890),(1230,520),(1130,520),(1130,500)])
a('text-slots',[(1220,470),(1260,470)])
a('vo-slots',[(1490,470),(1440,470)])
a('links-slots',[(1440,695),(1240,695),(1240,485),(1260,485)])
a('slots-object-kv',[(1350,440),(1350,410)],'#b28a62')
a('slots-relation-kv',[(1350,500),(1350,540)],'#528e94')
label(1390,432,'K,V',19);label(1390,532,'K,V',19)
a('upt-boxes-matching',[(720,915),(720,960),(1530,960),(1530,730)])
label(1000,950,'UPT boxes',21)
a('boxes-matching',[(590,555),(620,555),(620,1030),(1620,1030),(1620,695)])
label(880,1056,'YOLO boxes B',21)

# Decoder and inference selection are a compact end column.
label(1700,260,'03  Caption',26,True,'start')
box('prefix',1720,280,190,55,'Token prefix',data=True,size=25)
box('decoder',1700,430,230,100,['Transformer','decoder'],kind='train',size=27)
a('prefix-decoder',[(1815,335),(1815,430)])
a('vo-prime-decoder',[(1613,375),(1660,375),(1660,450),(1700,450)])
a('vr-prime-decoder',[(1613,575),(1660,575),(1660,500),(1700,500)])
label(1644,355,'Vo′',24);label(1644,605,'Vr′',24)
box('beam',1700,600,230,65,'Beam search',size=27)
box('candidates',1700,710,230,55,'5 captions',data=True,size=27)
a('decoder-beam',[(1815,530),(1815,600)])
a('beam-candidates',[(1815,665),(1815,710)])
label(1815,803,'Score candidates',23,True)
for key,x,s in [('score-d',1700,'D'),('score-c',1785,'C'),('score-o',1870,'O')]:
    box(key,x,820,60,55,s,size=27,kind='frozen' if key=='score-c' else None)
a('candidate-scores',[(1815,765),(1815,774),(1680,774),(1680,810),(1900,810),(1900,820)])
a('candidate-c',[(1815,810),(1815,820)])
a('candidate-d',[(1730,810),(1730,820)])
box('selection-final',1700,930,230,65,'Weighted selection',size=23)
box('final',1700,1040,230,55,'Final caption',data=True,size=27)
a('d-selection',[(1730,875),(1730,900),(1760,900),(1760,930)])
a('c-selection',[(1815,875),(1815,930)])
a('o-selection',[(1900,875),(1900,900),(1870,900),(1870,930)])
a('selection-caption',[(1815,995),(1815,1040)])

# Concise on-slide legend; longer implementation explanations belong to notes.
labels.append(icon(75,1115,'frozen'));label(95,1122,'Frozen',20,anchor='start')
labels.append(icon(205,1115,'train'));label(225,1122,'Trainable',20,anchor='start')
label(930,1122,'D: decoder likelihood     C: CLIP alignment     O: object consistency',21)
for x,y in [(220,500),(965,460),(1210,375),(1210,575)]:nodes.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{LINE}"/>')

svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0L8 4L0 8Z" fill="{LINE}"/></marker></defs><rect width="100%" height="100%" fill="white"/><g font-family="Arial, sans-serif">'+''.join(back+edges+nodes+labels)+'</g></svg>'
ET.fromstring(svg)
(R/'pipeline-teacher.svg').write_text(svg,encoding='utf-8')
(R/'pipeline-teacher.html').write_text(f'<!doctype html><meta charset="utf-8"><style>@page{{size:{W}px {H}px;margin:0}}html,body{{margin:0;width:{W}px;height:{H}px}}svg{{display:block}}</style>'+svg,encoding='utf-8')
(R/'geometry.json').write_text(json.dumps(geometry,indent=2),encoding='utf-8')
print('Teacher overview SVG written; no model execution.')
