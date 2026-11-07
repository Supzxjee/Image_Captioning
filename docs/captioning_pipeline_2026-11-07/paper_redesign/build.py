"""Reference-led serif architecture; native SVG, no model execution."""
from pathlib import Path
import base64
import html
import json
import xml.etree.ElementTree as ET

R=Path(__file__).resolve().parent
W,H=2000,1040
INK,LINE='#252b2e','#465158'
NEUTRAL,OBJ,REL='#f4f7f8','#fff0e1','#eaf4f6'
back,edges,nodes,labels=[],[],[],[]
geometry={}


def text(x,y,s,size=23,bold=False,italic=False,anchor='middle'):
    rows=s if isinstance(s,list) else [s]
    t=''.join(f'<tspan x="{x}" dy="{0 if i==0 else size*1.13}">{html.escape(v)}</tspan>' for i,v in enumerate(rows))
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{700 if bold else 400}" font-style="{"italic" if italic else "normal"}" fill="{INK}">{t}</text>'


def label(x,y,s,size=23,bold=False,italic=False,anchor='middle'):
    labels.append(text(x,y,s,size,bold,italic,anchor))


def math(x,y,base,sub='',prime=False,size=25):
    subtag=f'<tspan baseline-shift="sub" font-size="{size*.7}">{html.escape(sub)}</tspan>' if sub else ''
    p='<tspan baseline-shift="super" font-size="18">′</tspan>' if prime else ''
    labels.append(f'<text x="{x}" y="{y}" font-size="{size}" font-style="italic" fill="{INK}" text-anchor="middle">{html.escape(base)}{subtag}{p}</text>')


def icon(x,y,kind):
    if kind=='frozen':return f'<path d="M{x} {y-7}V{y+7} M{x-6} {y-3.5}L{x+6} {y+3.5} M{x-6} {y+3.5}L{x+6} {y-3.5}" fill="none" stroke="#4c96b5" stroke-width="1.6"/>'
    return f'<path d="M{x} {y-8}C{x+1} {y-1} {x+8} {y} {x+5} {y+6}C{x+1} {y+10} {x-7} {y+6} {x-4} {y+1}Z" fill="#e9a06c" stroke="#b87547" stroke-width=".8"/>'


def box(key,x,y,w,h,s,fill=NEUTRAL,data=False,kind=None,size=23,stack=False):
    rows=s if isinstance(s,list) else [s]
    geometry[key]=dict(x=x,y=y,w=w,h=h,lines=rows,size=size)
    if stack:nodes.append(f'<rect x="{x+6}" y="{y-6}" width="{w}" height="{h}" rx="5" fill="{fill}" stroke="#a5afb3" stroke-width=".8"/>')
    nodes.append(f'<g id="{key}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{0 if data else 6}" fill="{fill}" stroke="{LINE}" stroke-width="1"/>')
    nodes.append(text(x+w/2,y+h/2-(len(rows)-1)*size*.565+size*.34,rows,size))
    if kind:nodes.append(icon(x+w-8,y+3,kind))
    nodes.append('</g>')


def frame(x,y,w,h,title,title_x=None):
    back.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="white" stroke="{LINE}" stroke-width=".9" stroke-dasharray="6 5"/>')
    label(title_x if title_x is not None else x+w/2,y-11,title,22,True,True)


def bank(key,x,y,s,fill):
    nodes.append(f'<g id="{key}"><path d="M{x} {y+8}V{y+45}C{x} {y+57} {x+150} {y+57} {x+150} {y+45}V{y+8}Z" fill="{fill}" stroke="{LINE}" stroke-width="1"/><ellipse cx="{x+75}" cy="{y+8}" rx="75" ry="9" fill="{fill}" stroke="{LINE}" stroke-width="1"/>'+text(x+75,y+35,s,23)+'</g>')


def feature(key,x,y,base,sub='',prime=False,fill=NEUTRAL):
    nodes.append(f'<g id="{key}">')
    for i,w in enumerate((22,29,25)):
        nodes.append(f'<rect x="{x}" y="{y+i*10}" width="{w}" height="6" fill="{fill}" stroke="{LINE}" stroke-width=".6"/>')
    nodes.append('</g>');math(x+13,y-9,base,sub,prime)


def arrow(key,pts):
    assert all(x1==x2 or y1==y2 for (x1,y1),(x2,y2) in zip(pts,pts[1:])), f'Diagonal connector: {key}'
    p=' '.join(f'{x},{y}' for x,y in pts)
    edges.append(f'<g id="{key}"><polyline points="{p}" fill="none" stroke="white" stroke-width="6" stroke-linejoin="miter" stroke-linecap="butt"/><polyline points="{p}" fill="none" stroke="{LINE}" stroke-width="1.25" stroke-linejoin="miter" stroke-linecap="butt" marker-end="url(#arrow)"/></g>')


def plus(key,x,y):
    nodes.append(f'<g id="{key}"><circle cx="{x}" cy="{y}" r="11" fill="white" stroke="{LINE}" stroke-width="1"/>'+text(x,y+7,'+',24)+'</g>')


# Two compact upper preparation strips, as in the supplied paper-style reference.
frame(300,38,870,220,'Semantic bank construction (TRAIN only)')
box('lemmas',320,80,170,65,['Noun / verb','lemmas'],data=True)
box('bank-text',530,80,170,65,['CLIP text','encoder'],kind='frozen')
bank('noun-bank',750,58,'Noun bank',OBJ)
bank('verb-bank',750,155,'Verb bank',REL)
box('noun-agg',950,62,180,50,['Semantic','aggregation'],OBJ,kind='train',size=21)
box('verb-agg',950,159,180,50,['Semantic','aggregation'],REL,kind='train',size=21)
arrow('lemmas-text',[(490,112),(530,112)])
arrow('text-noun',[(700,112),(720,112),(720,85),(750,85)])
arrow('text-verb',[(720,112),(720,182),(750,182)])
arrow('noun-agg-input',[(900,85),(950,85)])
arrow('verb-agg-input',[(900,182),(950,182)])
arrow('pn-alignment',[(1130,87),(1150,87),(1150,280),(840,280),(840,400)])
arrow('pv-alignment',[(1130,184),(1180,184),(1180,520),(1090,520),(1090,550)])
math(870,304,'P','n');math(1120,514,'P','v')

frame(1210,38,750,220,'Image-specific object–relation prompts')
box('image-source',1230,62,170,55,'Image I',data=True)
box('upt',1450,62,170,55,'UPT detector',kind='frozen')
box('prompts',1670,62,220,55,['Object / relation','phrases'],data=True,size=22)
box('private-text',1230,165,170,60,['CLIP text','encoder'],kind='frozen')
box('matching',1460,165,190,60,['ROI matching','Class + IoU'],size=22)
box('box-source',1700,165,190,60,'YOLO boxes B',data=True)
arrow('image-upt',[(1400,89),(1450,89)])
arrow('upt-prompts',[(1620,89),(1670,89)])
arrow('phrases-text',[(1780,117),(1780,140),(1315,140),(1315,165)])
arrow('upt-matching',[(1535,117),(1535,165)])
arrow('b-matching',[(1700,195),(1650,195)])

# Visual extraction, with a single real input image and compact operation sizes.
photo=R.parent.parent/'csa_pipeline_2026-11-02/v3/assets/COCO_train2014_000000263974.jpg'
im=base64.b64encode(photo.read_bytes()).decode()
nodes.append(f'<image id="input-image" x="45" y="385" width="155" height="116.25" href="data:image/jpeg;base64,{im}"/>')
label(122,535,'Input image I',25,False,True)
box('visual',250,330,160,70,['CLIP ViT-B/16','Visual encoder'],kind='frozen',size=21)
box('yolo',250,500,160,60,'YOLO detector',kind='frozen',size=22)
box('roi',445,330,160,70,'ROI pooling',size=25)
box('project',445,450,160,65,['Linear','projection'],kind='train',size=24)
box('boxes',445,550,160,45,'Boxes B',data=True,size=25)
arrow('image-visual',[(200,420),(225,420),(225,365),(250,365)])
arrow('image-yolo',[(200,470),(225,470),(225,530),(250,530)])
arrow('visual-roi',[(410,365),(445,365)])
label(427,318,'patches',19,False,True)
arrow('yolo-boxes',[(410,530),(427,530),(427,572),(445,572)])
arrow('boxes-roi',[(605,572),(630,572),(630,420),(565,420),(565,400)])
arrow('roi-project',[(525,400),(525,450)])
feature('v',665,471,'V',fill='#d8e7ee')
arrow('project-v',[(605,482),(665,482)])
arrow('v-object',[(695,482),(720,482),(720,440),(750,440)])

# Main semantic cascade, deliberately the visual centre of the figure.
box('object-alignment',750,400,180,80,['Object','alignment'],OBJ,kind='train',size=25,stack=True)
label(840,513,'Two cascaded stages',20,False,True)
box('relation-alignment',1000,550,180,80,['Relation','alignment'],REL,kind='train',size=25)
arrow('object-relation',[(930,440),(990,440),(990,590),(1000,590)])
arrow('vo-query',[(990,440),(990,365),(1440,365)])
arrow('vr-query',[(1180,590),(1210,590),(1210,565),(1440,565)])
math(1085,352,'V','o');math(1220,608,'V','r')

# The only expanded mechanism: sources sit beside their exact consumers.
frame(1200,307,670,345,'Region-linked prompt alignment',title_x=1640)
box('slots',1210,430,180,60,'Slot MLPs',kind='train',size=25,stack=True)
box('object-attn',1440,330,180,70,['Masked cross-','attention'],OBJ,kind='train',size=22)
box('relation-attn',1440,530,180,70,['Masked cross-','attention'],REL,kind='train',size=22)
box('object-output',1660,342,90,45,'Projection',OBJ,kind='train',size=18)
box('relation-output',1660,542,90,45,'Projection',REL,kind='train',size=18)
plus('object-sum',1800,365);plus('relation-sum',1800,565)
arrow('text-slots',[(1315,225),(1315,430)])
arrow('links-slots',[(1555,225),(1410,225),(1410,445),(1390,445)])
arrow('vo-slots',[(990,440),(990,455),(1210,455)])
arrow('slots-object',[(1390,460),(1530,460),(1530,400)])
arrow('slots-relation',[(1410,460),(1410,500),(1530,500),(1530,530)])
arrow('object-attn-output',[(1620,365),(1660,365)])
arrow('relation-attn-output',[(1620,565),(1660,565)])
arrow('object-output-sum',[(1750,365),(1789,365)])
arrow('relation-output-sum',[(1750,565),(1789,565)])
arrow('object-residual',[(1380,365),(1380,315),(1800,315),(1800,354)])
arrow('relation-residual',[(1380,565),(1380,640),(1800,640),(1800,576)])
label(1420,351,'Q',18,False,True);label(1420,551,'Q',18,False,True)
label(1575,426,'K,V',18,False,True);label(1575,518,'K,V',18,False,True)
math(1830,353,'V','o',True);math(1830,606,'V','r',True)

# Generation folds into the bottom axis, like the selected paper reference.
box('decoder',1690,710,250,80,['Transformer decoder','×3'],kind='train',size=25)
box('prefix',1690,920,250,55,'Token prefix',data=True,size=25)
box('beam',1510,730,160,70,['Beam search','K = 5'],size=23)
box('candidates',1310,730,160,70,['Five candidate','captions'],data=True,size=23)
arrow('vo-decoder',[(1811,365),(1960,365),(1960,735),(1940,735)])
arrow('vr-decoder',[(1811,565),(1980,565),(1980,770),(1940,770)])
arrow('prefix-decoder',[(1815,920),(1815,790)])
arrow('decoder-beam',[(1690,765),(1670,765)])
arrow('beam-captions',[(1510,765),(1470,765)])

frame(600,700,650,240,'Candidate scoring and selection')
box('score-d',620,755,170,65,['Decoder','likelihood'],size=23)
box('score-c',830,755,170,65,'CLIPScore',kind='frozen',size=25)
box('score-o',1040,755,170,65,['Object','consistency'],size=23)
box('select',820,865,200,55,'Weighted selection',size=23)
box('final',360,865,200,55,'Final caption',data=True,size=26)
arrow('candidates-scores',[(1310,765),(1270,765),(1270,735),(705,735),(705,755)])
arrow('candidate-c',[(915,735),(915,755)])
arrow('candidate-o',[(1125,735),(1125,755)])
arrow('d-select',[(705,820),(705,842),(870,842),(870,865)])
arrow('c-select',[(915,820),(915,865)])
arrow('o-select',[(1125,820),(1125,842),(970,842),(970,865)])
arrow('selection-caption',[(820,892),(560,892)])
label(920,970,'S = 0.3D + 0.4C + 0.3O',21,False,True)

# Restrained publication legend, outside all data paths.
labels.append(icon(85,1000,'frozen'));label(103,1007,'Frozen',20,anchor='start')
labels.append(icon(235,1000,'train'));label(253,1007,'Trainable',20,anchor='start')
label(1300,1020,'Training: cross-entropy → SCST + cross-entropy',21,False,True)
for x,y in [(990,440),(1380,365),(1380,565),(1410,460)]:nodes.append(f'<circle cx="{x}" cy="{y}" r="2.3" fill="{LINE}"/>')

svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0L7 3.5L0 7Z" fill="{LINE}"/></marker></defs><rect width="100%" height="100%" fill="white"/><g font-family="Times New Roman, serif">'+''.join(back+edges+nodes+labels)+'</g></svg>'
ET.fromstring(svg)
(R/'captioning-paper.svg').write_text(svg,encoding='utf-8')
(R/'captioning-paper.html').write_text(f'<!doctype html><meta charset="utf-8"><style>@page{{size:{W}px {H}px;margin:0}}html,body{{margin:0;width:{W}px;height:{H}px}}svg{{display:block}}</style>'+svg,encoding='utf-8')
(R/'geometry.json').write_text(json.dumps(geometry,indent=2),encoding='utf-8')
print('Fresh serif/reference-style SVG generated; no model run.')
