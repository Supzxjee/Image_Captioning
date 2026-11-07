"""One connected publication figure; vector geometry only, no model execution."""
from pathlib import Path
import base64
import html
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
W, H = 2200, 1370
INK, LINE = '#26333c', '#52616b'
GRAY, ORANGE, BLUE = '#edf3f7', '#fcebdc', '#e2f1f5'
parts, edges, nodes, labels = [], [], [], []
geometry = {}


def text(x, y, lines, size=25, anchor='middle', bold=False, color=INK):
    if isinstance(lines, str):
        lines = [lines]
    content = ''.join(f'<tspan x="{x}" dy="{0 if i == 0 else size * 1.18}">{html.escape(s)}</tspan>' for i, s in enumerate(lines))
    return f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{600 if bold else 400}" fill="{color}">{content}</text>'


def icon(x, y, kind):
    if kind == 'frozen':
        paths = []
        for dx, dy in [(0, 10), (8.66, 5), (8.66, -5)]:
            paths.append(f'M{x-dx},{y-dy} L{x+dx},{y+dy}')
        return f'<path d="{" ".join(paths)}" fill="none" stroke="#318bb6" stroke-width="2.5"/>'
    return f'<path d="M{x},{y-12} C{x+2},{y-2} {x+12},{y-1} {x+8},{y+8} C{x+3},{y+16} {x-10},{y+11} {x-8},{y+3} C{x-7},{y-2} {x-2},{y-3} {x},{y-12}Z" fill="#f69a43" stroke="#cc652b" stroke-width="1.3"/>'


def box(key, x, y, w, h, lines, fill=GRAY, process=True, kind=None, size=25):
    geometry[key] = dict(x=x, y=y, w=w, h=h, process=process)
    lines = [lines] if isinstance(lines, str) else lines
    nodes.append(f'<g id="{key}"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{10 if process else 0}" fill="{fill}" stroke="#87959e" stroke-width="1.5"/>')
    nodes.append(text(x+w/2, y+h/2-(len(lines)-1)*size*.59+size*.34, lines, size))
    if kind:
        nodes.append(icon(x+w-13, y+8, kind))
    nodes.append('</g>')


def wire(key, points, color=LINE, arrow=True, dash=False):
    edges.append(f'<polyline id="{key}" points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="white" stroke-width="7" stroke-linejoin="round"/>')
    attr = (' marker-end="url(#arrow)"' if arrow else '') + (' stroke-dasharray="7 6"' if dash else '')
    edges.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round"{attr}/>')


def label(x, y, s, size=23, color=INK, anchor='middle'):
    labels.append(text(x, y, s, size, anchor, color=color))


def group(x, y, w, h, title, dash=False):
    attr = ' stroke-dasharray="8 6"' if dash else ''
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="white" stroke="#b4c0c7" stroke-width="1.5"{attr}/>')
    parts.append(text(x+20, y+34, title, 28, 'start', True))


# TRAIN preparation is drawn apart from the trainable aggregation step.
group(30, 25, 1500, 235, 'Global semantic banks (TRAIN only)', True)
box('captions', 55, 100, 200, 70, ['TRAIN captions', 'train + restval'], process=False, size=24)
box('lemmas', 295, 100, 220, 70, ['POS tagging', '+ lemmatization'], size=24)
box('bank-clip', 555, 100, 230, 70, ['CLIP text encoder', 'Frozen'], kind='frozen', size=24)
box('noun-bank', 835, 70, 210, 60, 'Fixed noun bank', ORANGE, False, size=24)
box('verb-bank', 835, 165, 210, 60, 'Fixed verb bank', BLUE, False, size=24)
box('noun-agg', 1100, 70, 225, 60, ['Learned semantic', 'aggregation'], ORANGE, kind='train', size=23)
box('verb-agg', 1100, 165, 225, 60, ['Learned semantic', 'aggregation'], BLUE, kind='train', size=23)
box('pn', 1380, 70, 115, 60, 'Pn', ORANGE, False)
box('pv', 1380, 165, 115, 60, 'Pv', BLUE, False)
for key, a, b in [('captions-lemmas', (255,135),(295,135)), ('lemmas-text',(515,135),(555,135)), ('noun-agg-link',(1045,100),(1100,100)), ('verb-agg-link',(1045,195),(1100,195)), ('pn-link',(1325,100),(1380,100)), ('pv-link',(1325,195),(1380,195))]:
    wire(key,[a,b])
wire('text-nouns',[(785,135),(805,135),(805,100),(835,100)])
wire('text-verbs',[(805,135),(805,195),(835,195)])
label(1630,90,['Pn, Pv: 150 × 512'],24,anchor='start')
label(1570,145,['Prepared once:', 'lemmas and fixed banks.', 'Aggregation remains trainable.'],23,anchor='start')

# Visual branch: clean photo, no illustrative/fabricated detector boxes.
group(30, 300, 450, 740, 'Visual region extraction')
photo = ROOT.parents[1]/'csa_pipeline_2026-11-02/v3/assets/COCO_train2014_000000263974.jpg'
photo_data = base64.b64encode(photo.read_bytes()).decode('ascii')
nodes.append(f'<image id="input-image" x="145" y="815" width="220" height="165" href="data:image/jpeg;base64,{photo_data}"/>')
label(255,1014,'Input image I',27)
box('visual-clip', 60, 665, 175, 75, ['CLIP ViT-B/16', 'Visual encoder'], kind='frozen', size=23)
box('yolo', 275, 665, 175, 75, ['YOLO', 'Detector'], kind='frozen', size=24)
box('patches',60,550,175,65,['Patch features','14 × 14 × 768'],process=False,size=22)
box('boxes',275,550,175,65,['Boxes B','(x1, y1, x2, y2)'],process=False,size=21)
box('pool',150,445,240,65,'ROI pooling',size=26)
box('project',150,345,240,65,['Linear projection','768 → 512'],kind='train',size=24)
wire('image-clip',[(200,815),(200,780),(147,780),(147,740)])
wire('image-yolo',[(310,815),(310,780),(362,780),(362,740)])
wire('clip-patches',[(147,665),(147,615)])
wire('yolo-boxes',[(362,665),(362,615)])
wire('patch-pool',[(147,550),(147,530),(220,530),(220,510)])
wire('box-pool',[(362,550),(362,510)])
wire('pool-project',[(270,445),(270,410)])

# Three real fusion instances, with the Vo -> relation branch explicit.
group(525,300,690,430,'Semantic bank–region alignment')
box('object1',570,365,235,75,['Object fusion 1','Self / cross-attn + gate'],ORANGE,kind='train',size=22)
box('object2',570,510,235,75,['Object fusion 2','Self / cross-attn + gate'],ORANGE,kind='train',size=22)
box('relation',930,600,235,75,['Relation fusion','Self / cross-attn + gate'],BLUE,kind='train',size=22)
wire('v-object1',[(390,377),(480,377),(480,402),(570,402)])
label(515,388,'V',25)
wire('object1-object2',[(687,440),(687,510)])
label(755,483,'V1, T1',24)
# Prototypes descend outside the main modules; one crossing has no junction.
wire('pn-object1',[(1495,100),(1520,100),(1520,270),(510,270),(510,350),(687,350),(687,365)],'#b47c45')
wire('pv-relation',[(1495,195),(1550,195),(1550,260),(1190,260),(1190,690),(1047,690),(1047,675)],'#43849b')
wire('vo-object-adapter',[(805,547),(850,547),(850,410),(1300,410)])
label(1150,395,'Vo',25)
wire('vo-relation',[(850,547),(850,637),(930,637)])
wire('vr-relation-adapter',[(1165,637),(1240,637),(1240,650),(1300,650)])
label(1210,623,'Vr',25)
label(850,713,'All region memories: N × 512',22)

# Adapters explicitly expose slot construction, masked attention and residual.
def adapter(prefix,y,name,fill,mem,out):
    group(1290,y,350,210,name)
    slot_label = 'Slot MLP: [to; Vo[i]; bi]' if prefix == 'object' else 'Slot MLP: [tr; Vo[s,o]; bs,bo]'
    box(prefix+'-slot',1310,y+54,310,48,slot_label,fill,kind='train',size=21)
    box(prefix+'-attn',1360,y+120,260,50,'Masked cross-attn',fill,kind='train',size=23)
    nodes.append(f'<circle cx="1640" cy="{y+145}" r="15" fill="white" stroke="{LINE}" stroke-width="1.5"/>')
    nodes.append(text(1640,y+153,'+',26))
    wire(prefix+'-slot-attn',[(1490,y+102),(1490,y+120)])
    wire(prefix+'-memory-query',[(1300,y+145),(1360,y+145)])
    wire(prefix+'-attn-out',[(1620,y+145),(1625,y+145)])
    wire(prefix+'-bypass',[(1325,y+145),(1325,y+188),(1640,y+188),(1640,y+160)])
    label(1330,y+130,'Q',20)
    label(1540,y+116,'K, V',20)
    label(1465,y+203,'Output projection before residual sum',17)
    label(1683,y+130,out,25)

adapter('object',265,'Object prompt adapter',ORANGE,'Vo','Vo′')
adapter('relation',505,'Relation prompt adapter',BLUE,'Vr','Vr′')
# Main incoming memory coordinates align with adapter query ports.
# Slot text/ROI sources are a shared bus, drawn continuously into both MLPs.
wire('vo-query-final',[(1300,410),(1300,410)])
wire('vo-prime-decoder',[(1655,410),(1745,410)])
wire('vr-prime-decoder',[(1655,650),(1700,650),(1700,490),(1745,490)])
label(1470,758,'Both slot MLPs use original Vo.',23)
label(1470,791,'Mask: linked ROI / pair endpoints only.',21)

# Prompt branch below feeds two separate continuous buses: text and ROI links.
group(525,825,1115,215,'Image-specific prompts and region links')
box('upt',550,890,185,65,['UPT detector','Input image I'],kind='frozen',size=24)
box('phrases',780,870,200,65,['Object / relation','phrases'],process=False,size=24)
box('prompt-clip',1030,870,215,65,['CLIP text encoder','to / tr'],kind='frozen',size=24)
box('matching',810,970,360,50,'Class + IoU ≥ 0.5 matching',size=23)
wire('upt-phrases',[(735,915),(755,915),(755,902),(780,902)])
wire('phrases-prompt-clip',[(980,902),(1030,902)])
wire('upt-box-matching',[(642,955),(642,995),(810,995)])
label(693,985,'UPT boxes',19)
wire('yolo-matching',[(450,582),(500,582),(500,1065),(990,1065),(990,1020)])
label(660,1091,'YOLO boxes B',21)
wire('prompt-text-consumers',[(1245,902),(1270,902),(1270,337),(1310,337)],'#8d745d')
wire('prompt-text-relation',[(1270,577),(1310,577)],'#8d745d')
label(1450,861,'Text to, tr → both Slot MLPs',21)
wire('roi-links-consumers',[(1170,995),(1250,995),(1250,352),(1310,352)],'#43849b')
wire('roi-links-relation',[(1250,592),(1310,592)],'#43849b')
label(1410,1020,'ROI links → both Slot MLPs / masks',20)
label(1440,968,'Unmatched slots omitted',21)
wire('image-upt',[(365,900),(525,900),(525,922),(550,922)])

# One decoder, two explicit memory inputs and a separate caption selection stage.
group(1740,300,430,850,'Caption generation')
box('decoder',1745,380,400,150,['Transformer decoder ×3','Masked self-attention','Parallel object / relation cross-attn','Gated fusion + FFN'],kind='train',size=23)
label(1960,362,'Vo′ and Vr′ stay separate',23)
box('prefix',1770,585,350,60,['Token prefix → Word embedding','+ fixed positional encoding'],kind='train',size=22)
wire('prefix-decoder',[(1945,585),(1945,530)])
box('head-beam',1770,700,350,75,['Vocabulary projection + softmax','Beam search (width 5)'],kind='train',size=21)
wire('decoder-head',[(2145,450),(2157,450),(2157,680),(1945,680),(1945,700)])
box('candidates',1770,820,350,50,'Five candidate captions',process=False,size=25)
wire('beam-candidates',[(1945,775),(1945,820)])
box('rank',1770,910,350,65,['D: likelihood · C: CLIP alignment','O: object consistency'],size=21)
wire('candidates-rank',[(1945,870),(1945,910)])
box('final',1770,1080,350,55,'Final caption',process=False,size=27)
wire('rank-final',[(1945,975),(1945,1080)])
label(1945,1009,'S = 0.3D + 0.4C + 0.3O',22)
label(1945,1043,'Select highest-scoring candidate',21)
label(1945,1170,'Reranking inputs: image I + YOLO labels',20)

# One compact shared detail: no replicated atlas of all implementation steps.
group(30,1140,1630,155,'Semantic fusion mechanism (shared structure; separate parameters)',True)
box('detail-selection',55,1200,275,65,['Soft semantic selection','Cosine relevance → T'],size=23)
box('detail-self',375,1200,230,65,['Self-attention','Q = K = V = X'],kind='train',size=23)
box('detail-cross',685,1200,275,65,['Cross-attention','Q = H; K = V = T'],kind='train',size=23)
box('detail-gate',1045,1200,260,65,['Learned sigmoid gate','Mix H and cross context'],kind='train',size=23)
box('detail-out',1395,1200,225,65,'Updated regions',process=False,size=24)
wire('detail-self-cross',[(605,1232),(685,1232)])
wire('detail-cross-gate',[(960,1232),(1045,1232)])
wire('detail-h-gate',[(645,1232),(645,1280),(1175,1280),(1175,1265)])
label(920,1290,'H bypass',18)
wire('detail-gate-out',[(1305,1232),(1395,1232)])
wire('detail-text-kv',[(330,1232),(350,1232),(350,1185),(820,1185),(820,1200)],'#8d745d')
label(660,1178,'T',20)
# Norm / FFN details are intentionally described, not extra node chains.
label(840,1322,'Fusion includes residual normalization and FFN; adapters include a zero-initialized output projection.',22)
label(1100,1356,'Training: cross-entropy → SCST + cross-entropy. No reference captions at inference.',23)
nodes.append(icon(1680,1320,'frozen')); labels.append(text(1700,1327,'Frozen',21,'start'))
nodes.append(icon(1840,1320,'train')); labels.append(text(1860,1327,'Trainable',21,'start'))

svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0L8 4L0 8Z" fill="{LINE}"/></marker></defs><rect width="100%" height="100%" fill="white"/><g font-family="Arial, sans-serif">'+''.join(parts+edges+nodes+labels)+'</g></svg>'
ET.fromstring(svg)
(ROOT/'pipeline-compact.svg').write_text(svg,encoding='utf-8')
(ROOT/'pipeline-compact.html').write_text(f'<!doctype html><meta charset="utf-8"><style>@page{{size:{W}px {H}px;margin:0}}html,body{{margin:0;width:{W}px;height:{H}px}}svg{{display:block}}</style>'+svg,encoding='utf-8')
(ROOT/'geometry.json').write_text(json.dumps(geometry,indent=2),encoding='utf-8')
print('SVG + HTML generated; no model execution.')
