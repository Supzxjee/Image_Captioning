"""Six editable paper figures. Drawing/XML only; no model imports or execution."""
from copy import deepcopy
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'module_m3_2026-11-19'))
import build_diagrams as art
import build_expanded_overview as expanded
import build_concise_overview as concise
import build_layout_proof as compact

OUT = ROOT/'figures'
OUT.mkdir(exist_ok=True)
art.ROOT = OUT
S = art.sub
VO,VR,VOP,VRP = art.VO,art.VR,art.VOP,art.VRP
PALE = '#F6F8FA'


class Figure(expanded.Figure):
    def title(self,value):
        self.text(self.width/2,30,value,18)

    def cylinder(self,x,y,w,h,title,shape):
        # Both formats retain native editable bank shapes and labels.
        self.svg.append(f'<path d="M{x},{y+12} C{x},{y-3} {x+w},{y-3} {x+w},{y+12} V{y+h-12} C{x+w},{y+h+3} {x},{y+h+3} {x},{y+h-12} Z" fill="white" stroke="#697A87" stroke-width="1"/>')
        self.svg.append(f'<ellipse cx="{x+w/2}" cy="{y+12}" rx="{w/2}" ry="12" fill="white" stroke="#697A87" stroke-width="1"/>')
        cell=self.cell(value='',vertex='1',style='shape=cylinder;size=12;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#697A87;strokeWidth=1;')
        ET.SubElement(cell,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),**{'as':'geometry'})
        self.text(x+w/2,y+42,title,17)
        self.text(x+w/2,y+67,shape,18)


def m1():
    f=Figure(1000,700)
    f.title('M1 · Semantic Bank Construction')
    f.box(40,65,160,60,['TRAIN captions'],sizes=[17])
    f.box(270,65,260,60,['POS Tagging + Lemmatization','spaCy en_core_web_sm'],PALE,True,[17,15])
    f.box(600,65,300,60,['Filter + Deduplicate','Minimum frequency = 5'],PALE,True,[17,15])
    f.edge([(200,95),(270,95)])
    f.edge([(530,95),(600,95)])
    f.box(190,175,230,60,['Noun lemmas','person, bicycle, …'],sizes=[17,16])
    f.box(610,175,230,60,['Verb lemmas','ride, stand, …'],sizes=[17,16])
    f.edge([(750,125),(750,150),(305,150),(305,175)])
    f.edge([(725,150),(725,175)])
    f.dot(725,150)
    f.box(170,280,270,65,['Object prompt template','a photo of a {lemma}'],PALE,True,[17,16])
    f.box(580,280,290,65,['Verb prompt template','a photo contains the action of {lemma}'],PALE,True,[17,14])
    f.edge([(305,235),(305,280)])
    f.edge([(725,235),(725,280)])
    f.box(170,395,700,60,['CLIP Text Encoder','CLIP ViT-B/16'],PALE,True,[18,16])
    f.icon(859,395,True)
    f.edge([(305,345),(305,395)])
    f.edge([(725,345),(725,395)])
    f.box(210,500,190,50,['L2 normalization'],PALE,True,[16])
    f.box(630,500,190,50,['L2 normalization'],PALE,True,[16])
    f.edge([(305,455),(305,500)])
    f.edge([(725,455),(725,500)])
    f.cylinder(210,590,190,85,'Noun bank',S('B','n')+': 5269 × 512')
    f.cylinder(630,590,190,85,'Verb bank',S('B','v')+': 1594 × 512')
    f.edge([(305,550),(305,590)])
    f.edge([(725,550),(725,590)])
    f.save('01_M1_semantic_bank')


def m2():
    f=Figure(1000,610)
    f.box(120,40,170,60,['Noun bank '+S('B','n'),'5269 × 512'],sizes=[17,16])
    f.box(730,40,170,60,['Verb bank '+S('B','v'),'1594 × 512'],sizes=[17,16])
    f.box(40,280,140,60,['Visual memory V','196 × 512'],sizes=[15,16])
    f.frame(200,130,760,450,'M2 · Semantic Bank–Visual Alignment')
    f.icon(945,130)
    f.box(220,180,160,40,['Noun aggregation'],PALE,True,[16])
    f.box(726,180,160,40,['Verb aggregation'],PALE,True,[16])
    f.edge([(205,100),(205,200),(220,200)])
    f.edge([(815,100),(815,160),(806,160),(806,180)])
    for x,title in [(220,'Object alignment 1'),(420,'Object alignment 2'),(726,'Relation alignment')]:
        f.box(x,285,160,50,[title,'Self-/cross-attention'],PALE,True,[15,13])
        f.box(x,395,160,40,['Gated fusion'],PALE,True,[16])
        f.edge([(x+80,335),(x+80,395)])
    f.edge([(180,310),(220,310)])
    f.edge([(300,220),(300,285)])
    f.edge([(806,220),(806,285)])
    f.edge([(380,415),(400,415),(400,310),(420,310)])
    f.edge([(580,415),(675,415),(675,310),(726,310)])
    f.box(700,495,190,65,['Visual memories',VO+', '+VR],sizes=[16,20])
    f.edge([(500,435),(500,527.5),(700,527.5)])
    f.edge([(806,435),(806,495)])
    f.save('02_M2_bank_visual_alignment')


def baseline():
    f=Figure(1350,580)
    f.title('Baseline · CLIP ViT-B/16 + Transformer Decoder')
    f.photo(40,65,130,95)
    f.box(40,160,130,30,['Input image I'],sizes=[15])
    f.box(220,85,170,60,['CLIP Vision Encoder','CLIP ViT-B/16'],PALE,True,[15,16])
    f.icon(379,85,True)
    f.box(440,85,180,60,['Patch tokens · no CLS','X: 196 × 768'],sizes=[15,17])
    f.box(670,85,140,60,['Linear projection','768 → 512'],PALE,True,[15,16])
    f.icon(799,85)
    f.box(855,85,130,60,['LayerNorm'],PALE,True,[16])
    f.icon(974,85)
    f.box(1040,85,210,60,['Visual memory','V: 196 × 512'],sizes=[16,18])
    for a,b in [(170,220),(390,440),(620,670),(810,855),(985,1040)]:
        f.edge([(a,115),(b,115)])
    f.box(1040,280,210,65,['Transformer Decoder','4 layers'],PALE,True,[18,16])
    f.icon(1239,280)
    f.edge([(1145,145),(1145,280)])
    f.box(855,280,130,65,['Final LayerNorm'],PALE,True,[15])
    f.icon(974,280)
    f.box(650,280,155,65,['Vocabulary projection','+ Softmax'],PALE,True,[14,16])
    f.icon(794,280)
    f.box(430,280,150,65,['Beam search','K = 5'],PALE,True,[18,16])
    f.box(160,280,210,65,['Generated caption'],sizes=[18])
    for a,b in [(1040,985),(855,805),(650,580),(430,370)]:
        f.edge([(a,312.5),(b,312.5)])
    f.box(1040,400,210,60,['Word Embedding','+ Position Encoding'],PALE,True,[17,16])
    f.icon(1239,400)
    f.box(1040,510,210,35,['Caption prefix'],sizes=[17])
    f.edge([(1145,510),(1145,460)])
    f.edge([(1145,400),(1145,345)])
    f.save('04_baseline')


def mini_m1(f):
    # The full overview opens M1 as well as M2/M3, preserving the existing lower graph.
    f.frame(230,15,730,255,'M1 · Semantic Bank Construction')
    box=lambda x,y,w,h,lines,fill='white',rounded=False,sizes=None: art.Figure.box(f,x,y,w,h,lines,fill,rounded,sizes or [12]*len(lines),False)
    edge=lambda p,arrow=True: art.Figure.edge(f,p,arrow)
    box(250,55,110,40,['TRAIN captions'],sizes=[13])
    box(390,55,180,40,['POS + Lemmatization','spaCy en_core_web_sm'],PALE,True,[12,11])
    box(610,55,135,40,['Filter + Deduplicate','Min frequency = 5'],PALE,True,[12,11])
    edge([(360,75),(390,75)])
    edge([(570,75),(610,75)])
    box(260,135,230,45,['Noun prompt template','a photo of a {lemma}'],PALE,True,[13,12])
    box(260,202.5,230,50,['Verb prompt template','a photo contains the action','of {lemma}'],PALE,True,[12,10,10])
    edge([(677.5,95),(677.5,115),(245,115),(245,227.5)],False)
    edge([(245,157.5),(260,157.5)])
    edge([(245,227.5),(260,227.5)])
    f.dot(245,157.5)
    box(560,155,155,65,['CLIP Text Encoder','CLIP ViT-B/16'],PALE,True,[13,13])
    f.icon(704,155,True)
    edge([(490,157.5),(525,157.5),(525,177.5),(560,177.5)])
    edge([(490,227.5),(525,227.5),(525,202.5),(560,202.5)])
    box(745,155,90,65,['L2','normalization'],PALE,True,[13,11])
    box(850,135,95,115,['Noun bank','5269 × 512','Verb bank','1594 × 512'],sizes=[12,12,12,12])
    edge([(715,187.5),(745,187.5)])
    edge([(835,187.5),(850,187.5)])


class FullFigure(concise.ConciseFigure):
    def __init__(self,w,h):
        super().__init__(w,h)
        mini_m1(self)

    def box(self,x,y,w,h,lines,*args,**kwargs):
        if (x,y) in ((40,40),(230,40),(455,20)):
            return
        return super().box(x,y,w,h,lines,*args,**kwargs)

    def edge(self,points,arrow=True):
        if points in ([(180,65),(230,65)],[(400,65),(455,65)]):
            return
        if points == [(550,145),(550,250),(600,250),(600,337),(340,337),(340,350)]:
            points=[(875,250),(875,280),(600,280),(600,337),(340,337),(340,350)]
        elif points == [(800,145),(800,270),(846,270),(846,350)]:
            points=[(925,250),(925,292),(846,292),(846,350)]
        return super().edge(points,arrow)


class ModularFigure(compact.Figure):
    def box(self,x,y,w,h,lines,fill='white',rounded=False,sizes=None,bold_first=False):
        if lines[0].startswith(('M1 ·','M2 ·','M3 ·')):
            lines,sizes=[lines[0][:2]],[23]
        elif lines[0]=='Transformer Decoder':
            lines,sizes=lines[:2],[15,14]
        elif lines[0]=='Caption reranking':
            lines,sizes=lines[:1],[16]
        return super().box(x,y,w,h,lines,fill,rounded,sizes,False)


def rename(old,new):
    for ext in ('.svg','.drawio'):
        (OUT/(old+ext)).replace(OUT/(new+ext))


def crop_m3():
    # Extract editable vector/native objects; no raster cropping or model execution.
    source=OUT/'05_full_pipeline.svg'
    ns='{http://www.w3.org/2000/svg}'
    ET.register_namespace('',ns[1:-1])
    root=ET.parse(source).getroot()
    x0,y0,w,h=1020,280,820,590
    inside=lambda x,y: x0<=x<=x0+w and y0<=y<=y0+h
    target=ET.Element(ns+'svg',width=str(w),height=str(h),viewBox=f'{x0} {y0} {w} {h}')
    ET.SubElement(target,ns+'rect',x=str(x0),y=str(y0),width=str(w),height=str(h),fill='white')
    for el in root:
        tag=el.tag.split('}')[-1]
        keep=tag in ('defs','style')
        if tag=='rect':
            x,y,rw,rh=[float(el.get(k,0)) for k in ('x','y','width','height')]
            keep=inside(x,y) and inside(x+rw,y+rh)
        elif tag=='text':
            keep=inside(float(el.get('x')),float(el.get('y')))
        elif tag=='path' and el.get('class')=='edge':
            pts=[tuple(map(float,p.split(','))) for p in re.findall(r'[ML]([0-9.,]+)',el.get('d'))]
            keep=bool(pts) and all(inside(*p) for p in pts)
        elif tag=='circle':
            keep=inside(float(el.get('cx')),float(el.get('cy')))
        elif tag=='g':
            m=re.search(r'translate\(([0-9.]+),([0-9.]+)\)',el.get('transform',''))
            keep=bool(m and inside(float(m[1]),float(m[2])))
        if keep:target.append(deepcopy(el))
    ET.ElementTree(target).write(OUT/'03_M3_prompt_visual_alignment.svg',encoding='utf-8',xml_declaration=True)
    source=ET.parse(OUT/'05_full_pipeline.drawio').getroot()
    mx=ET.Element('mxfile',host='app.diagrams.net')
    diag=ET.SubElement(mx,'diagram',id='m3',name='M3')
    graph=ET.SubElement(diag,'mxGraphModel',page='1',pageWidth=str(w),pageHeight=str(h),grid='1',gridSize='10')
    cells=ET.SubElement(graph,'root')
    ET.SubElement(cells,'mxCell',id='0')
    ET.SubElement(cells,'mxCell',id='1',parent='0')
    for cell in source.findall('.//mxCell'):
        if cell.get('id') in ('0','1'):continue
        geom=cell.find('mxGeometry')
        if geom is None:continue
        keep=False
        if cell.get('edge')=='1':
            pts=[(float(p.get('x')),float(p.get('y'))) for p in geom.findall('.//mxPoint')]
            keep=bool(pts) and all(inside(*p) for p in pts)
        elif cell.get('vertex')=='1':
            x,y,rw,rh=[float(geom.get(k,0)) for k in ('x','y','width','height')]
            keep=inside(x+rw/2,y+rh/2)
        if not keep:continue
        clone=deepcopy(cell)
        g=clone.find('mxGeometry')
        if clone.get('edge')=='1':
            for p in g.findall('.//mxPoint'):
                p.set('x',str(float(p.get('x'))-x0));p.set('y',str(float(p.get('y'))-y0))
        else:
            g.set('x',str(float(g.get('x',0))-x0));g.set('y',str(float(g.get('y',0))-y0))
        cells.append(clone)
    ET.ElementTree(mx).write(OUT/'03_M3_prompt_visual_alignment.drawio',encoding='utf-8',xml_declaration=True)


def combine_pages():
    result=ET.Element('mxfile',host='app.diagrams.net')
    for i,path in enumerate(sorted(OUT.glob('[0-9][0-9]_*.drawio')),1):
        original=ET.parse(path).getroot().find('diagram')
        diag=deepcopy(original)
        diag.set('id',f'figure-{i}')
        diag.set('name',path.stem)
        result.append(diag)
    ET.ElementTree(result).write(OUT/'ALL_6_FIGURES.drawio',encoding='utf-8',xml_declaration=True)


if __name__=='__main__':
    m1();m2();baseline()
    expanded.ROOT=OUT;art.ROOT=OUT;expanded.Figure=FullFigure
    expanded.main();rename('pipeline_m2_m3_expanded','05_full_pipeline')
    crop_m3()
    compact.ROOT=OUT;art.ROOT=OUT;compact.Figure=ModularFigure
    compact.main();rename('pipeline_layout_proof','06_modular_pipeline')
    combine_pages()
    for ext in ('.svg','.drawio'):
        paths=list(OUT.glob('[0-9][0-9]_*'+ext))
        assert len(paths)==6,(ext,len(paths))
        for path in paths:ET.parse(path)
    print('Six SVG/native draw.io pairs and one six-page draw.io generated. No models executed.')
