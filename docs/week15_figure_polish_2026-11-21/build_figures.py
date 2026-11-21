"""Restyle the four approved report graphs; drawing only, no research execution."""
from pathlib import Path
import base64
import copy
import hashlib
import html
import importlib.util
import json
import sys
import xml.etree.ElementTree as ET
from PIL import ImageFont

ROOT = Path(__file__).resolve().parent
DOCS = ROOT.parent
sys.path.insert(0, str(DOCS / 'module_m3_2026-11-19'))
import build_diagrams as art

INK, BORDER, WIRE = '#243747', '#738592', '#40525E'
NEUTRAL, OBJECT, RELATION = '#F3F6F9', '#FFF0E3', '#E8F3FA'
REPORT = Path('C:/Users/leduc/Downloads/Bao_cao_nghien_cuu_tuan_15.docx')
GENERATED = []


def mapping(knots):
    def transform(v):
        for (a, b), (c, d) in zip(knots, knots[1:]):
            if a <= v <= c:
                return b + (v-a)*(d-b)/(c-a)
        return v + (knots[0][1]-knots[0][0] if v < knots[0][0] else knots[-1][1]-knots[-1][0])
    return transform


class Figure(art.Figure):
    def __init__(self, name, width, height, xm=lambda x:x, ym=lambda y:y):
        super().__init__(width, height)
        self.name, self.xm, self.ym = name, xm, ym
        self.shapes, self.routes, self.issues = [], [], []
        self.svg[1] = '<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="7" refY="3.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L7 3.5 L0 7 Z" fill="'+WIRE+'"/></marker></defs>'
        self.svg[3] = '<style>text{font-family:"Times New Roman",serif;fill:'+INK+'}.edge{fill:none;stroke:'+WIRE+';stroke-width:1.35;stroke-linejoin:miter;marker-end:url(#arrow)}</style>'
        GENERATED.append(self)

    def text(self, x, y, value, size=20, bold=False, anchor='middle', italic=False):
        value = html.unescape(value)
        super().text(self.xm(x), self.ym(y), html.escape(value), size, bold, anchor, italic)

    def frame(self, x, y, w, h, title):
        xx, yy = self.xm(x), self.ym(y)
        ww, hh = self.xm(x+w)-xx, self.ym(y+h)-yy
        self.svg.append(f'<rect x="{xx}" y="{yy}" width="{ww}" height="{hh}" fill="none" stroke="#B3C2CD" stroke-width="1.1" stroke-dasharray="5 4"/>')
        cell=self.cell(value='',vertex='1',connectable='0',style='fillColor=none;strokeColor=#B3C2CD;dashed=1;dashPattern=5 4;')
        ET.SubElement(cell,'mxGeometry',x=str(xx),y=str(yy),width=str(ww),height=str(hh),**{'as':'geometry'})
        if self.name.startswith('03'):
            self.text(x+281,y+28,title,24,True)
        else:
            self.text(x+20,y+29,title,24,True,anchor='start')

    def node(self, x, y, w, h, lines, fill='white', process=False, sizes=None, cylinder=False):
        if self.name.startswith('03'):
            if lines[0].startswith('Visual memories'): return
            if lines[0] in ('Noun aggregation','Verb aggregation'): x-=17.5
            if lines[0] in ('Self-attention','Cross-attention','Gated fusion'):
                lines, sizes = [lines[0]], [25]
            if lines[0].startswith('Add & Norm'): lines, sizes = ['Add & Norm'],[23]
            if lines[0]=='Vₒ': lines,sizes=['Object memory Vₒ','196 × 512'],[22,21]
            if lines[0]=='Vᵣ': lines,sizes=['Relation memory Vᵣ','196 × 512'],[21,21]
        if self.name.startswith('04'):
            if lines[0]=='Residual add':
                self.plus(self.xm(x+w/2),self.ym(y+h/2)); return
            if lines[0].startswith('Updated '):
                lines,sizes=[('Relation memory Vᵣ′' if 'relation' in lines[0] else 'Object memory Vₒ′'),'196 × 512'],[23,21]
        if self.name.startswith('01') and lines[0]=='Patch tokens':
            lines,sizes=['Patches · no CLS','196 × 768'],[20,23]
        if fill in ('#FFF1E6',art.OBJECT,OBJECT): fill=OBJECT
        elif fill in ('#EAF4F8',art.RELATION,RELATION): fill=RELATION
        elif fill not in ('white','#FFFFFF'): fill=NEUTRAL
        xx,yy=self.xm(x),self.ym(y)
        ww,hh=self.xm(x+w)-xx,self.ym(y+h)-yy
        sizes=list(sizes or [23]*len(lines))
        # Measure the actual serif face; fit to the existing box without line wrapping.
        for i,(line,size) in enumerate(zip(lines,sizes)):
            font=ImageFont.truetype('C:/Windows/Fonts/'+('timesbd.ttf' if i==0 else 'times.ttf'),round(size*4))
            measured=font.getlength(line)/4
            sizes[i]=min(size,size*(ww-18)/max(measured,1))
        total=sum(sizes)+4*(len(sizes)-1)
        if total>hh-9:
            sizes=[s*(hh-9-4*(len(sizes)-1))/sum(sizes) for s in sizes]
        if min(sizes)<15: self.issues.append(('small_font',lines[0],min(sizes)))
        rx=8 if process else 0
        if cylinder:
            ry=10
            self.svg.append(f'<path d="M{xx},{yy+ry} V{yy+hh-ry} A{ww/2},{ry} 0 0 0 {xx+ww},{yy+hh-ry} V{yy+ry} Z" fill="{fill}" stroke="{BORDER}" stroke-width="1.2"/>')
            self.svg.append(f'<ellipse cx="{xx+ww/2}" cy="{yy+ry}" rx="{ww/2}" ry="{ry}" fill="{fill}" stroke="{BORDER}" stroke-width="1.2"/>')
        else:
            self.svg.append(f'<rect x="{xx}" y="{yy}" width="{ww}" height="{hh}" rx="{rx}" fill="{fill}" stroke="{BORDER}" stroke-width="1.2"/>')
        cursor=yy+(hh-total)/2+(5 if cylinder else 0)
        labels=[]
        for i,(line,size) in enumerate(zip(lines,sizes)):
            cursor+=size
            escaped=html.escape(line)
            weight='bold' if i==0 else 'normal'
            self.svg.append(f'<text x="{xx+ww/2}" y="{cursor-3}" text-anchor="middle" font-size="{size}" font-weight="{weight}">{escaped}</text>')
            labels.append(f'<div style="font-size:{size}px;font-weight:{weight};line-height:1.15;">{escaped}</div>')
            cursor+=4
        shape='shape=cylinder;size=10;' if cylinder else f'rounded={int(process)};arcSize=12;'
        cell=self.cell(value=''.join(labels),vertex='1',style=shape+f'whiteSpace=wrap;html=1;fillColor={fill};strokeColor={BORDER};strokeWidth=1.2;fontColor={INK};fontFamily=Times New Roman;align=center;verticalAlign=middle;spacing=3;')
        ET.SubElement(cell,'mxGeometry',x=str(xx),y=str(yy),width=str(ww),height=str(hh),**{'as':'geometry'})
        self.shapes.append((xx,yy,ww,hh,lines[0],cell))

    def plus(self,x,y):
        r=17
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="white" stroke="{BORDER}" stroke-width="1.3"/><path d="M{x-8} {y}H{x+8} M{x} {y-8}V{y+8}" stroke="{INK}" stroke-width="1.25"/>')
        cell=self.cell(value='+',vertex='1',style=f'ellipse;html=1;fillColor=white;strokeColor={BORDER};strokeWidth=1.3;fontFamily=Times New Roman;fontSize=27;fontColor={INK};')
        ET.SubElement(cell,'mxGeometry',x=str(x-r),y=str(y-r),width=str(2*r),height=str(2*r),**{'as':'geometry'})
        self.shapes.append((x-r,y-r,2*r,2*r,'+',cell))

    def dot(self,x,y):
        if self.name.startswith('03') and (x,y)==(790,1280): return
        x,y=self.xm(x),self.ym(y)
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="2.7" fill="{WIRE}"/>')
        cell=self.cell(value='',vertex='1',connectable='0',style=f'ellipse;fillColor={WIRE};strokeColor=none;')
        ET.SubElement(cell,'mxGeometry',x=str(x-2.7),y=str(y-2.7),width='5.4',height='5.4',**{'as':'geometry'})

    def edge(self,points,arrow=True,control=False):
        if self.name.startswith('03') and (points[0] in [(790,1280),(1370,1280)]): return
        if self.name.startswith('03'):
            if points==[(260,100),(350,100)]: points=[(260,100),(332.5,100)]
            if points==[(1420,100),(1510,100)]: points=[(1420,100),(1492.5,100)]
            if points[0]==(450,130): points=[(432.5,130),(432.5,240)]
            if points[0]==(1610,130): points=[(1592.5,130),(1592.5,240)]
        pts=[(self.xm(x),self.ym(y)) for x,y in points]
        if self.name.startswith('04'):
            for idx in (0,len(points)-1):
                x,y=points[idx]
                for row in (310,750):
                    cy=self.ym(row+35);cx=self.xm(1345)
                    if (x,y)==(1270,row+35): pts[idx]=(cx-17,cy)
                    if (x,y)==(1420,row+35): pts[idx]=(cx+17,cy)
                    if (x,y)==(1345,row+70): pts[idx]=(cx,cy+17)
        path='M'+' L'.join(f'{x},{y}' for x,y in pts)
        styles=('stroke-dasharray:6 4;' if control else '')+('marker-end:none;' if not arrow else '')
        self.svg.append(f'<path d="{path}" class="edge" style="{styles}"/>')
        cell=self.cell(value='',edge='1',style=f'edgeStyle=none;html=1;endArrow={"block" if arrow else "none"};endFill=1;endSize=7;strokeColor={WIRE};strokeWidth=1.35;rounded=0;'+('dashed=1;dashPattern=6 4;' if control else ''))
        geom=ET.SubElement(cell,'mxGeometry',relative='1',**{'as':'geometry'})
        for tag,(x,y) in [('sourcePoint',pts[0]),('targetPoint',pts[-1])]:
            ET.SubElement(geom,'mxPoint',x=str(x),y=str(y),**{'as':tag})
        if len(pts)>2:
            arr=ET.SubElement(geom,'Array',**{'as':'points'})
            for x,y in pts[1:-1]: ET.SubElement(arr,'mxPoint',x=str(x),y=str(y))
        self.routes.append((pts,cell,control))

    def icon(self,x,y,frozen=False):
        super().icon(self.xm(x),self.ym(y),frozen)

    def photo(self,x,y,w,h):
        super().photo(self.xm(x),self.ym(y),self.xm(x+w)-self.xm(x),self.ym(y+h)-self.ym(y))

    def save(self,ignored=None):
        collisions=[];anchored=0
        for pts,cell,_ in self.routes:
            for (ax,ay),(bx,by) in zip(pts,pts[1:]):
                assert abs(ax-bx)<1e-7 or abs(ay-by)<1e-7,('diagonal',pts)
                for x,y,w,h,label,_ in self.shapes:
                    if abs(ay-by)<1e-7 and y+1e-5<ay<y+h-1e-5 and max(min(ax,bx),x)<min(max(ax,bx),x+w)-1e-5:
                        collisions.append((label,pts))
                    if abs(ax-bx)<1e-7 and x+1e-5<ax<x+w-1e-5 and max(min(ay,by),y)<min(max(ay,by),y+h)-1e-5:
                        collisions.append((label,pts))
            for role,p in [('source',pts[0]),('target',pts[-1])]:
                px,py=p
                for x,y,w,h,label,shape in self.shapes:
                    inside=(x-1e-6<=px<=x+w+1e-6 and y-1e-6<=py<=y+h+1e-6)
                    boundary=min(abs(px-x),abs(px-x-w),abs(py-y),abs(py-y-h))<1e-6
                    if inside and boundary:
                        cell.set(role,shape.get('id'))
                        prefix='exit' if role=='source' else 'entry'
                        cell.set('style',cell.get('style')+f'{prefix}X={(px-x)/w};{prefix}Y={(py-y)/h};{prefix}Perimeter=0;')
                        anchored+=1;break
        assert not collisions,collisions
        assert not self.issues,self.issues
        (ROOT/(self.name+'.svg')).write_text('\n'.join(self.svg+['</svg>']),encoding='utf-8')
        self.mx.find('diagram').set('name',self.name.split('_',1)[1])
        self.mx.find('diagram').set('id',self.name)
        ET.indent(self.mx)
        ET.ElementTree(self.mx).write(ROOT/(self.name+'.drawio'),encoding='utf-8',xml_declaration=True)
        self.checks=dict(shapes=len(self.shapes),connectors=len(self.routes),orthogonal=True,unrelated_box_crossings=0,native_anchored_endpoints=anchored)
        print(self.name,self.checks)


def m1():
    f=Figure('02_M1_semantic_bank',1150,810)
    f.text(575,35,'M1 · Semantic Bank Construction',28,True)
    f.node(60,80,180,65,['TRAIN captions'])
    f.node(310,80,300,65,['POS tagging + Lemmatization','spaCy en_core_web_sm'],NEUTRAL,True,[23,20])
    f.node(720,80,300,65,['Filter + Deduplicate','Minimum frequency = 5'],NEUTRAL,True,[24,21])
    f.edge([(240,112.5),(310,112.5)]);f.edge([(610,112.5),(720,112.5)])
    for cx,verb in [(410,False),(835,True)]:
        fill=RELATION if verb else OBJECT
        f.node(cx-130,210,260,65,['Verb lemmas' if verb else 'Noun lemmas','e.g., ride, stand' if verb else 'e.g., person, bicycle'],fill,sizes=[24,21])
        f.node(cx-185,330,370,80,['Verb prompt' if verb else 'Object prompt','e.g., a photo contains the action of ride' if verb else 'e.g., a photo of a bicycle'],fill,True,[24,20])
        f.edge([(cx,275),(cx,330)]);f.edge([(cx,410),(cx,465)])
        f.node(cx-110,585,220,50,['L₂ normalization'],NEUTRAL,True,[23])
        f.edge([(cx,535),(cx,585)])
        f.node(cx-130,690,260,85,['Verb bank Bᵥ' if verb else 'Noun bank Bₙ','1594 × 512' if verb else '5269 × 512'],fill,sizes=[25,23],cylinder=True)
        f.edge([(cx,635),(cx,690)])
    f.node(225,465,795,70,['CLIP Text Encoder','CLIP ViT-B/16'],NEUTRAL,True,[25,22]);f.icon(1007,465,True)
    f.edge([(870,145),(870,178),(410,178),(410,210)])
    f.edge([(835,178),(835,210)]);f.dot(835,178)
    f.save()


def source(name,path):
    spec=importlib.util.spec_from_file_location(name,DOCS/path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.validate=lambda f:None
    return module


def main():
    original=hashlib.sha256(REPORT.read_bytes()).hexdigest()
    # Authoritative current report is preserved byte-for-byte.
    mod=source('overview_source',Path('full_pipeline_split_2026-11-21/build_full.py'))
    mod.Figure=lambda:Figure('01_full_pipeline',1840,1160)
    mod.main()
    m1()
    ym=mapping(list(zip([0,35,70,130,170,190,240,295,320,348,355,370,390,460,500,540,555,570,614,650,690,715,750,820,860,900,915,930,974,1010,1050,1100,1160,1190,1225,1280,1400],[0,30,60,115,140,160,200,250,280,306,320,335,355,410,440,475,490,510,550,580,615,640,670,725,755,790,805,825,865,895,930,965,1015,1045,1075,1125,1160])))
    mod=source('m2_source',Path('m2_expanded_2026-11-20/build_m2.py'))
    mod.Figure=lambda:Figure('03_M2_bank_visual_alignment',1760,1160,ym=ym)
    mod.main()
    xm=mapping(list(zip([0,40,220,250,280,300,470,510,540,710,740,780,920,1000,1060,1100,1200,1270,1345,1420,1490,1710,1730,1750],[0,40,220,250,280,300,470,510,540,710,740,780,920,980,1040,1080,1180,1240,1280,1320,1380,1600,1620,1640])))
    ym=mapping(list(zip([0,35,65,100,125,150,220,310,380,440,485,510,537,565,590,660,750,820,880,915,950,970,1000],[0,32,60,86,105,130,185,240,300,350,380,410,435,455,480,535,590,650,690,725,760,790,820])))
    mod=source('m3_source',Path('m3_clear_routing_2026-11-21/build_m3.py'))
    mod.Figure=lambda:Figure('04_M3_prompt_visual_alignment',1640,820,xm,ym)
    mod.main()
    assert hashlib.sha256(REPORT.read_bytes()).hexdigest()==original
    combined=ET.Element('mxfile',host='app.diagrams.net',version='24.7.17')
    for f in GENERATED:
        ET.parse(ROOT/(f.name+'.svg'))
        combined.append(copy.deepcopy(f.mx.find('diagram')))
    ET.indent(combined)
    ET.ElementTree(combined).write(ROOT/'Tat_ca_4_hinh.drawio',encoding='utf-8',xml_declaration=True)
    checks=dict(status='GEOMETRY_PASS_VISUAL_REVIEW_PENDING',report_sha256=original,report_modified=False,models_executed=False,figures={f.name:f.checks for f in GENERATED})
    (ROOT/'CHECKS.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
