"""Publication-style vector architecture figure, derived from implemented modules."""
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from html import escape
import math,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
W,H=3600,2030
im=Image.new('RGB',(W,H),'white');d=ImageDraw.Draw(im);v=[]
INK='#222831';BLUE='#EAF1F8';GREEN='#EDF6F0';SAND='#FBF2E7';PURPLE='#F2EEF8'
def font(s,b=False):return ImageFont.truetype('C:/Windows/Fonts/'+('arialbd.ttf' if b else 'arial.ttf'),s)
def text(x,y,s,size=30,b=False,center=False,color=INK):
 lines=s.split('\n')
 for j,line in enumerate(lines):
  yy=y+j*(size+9);xx=x-d.textlength(line,font=font(size,b))/2 if center else x
  d.text((xx,yy),line,font=font(size,b),fill=color)
  v.append(f'<text x="{x if center else xx}" y="{yy+size}" font-family="Arial" font-size="{size}" fill="{color}"'+(' font-weight="bold"' if b else '')+(' text-anchor="middle"' if center else '')+'>'+escape(line)+'</text>')
def rect(x,y,w,h,fill='white',dash=False,r=9):
 d.rounded_rectangle((x,y,x+w,y+h),radius=r,fill=fill,outline=INK,width=2)
 v.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{INK}" stroke-width="2"'+(' stroke-dasharray="11 8"' if dash else '')+'/>')
def box(x,y,w,h,s,fill=BLUE,size=30,b=False):
 rect(x,y,w,h,fill)
 while max(d.textlength(z,font=font(size,b)) for z in s.split('\n'))>w-24:size-=1
 assert size>=24,(s,size)
 text(x+w/2,y+(h-len(s.split('\n'))*(size+9)+9)/2-2,s,size,b,True)
def line(points,dash=False,color=INK,arrow=True):
 if not dash:d.line(points,fill=color,width=3)
 else:
  for (a,b),(c,e) in zip(points,points[1:]):
   length=math.hypot(c-a,e-b)
   for k in range(0,int(length),20):
    t=k/length;u=min(k+11,length)/length
    d.line((a+(c-a)*t,b+(e-b)*t,a+(c-a)*u,b+(e-b)*u),fill=color,width=3)
 v.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="3"'+(' stroke-dasharray="11 9"' if dash else '')+'/>')
 if arrow:
  (x,y),(a,b)=points[-1],points[-2];t=math.atan2(y-b,x-a)
  pts=[(x,y),(x-15*math.cos(t-.4),y-15*math.sin(t-.4)),(x-15*math.cos(t+.4),y-15*math.sin(t+.4))]
  d.polygon(pts,fill=color);v.append('<polygon points="'+' '.join(f'{a:.2f},{b:.2f}' for a,b in pts)+f'" fill="{color}"/>')
def plus(x,y):
 d.ellipse((x-24,y-24,x+24,y+24),fill='white',outline=INK,width=2)
 v.append(f'<circle cx="{x}" cy="{y}" r="24" fill="white" stroke="{INK}" stroke-width="2"/>');text(x,y-21,'+',32,center=True)
def stack(x,y,label,color=BLUE):
 for j in (2,1,0):rect(x+j*9,y-j*9,135,105,color)
 text(x+68,y+27,label,30,center=True)

text(50,15,'(a) Overall architecture',40,True)
rect(40,80,3520,980,'white',True)
# Input is an icon, not a synthetic research example.
rect(85,390,190,170,'#FAFAFA')
line([(110,525),(155,460),(191,495),(219,451),(253,525)],arrow=False,color='#8392A2')
d.ellipse((112,411,137,436),fill='#D6DEE7');v.append('<circle cx="124" cy="424" r="12" fill="#D6DEE7"/>')
text(180,580,'Image I',32,True,True)
box(365,270,260,135,'CLIP visual\nencoder',BLUE)
box(365,485,260,115,'YOLO\ndetections',BLUE)
box(365,740,260,115,'UPT\nHOI detector',GREEN)
line([(275,450),(315,450),(315,337),(365,337)])
line([(315,450),(315,542),(365,542)])
line([(315,542),(315,797),(365,797)])
text(495,420,'patch tokens',27,center=True)
box(735,345,260,130,'ROI pooling\nR: N × 768',BLUE)
line([(625,337),(690,337),(690,385),(735,385)])
line([(625,542),(690,542),(690,445),(735,445)])
text(687,561,'boxes',25,center=True)

# CSA: preserve object1 -> object2 -> relation and the two memories.
rect(1040,145,960,610,'#FCFDFE',True)
text(1070,157,'Cascade semantic alignment (CSA)',32,True)
box(1090,230,350,90,'Noun bank → SA',SAND,28)
box(1610,230,350,90,'Verb bank → SA',SAND,28)
text(1080,95,'TRAIN lexicon → frozen CLIP text embeddings',27)
box(1090,385,190,105,'Linear\n768 → 512',BLUE,27)
box(1350,385,240,105,'Object\nfusion 1',BLUE)
box(1660,385,240,105,'Object\nfusion 2',BLUE)
box(1660,605,240,105,'Relation\nfusion',BLUE)
line([(995,410),(1090,410)])
line([(1280,437),(1350,437)])
line([(1590,437),(1660,437)])
line([(1265,320),(1265,350),(1470,350),(1470,385)])
text(1265,335,'Pn',25,center=True)
line([(1785,320),(1970,320),(1970,657),(1900,657)])
text(1930,565,'Pv',25)
line([(1780,490),(1780,605)])
text(1803,543,'Vo',28)
text(1080,701,'SA: 150 prototypes per bank',25)

# Prompt creation and cached text/ROI associations.
box(735,740,260,115,'Prompt\nconstruction',GREEN)
box(735,915,260,90,'UPT–YOLO matching\nIoU ≥ 0.5',GREEN,27)
box(1090,845,240,115,'CLIP text\nencoder',GREEN)
stack(1480,855,'T',GREEN)
text(1550,982,'Text + ROI links',27,center=True)
line([(625,797),(735,797)])
line([(690,797),(690,960),(735,960)])
line([(995,797),(1030,797),(1030,902),(1090,902)])
line([(1330,902),(1480,902)])
line([(995,960),(1015,960),(1015,1020),(1420,1020),(1420,935),(1480,935)])
# Adapter lane has explicit object/relation outputs.
box(2110,385,300,115,'Object prompt\nadapter',GREEN)
box(2110,605,300,115,'Relation prompt\nadapter',GREEN)
line([(1900,437),(2110,437)])
text(2010,397,'Vo',28,center=True)
line([(1900,657),(2110,657)])
text(2010,618,'Vr',28,center=True)
line([(1633,900),(2055,900),(2055,490),(2110,490)])
line([(2055,900),(2055,710),(2110,710)])
text(2200,805,'Details in (b)',27)

# Dual-memory decoder and fixed selection.
box(2560,410,290,250,'Transformer\ndecoder ×3\n\nDual cross-attention',PURPLE,29)
line([(2410,437),(2480,437),(2480,470),(2560,470)])
line([(2410,657),(2480,657),(2480,600),(2560,600)])
text(2470,391,"Vo′",27)
text(2470,671,"Vr′",27)
box(2570,775,270,90,'Token prefix',PURPLE,29)
line([(2705,775),(2705,660)])
text(2705,890,'BOS / previous tokens',25,center=True)
stack(2965,485,'y1 … y5',PURPLE)
line([(2850,535),(2965,535)])
text(3040,625,'Beam search (K=5)',27,center=True)
box(3250,440,260,195,'Fixed\nreranking\n0.3 D + 0.4 C + 0.3 O',SAND,27)
line([(3118,535),(3250,535)])
box(3250,230,260,110,'Image +\nYOLO labels',SAND,28)
line([(3380,340),(3380,440)])
box(3250,795,260,95,'Final caption',SAND,30)
line([(3380,635),(3380,795)])
text(3260,920,'D: decoder score\nC: CLIPScore\nO: object consistency',25)

# Bottom inset: show computation rather than prose.
text(50,1120,'(b) Region-linked prompt adapters',40,True)
rect(40,1180,2220,735,'white',True)
text(80,1210,'Object branch',31,True)
text(80,1265,'[to; Vo[i]; bi]',30)
box(410,1240,260,100,'Slot MLP',GREEN)
box(850,1240,300,100,'Masked MHA',GREEN)
line([(340,1290),(410,1290)]);line([(670,1290),(850,1290)])
text(750,1244,'K,V',27,center=True)
text(1000,1375,'Q = Vo',29,center=True)
line([(1000,1440),(1000,1340)])
box(1250,1240,220,100,'Wout',GREEN)
line([(1150,1290),(1250,1290)]);line([(1470,1290),(1636,1290)])
plus(1660,1290);line([(1684,1290),(1930,1290)])
text(1990,1265,"Vo′",36,True)
line([(760,1440),(1660,1440),(1660,1314)])
text(1380,1395,'Vo (residual)',27,center=True)
text(80,1402,'Mask: only linked ROI i',27)

text(80,1510,'Relation branch',31,True)
text(80,1565,'[tr; Vo[s]; Vo[o]; bs; bo]',28)
box(410,1610,260,100,'Slot MLP',GREEN)
box(850,1610,300,100,'Masked MHA',GREEN)
line([(300,1615),(300,1660),(410,1660)])
line([(670,1660),(850,1660)]);text(750,1614,'K,V',27,center=True)
box(1250,1610,220,100,'Wout',GREEN)
line([(1150,1660),(1250,1660)]);line([(1470,1660),(1636,1660)])
plus(1660,1660);line([(1684,1660),(1930,1660)]);text(1990,1635,"Vr′",36,True)
line([(1000,1810),(1000,1710)]);text(1000,1745,'Q = Vr',29,center=True)
line([(760,1810),(1660,1810),(1660,1684)]);text(1380,1765,'Vr (residual)',27,center=True)
text(80,1802,'Mask: only endpoints s or o',27)
text(80,1860,'Unmatched prompts are omitted; an empty mask gives zero residual.',27,color='#58636D')

text(2300,1120,'(c) Training objective',40,True)
rect(2290,1180,1270,735,'#FFFDFA',True)
box(2340,1235,510,105,'Sampled caption ys',PURPLE,31)
box(2910,1235,570,105,'Greedy caption yg',PURPLE,31)
box(2480,1430,900,115,'A = CIDEr(ys, Ytrain) − CIDEr(yg, Ytrain)',SAND,30)
line([(2595,1340),(2595,1430)]);line([(3195,1340),(3195,1430)])
box(2350,1630,510,100,'LCE',SAND,34)
box(2960,1630,540,100,'LSCST = −A log p(ys | I)',SAND,29)
line([(2930,1545),(3230,1545),(3230,1630)])
line([(2605,1730),(2605,1765),(2730,1765),(2730,1783)])
line([(3230,1730),(3230,1765),(3090,1765),(3090,1783)])
text(2910,1790,'L = LCE + 0.05 LSCST',38,True,True)
text(2910,1860,'TRAIN references only • no object-reward loss',27,center=True,color='#58636D')
text(50,1950,'Frozen: CLIP encoders, semantic banks and detectors. Learned: CSA, prompt adapters and decoder. N = valid ROIs; d = 512.',28)
text(50,1990,'D and C are normalized within the candidate set; O is normalized detector-based object consistency. No reference captions are used for reranking.',26,color='#58636D')

im.save(P/'pipeline_paper_style.png',dpi=(300,300))
(P/'pipeline_paper_style.svg').write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/>'+''.join(v)+'</svg>',encoding='utf-8')
ET.parse(P/'pipeline_paper_style.svg')
print('Scientific architecture figure: PNG/SVG PASS')

# Vector PDF from the same SVG primitives; no rasterization.
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import toColor
pdfmetrics.registerFont(TTFont('ArialFigure','C:/Windows/Fonts/arial.ttf'))
pdfmetrics.registerFont(TTFont('ArialFigureBold','C:/Windows/Fonts/arialbd.ttf'))
c=canvas.Canvas(str(P/'pipeline_paper_style.pdf'),pagesize=(W,H))
c.setTitle('CSA with region-linked object-relation prompts')
for e in ET.parse(P/'pipeline_paper_style.svg').getroot():
 tag=e.tag.split('}')[-1];a=e.attrib
 c.setDash([11,8] if 'stroke-dasharray' in a else [])
 c.setLineWidth(float(a.get('stroke-width',2)))
 fill=a.get('fill','none');stroke=a.get('stroke','none')
 if fill!='none':c.setFillColor(toColor(fill))
 if stroke!='none':c.setStrokeColor(toColor(stroke))
 if tag=='rect':
  if a.get('width')=='100%':continue
  c.roundRect(float(a['x']),H-float(a['y'])-float(a['height']),float(a['width']),float(a['height']),float(a.get('rx',0)),fill=int(fill!='none'),stroke=int(stroke!='none'))
 elif tag=='circle':c.circle(float(a['cx']),H-float(a['cy']),float(a['r']),fill=int(fill!='none'),stroke=int(stroke!='none'))
 elif tag=='text':
  c.setFont('ArialFigureBold' if a.get('font-weight')=='bold' else 'ArialFigure',float(a['font-size']))
  if a.get('text-anchor')=='middle':c.drawCentredString(float(a['x']),H-float(a['y']),e.text or '')
  else:c.drawString(float(a['x']),H-float(a['y']),e.text or '')
 elif tag in ('polyline','polygon'):
  points=[tuple(map(float,s.split(','))) for s in a['points'].split()];q=c.beginPath();q.moveTo(points[0][0],H-points[0][1])
  for x,y in points[1:]:q.lineTo(x,H-y)
  if tag=='polygon':q.close()
  c.drawPath(q,stroke=int(stroke!='none'),fill=int(fill!='none'))
c.showPage();c.save()
print('Vector PDF written')

