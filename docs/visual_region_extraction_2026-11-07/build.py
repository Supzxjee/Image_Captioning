"""Editable engineering diagram, no detector/model run or photo modification."""
from pathlib import Path
from html import escape
import base64,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
asset=P.parent/'csa_pipeline_2026-11-02/v3/assets/COCO_train2014_000000263974.jpg'
W,H=900,1440
v=[]
def rect(x,y,w,h,fill='#f3f6f9',radius=0,stroke='#607181',role='data'):
 v.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2" data-role="{role}"/>')
def text(x,y,s,size=29,bold=False,color='#233442'):
 v.append(f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="{size}" text-anchor="middle" fill="{color}"'+(' font-weight="bold"' if bold else '')+'>'+escape(s)+'</text>')
def arrow(points):
 v.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+'" fill="none" stroke="#536676" stroke-width="2.5" marker-end="url(#arrow)"/>')
rect(20,20,860,1400,'white',12,'#81909b','container')
text(450,72,'Visual region extraction',36,True)
# All data nodes use square corners; all processing modules have rounded corners.
for dx,dy in [(12,-12),(6,-6),(0,0)]:rect(300+dx,150+dy,300,86,'#edf4f8')
text(450,184,'Region features V',28,True);text(450,218,'N × 512',24)
rect(300,300,300,96,'#f0f3f6',16,role='module')
text(450,341,'Linear projection',30,True);text(450,376,'768 → 512',25)
rect(300,475,300,96,'#f0f3f6',16,role='module')
text(450,534,'ROI pooling',32,True)
for dx,dy in [(12,-12),(6,-6),(0,0)]:rect(90+dx,665+dy,280,110,'#f6f8fa')
text(230,708,'Patch features',29,True);text(230,747,'14 × 14 × 768',24)
rect(530,665,280,110,'#f6f8fa')
text(670,708,'Bounding boxes B',28,True);text(670,747,'xyxy coordinates',24)
rect(90,860,280,110,'#dfedf7',16,role='module')
text(230,905,'CLIP ViT-B/16',29,True);text(230,946,'Frozen visual encoder',23)
rect(530,860,280,110,'#f0f3f6',16,role='module')
text(670,905,'YOLO detector',29,True);text(670,946,'Frozen',24)
raw=base64.b64encode(asset.read_bytes()).decode()
v.append(f'<image x="260" y="1055" width="380" height="285" href="data:image/jpeg;base64,{raw}"/>')
rect(260,1055,380,285,'none',0,'#89959f')
text(450,1378,'Input image I',31,True)
arrow([(450,1055),(450,1015),(230,1015),(230,970)])
arrow([(450,1015),(670,1015),(670,970)])
v.append('<circle cx="450" cy="1015" r="4" fill="#536676"/>')
arrow([(230,860),(230,775)]);arrow([(670,860),(670,775)])
arrow([(230,653),(230,625),(380,625),(380,571)])
arrow([(670,665),(670,625),(520,625),(520,571)])
arrow([(450,475),(450,396)]);arrow([(450,300),(450,236)])
text(450,1404,'Rounded: modules · Square / stack: data',18,color='#607181')
svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L9,4.5 L0,9 Z" fill="#536676"/></marker></defs><rect width="100%" height="100%" fill="white"/>'+''.join(v)+'</svg>'
r=ET.fromstring(svg)
assert len([e for e in r.iter() if e.get('data-role')=='module' and e.get('rx')=='16'])==4
assert all(e.get('rx')=='0' for e in r.iter() if e.get('data-role')=='data')
(P/'visual_region_extraction.svg').write_text(svg,encoding='utf-8')
print('SVG created; four rounded modules, square data, clean input image')
