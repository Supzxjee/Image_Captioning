"""Draw the complete system overview with independent semantic data paths."""
from pathlib import Path
import hashlib
import html
import json
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'module_m3_2026-11-19'))
import build_diagrams as art
art.ROOT=ROOT
PALE,OBJECT,RELATION='#F5F8FA','#FFF1E6','#EAF4F8'


class Figure(art.Figure):
    def __init__(self):
        super().__init__(1840,1160)
        self.nodes,self.routes=[],[]

    def node(self,x,y,w,h,lines,fill='white',process=False,sizes=None):
        self.nodes.append((x,y,w,h,lines[0]))
        self.box(x,y,w,h,[html.escape(s) for s in lines],fill,process,sizes or [21]*len(lines))

    def edge(self,points,arrow=True):
        self.routes.append(points)
        super().edge(points,arrow)


def validate(f):
    for points in f.routes:
        for (ax,ay),(bx,by) in zip(points,points[1:]):
            assert ax==bx or ay==by,('Not orthogonal',points)
            for x,y,w,h,label in f.nodes:
                if ay==by and y<ay<y+h:
                    assert max(min(ax,bx),x)>=min(max(ax,bx),x+w),('Wire through box',label,points)
                if ax==bx and x<ax<x+w:
                    assert max(min(ay,by),y)>=min(max(ay,by),y+h),('Wire through box',label,points)
    for ext in ('svg','drawio'):
        root=ET.parse(ROOT/('Full_pipeline_split.'+ext)).getroot()
        if ext=='drawio':
            image_cells=[c for c in root.findall('.//mxCell') if c.get('style','').startswith('shape=image;')]
            assert len(image_cells)==1, 'Only the input photograph should be raster.'
    record=dict(status='STATIC_PASS_VISUAL_QA_PENDING',nodes=len(f.nodes),connectors=len(f.routes),
                all_connectors_orthogonal=True,unrelated_box_crossings=0,
                separate_data=['Bₙ','Bᵥ','Vₒ','Vᵣ','Vₒ′','Vᵣ′'],
                models_executed=False,formal_reports_modified=False)
    (ROOT/'CHECKS.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    f=Figure()
    f.text(920,28,'Image Captioning Pipeline',28)
    # M1 is represented at module scale; fixed banks retain separate identities.
    f.node(40,65,170,80,['TRAIN captions'],sizes=[23])
    f.node(300,50,210,110,['M1','Semantic bank','construction'],PALE,True,[26,22,22])
    f.node(260,230,180,65,['Noun bank Bₙ','5269 × 512'],OBJECT,sizes=[23,21])
    f.node(535,230,180,65,['Verb bank Bᵥ','1594 × 512'],RELATION,sizes=[23,21])
    f.edge([(210,105),(300,105)])
    f.edge([(350,160),(350,230)])
    f.edge([(450,160),(450,190),(625,190),(625,230)])

    # Cached image-specific prompt preparation and metadata bypass.
    f.node(900,40,130,55,['Image I'],sizes=[23])
    f.node(1070,40,160,55,['UPT detector'],PALE,True,[22]);f.icon(1218,40,True)
    f.node(1270,40,220,55,['Class + IoU matching','IoU ≥ 0.5'],PALE,True,[20,21])
    f.node(1570,40,170,55,['YOLO boxes'],sizes=[22])
    f.edge([(1030,67.5),(1070,67.5)])
    f.edge([(1230,67.5),(1270,67.5)])
    f.edge([(1570,67.5),(1490,67.5)])
    f.node(1010,145,320,100,['Matched prompt record','e.g., person; bicycle','e.g., person riding bicycle'],sizes=[24,21,20])
    f.edge([(1380,95),(1380,120),(1170,120),(1170,145)])
    f.node(1390,160,170,70,['CLIP Text Encoder','ViT-B/16'],PALE,True,[20,21]);f.icon(1548,160,True)
    f.node(1610,145,170,100,['Prompt embeddings','tₒ, tᵣ','L₂-normalized · 512-D'],sizes=[19,27,16])
    f.node(1010,330,260,65,['ROI links + valid slots','Box / pair geometry'],sizes=[22,21])
    f.edge([(1330,195),(1390,195)])
    f.edge([(1560,195),(1610,195)])
    f.edge([(1170,245),(1170,330)])

    # Frozen visual encoding and the main pair of alignment modules.
    f.photo(40,410,160,95)
    f.node(40,505,160,30,['Input image I'],sizes=[20])
    f.node(40,560,160,60,['CLIP Vision Encoder','ViT-B/16'],PALE,True,[17,22]);f.icon(188,560,True)
    f.node(40,650,160,80,['Patch tokens','without CLS','196 × 768'],sizes=[23,19,23])
    f.node(40,770,160,60,['Linear projection','768 → 512'],PALE,True,[21,23]);f.icon(188,770)
    f.node(260,770,160,60,['Visual memory V','196 × 512'],sizes=[20,22])
    f.edge([(120,535),(120,560)])
    f.edge([(120,620),(120,650)])
    f.edge([(120,730),(120,770)])
    f.edge([(200,800),(260,800)])
    f.node(500,600,220,150,['M2','Semantic bank–','visual alignment'],PALE,True,[28,24,24]);f.icon(708,600)
    f.node(820,550,180,65,['Object memory Vₒ','196 × 512'],OBJECT,sizes=[22,21])
    f.node(820,710,180,65,['Relation memory Vᵣ','196 × 512'],RELATION,sizes=[20,21])
    f.edge([(350,295),(350,430),(560,430),(560,600)])
    f.edge([(625,295),(625,600)])
    f.edge([(420,800),(460,800),(460,675),(500,675)])
    f.edge([(720,640),(770,640),(770,582.5),(820,582.5)])
    f.edge([(720,710),(770,710),(770,742.5),(820,742.5)])
    f.node(1180,600,240,150,['M3','Image-specific prompt–','visual alignment'],PALE,True,[28,21,24]);f.icon(1408,600)
    f.edge([(1000,582.5),(1090,582.5),(1090,640),(1180,640)])
    f.edge([(1000,742.5),(1090,742.5),(1090,710),(1180,710)])
    f.edge([(1140,395),(1140,520),(1260,520),(1260,600)])
    f.edge([(1695,245),(1695,410),(1350,410),(1350,600)])

    # Both updated memories enter the parallel decoder independently.
    f.node(1520,550,200,65,['Object memory Vₒ′','196 × 512'],OBJECT,sizes=[23,21])
    f.node(1520,710,200,65,['Relation memory Vᵣ′','196 × 512'],RELATION,sizes=[21,21])
    f.edge([(1420,640),(1470,640),(1470,582.5),(1520,582.5)])
    f.edge([(1420,710),(1470,710),(1470,742.5),(1520,742.5)])
    f.node(1520,860,200,80,['Transformer Decoder','3 layers · parallel attention'],PALE,True,[21,17]);f.icon(1708,860)
    f.edge([(1720,582.5),(1770,582.5),(1770,840),(1660,840),(1660,860)])
    f.edge([(1620,775),(1620,860)])
    f.node(1520,980,200,60,['Word embedding','+ Position encoding'],PALE,True,[22,20]);f.icon(1708,980)
    f.node(1520,1080,200,50,['Caption prefix'],sizes=[23])
    f.edge([(1620,1080),(1620,1040)])
    f.edge([(1620,980),(1620,940)])
    f.node(1260,860,200,80,['Vocabulary projection','+ Softmax'],PALE,True,[20,24]);f.icon(1448,860)
    f.node(1050,860,150,80,['Beam search','K = 5'],PALE,True,[23,24])
    f.node(820,860,170,80,['Candidate captions','5 per image'],sizes=[21,21])
    f.node(530,845,220,110,['Caption reranking','Decoder likelihood','CLIP alignment','Object consistency'],PALE,True,[23,19,19,19])
    f.node(260,860,200,80,['Final caption'],sizes=[25])
    for start,end in [(1520,1460),(1260,1200),(1050,990),(820,750),(530,460)]:
        f.edge([(start,900),(end,900)])
    f.node(530,1000,220,60,['Image I + YOLO labels'],sizes=[21])
    f.edge([(640,1000),(640,955)])
    f.save('Full_pipeline_split')
    validate(f)
    print('Full pipeline split-bank/memory SVG and draw.io generated; XML/routes PASS.')


if __name__=='__main__':
    main()
