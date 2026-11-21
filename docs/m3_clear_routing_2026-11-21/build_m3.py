"""Draw M3 with explicit memory and ROI-control routes; no model execution."""
from pathlib import Path
import hashlib
import html
import json
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'module_m3_2026-11-19'))
import build_diagrams as art

art.ROOT = ROOT


class Figure(art.Figure):
    def __init__(self):
        super().__init__(1750, 1000)
        self.nodes, self.routes = [], []

    def node(self, x, y, w, h, lines, fill='white', process=False, sizes=None):
        self.nodes.append((x,y,w,h,lines[0]))
        self.box(x,y,w,h,[html.escape(s) for s in lines],fill,process,sizes or [23]*len(lines))

    def edge(self, points, arrow=True, control=False):
        self.routes.append((points,control))
        super().edge(points,arrow)
        if control:
            self.svg[-1] = self.svg[-1].replace('class="edge"','class="edge" style="stroke-dasharray:6 4"')
            cell = self.cells[-1]
            cell.set('style',cell.get('style')+'dashed=1;dashPattern=6 4;')


def branch(f, row, relation=False):
    fill = '#EAF4F8' if relation else '#FFF1E6'
    control_y = row-160
    mask_y = control_y-25
    f.node(300,control_y,170,70,['ROI links','valid slots'],sizes=[24,21])
    f.node(510,control_y,230,70,
           ['Relation prompt','tᵣ + pair geometry'] if relation else ['Object prompt','tₒ + box geometry'],sizes=[24,21])
    f.node(300,row,170,70,['Subject–object','ROI pooling'] if relation else ['Object ROI','pooling'],fill,True,[23,24])
    f.node(540,row,170,70,['Concatenate'],fill,True,[24])
    f.node(780,row,140,70,['Slot MLP'],fill,True,[24])
    f.node(1000,row,200,70,['Masked cross-','attention'],fill,True,[24,24])
    f.node(1270,row,150,70,['Residual add'],fill,True,[22])
    f.node(1490,row-10,220,90,
           ['Updated relation','memory Vᵣ′','196 × 512'] if relation else ['Updated object','memory Vₒ′','196 × 512'],sizes=[23,25,20])
    for start,end in [(470,540),(710,780),(920,1000),(1200,1270),(1420,1490)]:
        f.edge([(start,row+35),(end,row+35)])
    f.edge([(625,control_y+70),(625,row)])
    f.edge([(385,control_y+70),(385,row)],control=True)
    f.edge([(385,control_y),(385,mask_y),(1100,mask_y),(1100,row)],control=True)


def validate(f):
    for points,_ in f.routes:
        for (ax,ay),(bx,by) in zip(points,points[1:]):
            assert ax==bx or ay==by, ('Non-orthogonal route',points)
            for x,y,w,h,label in f.nodes:
                if ay==by and y<ay<y+h:
                    assert max(min(ax,bx),x)>=min(max(ax,bx),x+w), ('Wire through box',label,points)
                if ax==bx and x<ax<x+w:
                    assert max(min(ay,by),y)>=min(max(ay,by),y+h), ('Wire through box',label,points)
    for suffix in ('svg','drawio'):
        ET.parse(ROOT/('M3_clear_routing.'+suffix))
    repo=ROOT.parents[1]
    sources=['baseline_final_pipeline/final_pipeline/xe/csa_patch_model.py','baseline_final_pipeline/final_pipeline/xe/semantic_patch_model.py']
    checks=dict(status='STATIC_PASS_VISUAL_QA_PENDING',nodes=len(f.nodes),connectors=len(f.routes),
                all_connectors_orthogonal=True,unrelated_box_crossings=0,
                source_sha256={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sources},
                models_executed=False,formal_reports_modified=False)
    (ROOT/'CHECKS.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')


def main():
    f=Figure()
    f.text(40,35,'M3 · Image-Specific Prompt–Visual Alignment',27,anchor='start')
    f.frame(280,65,1450,420,'Object prompt alignment')
    f.frame(280,510,1450,460,'Relation prompt alignment')
    f.node(40,310,180,70,['Object memory Vₒ','196 × 512'],sizes=[20,20])
    f.node(40,880,180,70,['Relation memory Vᵣ','196 × 512'],sizes=[19,20])
    branch(f,310)
    branch(f,750,True)

    # A single original Vo source serves both pooling rows and the object update.
    f.edge([(220,345),(300,345)])
    f.edge([(250,345),(250,785),(300,785)])
    f.dot(250,345)
    f.edge([(250,440),(1060,440),(1060,380)])
    f.edge([(1060,440),(1345,440),(1345,380)])
    f.dot(250,440)
    f.dot(1060,440)

    # Relation query and residual are original Vr, not the Vo used in pair pooling.
    f.edge([(220,915),(1060,915),(1060,820)])
    f.edge([(1060,915),(1345,915),(1345,820)])
    f.dot(1060,915)
    f.save('M3_clear_routing')
    validate(f)
    print('M3 SVG/native draw.io, XML and orthogonal routes PASS.')


if __name__=='__main__':
    main()
