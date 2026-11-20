"""Editable integrated M2 mechanism figure. Drawing only; no model imports."""
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
        super().__init__(1760, 1400)
        self.boxes, self.paths = [], []

    def node(self, x, y, w, h, lines, fill='white', process=False, sizes=None):
        self.boxes.append((x, y, w, h, lines[0]))
        self.box(x, y, w, h, [html.escape(t) for t in lines], fill, process,
                 sizes or [24] * len(lines))

    def edge(self, points, arrow=True):
        self.paths.append(points)
        super().edge(points, arrow)

    def frame(self, x, y, w, h, title):
        super().frame(x, y, w, h, title)
        self.svg.pop()
        self.cells.remove(self.cells[-1])
        # Keep the heading between the visual and semantic input wires.
        self.text(x+281, y+28, html.escape(title), 24, True)


def stage(f, x, title, visual, semantic, selected, output, fill):
    f.frame(x, 320, 500, 870, title)
    f.node(x+60, 240, 220, 55, ['Visual memory '+visual, '196 × 512'], sizes=[23, 21])
    f.node(x+315, 240, 155, 55, [semantic, '150 × 512'], sizes=[25, 21])
    main = x+170
    f.node(x+60, 390, 220, 70, ['Self-attention', 'Q = K = V: '+visual], fill, True, [26, 20])
    f.node(x+315, 390, 155, 70, ['Cosine', 'reweighting'], fill, True, [24, 23])
    f.node(x+315, 530, 155, 55, [selected, '150 × 512'], sizes=[25, 21])
    f.node(x+60, 500, 220, 40, ['Add & Norm'], process=True, sizes=[24])
    f.node(x+60, 570, 220, 44, ['FFN'], fill, True, [26])
    f.node(x+60, 650, 220, 40, ['Add & Norm · H'], process=True, sizes=[23])
    f.node(x+60, 750, 220, 70, ['Cross-attention', 'Q: H · K,V: '+selected], fill, True, [26, 19])
    f.node(x+60, 860, 220, 40, ['Add & Norm'], process=True, sizes=[24])
    f.node(x+60, 930, 220, 44, ['FFN'], fill, True, [26])
    f.node(x+60, 1010, 220, 40, ['Add & Norm · C'], process=True, sizes=[23])
    f.node(x+60, 1100, 220, 60, ['Gated fusion', 'gH + (1 − g)C'], fill, True, [26, 22])
    f.node(x+60, 1225, 220, 55, [output, '196 × 512'], sizes=[26, 21])

    # Visual input serves self-attention and image-dependent prototype weighting.
    f.edge([(main,295),(main,390)])
    f.edge([(main,355),(x+295,355),(x+295,425),(x+315,425)])
    f.dot(main,355)
    f.edge([(x+392.5,295),(x+392.5,390)])
    f.edge([(x+392.5,460),(x+392.5,530)])
    f.edge([(x+392.5,585),(x+392.5,795),(x+280,795)])
    for start,end in [(460,500),(540,570),(614,650),(690,750),(820,860),(900,930),(974,1010),(1050,1100),(1160,1225)]:
        f.edge([(main,start),(main,end)])

    # Each Add & Norm includes the residual represented by its explicit side wire.
    for start,end in [(370,520),(555,670),(715,880),(915,1030)]:
        f.edge([(main,start),(x+45,start),(x+45,end),(x+60,end)])
        f.dot(main,start)
    # H bypasses the cross-attention branch and reaches the scalar gate separately.
    f.edge([(x+45,715),(x+20,715),(x+20,1130),(x+60,1130)])
    f.dot(x+45,715)


def validate(f):
    collisions = []
    for points in f.paths:
        for (ax,ay),(bx,by) in zip(points,points[1:]):
            assert ax == bx or ay == by, ('Non-orthogonal edge',points)
            for x,y,w,h,label in f.boxes:
                if ay == by and y < ay < y+h and max(min(ax,bx),x) < min(max(ax,bx),x+w):
                    collisions.append((label,(ax,ay),(bx,by)))
                if ax == bx and x < ax < x+w and max(min(ay,by),y) < min(max(ay,by),y+h):
                    collisions.append((label,(ax,ay),(bx,by)))
    assert not collisions, collisions
    for suffix in ('svg','drawio'):
        ET.parse(ROOT / ('M2_expanded.'+suffix))
    source = ROOT.parents[1] / 'baseline_final_pipeline/final_pipeline/xe/bank_layers.py'
    record = dict(status='STATIC_GEOMETRY_PASS_VISUAL_QA_PENDING', processing_and_data_boxes=len(f.boxes),
                  connectors=len(f.paths), all_connectors_orthogonal=True, unrelated_box_crossings=0,
                  mechanism_source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  models_executed=False)
    (ROOT/'CHECKS.json').write_text(json.dumps(record,indent=2),encoding='utf-8')


def main():
    f = Figure()
    f.text(880, 35, 'M2 · Semantic Bank–Visual Alignment', 28)
    f.node(60,70,200,60,['Noun bank Bₙ','5269 × 512'],sizes=[25,22])
    f.node(350,70,200,60,['Noun aggregation'], '#F4F7F9',True,[23])
    f.node(1220,70,200,60,['Verb bank Bᵥ','1594 × 512'],sizes=[25,22])
    f.node(1510,70,200,60,['Verb aggregation'],'#F4F7F9',True,[23])
    f.edge([(260,100),(350,100)])
    f.edge([(1420,100),(1510,100)])
    f.edge([(450,130),(450,190),(432.5,190),(432.5,240)])
    f.edge([(1610,130),(1610,190),(1592.5,190),(1592.5,240)])
    stage(f,40,'Object alignment 1','V','Prototypes Pₙ','T₁','V₁','#FFF1E6')
    stage(f,620,'Object alignment 2','V₁','Input T₁','T₂','Vₒ','#FFF1E6')
    stage(f,1200,'Relation alignment','Vₒ','Prototypes Pᵥ','Tᵣ','Vᵣ','#EAF4F8')

    # Both outputs of stage1 enter stage2 through independent clear channels.
    f.edge([(510,557.5),(565,557.5),(565,170),(1012.5,170),(1012.5,240)])
    f.edge([(320,1252.5),(590,1252.5),(590,267.5),(680,267.5)])
    f.edge([(900,1252.5),(1170,1252.5),(1170,267.5),(1260,267.5)])
    f.dot(790,1280)
    f.node(1050,1310,640,60,['Visual memories Vₒ, Vᵣ','196 × 512 each'],sizes=[27,22])
    f.edge([(790,1280),(790,1340),(1050,1340)])
    f.edge([(1370,1280),(1370,1310)])
    f.save('M2_expanded')
    validate(f)
    print('M2 SVG/native draw.io generated; orthogonal geometry and XML PASS.')


if __name__ == '__main__':
    main()
