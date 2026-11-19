"""One restrained overview layout proof; all operations are drawing only."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import build_diagrams as art

ROOT = Path(__file__).resolve().parent / 'layout_proof'
ROOT.mkdir(exist_ok=True)
art.ROOT = ROOT
PALE, ACCENT = '#F5F8FA', '#EDF4F4'


class Figure(art.Figure):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.process_ports = []

    def box(self, x, y, width, height, lines, fill='white', rounded=False, sizes=None, bold_first=False):
        sizes = sizes or [16] * len(lines)
        if rounded:
            new_height = height * .82
            new_y = y + (height-new_height)/2
            self.process_ports.append((x, y, width, height, new_y, new_height))
            fill = '#EEF3F8' if lines[0] == 'Word Embedding' else ('#F0F5F5' if lines[0].startswith('M3') else 'white')
        else:
            new_y, new_height = y, height
        super().box(x, new_y, width, new_height, lines, fill, rounded, [s*.85 for s in sizes], False)

    def mapped(self, x, y):
        for bx, by, bw, bh, ny, nh in self.process_ports:
            if (x == bx or x == bx+bw) and by <= y <= by+bh:
                return x, ny+(y-by)/bh*nh
            if (y == by or y == by+bh) and bx <= x <= bx+bw:
                return x, ny if y == by else ny+nh
        return x, y

    def icon(self, x, y, frozen=False):
        super().icon(x, y, frozen)
        original = self.svg.pop()
        self.svg.append(f'<g transform="translate({x},{y}) scale(.60) translate({-x},{-y})">{original}</g>')
        cell = self.cells[-1]
        cell.set('style', cell.get('style').replace('fontSize=21', 'fontSize=14'))
        g = cell.find('mxGeometry')
        g.set('x', str(x-9))
        g.set('y', str(y-10))
        g.set('width', '18')
        g.set('height', '18')

    def save(self, name):
        for i, item in enumerate(self.svg):
            if 'class="edge"' in item:
                self.svg[i] = re.sub(r'([ML])([0-9.]+),([0-9.]+)',
                    lambda m: m[1] + ','.join(str(v) for v in self.mapped(float(m[2]),float(m[3]))), item)
        for cell in self.cells:
            if cell.get('edge') == '1':
                for point in cell.findall('.//mxPoint'):
                    px, py = self.mapped(float(point.get('x')), float(point.get('y')))
                    point.set('x', str(px))
                    point.set('y', str(py))
        self.svg = [item.replace('stroke-width="1.35"', 'stroke-width="1"')
                    .replace('stroke-width:1.6', 'stroke-width:.95')
                    .replace('markerWidth="8" markerHeight="8" refX="8" refY="4"',
                             'markerWidth="6" markerHeight="6" refX="6" refY="3"')
                    .replace('M0 0 L8 4 L0 8 Z', 'M0 0 L6 3 L0 6 Z')
                    .replace('baseline-shift="sub" font-size="14"', 'baseline-shift="sub" font-size="10"')
                    for item in self.svg]
        for cell in self.cells:
            if 'style' in cell.attrib:
                cell.set('style', cell.get('style').replace('strokeWidth=1.35', 'strokeWidth=1')
                         .replace('strokeWidth=1.6', 'strokeWidth=.95'))
        super().save(name)


def main():
    f = Figure(1260, 735)
    vo, vr, vop, vrp = art.VO, art.VR, art.VOP, art.VRP
    f.box(220, 47, 130, 52, ['TRAIN captions'], sizes=[16])
    f.box(390, 47, 145, 52, ['M1 · Semantic Bank', 'Construction'], PALE, True, [15, 16])
    f.box(575, 24, 300, 106, [
        'Semantic banks · TRAIN',
        'Nouns: person, bicycle, motorcycle, …',
        'Verbs: ride, stand, park, …',
        art.sub('B', 'n') + ': 5269×512;  ' + art.sub('B', 'v') + ': 1594×512'],
        sizes=[17, 15, 15, 15])
    f.edge([(589, 61), (861, 61)], False)
    f.edge([(350, 73), (390, 73)])
    f.edge([(535, 73), (575, 73)])
    f.edge([(725, 130), (725, 340), (675, 340), (675, 375)])

    f.box(995, 20, 160, 32, ['Image I'], sizes=[16])
    f.box(995, 80, 160, 48, ['UPT', 'Interaction detector'], PALE, True, [17, 14])
    f.icon(1144, 80, True)
    f.box(795, 165, 160, 48, ['YOLO boxes', 'Classes + coordinates'], sizes=[16, 14])
    f.box(995, 165, 160, 48, ['Class + IoU matching', 'IoU ≥ 0.5'], PALE, True, [14, 16])
    f.edge([(1075, 52), (1075, 80)])
    f.edge([(1075, 128), (1075, 165)])
    f.edge([(955, 189), (995, 189)])
    f.box(925, 245, 300, 95, [
        'Object–relation prompts · example',
        'Objects: “a person”, “a bicycle”, …',
        'Relation: “a person riding a bicycle”',
        'Matched box links retained'], sizes=[16, 14, 14, 13])
    f.edge([(939, 277), (1211, 277)], False)
    f.edge([(1075, 213), (1075, 245)])
    f.box(780, 265, 120, 65, ['CLIP Text Encoder', 'ViT-B/16', art.TO + ', ' + art.TR + ': 512-D'], PALE, True, [13, 14, 14])
    f.icon(889, 265, True)
    f.edge([(925, 297.5), (900, 297.5)])
    f.edge([(840, 330), (840, 350), (1075, 350), (1075, 375)])
    f.edge([(1170, 340), (1170, 360), (1135, 360), (1135, 375)])

    f.photo(40, 200, 140, 105)
    f.box(40, 305, 140, 27, ['Input image I'], sizes=[15])
    f.edge([(110, 332), (110, 375)])
    f.box(40, 375, 140, 70, ['CLIP Vision Encoder', 'CLIP ViT-B/16'], PALE, True, [14, 15])
    f.icon(169, 375, True)
    f.box(220, 375, 170, 70, ['Visual patch tokens', 'X: 196 × 768', '14 × 14 grid; no CLS'], sizes=[16, 18, 13])
    f.box(430, 375, 125, 70, ['Linear projection', '768 → 512', 'V: 196 × 512'], PALE, True, [14, 17, 14])
    f.icon(544, 375)
    f.box(595, 375, 160, 70, ['M2 · Bank–Visual', 'Alignment'], PALE, True, [16, 17])
    f.icon(744, 375)
    f.box(795, 375, 160, 70, ['Bank-aligned memories', vo + ', ' + vr, '196 × 512 each'], sizes=[14, 19, 14])
    f.box(995, 375, 160, 70, ['M3 · Prompt–Visual', 'Alignment'], ACCENT, True, [15, 17])
    f.icon(1144, 375)
    for a, b in [(180, 220), (390, 430), (555, 595), (755, 795), (955, 995)]:
        f.edge([(a, 410), (b, 410)])

    f.box(995, 495, 160, 70, ['Prompt-aligned memories', vop + ', ' + vrp, '196 × 512 each'], sizes=[13, 19, 14])
    f.edge([(1075, 445), (1075, 495)])
    f.box(795, 495, 160, 70, ['Transformer Decoder', 'Parallel attention · ×3', 'Memory K/V: ' + vop + ', ' + vrp], PALE, True, [15, 14, 14])
    f.icon(944, 495)
    f.box(595, 495, 160, 70, ['Vocabulary projection', '+ Softmax'], PALE, True, [15, 16])
    f.icon(744, 495)
    f.box(430, 495, 125, 70, ['Beam search', 'K = 5'], PALE, True, [17, 16])
    f.box(220, 495, 170, 70, ['Caption reranking', 'Decoder · CLIP · Objects'], PALE, True, [16, 13])
    f.box(40, 495, 140, 70, ['Generated caption'], sizes=[16])
    for a, b in [(995, 955), (795, 755), (595, 555), (430, 390), (220, 180)]:
        f.edge([(a, 530), (b, 530)])
    f.box(795, 605, 160, 55, ['Word Embedding', '+ Position Encoding'], PALE, True, [16, 14])
    f.icon(944, 605)
    f.box(795, 690, 160, 30, ['Caption prefix'], sizes=[16])
    f.edge([(875, 690), (875, 660)])
    f.edge([(875, 605), (875, 565)])
    f.box(220, 605, 170, 55, ['Image I + YOLO labels', 'Detector-based support'], sizes=[15, 14])
    f.edge([(305, 605), (305, 565)])
    f.save('pipeline_layout_proof')
    for ext in ('.svg', '.drawio'):
        ET.parse(ROOT / ('pipeline_layout_proof' + ext))
    print('One overview layout proof generated; XML PASS; no model execution.')


if __name__ == '__main__':
    main()
