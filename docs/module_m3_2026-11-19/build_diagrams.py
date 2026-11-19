"""Editable architecture figures; standard library only, no model execution."""
from pathlib import Path
import base64
import html
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
DOWNLOADS = Path('C:/Users/leduc/Downloads')
BLUE, OBJECT, RELATION = '#EAF2F8', '#FCEBDC', '#E4F2F5'
INK, BORDER = '#202B33', '#697A87'


class Figure:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                    '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="8" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L8 4 L0 8 Z" fill="#30383E"/></marker></defs>',
                    f'<rect width="{width}" height="{height}" fill="white"/>',
                    f'<style>text{{font-family:"Times New Roman",serif;fill:{INK}}}.edge{{fill:none;stroke:#30383E;stroke-width:1.6;marker-end:url(#arrow)}}</style>']
        self.mx = ET.Element('mxfile', host='app.diagrams.net', version='24.7.17')
        diagram = ET.SubElement(self.mx, 'diagram', name='Architecture', id='diagram')
        model = ET.SubElement(diagram, 'mxGraphModel', dx=str(width), dy=str(height), grid='1', gridSize='10', page='1', pageScale='1', pageWidth=str(width), pageHeight=str(height))
        self.cells = ET.SubElement(model, 'root')
        ET.SubElement(self.cells, 'mxCell', id='0')
        ET.SubElement(self.cells, 'mxCell', id='1', parent='0')
        self.index = 1

    def cell(self, **kwargs):
        self.index += 1
        return ET.SubElement(self.cells, 'mxCell', id=str(self.index), parent='1', **kwargs)

    def text(self, x, y, value, size=20, bold=False, anchor='middle', italic=False):
        weight = 'font-weight="bold"' if bold else ''
        style = 'font-style="italic"' if italic else ''
        self.svg.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" {weight} {style}>{value}</text>')
        label = value.replace('<tspan baseline-shift="sub" font-size="14">', '<sub>').replace('<tspan baseline-shift="super" font-size="14">', '<sup>').replace('</tspan>', '</sub>')
        label = re.sub(r'</sub>(?!.*<sup>)', '</sub>', label)
        # Draw.io HTML understands sub/sup; SVG uses typographic tspans.
        label = re.sub(r'<sup>(.*?)</sub>', r'<sup>\1</sup>', label)
        cell = self.cell(value=label, vertex='1', style=f'text;html=1;strokeColor=none;fillColor=none;align={"center" if anchor == "middle" else "left"};verticalAlign=middle;fontFamily=Times New Roman;fontSize={size};fontColor={INK};fontStyle={int(bold) + 2 * int(italic)};whiteSpace=wrap;')
        width = max(120, len(re.sub('<[^>]+>', '', html.unescape(value))) * size * .57)
        ET.SubElement(cell, 'mxGeometry', x=str(x - width / 2 if anchor == 'middle' else x), y=str(y - size), width=str(width), height=str(size * 1.4), **{'as': 'geometry'})

    def box(self, x, y, width, height, lines, fill='white', rounded=False, sizes=None, bold_first=False):
        self.svg.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="{9 if rounded else 0}" fill="{fill}" stroke="{BORDER}" stroke-width="1.35"/>')
        cell = self.cell(value='', vertex='1', style=f'rounded={int(rounded)};arcSize=12;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={BORDER};strokeWidth=1.35;')
        ET.SubElement(cell, 'mxGeometry', x=str(x), y=str(y), width=str(width), height=str(height), **{'as': 'geometry'})
        sizes = sizes or [20] * len(lines)
        total = sum(sizes) + 5 * (len(lines) - 1)
        cursor = y + (height - total) / 2
        for i, (line, size) in enumerate(zip(lines, sizes)):
            cursor += size
            self.text(x + width / 2, cursor - 2, line, size, bold_first and i == 0)
            cursor += 5

    def frame(self, x, y, width, height, title):
        self.svg.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="0" fill="none" stroke="#A9B8C2" stroke-width="1.2" stroke-dasharray="6 4"/>')
        cell = self.cell(value='', vertex='1', style='rounded=0;dashed=1;dashPattern=6 4;fillColor=none;strokeColor=#A9B8C2;')
        ET.SubElement(cell, 'mxGeometry', x=str(x), y=str(y), width=str(width), height=str(height), **{'as': 'geometry'})
        self.text(x + 16, y + 27, html.escape(title), 22, True, 'start')

    def edge(self, points, arrow=True):
        path = 'M' + ' L'.join(f'{x},{y}' for x, y in points)
        self.svg.append(f'<path d="{path}" class="edge"' + ('/>' if arrow else ' style="marker-end:none"/>'))
        cell = self.cell(value='', edge='1', style=f'edgeStyle=none;html=1;endArrow={"block" if arrow else "none"};endFill=1;strokeColor=#30383E;strokeWidth=1.6;rounded=0;')
        geometry = ET.SubElement(cell, 'mxGeometry', relative='1', **{'as': 'geometry'})
        ET.SubElement(geometry, 'mxPoint', x=str(points[0][0]), y=str(points[0][1]), **{'as': 'sourcePoint'})
        ET.SubElement(geometry, 'mxPoint', x=str(points[-1][0]), y=str(points[-1][1]), **{'as': 'targetPoint'})
        if len(points) > 2:
            arr = ET.SubElement(geometry, 'Array', **{'as': 'points'})
            for x, y in points[1:-1]:
                ET.SubElement(arr, 'mxPoint', x=str(x), y=str(y))

    def dot(self, x, y):
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="3" fill="#30383E"/>')

    def add(self, x, y):
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="16" fill="white" stroke="{BORDER}" stroke-width="1.4"/>')
        self.svg.append(f'<path d="M{x-8},{y}H{x+8} M{x},{y-8}V{y+8}" fill="none" stroke="{INK}" stroke-width="1.5"/>')
        cell = self.cell(value='+', vertex='1', style=f'ellipse;html=1;fillColor=#FFFFFF;strokeColor={BORDER};fontSize=25;fontFamily=Times New Roman;')
        ET.SubElement(cell, 'mxGeometry', x=str(x-16), y=str(y-16), width='32', height='32', **{'as': 'geometry'})

    def icon(self, x, y, frozen=False):
        if frozen:
            self.svg.append(f'<g transform="translate({x},{y})" stroke="#4094BB" stroke-width="1.6"><path d="M0,-8V8 M-7,-4L7,4 M-7,4L7,-4 M-3,-6L0,-3L3,-6 M-3,6L0,3L3,6"/></g>')
        else:
            self.svg.append(f'<g transform="translate({x-8},{y-14})"><path d="M8 0 C13 9 20 10 16 19 C12 27 1 25 0 17 C-1 11 4 8 5 4 C5 9 8 9 8 0Z" fill="#EEA65B" stroke="#BA642C" stroke-width="1"/><path d="M8 12 C12 17 12 20 8 21 C4 20 5 17 8 12Z" fill="#FFF2BD"/></g>')
        cell = self.cell(value='❄' if frozen else '🔥', vertex='1', style='text;html=1;strokeColor=none;fillColor=none;align=center;fontSize=21;')
        ET.SubElement(cell, 'mxGeometry', x=str(x-13), y=str(y-17), width='26', height='28', **{'as': 'geometry'})

    def photo(self, x, y, width, height):
        data = base64.b64encode((DOWNLOADS / 'input_image_I.jpg').read_bytes()).decode()
        self.svg.append(f'<image x="{x}" y="{y}" width="{width}" height="{height}" preserveAspectRatio="xMidYMid slice" href="data:image/jpeg;base64,{data}"/>')
        cell = self.cell(value='', vertex='1', style=f'shape=image;imageAspect=0;aspect=fixed;image=data:image/jpeg,{data};')
        ET.SubElement(cell, 'mxGeometry', x=str(x), y=str(y), width=str(width), height=str(height), **{'as': 'geometry'})

    def save(self, name):
        (ROOT / (name + '.svg')).write_text('\n'.join(self.svg + ['</svg>']), encoding='utf-8')
        ET.indent(self.mx)
        ET.ElementTree(self.mx).write(ROOT / (name + '.drawio'), encoding='utf-8', xml_declaration=True)


def sub(base, suffix):
    return base + '<tspan baseline-shift="sub" font-size="14">' + suffix + '</tspan>'


VO, VR = sub('V', 'o'), sub('V', 'r')
VOP, VRP = sub('V', 'o') + '′', sub('V', 'r') + '′'
TO, TR = sub('t', 'o'), sub('t', 'r')
WO, WS, WT = sub('W', 'o'), sub('W', 's'), sub('W', 't')


def m3():
    f = Figure(1500, 1020)
    f.text(750, 32, 'M3 · Image-Specific Prompt–Visual Alignment', 28, True)
    f.frame(20, 55, 1460, 260, 'Image-specific prompt preparation')
    f.photo(40, 100, 100, 100)
    f.text(90, 225, 'Image I', 20, True)
    f.box(185, 100, 170, 60, ['UPT', 'Interaction detector'], BLUE, True, [21, 18], True)
    f.icon(344, 98, True)
    f.box(405, 95, 240, 70, ['Object / relation phrases'], 'white', sizes=[20])
    f.box(695, 95, 220, 70, ['CLIP Text Encoder', 'ViT-B/16'], BLUE, True, [21, 18], True)
    f.icon(904, 94, True)
    f.box(970, 95, 160, 70, ['L2 normalization'], BLUE, True, [19])
    f.box(1180, 95, 260, 70, [TO + ', ' + TR, '512-D prompt embeddings'], 'white', sizes=[24, 19])
    f.box(185, 245, 170, 50, ['YOLO detector'], BLUE, True, [20], True)
    f.icon(344, 244, True)
    f.box(405, 225, 240, 70, ['Box matching', 'Same class; IoU ≥ 0.5'], BLUE, True, [21, 18], True)
    f.box(695, 225, 220, 70, ['Patch overlap weights', '14 × 14 grid'], BLUE, True, [20, 19], True)
    f.box(970, 225, 230, 70, [WO + ', ' + WS + ', ' + WT, 'Patch weights'], 'white', sizes=[24, 20])
    f.box(1245, 225, 195, 70, [sub('b', 'i') + ', g', 'Box / pair geometry'], 'white', sizes=[24, 18])
    f.edge([(140, 130), (185, 130)])
    f.edge([(160, 130), (160, 270), (185, 270)])
    f.dot(160, 130)
    f.edge([(355, 130), (405, 130)])
    f.edge([(645, 130), (695, 130)])
    f.edge([(915, 130), (970, 130)])
    f.edge([(1130, 130), (1180, 130)])
    f.edge([(270, 160), (270, 190), (525, 190), (525, 225)])
    f.text(435, 185, 'UPT boxes + classes', 17)
    f.edge([(355, 270), (405, 270)])
    f.edge([(645, 260), (695, 260)])
    f.edge([(915, 260), (970, 260)])
    f.edge([(525, 295), (525, 305), (1342, 305), (1342, 295)])

    f.frame(20, 340, 1460, 295, 'Object prompt–visual alignment')
    f.box(40, 445, 135, 65, [VO, '196 × 512'], sizes=[25, 18])
    f.box(225, 445, 180, 65, ['Box-weighted pooling'], BLUE, True, [19])
    f.box(225, 375, 180, 42, [WO], sizes=[24])
    f.box(465, 375, 190, 42, [TO + ', ' + sub('b', 'i')], sizes=[24])
    f.box(465, 445, 190, 65, ['Concatenation', '[' + TO + '; ' + sub('v', 'i') + '; ' + sub('b', 'i') + ']'], OBJECT, True, [20, 23], True)
    f.box(715, 445, 150, 65, ['Slot MLP', '1028 → 512 → 512'], OBJECT, True, [21, 16], True)
    f.icon(852, 443)
    f.box(915, 445, 120, 65, [sub('S', 'o'), '≤ 8 × 512'], OBJECT, sizes=[25, 18])
    f.box(1095, 365, 160, 52, [sub('M', 'o') + ': ' + WO + 'ᵀ &gt; 0', 'Valid slots only'], sizes=[19, 16])
    f.box(1095, 445, 160, 65, ['Masked', 'cross-attention'], OBJECT, True, [21, 20], True)
    f.icon(1243, 443)
    f.add(1320, 477)
    f.box(1370, 445, 90, 65, [VOP, '196 × 512'], sizes=[25, 16])
    f.edge([(175, 477), (225, 477)])
    f.edge([(315, 417), (315, 445)])
    f.edge([(405, 477), (465, 477)])
    f.text(435, 464, sub('v', 'i'), 22)
    f.edge([(560, 417), (560, 445)])
    for a, b in [(655, 715), (865, 915), (1035, 1095), (1255, 1304), (1336, 1370)]:
        f.edge([(a, 477), (b, 477)])
    f.text(1065, 464, 'K, V', 18)
    f.edge([(1175, 417), (1175, 445)])
    f.edge([(107, 510), (107, 565), (1175, 565), (1175, 510)])
    f.text(1189, 547, 'Q', 20)
    f.edge([(107, 565), (107, 605), (1320, 605), (1320, 493)])
    f.dot(107, 565)
    f.text(700, 594, 'Residual ' + VO, 19)

    f.frame(20, 660, 1460, 340, 'Relation prompt–visual alignment')
    f.box(40, 795, 135, 65, [VO, '196 × 512'], sizes=[25, 18])
    f.box(225, 725, 180, 42, [WS + ', ' + WT], sizes=[24])
    f.box(225, 795, 180, 65, ['Box-weighted pooling'], BLUE, True, [19])
    f.box(465, 725, 190, 42, [TR + ', g'], sizes=[24])
    f.box(465, 795, 190, 65, ['Concatenation', '[' + TR + '; ' + sub('v', 's') + '; ' + sub('v', 't') + '; g]'], RELATION, True, [20, 22], True)
    f.box(715, 795, 150, 65, ['Slot MLP', '1544 → 512 → 512'], RELATION, True, [21, 16], True)
    f.icon(852, 793)
    f.box(915, 795, 120, 65, [sub('S', 'r'), '≤ 3 × 512'], RELATION, sizes=[25, 18])
    f.box(1095, 715, 160, 52, ['(' + WS + ' + ' + WT + ')ᵀ &gt; 0', 'Valid slots only'], sizes=[18, 16])
    f.box(1095, 795, 160, 65, ['Masked', 'cross-attention'], RELATION, True, [21, 20], True)
    f.icon(1243, 793)
    f.add(1320, 827)
    f.box(1370, 795, 90, 65, [VRP, '196 × 512'], sizes=[25, 16])
    f.edge([(175, 827), (225, 827)])
    f.edge([(315, 767), (315, 795)])
    f.edge([(405, 827), (465, 827)])
    f.text(435, 810, sub('v', 's') + ', ' + sub('v', 't'), 19)
    f.edge([(560, 767), (560, 795)])
    for a, b in [(655, 715), (865, 915), (1035, 1095), (1255, 1304), (1336, 1370)]:
        f.edge([(a, 827), (b, 827)])
    f.text(1065, 814, 'K, V', 18)
    f.edge([(1175, 767), (1175, 795)])
    f.box(40, 900, 135, 55, [VR], sizes=[25])
    f.edge([(175, 927), (1175, 927), (1175, 860)])
    f.text(1189, 900, 'Q', 20)
    f.edge([(210, 927), (210, 973), (1320, 973), (1320, 843)])
    f.dot(210, 927)
    f.text(740, 962, 'Residual ' + VR, 19)
    f.save('m3_prompt_visual_alignment')


def overall():
    f = Figure(1700, 800)
    f.text(850, 32, 'Overall Image Captioning Pipeline', 29, True)
    f.box(190, 75, 190, 70, ['TRAIN captions'], sizes=[23], bold_first=True)
    f.box(440, 65, 240, 90, ['M1', 'Semantic Bank', 'Construction'], BLUE, True, [23, 22, 22], True)
    f.box(735, 60, 275, 100, ['Noun / verb banks', sub('B', 'n') + ': 5269 × 512', sub('B', 'v') + ': 1594 × 512'], sizes=[22, 21, 21], bold_first=True)
    f.edge([(380, 110), (440, 110)])
    f.edge([(680, 110), (735, 110)])
    f.edge([(905, 160), (905, 270)])
    f.photo(40, 285, 100, 100)
    f.text(90, 412, 'Input image I', 22, True)
    f.box(190, 295, 160, 80, ['CLIP Vision Encoder', 'ViT-B/16'], BLUE, True, [18, 20], True)
    f.icon(338, 293, True)
    f.box(395, 295, 155, 80, ['Patch tokens', 'X: 196 × 768'], sizes=[22, 21], bold_first=True)
    f.box(595, 295, 160, 80, ['Linear projection', '768 → 512'], BLUE, True, [21, 21], True)
    f.icon(743, 293)
    f.box(800, 270, 210, 130, ['M2', 'Semantic Bank–', 'Visual Alignment'], BLUE, True, [25, 23, 23], True)
    f.icon(996, 268)
    f.box(1080, 270, 210, 130, ['M3', 'Image-Specific Prompt–', 'Visual Alignment'], BLUE, True, [25, 20, 23], True)
    f.icon(1277, 268)
    f.box(1360, 270, 260, 130, ['Transformer Decoder', 'Parallel cross-attention', '+ gated fusion', '3 layers; d = 512'], BLUE, True, [24, 22, 21, 20], True)
    f.icon(1606, 268)
    for a, b in [(140, 190), (350, 395), (550, 595), (755, 800)]:
        f.edge([(a, 335), (b, 335)])
    f.text(777, 318, 'V', 23, italic=True)
    f.box(1080, 95, 210, 65, ['Image I'], sizes=[24], bold_first=True)
    f.edge([(1185, 160), (1185, 270)])
    for y, source_label, target_label in [(310, VO, VOP), (365, VR, VRP)]:
        f.edge([(1010, y), (1080, y)])
        f.text(1045, y-10, source_label, 23)
        f.edge([(1290, y), (1360, y)])
        f.text(1325, y-10, target_label, 23)
    f.text(1325, 425, 'K, V', 19)
    f.box(1370, 460, 240, 80, ['Word Embedding', '+ Positional Encoding'], BLUE, True, [23, 22], True)
    f.icon(1598, 458)
    f.box(1370, 575, 240, 55, ['Caption prefix'], sizes=[23], bold_first=True)
    f.edge([(1490, 575), (1490, 540)])
    f.edge([(1490, 460), (1490, 400)])
    f.box(1360, 670, 240, 80, ['Vocabulary projection', '+ Softmax'], BLUE, True, [23, 22], True)
    f.icon(1587, 668)
    f.edge([(1620, 335), (1665, 335), (1665, 710), (1600, 710)])
    f.box(1070, 670, 220, 80, ['Beam search', 'K = 5'], BLUE, True, [24, 22], True)
    f.edge([(1360, 710), (1290, 710)])
    f.box(670, 640, 330, 140, ['Caption reranking', 'D: Decoder likelihood', 'C: CLIP image–text alignment', 'O: Detector-based object support', '0.3 D + 0.4 C + 0.3 O'], BLUE, True, [24, 20, 20, 19, 22], True)
    f.edge([(1070, 710), (1000, 710)])
    f.text(1035, 695, '5 captions', 17)
    f.box(355, 680, 240, 60, ['Final caption'], sizes=[25], bold_first=True)
    f.edge([(670, 710), (595, 710)])
    f.edge([(1185, 400), (1185, 590), (835, 590), (835, 640)])
    f.text(1010, 579, 'Image I + YOLO labels', 22)
    f.save('pipeline_m1_m2_m3')


if __name__ == '__main__':
    m3()
    overall()
    for name in ('m3_prompt_visual_alignment', 'pipeline_m1_m2_m3'):
        for ext in ('.svg', '.drawio'):
            ET.parse(ROOT / (name + ext))
    print('Generated and XML-checked two SVG / draw.io pairs; no model execution.')
