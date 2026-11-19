"""Integrated M2/M3 architecture drawing; standard library, no model runs."""
from pathlib import Path
import xml.etree.ElementTree as ET
import build_diagrams as art
from build_layout_proof import Figure as ProofFigure

ROOT = Path(__file__).resolve().parent / 'expanded_overview'
ROOT.mkdir(exist_ok=True)
art.ROOT = ROOT
SUB = art.sub
VO, VR, VOP, VRP = art.VO, art.VR, art.VOP, art.VRP
TO, TR, WO, WS, WT = art.TO, art.TR, art.WO, art.WS, art.WT
WHITE, PALE, TEAL = 'white', '#F6F8FA', '#F0F6F5'


class Figure(ProofFigure):
    def box(self, x, y, w, h, lines, fill='white', rounded=False, sizes=None, bold_first=False):
        art.Figure.box(self, x, y, w, h, lines, fill, rounded, sizes or [14]*len(lines), False)

    def frame(self, x, y, w, h, title):
        art.Figure.frame(self, x, y, w, h, title)
        self.svg[-1] = self.svg[-1].replace('font-size="22"', 'font-size="17"').replace('font-weight="bold"', '')
        cell = self.cells[-1]
        cell.set('style', cell.get('style').replace('fontSize=22', 'fontSize=17').replace('fontStyle=1', 'fontStyle=0'))

    def gap(self, x, y):
        # A straight line gap distinguishes crossing wires without a curved bridge.
        self.svg.append(f'<rect x="{x-4}" y="{y-4}" width="8" height="8" fill="white"/>')
        cell = self.cell(value='', vertex='1', style='rounded=0;fillColor=#FFFFFF;strokeColor=none;')
        ET.SubElement(cell, 'mxGeometry', x=str(x-4), y=str(y-4), width='8', height='8', **{'as':'geometry'})


def main():
    f = Figure(1840, 1100)
    # M1 is unchanged; a fixed TRAIN-only knowledge source.
    f.box(40, 40, 140, 50, ['TRAIN captions'], sizes=[16])
    f.box(230, 40, 170, 50, ['M1 · Semantic Bank', 'Construction'], PALE, True, [15, 16])
    f.box(455, 20, 450, 125, ['Fixed semantic banks · TRAIN',
        'Nouns: person, bicycle, motorcycle, …',
        'Verbs: ride, stand, park, …',
        SUB('B','n') + ': 5269 × 512;  ' + SUB('B','v') + ': 1594 × 512'], sizes=[17,16,16,16])
    f.edge([(180,65),(230,65)])
    f.edge([(400,65),(455,65)])

    # Frozen image-specific prompt preparation; box metadata bypasses CLIP text.
    f.box(1000,30,140,45,['Image I'],sizes=[16])
    f.box(1180,30,140,45,['UPT detector'],PALE,True,[15])
    f.icon(1309,30,True)
    f.box(1360,30,155,45,['Class + IoU matching','IoU ≥ 0.5'],PALE,True,[13,15])
    f.box(1560,30,170,45,['YOLO boxes'],sizes=[16])
    f.edge([(1140,52.5),(1180,52.5)])
    f.edge([(1320,52.5),(1360,52.5)])
    f.edge([(1560,52.5),(1515,52.5)])
    f.box(1000,115,360,100,['Object–relation prompts · example',
        'Objects: “a person”, “a bicycle”, …',
        'Relation: “a person riding a bicycle”',
        'Matched box links + geometry'],sizes=[16,15,15,14])
    f.edge([(1437.5,75),(1437.5,95),(1180,95),(1180,115)])
    f.box(1410,125,150,75,['CLIP Text Encoder','ViT-B/16',TO+', '+TR+': 512-D (L2)'],PALE,True,[14,15,14])
    f.icon(1549,125,True)
    f.box(1600,125,155,75,['Cached prompt record',TO+', '+TR,'ROI weights + geometry'],sizes=[13,17,12])
    f.edge([(1360,165),(1410,165)])
    f.edge([(1560,162.5),(1600,162.5)])
    f.edge([(1180,215),(1180,240),(1677.5,240),(1677.5,200)])

    # The unannotated input and its CLIP patch extraction remain compact.
    f.photo(40,150,150,100)
    f.box(40,250,150,30,['Input image I'],sizes=[15])
    f.box(40,315,150,50,['CLIP Vision Encoder','CLIP ViT-B/16'],PALE,True,[14,15])
    f.icon(179,315,True)
    f.box(40,390,150,50,['Patch tokens · no CLS','X: 196 × 768'],sizes=[14,16])
    f.box(40,465,150,50,['Linear projection','768 → 512'],PALE,True,[15,16])
    f.icon(179,465)
    for a,b in [(280,315),(365,390),(440,465)]:
        f.edge([(115,a),(115,b)])

    # M2: actual sequential object stages followed by object-guided relation alignment.
    f.frame(240,300,750,470,'M2 · Semantic Bank–Visual Alignment')
    f.icon(975,301)
    f.box(260,350,160,70,['Noun aggregation',SUB('B','n')+' → '+SUB('P','n'),SUB('P','n')+': 150 × 512'],PALE,True,[14,16,14])
    f.box(766,350,160,70,['Verb aggregation',SUB('B','v')+' → '+SUB('P','v'),SUB('P','v')+': 150 × 512'],PALE,True,[14,16,14])
    f.edge([(550,145),(550,250),(600,250),(600,337),(340,337),(340,350)])
    f.edge([(800,145),(800,270),(846,270),(846,350)])
    for x,title,prototype in [(260,'Object alignment 1',SUB('P','n')),(460,'Object alignment 2',SUB('T','1')),(766,'Relation alignment',SUB('P','v'))]:
        f.box(x,445,160,90,[title,'Self-/cross-attention','Cosine-weighted K/V: '+prototype,'Contexts: H, C'],PALE,True,[15,13,11,14])
    f.edge([(190,490),(260,490)])
    f.edge([(340,420),(340,445)])
    f.edge([(846,420),(846,445)])
    for x,out in [(260,SUB('V','1')+', '+SUB('T','1')),(460,VO),(766,VR)]:
        f.box(x,565,160,70,['Gated visual update','gH + (1 − g)C','Output: '+out],PALE,True,[14,14,15])
        f.edge([(x+80,535),(x+80,565)])
    f.edge([(420,597.5),(440,597.5),(440,490),(460,490)])
    f.edge([(620,597.5),(715,597.5),(715,490),(766,490)])
    f.box(740,685,190,65,['Bank-aligned memories',VO+', '+VR,'196 × 512 each'],sizes=[14,19,14])
    f.edge([(540,635),(540,717.5),(740,717.5)])
    f.edge([(846,635),(846,685)])

    # M3 uses a named shared context record, plus an independent prompt/ROI record.
    f.frame(1040,300,760,470,'M3 · Image-Specific Prompt–Visual Alignment')
    f.icon(1785,301)
    f.box(1070,340,700,55,['Prompt inputs: object '+TO+', b, '+WO+'; relation '+TR+', g, '+WS+', '+WT,
        'Matched ROI weights + UPT box/pair geometry'],sizes=[16,13])
    f.box(1070,410,700,55,['Visual context from M2: '+VO+', '+VR+'   (196 × 512 each)',
        'Original '+VO+' is shared by both pooling branches'],sizes=[17,14])
    f.edge([(1755,162.5),(1810,162.5),(1810,330),(1700,330),(1700,340)])
    f.edge([(930,717.5),(1010,717.5),(1010,400),(1200,400),(1200,410)])
    # Context record selects Vo for pooling and the branch-specific query/residual.
    f.edge([(1070,437.5),(1050,437.5),(1050,750)],False)
    # Prompt record branches above each row; the connection is not a decoder prefix.
    f.edge([(1770,367.5),(1780,367.5),(1780,625)],False)
    f.edge([(1780,475),(1269,475),(1269,500)])
    f.edge([(1780,625),(1269,625),(1269,650)])
    f.dot(1780,475)

    for y,relation in [(500,False),(650,True)]:
        q = VR if relation else VO
        out = VRP if relation else VOP
        slot = SUB('S','r' if relation else 'o')
        pool = [SUB('v','s')+' = '+WS+' '+VO,SUB('v','t')+' = '+WT+' '+VO] if relation else [SUB('v','i')+' = '+WO+' '+VO,'Object region feature']
        f.box(1070,y,118,65,['Box pooling']+pool,WHITE,True,[14,13,12])
        cat = '['+TR+'; '+SUB('v','s')+'; '+SUB('v','t')+'; g]' if relation else '['+TO+'; '+SUB('v','i')+'; b]'
        f.box(1210,y,118,65,['Concatenation',cat],WHITE,True,[13,14])
        f.box(1350,y,118,65,['Slot MLP',('1544' if relation else '1028')+' → 512 → 512','Output: '+slot],TEAL,True,[15,11,15])
        f.box(1490,y,118,65,['Masked cross-attn','Q: '+q+'; K/V: '+slot,'Valid mask → Δ'],TEAL,True,[12,13,11])
        f.box(1630,y,118,65,['Residual add',out+' = '+q+' + Δ','196 × 512'],WHITE,True,[14,14,13])
        for a,b in [(1188,1210),(1328,1350),(1468,1490),(1608,1630)]:
            f.edge([(a,y+32.5),(b,y+32.5)])
        f.edge([(1050,y+32.5),(1070,y+32.5)])
        f.dot(1050,y+32.5)
        lane = 595 if not relation else 750
        f.edge([(1050,lane),(1549,lane),(1549,y+65)])
        f.edge([(1549,lane),(1660,lane),(1660,y+65)])
        f.dot(1050,lane)
        f.dot(1549,lane)

    # Updated object stream passes the prompt distribution bus with a straight gap.
    f.gap(1780,532.5)
    f.box(1540,800,190,60,['Prompt-aligned memories',VOP+', '+VRP,'196 × 512 each'],sizes=[13,19,14])
    f.edge([(1748,532.5),(1815,532.5),(1815,830),(1730,830)])
    f.edge([(1715,715),(1715,800)])

    # Compact conventional caption-generation path.
    f.box(1300,875,180,60,['Transformer Decoder','3 layers · parallel attention'],PALE,True,[16,12])
    f.icon(1469,875)
    f.box(1060,875,180,60,['Vocabulary projection','+ Softmax'],PALE,True,[15,16])
    f.icon(1229,875)
    f.box(840,875,150,60,['Beam search','K = 5'],PALE,True,[17,16])
    f.box(580,875,190,60,['Caption reranking','Decoder · CLIP · Objects'],PALE,True,[16,14])
    f.box(300,875,200,60,['Generated caption'],sizes=[17])
    f.edge([(1540,830),(1500,830),(1500,905),(1480,905)])
    for a,b in [(1300,1240),(1060,990),(840,770),(580,500)]:
        f.edge([(a,905),(b,905)])
    f.box(1300,970,180,50,['Word Embedding','+ Position Encoding'],PALE,True,[16,14])
    f.icon(1469,970)
    f.box(1300,1050,180,30,['Caption prefix'],sizes=[16])
    f.edge([(1390,1050),(1390,1020)])
    f.edge([(1390,970),(1390,935)])
    f.box(580,970,190,50,['Image I + YOLO labels','Detector-based support'],sizes=[15,14])
    f.edge([(675,970),(675,935)])
    f.save('pipeline_m2_m3_expanded')
    for ext in ('.svg','.drawio'):
        ET.parse(ROOT / ('pipeline_m2_m3_expanded'+ext))
    print('Expanded M2/M3 SVG and draw.io generated; no model execution.')


if __name__ == '__main__':
    main()
