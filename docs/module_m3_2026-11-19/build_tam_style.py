"""Presentation-only revision: compact reference-style overview and vertical M3."""
from pathlib import Path
import xml.etree.ElementTree as ET
import build_diagrams as art

ROOT = Path(__file__).resolve().parent / 'tam_style'
ROOT.mkdir(exist_ok=True)
art.ROOT = ROOT
F, B, O, R = art.Figure, art.BLUE, art.OBJECT, art.RELATION
VO, VR, VOP, VRP = art.VO, art.VR, art.VOP, art.VRP
TO, TR, WO, WS, WT = art.TO, art.TR, art.WO, art.WS, art.WT
sub = art.sub


def overview():
    f = F(1400, 790)
    f.text(365, 30, 'Semantic bank construction', 22, True, italic=True)
    f.box(50, 65, 150, 70, ['TRAIN captions'], sizes=[22])
    f.box(250, 65, 170, 70, ['M1 · Semantic Bank', 'Construction'], B, True, [19, 21])
    f.box(475, 55, 220, 90, ['Noun / verb banks', sub('B', 'n') + ': 5269 × 512', sub('B', 'v') + ': 1594 × 512'], sizes=[21, 20, 20])
    f.edge([(200, 100), (250, 100)])
    f.edge([(420, 100), (475, 100)])
    f.edge([(585, 145), (585, 350), (760, 350), (760, 390)])

    f.text(1015, 35, 'Image I', 22, True)
    f.edge([(1015, 43), (1015, 70)])
    f.box(925, 70, 180, 50, ['UPT', 'Interaction detector'], B, True, [21, 17])
    f.icon(1093, 68, True)
    f.box(1175, 150, 180, 50, ['YOLO boxes B'], sizes=[22])
    f.box(925, 150, 180, 50, ['Class + IoU matching', 'IoU ≥ 0.5'], B, True, [17, 19])
    f.edge([(1015, 120), (1015, 150)])
    f.edge([(1175, 175), (1105, 175)])
    f.box(925, 230, 180, 50, ['Object–relation', 'prompts'], sizes=[20, 20])
    f.edge([(1015, 200), (1015, 230)])
    f.box(925, 310, 180, 50, ['CLIP Text Encoder', 'ViT-B/16'], B, True, [18, 19])
    f.icon(1093, 308, True)
    f.edge([(1015, 280), (1015, 310)])
    f.edge([(1015, 360), (1015, 390)])
    f.text(1055, 380, TO + ', ' + TR, 19)
    f.edge([(1105, 190), (1140, 190), (1140, 370), (1080, 370), (1080, 390)])
    f.text(1150, 350, 'ROI links', 17, False, 'start')

    f.photo(50, 230, 150, 110)
    f.text(125, 215, 'Input image I', 21, True)
    f.edge([(125, 340), (125, 390)])
    f.box(50, 390, 150, 70, ['CLIP Vision Encoder', 'CLIP ViT-B/16'], B, True, [17, 20])
    f.icon(188, 388, True)
    f.box(250, 390, 170, 70, ['Visual patch tokens', 'X: 196 × 768'], sizes=[20, 21])
    f.box(475, 390, 140, 70, ['Linear projection', '768 → 512'], B, True, [19, 21])
    f.icon(603, 388)
    f.box(670, 390, 180, 70, ['M2 · Bank–Visual', 'Alignment'], B, True, [20, 21])
    f.icon(838, 388)
    f.box(925, 390, 180, 70, ['M3 · Prompt–Visual', 'Alignment'], B, True, [19, 21])
    f.icon(1093, 388)
    for a, b in [(200, 250), (420, 475), (615, 670)]:
        f.edge([(a, 425), (b, 425)])
    f.text(643, 409, 'V', 22, italic=True)
    f.edge([(850, 410), (925, 410)])
    f.edge([(850, 445), (925, 445)])
    f.text(888, 401, VO, 22)
    f.text(888, 437, VR, 22)
    f.box(1175, 390, 180, 70, [VOP + ', ' + VRP, '196 × 512 each'], sizes=[25, 20])
    f.edge([(1105, 410), (1175, 410)])
    f.edge([(1105, 445), (1175, 445)])

    f.box(1175, 525, 180, 70, ['Transformer Decoder', 'Parallel attention · ×3'], B, True, [20, 18])
    f.icon(1343, 523)
    f.edge([(1225, 460), (1225, 525)])
    f.edge([(1305, 460), (1305, 525)])
    f.text(1265, 500, 'K, V', 18)
    f.box(925, 525, 180, 70, ['Vocabulary projection', '+ Softmax'], B, True, [19, 20])
    f.icon(1093, 523)
    f.box(670, 525, 180, 70, ['Beam search', 'K = 5'], B, True, [22, 21])
    f.box(475, 525, 140, 70, ['Caption reranking'], B, True, [18])
    f.box(250, 525, 170, 70, ['Final caption'], sizes=[22])
    for a, b in [(1175, 1105), (925, 850), (670, 615), (475, 420)]:
        f.edge([(a, 560), (b, 560)])
    f.box(1175, 640, 180, 65, ['Word Embedding', '+ Position Encoding'], B, True, [21, 19])
    f.icon(1343, 638)
    f.box(1175, 740, 180, 40, ['Caption prefix'], sizes=[21])
    f.edge([(1265, 740), (1265, 705)])
    f.edge([(1265, 640), (1265, 595)])
    f.box(475, 650, 140, 70, ['Image I', 'YOLO labels'], sizes=[22, 20])
    f.edge([(545, 650), (545, 595)])
    f.save('pipeline_tam_style')


def adapter(f, offset, relation=False):
    fill = R if relation else O
    q, out = (VR, VRP) if relation else (VO, VOP)
    title = 'Relation prompt alignment' if relation else 'Object prompt alignment'
    f.frame(30 + offset, 60, 640, 725, title)
    prompt = TR + ', g, ' + WS + ', ' + WT if relation else TO + ', ' + sub('b', 'i') + ', ' + WO
    f.box(220 + offset, 110, 230, 65, ['Prompt + box links', prompt], sizes=[21, 23])
    f.box(40 + offset, 215, 130, 60, [VO, '196 × 512'], sizes=[25, 18])
    f.box(220 + offset, 215, 230, 60, ['Box-weighted pooling'], B, True, [22])
    f.edge([(170 + offset, 245), (220 + offset, 245)])
    f.edge([(335 + offset, 175), (335 + offset, 215)])
    f.text(398 + offset, 202, WS + ', ' + WT if relation else WO, 20)
    pool_label = sub('v', 's') + ', ' + sub('v', 't') if relation else sub('v', 'i')
    f.text(374 + offset, 301, pool_label, 23)
    concat = '[' + TR + '; ' + sub('v', 's') + '; ' + sub('v', 't') + '; g]' if relation else '[' + TO + '; ' + sub('v', 'i') + '; ' + sub('b', 'i') + ']'
    f.box(220 + offset, 320, 230, 65, ['Concatenation', concat], fill, True, [22, 23])
    f.edge([(335 + offset, 275), (335 + offset, 320)])
    f.edge([(450 + offset, 142), (505 + offset, 142), (505 + offset, 347), (450 + offset, 347)])
    f.box(220 + offset, 420, 230, 65, ['Slot MLP', ('1544' if relation else '1028') + ' → 512 → 512'], fill, True, [23, 20])
    f.icon(438 + offset, 418)
    f.edge([(335 + offset, 385), (335 + offset, 420)])
    f.box(220 + offset, 540, 230, 65, ['Masked multi-head', 'cross-attention'], fill, True, [22, 23])
    f.icon(438 + offset, 538)
    f.edge([(335 + offset, 485), (335 + offset, 540)])
    f.text(391 + offset, 521, 'Slots · K, V', 19)
    if relation:
        f.box(40 + offset, 540, 130, 65, [VR], sizes=[25])
        f.edge([(170 + offset, 572), (220 + offset, 572)])
        f.edge([(190 + offset, 572), (190 + offset, 655), (319 + offset, 655)])
        f.dot(190 + offset, 572)
        f.text(195 + offset, 559, 'Q', 18)
    else:
        f.edge([(105 + offset, 275), (105 + offset, 572), (220 + offset, 572)])
        f.edge([(105 + offset, 572), (105 + offset, 655), (319 + offset, 655)])
        f.dot(105 + offset, 572)
        f.text(195 + offset, 559, 'Q', 18)
    f.box(480 + offset, 490, 150, 85, ['Spatial mask', '(' + WS + ' + ' + WT + ')ᵀ &gt; 0' if relation else WO + 'ᵀ &gt; 0', 'Valid slots only'], sizes=[21, 19 if not relation else 17, 17])
    f.edge([(505 + offset, 142), (555 + offset, 142), (555 + offset, 490)])
    f.dot(505 + offset, 142)
    f.edge([(480 + offset, 560), (450 + offset, 560)])
    f.add(335 + offset, 655)
    f.edge([(335 + offset, 605), (335 + offset, 639)])
    f.text(181 + offset, 680, 'Residual ' + q, 19)
    f.box(220 + offset, 715, 230, 55, [out + ' : 196 × 512'], sizes=[25])
    f.edge([(335 + offset, 671), (335 + offset, 715)])


def m3():
    f = F(1400, 810)
    f.text(700, 32, 'M3 · Prompt–Visual Alignment', 26, True)
    adapter(f, 0)
    adapter(f, 700, True)
    f.save('m3_tam_style')


if __name__ == '__main__':
    overview()
    m3()
    for name in ('pipeline_tam_style', 'm3_tam_style'):
        for extension in ('.svg', '.drawio'):
            ET.parse(ROOT / (name + extension))
    print('Two reference-style figures generated; XML PASS; no model execution.')
