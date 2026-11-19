"""Tâm-style data blocks with example contents; no model or metric execution."""
from pathlib import Path
import xml.etree.ElementTree as ET
import build_diagrams as art

ROOT = Path(__file__).resolve().parent / 'data_blocks'
ROOT.mkdir(exist_ok=True)
art.ROOT = ROOT
F, BLUE, OBJECT, RELATION = art.Figure, art.BLUE, art.OBJECT, art.RELATION
VO, VR, VOP, VRP = art.VO, art.VR, art.VOP, art.VRP
TO, TR, WO, WS, WT = art.TO, art.TR, art.WO, art.WS, art.WT
sub = art.sub


def overview():
    f = F(1600, 950)
    f.box(40, 90, 140, 70, ['TRAIN captions'], sizes=[19])
    f.box(230, 90, 170, 70, ['M1 · Semantic Bank', 'Construction'], BLUE, True, [19, 21])
    f.box(455, 45, 460, 175, [
        'Semantic banks · TRAIN vocabulary',
        'Nouns: person, bicycle, motorcycle, …',
        'Verbs: ride, stand, park, …',
        sub('B', 'n') + ': 5269 × 512   |   ' + sub('B', 'v') + ': 1594 × 512',
        'Fixed CLIP text embeddings'], sizes=[24, 23, 23, 23, 20], bold_first=True)
    f.edge([(180, 125), (230, 125)])
    f.edge([(400, 125), (455, 125)])
    f.edge([(685, 220), (685, 350), (775, 350), (775, 500)])

    f.box(1010, 30, 170, 45, ['Input image I'], sizes=[22])
    f.box(1010, 105, 170, 55, ['UPT', 'Interaction detector'], BLUE, True, [22, 17])
    f.icon(1168, 103, True)
    f.box(1010, 195, 170, 60, ['Class + IoU matching', 'IoU ≥ 0.5'], BLUE, True, [16, 21])
    f.box(1240, 195, 270, 60, ['YOLO boxes', 'Classes + box coordinates'], sizes=[23, 21])
    f.edge([(1095, 75), (1095, 105)])
    f.edge([(1095, 160), (1095, 195)])
    f.edge([(1240, 225), (1180, 225)])
    f.box(960, 290, 550, 105, [
        'Instance-linked prompts · illustrative example',
        'Objects: “a person”; “a bicycle”; “a motorcycle”',
        'Relation: “a person riding a bicycle”'], sizes=[23, 23, 23], bold_first=True)
    f.edge([(1095, 255), (1095, 290)])
    f.box(1140, 420, 200, 55, ['CLIP Text Encoder · B/16', TO + ', ' + TR + ' : 512-D · L2 norm'], BLUE, True, [17, 18])
    f.icon(1328, 418, True)
    f.edge([(1240, 395), (1240, 420)])
    f.edge([(1240, 475), (1240, 500)])
    # Box/geometry links bypass the text encoder; arrow has no floating label.
    f.edge([(1175, 255), (1175, 275), (1545, 275), (1545, 485), (1300, 485), (1300, 500)])

    f.photo(40, 305, 150, 115)
    f.box(40, 420, 150, 35, ['Input image I'], sizes=[21])
    f.edge([(115, 455), (115, 500)])
    f.box(40, 500, 150, 85, ['CLIP Vision Encoder', 'CLIP ViT-B/16'], BLUE, True, [17, 20])
    f.icon(178, 498, True)
    f.box(230, 500, 180, 85, ['Visual patch tokens', 'X: 196 × 768', '14 × 14 patches · no CLS'], sizes=[21, 23, 16], bold_first=True)
    f.box(455, 500, 170, 85, ['Visual projection', '768 → 512', 'Output V: 196 × 512'], BLUE, True, [21, 23, 18])
    f.icon(613, 498)
    f.box(690, 500, 160, 85, ['M2 · Bank–Visual', 'Alignment'], BLUE, True, [18, 22])
    f.icon(838, 498)
    f.box(890, 500, 200, 85, ['Bank-aligned memories', VO + ', ' + VR, '196 × 512 each'], sizes=[18, 25, 19], bold_first=True)
    f.box(1150, 500, 190, 85, ['M3 · Prompt–Visual', 'Alignment', 'Input: memories + ROI links'], BLUE, True, [20, 22, 15])
    f.icon(1328, 498)
    f.box(1370, 500, 190, 85, ['Prompt-aligned memories', VOP + ', ' + VRP, '196 × 512 each'], sizes=[16, 25, 19], bold_first=True)
    for a, b in [(190, 230), (410, 455), (625, 690), (850, 890), (1090, 1150), (1340, 1370)]:
        f.edge([(a, 542), (b, 542)])

    f.box(1390, 675, 170, 85, ['Transformer Decoder', 'Parallel attention · ×3', 'K/V: ' + VOP + ', ' + VRP], BLUE, True, [19, 17, 20])
    f.icon(1548, 673)
    f.edge([(1475, 585), (1475, 675)])
    f.box(1150, 675, 190, 85, ['Vocabulary projection', '+ Softmax'], BLUE, True, [20, 22])
    f.icon(1328, 673)
    f.box(900, 675, 180, 85, ['Beam search', 'K = 5'], BLUE, True, [24, 22])
    f.box(550, 650, 300, 135, ['Candidate captions · examples',
        '1. A person rides a bicycle.',
        '2. A person on a bicycle.',
        '… 5 beam candidates'], sizes=[21, 22, 22, 20], bold_first=True)
    f.box(300, 675, 190, 85, ['Caption reranking', 'Decoder + CLIP', '+ detector object support'], BLUE, True, [21, 20, 16])
    f.box(40, 650, 210, 135, ['Final caption · example', '“A person rides', 'a bicycle.”'], sizes=[19, 23, 23], bold_first=True)
    for a, b in [(1390, 1340), (1150, 1080), (900, 850), (550, 490), (300, 250)]:
        f.edge([(a, 717), (b, 717)])
    f.box(1390, 805, 170, 60, ['Word Embedding', '+ Position Encoding'], BLUE, True, [20, 18])
    f.icon(1548, 803)
    f.box(1390, 900, 170, 40, ['Caption prefix'], sizes=[22])
    f.edge([(1475, 900), (1475, 865)])
    f.edge([(1475, 805), (1475, 760)])
    f.box(300, 820, 190, 70, ['Reranking evidence', 'Image I + YOLO labels'], sizes=[21, 19], bold_first=True)
    f.edge([(395, 820), (395, 760)])
    f.save('pipeline_data_blocks')


def adapter(f, offset, relation=False):
    fill = RELATION if relation else OBJECT
    q, out = (VR, VRP) if relation else (VO, VOP)
    f.frame(30 + offset, 60, 640, 810,
            'Relation prompt alignment' if relation else 'Object prompt alignment')
    prompt = TR + ', g, ' + WS + ', ' + WT if relation else TO + ', ' + sub('b', 'i') + ', ' + WO
    lines = (['Relation prompt inputs', prompt, 'Example: “a person', 'riding a bicycle”'] if relation else
             ['Object prompt inputs', prompt, 'Example: “a bicycle”'])
    f.box(220 + offset, 110, 230, 110, lines, sizes=[21, 22, 19, 19] if relation else [21, 23, 21], bold_first=True)
    f.box(40 + offset, 250, 130, 80, ['Object memory', VO, '196 × 512'], sizes=[17, 26, 17])
    pooling = (['Box-weighted pooling', sub('v', 's') + ' = ' + WS + ' ' + VO,
                sub('v', 't') + ' = ' + WT + ' ' + VO] if relation else
               ['Box-weighted pooling', sub('v', 'i') + ' = ' + WO + ' ' + VO, '512-D region feature'])
    f.box(220 + offset, 250, 230, 90, pooling, BLUE, True, [21, 23, 22 if relation else 18])
    f.edge([(170 + offset, 290), (220 + offset, 290)])
    f.edge([(335 + offset, 220), (335 + offset, 250)])
    concat = ('[' + TR + '; ' + sub('v', 's') + '; ' + sub('v', 't') + '; g]' if relation else
              '[' + TO + '; ' + sub('v', 'i') + '; ' + sub('b', 'i') + ']')
    f.box(220 + offset, 365, 230, 75, ['Concatenation', concat], fill, True, [22, 23])
    f.edge([(335 + offset, 340), (335 + offset, 365)])
    f.edge([(450 + offset, 160), (505 + offset, 160), (505 + offset, 397), (450 + offset, 397)])
    slot = sub('S', 'r' if relation else 'o')
    f.box(220 + offset, 465, 230, 95,
          ['Slot MLP', ('1544' if relation else '1028') + ' → 512 → 512', 'Output slots: ' + slot], fill, True, [23, 20, 22])
    f.icon(438 + offset, 463)
    f.edge([(335 + offset, 440), (335 + offset, 465)])
    f.box(220 + offset, 595, 230, 95,
          ['Masked multi-head', 'cross-attention', 'Q = ' + q + '; K = V = ' + slot], fill, True, [21, 23, 20])
    f.icon(438 + offset, 593)
    f.edge([(335 + offset, 560), (335 + offset, 595)])
    if relation:
        f.box(40 + offset, 595, 130, 90, ['Relation memory', VR, '196 × 512'], sizes=[16, 26, 17])
        f.edge([(170 + offset, 640), (220 + offset, 640)])
        f.edge([(190 + offset, 640), (190 + offset, 752), (220 + offset, 752)])
        f.dot(190 + offset, 640)
    else:
        f.edge([(105 + offset, 330), (105 + offset, 640), (220 + offset, 640)])
        f.edge([(105 + offset, 640), (105 + offset, 752), (220 + offset, 752)])
        f.dot(105 + offset, 640)
    f.box(480 + offset, 540, 150, 90,
          ['Spatial mask', '(' + WS + ' + ' + WT + ')ᵀ &gt; 0' if relation else WO + 'ᵀ &gt; 0', 'Valid slots only'], sizes=[21, 17 if relation else 20, 17])
    f.edge([(505 + offset, 160), (555 + offset, 160), (555 + offset, 540)])
    f.dot(505 + offset, 160)
    f.edge([(555 + offset, 630), (555 + offset, 655), (450 + offset, 655)])
    f.box(220 + offset, 720, 230, 65, ['Residual addition', q + ' + attention update'], fill, True, [22, 20])
    f.edge([(335 + offset, 690), (335 + offset, 720)])
    f.box(220 + offset, 800, 230, 65, ['Aligned ' + ('relation' if relation else 'object') + ' memory', out + ': 196 × 512'], sizes=[19, 24])
    f.edge([(335 + offset, 785), (335 + offset, 800)])


def m3():
    f = F(1400, 890)
    f.text(700, 32, 'M3 · Prompt–Visual Alignment', 26, True)
    adapter(f, 0)
    adapter(f, 700, True)
    f.save('m3_data_blocks')


if __name__ == '__main__':
    overview()
    m3()
    for name in ('pipeline_data_blocks', 'm3_data_blocks'):
        for ext in ('.svg', '.drawio'):
            ET.parse(ROOT / (name + ext))
    print('Data-block figures generated; XML PASS; examples are illustrative, not model outputs.')
