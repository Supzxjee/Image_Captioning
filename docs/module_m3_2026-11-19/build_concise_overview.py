"""Prune labels while preserving the approved expanded M2/M3 connectivity."""
from pathlib import Path
import xml.etree.ElementTree as ET
import build_diagrams as art
import build_expanded_overview as expanded

ROOT = Path(__file__).resolve().parent / 'concise_overview'
ROOT.mkdir(exist_ok=True)
expanded.ROOT = ROOT
art.ROOT = ROOT


class ConciseFigure(expanded.Figure):
    def box(self, x, y, w, h, lines, fill='white', rounded=False, sizes=None, bold_first=False):
        title = lines[0]
        if title.startswith('Fixed semantic banks'):
            lines = ['Semantic banks · TRAIN', 'Nouns: person, bicycle, …', 'Verbs: ride, stand, …']
            sizes = [17,16,16]
        elif title.startswith('Object–relation prompts'):
            lines, sizes = lines[:3], [16,15,15]
        elif title == 'Cached prompt record':
            lines, sizes = ['Prompt + ROI features',art.TO+', '+art.TR], [13,18]
        elif title == 'CLIP Text Encoder':
            lines, sizes = ['CLIP Text Encoder','ViT-B/16'], [14,15]
        elif title in ('Noun aggregation','Verb aggregation'):
            lines, sizes = [title], [16]
        elif title.startswith('Object alignment') or title == 'Relation alignment':
            lines, sizes = [title,'Self-/cross-attention'], [15,13]
        elif title == 'Gated visual update':
            lines, sizes = ['Gated fusion'], [16]
        elif title.startswith('Prompt inputs:'):
            x, w = 1290, 480
            lines, sizes = ['Prompt embeddings + ROI links'], [17]
        elif title.startswith('Visual context from M2:'):
            w = 480
            lines, sizes = ['Visual memories: '+art.VO+', '+art.VR,'196 × 512 each'], [17,14]
        elif title == 'Box pooling':
            relation = y == 650
            lines = ['Relation pooling' if relation else 'Object pooling','from '+art.VO]
            sizes = [14,15]
        elif title == 'Concatenation':
            lines, sizes = ['Concatenate'], [15]
        elif title == 'Slot MLP':
            lines, sizes = ['Slot MLP'], [16]
        elif title == 'Masked cross-attn':
            lines, sizes = ['Masked cross-','attention'], [14,15]
        elif title == 'Residual add':
            lines = ['Residual add',art.VRP if y == 650 else art.VOP]
            sizes = [14,18]
        elif title == 'Bank-aligned memories':
            lines, sizes = ['Visual memories',art.VO+', '+art.VR], [15,19]
        elif title == 'Prompt-aligned memories':
            lines, sizes = ['Aligned memories',art.VOP+', '+art.VRP], [15,19]
        elif title == 'Caption reranking':
            lines, sizes = ['Caption reranking'], [16]
        elif title == 'Image I + YOLO labels':
            lines, sizes = [title], [16]

        if rounded:
            new_h = 36 if len(lines) == 1 else 50
            new_h = min(h,new_h)
            new_y = y+(h-new_h)/2
            self.process_ports.append((x,y,w,h,new_y,new_h))
        else:
            new_y,new_h = y,h
        art.Figure.box(self,x,new_y,w,new_h,lines,fill,rounded,sizes,False)

    def mapped(self, x, y):
        for bx,by,bw,bh,ny,nh in self.process_ports:
            if (x == bx or x == bx+bw) and by <= y <= by+bh:
                # Preserve horizontal port heights unless the shortened box excludes it.
                return x,min(max(y,ny),ny+nh)
            if (y == by or y == by+bh) and bx <= x <= bx+bw:
                return x,ny if y == by else ny+nh
        return x,y

    def icon(self, x, y, frozen=False):
        for bx,by,bw,bh,ny,nh in self.process_ports:
            if y == by and bx <= x <= bx+bw:
                y = ny
                break
        super().icon(x,y,frozen)


if __name__ == '__main__':
    expanded.Figure = ConciseFigure
    expanded.main()
    for extension in ('.svg','.drawio'):
        source=ROOT/('pipeline_m2_m3_expanded'+extension)
        target=ROOT/('pipeline_m2_m3_concise'+extension)
        source.replace(target)
        ET.parse(target)
    print('Concise M2/M3 overview generated; no model execution.')
