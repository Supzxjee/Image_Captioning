"""Trial modular bank + source-grounded alignment detail; no model execution."""
from pathlib import Path
import xml.etree.ElementTree as ET
R=Path(__file__).resolve().parent
source=(R/'build.py').read_text(encoding='utf-8')
# Reuse established SVG primitives without executing the old figure body.
ns={'__file__':str(R/'build.py')}
exec(source[:source.index('# Two compact upper preparation strips')],ns)

def reset(w,h):
    ns.update(W=w,H=h,back=[],edges=[],nodes=[],labels=[],geometry={})

def emit(name):
    w,h=ns['W'],ns['H']
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0L7 3.5L0 7Z" fill="#465158"/></marker></defs><rect width="100%" height="100%" fill="white"/><g font-family="Times New Roman, serif">'+''.join(ns['back']+ns['edges']+ns['nodes']+ns['labels'])+'</g></svg>'
    ET.fromstring(svg)
    (R/(name+'.svg')).write_text(svg,encoding='utf-8')
    (R/(name+'.html')).write_text(f'<!doctype html><meta charset="utf-8"><style>@page{{size:{w}px {h}px;margin:0}}html,body{{margin:0;width:{w}px;height:{h}px}}svg{{display:block}}</style>'+svg,encoding='utf-8')

box,arrow,label,frame,bank,plus=[ns[k] for k in ('box','arrow','label','frame','bank','plus')]
OBJ,REL,NEUTRAL=[ns[k] for k in ('OBJ','REL','NEUTRAL')]
# Main drawing: replace only preparation strip; leave aggregation in the learned graph.
start=source.index('# Two compact upper preparation strips')
end=source.index('frame(1210,38',start)
replacement="""
frame(650,38,520,220,'Semantic banks (see bank detail)')
box('semantic-banks',675,93,200,95,['Semantic banks','Nouns / verbs'],data=True,size=24)
box('noun-agg',950,62,180,50,['Semantic','aggregation'],OBJ,kind='train',size=21)
box('verb-agg',950,159,180,50,['Semantic','aggregation'],REL,kind='train',size=21)
arrow('noun-agg-input',[(875,112),(910,112),(910,87),(950,87)])
arrow('verb-agg-input',[(875,165),(910,165),(910,184),(950,184)])
arrow('pn-alignment',[(1130,87),(1150,87),(1150,280),(840,280),(840,400)])
arrow('pv-alignment',[(1130,184),(1180,184),(1180,520),(1090,520),(1090,550)])
math(870,304,'P','n');math(1120,514,'P','v')
"""
main=source[:start]+replacement+source[end:source.index('svg=f')]
main=main.replace("['Object','alignment']","['Object','alignment (A)']").replace("['Relation','alignment']","['Relation','alignment (A)']").replace("'Region-linked prompt alignment'","'Region-linked prompt alignment (detail B)'")
exec(main,ns)
emit('pipeline-modular-bank')

reset(1550,285)
label(775,35,'Bank preparation — offline, TRAIN captions only',27,True,True)
box('train',35,110,200,70,['TRAIN','captions'],data=True,size=24)
box('lemma',285,110,230,70,['POS tagging','+ lemmatization'],size=24)
box('lemma-data',565,110,200,70,['Noun / verb','lemmas'],data=True,size=24)
box('encoder',815,110,210,70,['CLIP text','encoder'],kind='frozen',size=24)
bank('noun',1110,67,'Noun bank',OBJ);bank('verb',1110,173,'Verb bank',REL)
box('module',1320,110,200,70,['Semantic banks','(main figure)'],data=True,size=23)
for k,a,b in [('caption-lemma',235,285),('lemma-data-edge',515,565),('data-clip',765,815)]:arrow(k,[(a,145),(b,145)])
arrow('clip-noun',[(1025,145),(1065,145),(1065,94),(1110,94)])
arrow('clip-verb',[(1065,145),(1065,200),(1110,200)])
arrow('noun-module',[(1260,94),(1290,94),(1290,130),(1320,130)])
arrow('verb-module',[(1260,200),(1290,200),(1290,160),(1320,160)])
label(775,263,'Fixed text embeddings; learned semantic aggregation stays in the main model.',23,False,True)
emit('bank-preparation-detail')

reset(1780,970)
label(890,38,'Where visual–semantic alignment happens',30,True)
frame(30,85,1720,385,'A. Semantic bank–region alignment')
box('regions',65,145,190,65,['Region features','X'],data=True,size=23)
box('prototypes',65,320,190,65,['Semantic vectors','P'],data=True,size=23)
box('similarity',325,320,220,65,['Cosine similarity','P ↔ X'],size=23)
box('reweight',600,320,215,65,['Soft relevance','weighting'],size=23)
box('selected',870,320,190,65,['Image-weighted','semantics T'],data=True,fill=OBJ,size=23)
box('self',325,145,220,65,['Self-attention','+ Add & Norm / FFN'],kind='train',size=21)
box('cross',1120,220,225,80,['Cross-attention','Q: H   K,V: T'],fill=OBJ,kind='train',size=24)
box('gate',1405,220,195,80,['Learned gate','H ↔ cross output'],kind='train',size=22)
box('out',1640,220,85,80,['X′'],data=True,fill=OBJ,size=28)
arrow('x-self',[(255,177),(325,177)])
arrow('p-sim',[(255,352),(325,352)])
arrow('x-sim',[(280,177),(280,275),(435,275),(435,320)])
arrow('sim-reweight',[(545,352),(600,352)])
arrow('weighted-semantics',[(815,352),(870,352)])
arrow('semantics-kv',[(1060,352),(1232,352),(1232,300)])
arrow('self-q',[(545,177),(1090,177),(1090,250),(1120,250)])
arrow('self-gate',[(1000,177),(1502,177),(1502,220)])
arrow('cross-gate',[(1345,260),(1405,260)])
arrow('gate-out',[(1600,260),(1640,260)])
label(730,162,'H',23,False,True)
label(1228,327,'K,V',20,False,True)
label(890,429,'Object: (V, Pn) → (V₁, T₁) → (Vo, T₂); relation: (Vo, Pv) → Vr.',24,False,True)
label(890,456,'Second object stage uses T₁. Standard residual / normalization / FFN steps are abbreviated.',20,False,True)

frame(30,530,1720,380,'B. Image-specific prompt–region alignment')
box('text',65,590,210,70,['Prompt embedding','t'],data=True,fill=REL,size=23)
box('matched',65,760,210,85,['Matched Vo region(s)','+ box geometry'],data=True,size=23)
box('slot',355,675,190,70,'Slot MLP',kind='train',size=26)
box('slots',605,675,190,70,['Linked prompt','slots Z'],data=True,fill=REL,size=23)
box('memory',850,585,210,65,['Region memory','Vo or Vr'],data=True,size=23)
box('masked',850,740,210,80,['Masked cross-attn','Q: memory  K,V: Z'],kind='train',fill=REL,size=22)
box('projection',1130,750,180,60,'Projection',kind='train',fill=REL,size=25)
plus('sum',1430,780)
box('updated',1520,745,195,70,['Updated memory','Vo′ or Vr′'],data=True,fill=REL,size=24)
arrow('text-mlp',[(275,625),(310,625),(310,695),(355,695)])
arrow('regions-mlp',[(275,802),(310,802),(310,725),(355,725)])
arrow('mlp-slots',[(545,710),(605,710)])
arrow('slots-attn',[(795,710),(820,710),(820,780),(850,780)])
arrow('memory-query',[(955,650),(955,740)])
arrow('attn-project',[(1060,780),(1130,780)])
arrow('project-add',[(1310,780),(1419,780)])
arrow('memory-bypass',[(1060,617),(1430,617),(1430,769)])
arrow('add-updated',[(1441,780),(1520,780)])
label(890,883,'Attention is allowed only between each ROI and its linked prompt slots; unmatched ROIs stay unchanged.',23,False,True)
label(890,949,'A learns soft semantic correspondence; B restricts prompt attention using matched object / relation endpoints.',23,False,True)
emit('alignment-mechanism-detail')
print('Three editable trial diagrams generated; no model run.')
