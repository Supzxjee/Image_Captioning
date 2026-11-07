"""A single compact technical flow: no duplicated overview boxes or memory aliases."""
from pathlib import Path
import json,html,base64
ROOT=Path(__file__).resolve().parent
old=(ROOT.parent/'parameter_icons/build_pipeline.py').read_text()
exec(compile(old.split('# Left:')[0],'shared_primitives','exec'),globals())
exec(compile(old[old.index('def icon('):old.index("frozen=['clip-visual'")],'icon_primitives','exec'),globals())
# All grouped mechanisms are on the real processing path; no zoom leaders.
def join(id,x,y):
 c.nodes.append(f'<circle data-junction="{id}" cx="{x}" cy="{y}" r="3.2" fill="#56616B"/>')
 c.meta.append(dict(id=id,x=x-3.2,y=y-3.2,w=6.4,h=6.4,kind='junction'))
def wire(id,pts,src=None,dst=None,head=True):
 edge(id,pts,src,dst)
 if not head:c.edges[-1]=c.edges[-1].replace(' marker-end="url(#arrow)"','')
frame('visual',24,48,360,848,'Visual region extraction',fs=26)
frame('semantic',408,48,608,816,'Semantic bank–region alignment',fs=25)
frame('conditioning',1040,48,648,816,'Image-specific prompt–region alignment',fs=25)
# A. Compact visual extraction; one original photo only.
photo=ROOT.parents[1]/'csa_pipeline_2026-11-02/v3/assets/COCO_train2014_000000263974.jpg'
data=base64.b64encode(photo.read_bytes()).decode()
c.nodes.append(f'<g id="input-image"><image x="56" y="656" width="256" height="192" href="data:image/jpeg;base64,{data}"/></g>')
c.meta.append(dict(id='input-image',x=56,y=656,w=256,h=192,kind='image'))
c.plain(184,880,'Input image I',27,600)
c.box('clip-visual',48,520,168,80,'CLIP ViT-B/16',('Frozen visual encoder',),fill='visual',fs=21,ss=15)
c.box('yolo',240,520,120,80,'YOLO',('Detector',),fill='visual',fs=27,ss=20)
c.box('patches',48,400,168,80,'Patch features',('14×14×768',),fill='visual',fs=22,ss=20,stack=True)
c.box('boxes',240,400,120,80,'Boxes B',(),fill='visual',fs=25)
c.box('roi-pooling',96,280,240,64,'ROI pooling',(),fill='visual',fs=28)
c.box('projection',96,160,240,64,'Linear projection',('768 → 512',),fill='visual',fs=25,ss=23)
wire('image-clip',[(140,656),(140,600)],'input-image','clip-visual')
wire('image-yolo',[(300,656),(300,600)],'input-image','yolo')
wire('clip-patches',[(132,520),(132,480)],'clip-visual','patches')
wire('yolo-boxes',[(300,520),(300,480)],'yolo','boxes')
wire('patches-pool',[(160,400),(160,344)],'patches','roi-pooling')
wire('boxes-pool',[(300,400),(300,344)],'boxes','roi-pooling')
wire('pool-project',[(216,280),(216,224)],'roi-pooling','projection')
token('v',384,176,'V',C['visual'])
wire('project-v',[(336,192),(384,192)],'projection','v')
# B. Fusions occur once, directly on the main visual path. Banks feed them upward.
c.box('object-fusion-1',432,160,224,64,'Object fusion 1',(),fill='semantic',fs=28)
c.box('object-fusion-2',720,160,224,64,'Object fusion 2',(),fill='semantic',fs=28)
c.box('relation-fusion',720,352,224,64,'Relation fusion',(),fill='relation',fs=28)
wire('v-object1',[(404,192),(432,192)],'v','object-fusion-1')
wire('object1-object2',[(656,192),(720,192)],'object-fusion-1','object-fusion-2')
c.plain(688,180,'V₁,T₁',18)
token('vo',984,176,'Vₒ',ORANGE)
token('vr',984,368,'Vᵣ',BLUE)
wire('object2-vo',[(944,192),(984,192)],'object-fusion-2','vo')
wire('object2-relation',[(832,224),(832,352)],'object-fusion-2','relation-fusion')
c.plain(864,300,'Vₒ',22)
wire('relation-vr',[(944,384),(984,384)],'relation-fusion','vr')
for key,x,colour in [('noun',432,'semantic'),('verb',720,'relation')]:
 c.box(key+'-bank',x,624,224,64,'TRAIN '+key+' bank',('Fixed embeddings',),fill=colour,fs=24,ss=20)
 c.box(key+'-aggregation',x,512,224,64,'Learned semantic',('aggregation',),fill=colour,fs=23,ss=23)
 centre=x+112
 token(key+'-prototypes',centre-10,448,'Pₙ' if key=='noun' else 'Pᵥ',ORANGE if key=='noun' else BLUE,'right')
 wire(key+'-bank-aggregation',[(centre,624),(centre,576)],key+'-bank',key+'-aggregation')
 wire(key+'-aggregate-prototypes',[(centre,512),(centre,480)],key+'-aggregation',key+'-prototypes')
 c.plain(centre+72,504,'150×512',17)
wire('noun-object1',[(544,448),(544,224)],'noun-prototypes','object-fusion-1')
wire('verb-relation',[(832,448),(832,416)],'verb-prototypes','relation-fusion')
c.box('train-lemmas',432,752,96,80,'TRAIN',('lemmas',),fill='plain',fs=20,ss=20)
c.box('bank-text-encoder',560,752,384,80,'CLIP text encoder',('Frozen',),fill='visual',fs=24,ss=22)
wire('lemmas-text',[(528,792),(560,792)],'train-lemmas','bank-text-encoder')
wire('text-noun-bank',[(592,752),(592,688)],'bank-text-encoder','noun-bank')
wire('text-verb-bank',[(888,752),(888,688)],'bank-text-encoder','verb-bank')
c.plain(584,856,'V, Vₒ, Vᵣ: N×512',20)
# C. A single preparation branch supplies text and valid links to both adapters.
c.box('upt',1064,152,224,64,'Image I → UPT',('Interaction detector',),fill='visual',fs=24,ss=21)
c.box('object-phrases',1064,248,112,64,'Object',('phrases',),fill='semantic',fs=22,ss=21)
c.box('relation-phrases',1200,248,112,64,'Relation',('phrases',),fill='relation',fs=21,ss=21)
c.box('clip-text',1064,368,248,64,'CLIP text encoder',('Frozen',),fill='visual',fs=25,ss=22)
wire('upt-object',[(1120,216),(1120,248)],'upt','object-phrases')
wire('upt-relation',[(1256,216),(1256,248)],'upt','relation-phrases')
wire('object-text',[(1120,312),(1120,368)],'object-phrases','clip-text')
wire('relation-text',[(1256,312),(1256,368)],'relation-phrases','clip-text')
c.plain(1176,464,'Embeddings tₒ, tᵣ',21)
c.box('region-matching',1064,584,248,80,'Region matching',('Class + IoU ≥0.5',),fill='visual',fs=25,ss=22)
c.plain(1188,560,'YOLO boxes B',21)
wire('yolo-matching',[(1188,568),(1188,584)],None,'region-matching')
wire('upt-matching',[(1064,168),(1056,168),(1056,624),(1064,624)],'upt','region-matching')
# Junctions only collect existing text/link metadata; there is no extra learned module.
join('prepared-input',1304,448);join('prepared-fork',1392,448)
wire('text-prepared',[(1304,432),(1304,448)],'clip-text','prepared-input',False)
wire('links-prepared',[(1304,584),(1304,448)],'region-matching','prepared-input',False)
wire('prepared-branch',[(1304,448),(1392,448)],'prepared-input','prepared-fork',False)
for key,y,mem,out,colour in [('object',144,'Vₒ','Vₒ′','semantic'),('relation',464,'Vᵣ','Vᵣ′','relation')]:
 c.back.append(f'<rect x="1312" y="{y}" width="352" height="184" rx="6" fill="#fff" stroke="#bac4cb" stroke-width="1"/>')
 c.nodes.append(c.text(1336 if key=='object' else 1432,y+32,('Object' if key=='object' else 'Relation')+' adapter',23,600,'start'))
 c.box(key+'-slots',1336,y+80,112,64,'Slot MLP',(),fill=colour,fs=21)
 c.box(key+'-attention',1480,y+80,112,64,'Masked',('attention',),fill=colour,fs=21,ss=21)
 c.plus(key+'-sum',1624,y+112)
 wire(key+'-slots-attention',[(1448,y+112),(1480,y+112)],key+'-slots',key+'-attention')
 wire(key+'-attention-sum',[(1592,y+112),(1606,y+112)],key+'-attention',key+'-sum')
 token(key+'-updated',1664,y+96,out,ORANGE if key=='object' else BLUE)
 wire(key+'-sum-output',[(1642,y+112),(1664,y+112)],key+'-sum',key+'-updated')
wire('prepared-object-slots',[(1392,448),(1392,288)],'prepared-fork','object-slots')
wire('prepared-relation-slots',[(1392,448),(1392,544)],'prepared-fork','relation-slots')
c.plain(1404,316,'tₒ + Vₒ[i] + box',16,400,'start')
c.plain(1404,528,'tᵣ + Vₒ[s], Vₒ[o] + boxes',16,400,'start')
c.plain(1376,824,'Slot regions use original Vₒ',22)
# Explicit visual-query/residual forks. Upper object fork, lower relation fork.
join('object-visual-fork',1536,112)
wire('vo-adapter',[(1004,192),(1024,192),(1024,112),(1536,112)],'vo','object-visual-fork',False)
wire('object-visual-query',[(1536,112),(1536,224)],'object-visual-fork','object-attention')
wire('object-residual',[(1536,112),(1624,112),(1624,238)],'object-visual-fork','object-sum')
join('relation-visual-fork',1536,704)
wire('vr-adapter',[(1004,384),(1024,384),(1024,704),(1536,704)],'vr','relation-visual-fork',False)
wire('relation-visual-query',[(1536,704),(1536,608)],'relation-visual-fork','relation-attention')
wire('relation-residual',[(1536,704),(1624,704),(1624,594)],'relation-visual-fork','relation-sum')
# D. Updated memories connect once and directly to the nearby decoder.
c.plain(1720,28,'Caption generation',26,600,'start')
frame('decoder',1720,56,288,368,'Transformer decoder',fs=24)
c.box('gated-ffn',1744,128,240,64,'Gated fusion + FFN',(),fill='decoder',fs=24)
c.box('parallel-attention',1744,224,240,80,'Parallel',('cross-attention',),fill='decoder',fs=28,ss=26)
c.box('masked-self-attention',1744,336,240,64,'Masked self-attention',(),fill='decoder',fs=22)
c.plain(1976,416,'×3',23,600)
wire('object-memory-decoder',[(1684,256),(1744,256)],'object-updated','parallel-attention')
wire('relation-memory-decoder',[(1684,576),(1704,576),(1704,280),(1744,280)],'relation-updated','parallel-attention')
wire('self-cross',[(1864,336),(1864,304)],'masked-self-attention','parallel-attention')
wire('cross-ffn',[(1864,224),(1864,192)],'parallel-attention','gated-ffn')
c.box('embedding',1744,488,240,72,'Word embedding',('Fixed position',),fill='decoder',fs=24,ss=23)
c.box('prefix',1744,608,240,56,'Token prefix',(),fill='plain',fs=28)
wire('prefix-embedding',[(1864,608),(1864,560)],'prefix','embedding')
wire('embedding-self',[(1864,488),(1864,400)],'embedding','masked-self-attention')
c.box('vocab',2024,128,320,64,'Vocabulary projection',('+ softmax',),fill='decoder',fs=25,ss=23)
c.box('beam',2024,248,320,64,'Beam search',('K = 5',),fill='decoder',fs=25,ss=23)
wire('decoder-vocab',[(1984,160),(2024,160)],'gated-ffn','vocab')
wire('vocab-beam',[(2184,192),(2184,248)],'vocab','beam')
c.nodes.append('<g id="candidates"><rect data-node-id="candidates" x="2024" y="360" width="320" height="80" fill="#fff" stroke="#8a969e" stroke-width="1.4"/>'+c.text(2184,388,'Five candidate captions',24,600,owner='candidates')+'</g>')
c.meta.append(dict(id='candidates',x=2024,y=360,w=320,h=80,kind='data'))
for j in range(5):
 x=2048+j*56
 c.nodes.append(f'<rect x="{x}" y="400" width="40" height="28" fill="#fff" stroke="#adb7be" stroke-width=".9"/>')
 c.plain(x+20,420,'y'+str(j+1),18)
wire('beam-candidates',[(2184,312),(2184,360)],'beam','candidates')
frame('reranking',1720,688,640,240,'Caption reranking',fs=26)
for key,x,a,b in [('likelihood',1744,'D · Decoder','likelihood'),('alignment',1952,'C · Image–text','alignment'),('consistency',2160,'O · Object','consistency')]:
 c.box(key,x,752,176,64,a,(b,),fill='rank',fs=20,ss=21)
c.box('select',1744,848,592,64,'Weighted scoring',('and selection',),fill='rank',fs=25,ss=23)
wire('candidates-rerank',[(2184,440),(2184,688)],'candidates',None)
wire('d-select',[(1832,816),(1832,848)],'likelihood','select')
wire('c-select',[(2040,816),(2040,848)],'alignment','select')
wire('o-select',[(2248,816),(2248,848)],'consistency','select')
c.plain(1744,968,'I + YOLO labels',21,600,'start')
wire('rerank-evidence',[(1800,940),(1800,928)],None,None)
c.box('final-caption',2024,952,320,64,'Final caption',(),fill='rank',fs=30)
wire('selection-final',[(2272,912),(2272,952)],None,'final-caption')
# Parameter icons retain the user's blue snowflake / orange flame convention.
frozen=['clip-visual','yolo','bank-text-encoder','upt','clip-text','alignment']
learned=['projection','noun-aggregation','verb-aggregation','object-fusion-1','object-fusion-2','relation-fusion','embedding','vocab']
node_map={n['id']:n for n in c.meta}
for key in frozen+learned:
 n=node_map[key]
 bx=n['x']+n['w']+16 if key.endswith('-aggregation') else n['x']+n['w']-22
 c.nodes.append(icon('frozen' if key in frozen else 'trainable',bx,n['y']-22,key))
c.nodes.append(icon('trainable',1986,34,'decoder'))
for key,y in [('object',144),('relation',464)]:c.nodes.append(icon('trainable',1642,y-22,key+'-adapter'))
c.plain(1200,1048,'Training: cross-entropy pretraining followed by SCST + cross-entropy.',25)
for x,kind,label in [(848,'frozen','Frozen parameters'),(1256,'trainable','Trainable parameters')]:
 c.nodes.append(icon(kind,x,1070));c.plain(x+36,1090,label,23,400,'start')
# Square corners for data only.
for j,item in enumerate(c.nodes):
 for key in ['patches','boxes','noun-bank','verb-bank','train-lemmas','object-phrases','relation-phrases','prefix','final-caption']:
  if item.startswith('<g id="'+key+'">'):item=item.replace('rx="6"','rx="0"')
 c.nodes[j]=item
slug='captioning-pipeline-compact'
svg=c.svg(slug,(0,0,2400,1120))
caption='Luồng kỹ thuật được trình bày liên tục: các vùng ảnh đi trực tiếp qua hai object fusion và một relation fusion, rồi hai prompt adapter cập nhật memory trước decoder. Không lặp các block tổng quan hoặc memory sau adapter. Nhánh chuẩn bị prompt kết hợp text embeddings với valid region links; các chấm nhỏ chỉ điểm nối/tách dữ liệu, không phải module mới. Relation slot dùng Vo gốc. Icon tuyết xanh/lửa cam áp dụng khi huấn luyện; fixed position và softmax không có trọng số học. Ảnh chỉ minh họa. Reranking: S = 0.3D + 0.4C + 0.3O; D/C được min–max trong cùng 5 ứng viên.'
doc='<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Compact image captioning architecture</title><style>body{margin:0;background:#fff;font-family:Arial,"Liberation Sans",sans-serif}.diagram-container{width:100%;overflow-x:auto}svg{width:100%;min-width:2400px;display:block}figcaption{max-width:1100px;margin:20px auto;padding:0 24px;line-height:1.6;font-size:16px}@media print{.diagram-container{overflow:visible}svg{min-width:0}figcaption{display:none}}</style></head><body><main><div class="diagram-container">'+svg+'</div><figcaption lang="vi">'+html.escape(caption)+'</figcaption></main></body></html>'
(ROOT/(slug+'.html')).write_text(doc)
(ROOT/'diagram_geometry.json').write_text(json.dumps({'canvas':[2400,1120],'nodes':c.meta,'routes':c.routes},indent=2))
print('Built a single compact technical flow.')
