// Editable scientific diagrams. No model imports or model execution.
// Run with the supplied Codex Node runtime; see README.md.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL, fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
const R=process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES;
if(!R) throw new Error('Missing supplied CODEX_PRIMARY_RUNTIME_NODE_MODULES');
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(R,'@oai/artifact-tool/dist/artifact_tool.mjs')).href).catch(async()=>await import('@oai/artifact-tool'));
const out=path.dirname(fileURLToPath(import.meta.url));
const work='/workspace/csa-pipeline-build-v2';
await fs.mkdir(work,{recursive:true});
const W=1920,H=1180,FONT='Liberation Serif';
const C={ink:'#151515',muted:'#444444',blue:'#222222',learn:'#ffffff',frozen:'#f7f7f7',purple:'#ffffff',orange:'#ffffff',green:'#ffffff',line:'#222222',white:'#ffffff'};
const require=createRequire(path.join(R,'@oai/artifact-tool/package.json'));
const {Canvas}=require('skia-canvas');const measure=new Canvas(1,1).getContext('2d');
const pages=[];let p;
function page(stem,title,sub){p={stem,title,items:[]};pages.push(p);text(40,22,1840,55,title,36,true);text(42,79,1835,35,sub,21,false,C.muted);}
function text(x,y,w,h,t,size=22,bold=false,color=C.ink,align='left'){p.items.push({type:'text',x,y,w,h,t,size,bold,color,align});}
function frame(x,y,w,h,t){p.items.push({type:'frame',x,y,w,h,t});text(x+16,y+8,Math.min(w-32,t.length*12+25),30,t,21,true,C.muted);}
function box(x,y,w,h,t,kind='learn',size=22){p.items.push({type:'box',x,y,w,h,t,kind,size});}
function arrow(pts,label='',lx=0,ly=0,lw=200,color=C.line,dash=false){p.items.push({type:'arrow',pts,color,dash});if(label)text(lx,ly,lw,30,label,20,false,color,'center');}
function footer(n){text(42,1132,1500,28,'CSA adaptation • cached CLIP ROI + semantic banks • 02/11/2026',19,false,C.muted);text(1800,1132,75,28,`${n}/3`,19,false,C.muted,'center');}
function legend(y=990){box(45,y,195,32,'Frozen encoder / cache','frozen',18);box(260,y,195,32,'Learned modules','learn',18);box(475,y,195,32,'TRAIN only','orange',18);text(700,y,1165,32,'Khung nét đứt: nhóm xử lý • Mũi tên: luồng dữ liệu',18,false,C.muted);}

page('01_overview','CSA adaptation','Cached CLIP ROI + semantic banks');
// Offline semantic path follows the reference layout: a compact upper band.
frame(55,125,1270,220,'Offline semantic bank preparation · TRAIN captions only');
box(95,195,225,80,'TRAIN captions','purple',27);
box(370,195,225,80,'POS + lemma','purple',27);
box(645,195,255,80,'Frozen CLIP\ntext encoder','frozen',26);
box(955,170,325,130,'Global semantic banks\nNouns: 5269×512\nVerbs: 1594×512','purple',25);
arrow([[320,235],[370,235]]);arrow([[595,235],[645,235]]);arrow([[900,235],[955,235]]);
box(1420,165,380,130,'Two learned aggregators\nNouns → To · Verbs → Tr\nK=150 prototypes, d=512','learn',26);
arrow([[1280,235],[1420,235]]);
// Visual branch runs left to right; no return wire crossing an upper title.
box(70,500,165,100,'Input image\nI (RGB)','purple',28);
frame(280,360,835,300,'Offline visual feature preparation');
box(300,410,225,75,'Predicted boxes\nproducer cache','purple',25);
box(300,535,225,100,'Frozen CLIP vision\nViT-B/16','frozen',26);
box(580,535,235,100,'Patch features\n196×768 (14×14)','purple',26);
box(870,535,225,100,'Fractional ROI pooling\nRaw ROI cache\nN×768','purple',24);
arrow([[152,500],[152,447],[300,447]]);
arrow([[235,550],[260,550],[260,585],[300,585]]);
arrow([[525,585],[580,585]]);arrow([[815,585],[870,585]]);
arrow([[525,447],[982,447],[982,535]],'Boxes',690,450,100);
text(295,676,810,55,'N varies per image (max 87); empty detections → full-frame fallback.\nContextual CLIP patches; not isolated object pixels.',21,false,C.muted);
box(1150,535,225,100,'Visual projector\n768 → 512','learn',27);
arrow([[1095,585],[1150,585]]);
text(1130,653,260,35,'V0: N×512',25,false,C.ink,'center');
// CSA is vertical, as the fusion block in the user's reference.
frame(1395,380,440,415,'CSA');
box(1420,435,360,80,'Object alignment 1\n(V0, To) → (V1, T1)','learn',26);
box(1420,565,360,80,'Object alignment 2\n(V1, T1) → Vo','learn',26);
box(1420,695,360,80,'Relation alignment\n(Vo, Tr) → Vr','learn',26);
arrow([[1375,585],[1385,585],[1385,475],[1420,475]]);
arrow([[1500,295],[1500,425],[1600,425],[1600,435]],'To',1512,335,70);
arrow([[1760,295],[1875,295],[1875,735],[1780,735]],'Tr',1805,340,60);
arrow([[1600,515],[1600,565]],'V1 + T1',1620,525,150);
arrow([[1600,645],[1600,695]],'Vo',1620,655,70);
// Object and relation streams remain separate into the decoder.
box(950,850,300,90,'Parallel adaptive\ndecoder (×3)','learn',28);
arrow([[1780,605],[1845,605],[1845,895],[1250,895]],'Vo',1770,820,65);
arrow([[1600,775],[1600,815],[1100,815],[1100,850]],'Vr',1425,780,65);
box(620,850,250,90,'Linear + softmax\nvocabulary 10,000','learn',26);
box(300,850,250,90,'Generated caption','purple',28);
arrow([[950,895],[870,895]]);arrow([[620,895],[550,895]]);
box(950,980,300,70,'Embedding + positions','learn',27);
box(1330,980,330,70,'GT captions (shift right)\nTRAIN only','purple',25);
box(550,980,300,70,'BOS / generated prefix\nVALID & inference','purple',24);
arrow([[1330,1015],[1250,1015]]);arrow([[850,1015],[950,1015]]);arrow([[1100,980],[1100,940]]);
text(75,1080,1780,50,'TRAIN: XE loss on next tokens. VALID: beam 5 → CIDEr → best checkpoint; no backward.\nFrozen encoders and caches are prepared offline; learned aggregators, projector, CSA and decoder train together.',21,false,C.muted,'center');
footer(1);

page('02_encoder','Encoder CSA — hai stage object, một stage relation','Object → object → relation · Chi tiết selection và visual–semantic fusion');
box(45,160,245,90,'ROI cache frozen\nN×768; explicit valid mask','frozen',22);box(345,160,245,90,'Learnable projection\nV0: N×512');arrow([[290,205],[345,205]]);
box(650,155,300,100,'Noun bank 5269×512\nfrozen → learned aggregator\nTo: 150×512','purple',21);
box(1230,155,300,100,'Verb bank 1594×512\nfrozen → learned aggregator\nTr: 150×512','purple',21);
box(345,340,310,115,'Object stage 1\nV0 + To\n→ V1 và selected noun T1');
box(835,340,310,115,'Object stage 2\nV1 + T1\n→ Vo');box(1375,340,310,115,'Relation stage\nVo + Tr\n→ Vr');
arrow([[467,250],[467,340]]);arrow([[800,255],[800,295],[580,295],[580,340]],'To',650,263,70);
arrow([[655,380],[835,380]],'V1',690,340,90);arrow([[655,430],[835,430]],'T1 · carry',670,455,150);
arrow([[1145,390],[1375,390]],'Vo',1210,350,80);arrow([[1380,255],[1530,255],[1530,340]],'Tr',1450,270,70);
arrow([[1050,455],[1050,510],[1220,510]],'Vo',1100,472,100);arrow([[1685,400],[1820,400],[1820,510],[1490,510]],'Vr → decoder',1650,450,200);
box(1220,480,270,60,'Decoder nhận Vo và Vr','purple',21);text(45,515,900,34,'BÊN TRONG MỖI STAGE',25,true);
frame(40,560,1835,340,'Alignment / selection');
box(65,625,235,105,'Incoming prototypes P\n+ incoming regions V\nP: K×512, V: N×512','purple',21);
box(355,625,270,105,'Cosine S: K×N\nSoftmax theo vùng, τ=0.1\nRk = Σn wkn Skn','learn',21);
box(680,625,275,105,'Softmax theo prototype\nSelected T = softmax(R)⊙P\nK=150, không top-K retrieval','learn',20);
arrow([[300,678],[355,678]]);arrow([[625,678],[680,678]]);
box(1010,605,310,115,'Visual self-attention\nAdd & Norm → FFN\nOwn stream: N×512','learn',22);
box(1390,605,430,115,'Cross-attention\nQ = own visual stream\nK,V = selected semantic T\nAdd & Norm → FFN = cross','learn',22);
arrow([[180,625],[180,600],[1165,600],[1165,605]]);
text(725,565,200,28,'V từ stage',18,false,C.muted);
arrow([[1320,660],[1390,660]],'Q',1325,620,60);arrow([[955,700],[980,700],[980,750],[1605,750],[1605,720]],'Selected T → K,V',680,755,280);
box(1010,810,810,65,'Scalar sigmoid gate G: N×1 → G⊙own + (1−G)⊙cross','learn',23);
arrow([[1165,720],[1165,810]]);arrow([[1775,720],[1840,720],[1840,842],[1820,842]]);
text(55,920,1800,57,'N thay đổi theo ảnh, max 87 trong cache full; empty detections → 1 full-frame fallback.\nROI dùng contextual CLIP patches; selection/attention chưa chứng minh mỗi từ caption gắn chính xác một box.',22,false,C.muted);
legend(990);footer(2);

page('03_decoder','Parallel adaptive decoder và TRAIN / VALID','Vo và Vr được đọc bằng hai cross-attention riêng • Cùng Q là hidden text sau causal self-attention');
box(45,180,215,80,'Prefix token IDs\nTRAIN: reference shift','purple',22);
box(315,180,255,100,'Word embedding 512D\nraw embedding e_prev\n+ sinusoidal position','learn',21);
box(650,180,285,100,'Causal self-attention\nAdd & Norm\nHidden text H','learn',22);
arrow([[260,220],[315,220]]);arrow([[570,220],[650,220]]);
frame(1000,135,850,550,'PARALLEL CA + GATES · trong mỗi layer');
box(1040,230,285,95,'Object cross-attention\nQ=H; K,V=Vo\nContext co','learn',22);
box(1040,430,285,95,'Relation cross-attention\nQ=H; K,V=Vr\nContext cr','learn',22);
box(1385,230,400,95,'go = sigmoid(Wo[co; e_prev])\nVector gate riêng: 512D','learn',23);
box(1385,430,400,95,'gr = sigmoid(Wr[cr; e_prev])\nVector gate riêng: 512D','learn',23);
arrow([[935,225],[970,225],[970,278],[1040,278]],'H',933,270,30);arrow([[970,278],[970,478],[1040,478]]);
box(655,365,235,70,'Object stream Vo\nN×512','purple',22);box(655,520,235,70,'Relation stream Vr\nN×512','purple',22);
arrow([[890,400],[920,400],[920,310],[1040,310]]);arrow([[890,555],[1020,555],[1020,510],[1040,510]]);
arrow([[1325,278],[1385,278]]);arrow([[1325,478],[1385,478]]);
arrow([[443,180],[443,120],[1585,120],[1585,230]],'raw e_prev · không phải prompt phụ',445,138,500,C.blue);
arrow([[1585,190],[1818,190],[1818,478],[1785,478]],'',0,0,0,C.blue);
box(1150,595,600,65,'go⊙co + gr⊙cr → FFN → layer tiếp','learn',25);
arrow([[1585,325],[1585,355],[1832,355],[1832,570],[1630,570],[1630,595]]);
arrow([[1585,525],[1585,595]]);
box(1070,720,350,70,'Linear 512 → vocab10000\nNext-token distribution','learn',23);arrow([[1450,660],[1450,695],[1245,695],[1245,720]]);
box(1470,720,380,70,'BOS → beam5 → caption\nmax20 · raw logprob · stop EOS','green',21);arrow([[1420,755],[1470,755]]);
box(45,695,900,110,'TRAIN · targets next token → XE loss → backward\n113,287 ảnh · 566,747 caption events/epoch\n10 epochs; global32; không SCST/RL','orange',24);
box(45,845,900,125,'VALID · eval, không nhận reference prefix khi sinh\n5000 ảnh → beam5 captions + raw refs → PTB / CIDEr\nTeacher-forced NLL trên25,010 refs là nhánh đo riêng','green',23);
box(1010,845,840,125,'Checkpoint selection · không có gradient VALID\nbest.bin: CIDEr lớn nhất; hòa chọn epoch sớm\ncheckpoint.bin: latest để resume · TEST chưa dùng','green',23);
legend(990);footer(3);

const esc=s=>s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
function svgSegments(pts,color,dash,width=2){let z='';for(let k=1;k<pts.length;k++){const [x,y]=pts[k-1],[u,v]=pts[k],d=Math.hypot(u-x,v-y);if(!d)continue;const dx=(u-x)/d,dy=(v-y)/d;if(!dash)z+=`<path d="M${x},${y} L${u},${v}" fill="none" stroke="${color}" stroke-width="${width}"/>`;else for(let a=0;a<d;a+=12){const b=Math.min(a+7,d);z+=`<path d="M${x+a*dx},${y+a*dy} L${x+b*dx},${y+b*dy}" stroke="${color}" stroke-width="${width}"/>`;}}return z;}
const deck=Presentation.create({slideSize:{width:W,height:H}});
for(const page of pages){
 const slide=deck.slides.add();slide.background.fill=C.white;
 slide.speakerNotes.text='Sources: RESEARCH_EXCHANGE.md X051/X052; experiments/csa_paper/model.py (read as text only); FULL_XE_10_PROTOCOL.md and run_full_xe.py, full_xe_decode.py at 5a14d1b. Architecture diagram only; first XE10 attempt stopped before TRAIN; V2 runtime pending. No exact benchmark reproduction or new-method claim.';
 let svg=`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><defs><marker id="arrow" markerWidth="9" markerHeight="9" refX="8" refY="4" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L8,4 L0,8" fill="${C.line}"/></marker></defs><rect width="${W}" height="${H}" fill="white"/>`;
 let i=0;
 function addText(input){const a={...input}; let lines; do { measure.font=`${a.bold?'bold ':''}${a.size}px ${FONT}`; lines=[]; for(const line of a.t.split('\n')) {let current=''; for(const word of line.split(' ')){if(current && measure.measureText(current+' '+word).width>a.w-4){lines.push(current);current=word;}else current=current?current+' '+word:word;} lines.push(current);} if(lines.length*a.size*1.25>a.h && a.size>17){a.size-=1;}else break;}while(true); a.t=lines.join('\n');const shape=slide.shapes.add({geometry:'textbox',name:`text-${i++}`,position:{left:a.x,top:a.y,width:a.w,height:a.h},fill:'none',line:{style:'solid',fill:'none',width:0}});shape.text=a.t;shape.text.style={typeface:FONT,fontSize:a.size,bold:!!a.bold,color:a.color||C.ink,alignment:a.align||'center',verticalAlignment:'middle',insets:0};
 const lineH=a.size*1.25,base=a.y+(a.h-lines.length*lineH)/2+a.size;
 for(let k=0;k<lines.length;k++)svg+=`<text x="${a.align==='left'?a.x:a.x+a.w/2}" y="${base+k*lineH}" text-anchor="${a.align==='left'?'start':'middle'}" font-family="${FONT}" font-size="${a.size}" font-weight="${a.bold?'700':'400'}" fill="${a.color||C.ink}">${esc(lines[k])}</text>`;}
 for(const a of page.items){
  if(a.type==='text'){addText(a);continue;}
  if(a.type==='box'||a.type==='frame'){
   const isFrame=a.type==='frame',fill=isFrame?'none':C[a.kind]||C.learn,stroke='#333333';
   slide.shapes.add({geometry:a.type==='box'&&a.kind==='learn'?'roundRect':'rect',name:`${a.type}-${i++}`,position:{left:a.x,top:a.y,width:a.w,height:a.h},fill,line:{style:isFrame?'dashed':'solid',fill:stroke,width:1.6}});
   if(isFrame)svg+=svgSegments([[a.x,a.y],[a.x+a.w,a.y],[a.x+a.w,a.y+a.h],[a.x,a.y+a.h],[a.x,a.y]],stroke,true,1.3);else svg+=`<rect x="${a.x}" y="${a.y}" width="${a.w}" height="${a.h}" fill="${fill}" stroke="${stroke}" stroke-width="1.3" rx="${a.kind==='learn'?10:0}"/>`;
   if(!isFrame){addText({...a,x:a.x+9,y:a.y+9,w:a.w-18,h:a.h-18,color:C.ink,align:'center'});}
  }else if(a.type==='arrow'){
   const pts=a.pts;
   for(let j=1;j<pts.length;j++){const [x,y]=pts[j-1],[x2,y2]=pts[j];if(x===x2&&y===y2)continue;slide.shapes.add({geometry:'connector',kind:'straight',name:`connector-${i++}`,position:{left:Math.min(x,x2),top:Math.min(y,y2),width:Math.abs(x2-x),height:Math.abs(y2-y),horizontalFlip:x2<x,verticalFlip:y2<y},line:{style:a.dash?'dashed':'solid',fill:a.color,width:2},tail:{type:j===pts.length-1?'triangle':'none',width:'sm',length:'sm'}});}
   svg+=svgSegments(pts,a.color,a.dash);const [ex,ey]=pts.at(-1),[sx,sy]=pts.at(-2),len=Math.hypot(ex-sx,ey-sy),dx=(ex-sx)/len,dy=(ey-sy)/len;svg+=`<polygon points="${ex},${ey} ${ex-8*dx+4*dy},${ey-8*dy-4*dx} ${ex-8*dx-4*dy},${ey-8*dy+4*dx}" fill="${a.color}"/>`;
  }
 }
 svg+='</svg>';await fs.writeFile(path.join(out,`${page.stem}.svg`),svg);
 await fs.writeFile(path.join(out,`${page.stem}.png`),new Uint8Array(await (await deck.export({slide,format:'png',scale:1.25})).arrayBuffer()));
}
await fs.writeFile(path.join(out,'diagram_layout.json'),JSON.stringify(pages,null,2));
const candidate=path.join(work,'candidate.pptx');await(await PresentationFile.exportPptx(deck)).save(candidate);
const {finalizePresentation}=await import(pathToFileURL('/home/agent/.codex/skills/builtins/presentations/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:'/workspace',candidatePath:candidate,finalPath:path.join(out,'CSA_pipeline_editable.pptx'),pythonExecutable:process.env.CODEX_PRIMARY_RUNTIME_PYTHON,
 integrityValidatorPath:'/home/agent/.codex/skills/builtins/presentations/container_tools/inspect_presentation_package_integrity.py',
 layoutValidatorPath:'/home/agent/.codex/skills/builtins/presentations/container_tools/inspect_presentation_layout_geometry.py',
 layoutArgs:['--expected-slide-size-emu',`${W*9525},${H*9525}`,'--validate-bullet-geometry','--validate-heading-fit'],fontPolicy:{basis:'design',families:[FONT]},explicitTotalSlideCount:3,verifyArtifactToolImport:true,receiptPath:path.join(work,`validation-${Date.now()}.json`)});
const {spawnSync}=await import('node:child_process');
const pdf=spawnSync(process.env.CODEX_PRIMARY_RUNTIME_PYTHON,['-c',`import fitz,sys,pathlib
p=pathlib.Path(sys.argv[1]);d=fitz.open()
for name in ['01_overview','02_encoder','03_decoder']:
 s=fitz.open(str(p/(name+'.svg')));v=fitz.open('pdf',s.convert_to_pdf());d.insert_pdf(v)
d.save(str(p/'CSA_pipeline_vector.pdf'))
`,out],{encoding:'utf8'});
if(pdf.status!==0)throw new Error(pdf.stderr||'Vector PDF conversion failed');
console.log('Created 3 editable diagrams and SVG/PNG/vector PDF exports.');
