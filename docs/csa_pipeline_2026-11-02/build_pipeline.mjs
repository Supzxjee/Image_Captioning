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
const work='/workspace/csa-pipeline-build';
await fs.mkdir(work,{recursive:true});
const W=1920,H=1080,FONT='Liberation Serif';
const C={ink:'#151515',muted:'#444444',blue:'#222222',learn:'#ffffff',frozen:'#f7f7f7',purple:'#ffffff',orange:'#ffffff',green:'#ffffff',line:'#222222',white:'#ffffff'};
const require=createRequire(path.join(R,'@oai/artifact-tool/package.json'));
const {Canvas}=require('skia-canvas');const measure=new Canvas(1,1).getContext('2d');
const pages=[];let p;
function page(stem,title,sub){p={stem,title,items:[]};pages.push(p);text(40,22,1840,55,title,36,true);text(42,79,1835,35,sub,21,false,C.muted);}
function text(x,y,w,h,t,size=22,bold=false,color=C.ink,align='left'){p.items.push({type:'text',x,y,w,h,t,size,bold,color,align});}
function frame(x,y,w,h,t){p.items.push({type:'frame',x,y,w,h,t});text(x+16,y+8,Math.min(w-32,t.length*12+25),30,t,21,true,C.muted);}
function box(x,y,w,h,t,kind='learn',size=22){p.items.push({type:'box',x,y,w,h,t,kind,size});}
function arrow(pts,label='',lx=0,ly=0,lw=200,color=C.line,dash=false){p.items.push({type:'arrow',pts,color,dash});if(label)text(lx,ly,lw,30,label,20,false,color,'center');}
function footer(n){text(42,1032,1830,28,`CSA adaptation • cached CLIP ROI + semantic banks • 02/11/2026                                      ${n}/3`,19,false,C.muted);}
function legend(y=990){box(45,y,195,32,'F · frozen / cache','frozen',18);box(260,y,195,32,'T · trainable','learn',18);box(475,y,195,32,'TRAIN only','orange',18);text(700,y,1165,32,'Khung OFFLINE: chuẩn bị trước • Mũi tên: luồng dữ liệu • Nét đứt: gradient TRAIN',18,false,C.muted);}

page('01_overview','CSA adaptation — cached CLIP ROI + semantic banks','Pipeline hiện tại • tự triển khai theo CSA, thay feature/text encoder • sơ đồ kiến trúc và run XE10, chưa phải kết quả huấn luyện');
frame(40,125,860,245,'OFFLINE · visual inputs');frame(925,125,950,245,'OFFLINE · semantic banks toàn cục từ TRAIN');
box(60,225,120,65,'Ảnh RGB','frozen');box(220,175,210,65,'Predicted boxes\nproducer cache','frozen',20);
box(220,265,210,85,'CLIP ViT-B/16\n197×768 → 196×768\npatch grid 14×14','frozen',19);
box(480,230,195,90,'Fractional-area\nROI pooling','frozen');box(720,235,155,80,'ROI cache\nN×768','frozen');
arrow([[180,250],[195,250],[195,207],[220,207]]);arrow([[180,260],[195,260],[195,307],[220,307]]);
arrow([[430,207],[575,207],[575,230]]);arrow([[430,307],[455,307],[455,285],[480,285]]);arrow([[675,275],[720,275]]);
box(945,185,180,80,'Captions TRAIN\n113,287 ảnh','frozen',21);box(1160,185,180,80,'POS + lemma\nriding/rides → ride','frozen',20);
box(1375,185,190,80,'Frozen CLIP text\nL2-normalized 512D','frozen',20);
box(1610,175,245,145,'Noun bank: 5269×512\n\nVerb bank: 1594×512','frozen',22);
arrow([[1125,225],[1160,225]]);arrow([[1340,225],[1375,225]]);arrow([[1565,225],[1610,225]]);
text(945,292,610,53,'Banks không đổi theo ảnh suy luận;\nTRAIN XE10 không chạy lại detector/CLIP.',19,false,C.muted);
frame(40,390,1835,385,'CSA encoder + decoder · T');
box(325,440,300,70,'Noun semantic aggregator\nTo: 150×512','learn',21);box(965,440,300,70,'Verb semantic aggregator\nTr: 150×512','learn',21);
arrow([[1680,320],[1680,377],[475,377],[475,440]]);arrow([[1805,320],[1805,385],[1115,385],[1115,440]]);
box(65,595,190,95,'Visual projection\n768 → 512\nV0: N×512');
box(325,595,250,95,'Object stage 1\n(V0, To) → (V1, T1)');box(645,595,250,95,'Object stage 2\n(V1, T1) → Vo');
box(965,595,250,95,'Relation stage\n(Vo, Tr) → Vr');box(1305,595,260,95,'Parallel decoder\n3 layers · Vo và Vr');
box(1640,595,225,95,'Linear 512 → 10000\nNext-token\ndistribution', 'learn',20);
arrow([[798,315],[798,360],[25,360],[25,642],[65,642]]);arrow([[255,642],[325,642]],'V0',260,607,65);
arrow([[475,510],[475,595]],'To',483,545,60);arrow([[575,642],[645,642]],'V1 + T1',568,607,85);
arrow([[895,642],[965,642]],'Vo',900,607,60);arrow([[1115,510],[1115,595]],'Tr',1120,545,65);
arrow([[1215,642],[1305,642]],'Vr',1220,607,75);arrow([[895,665],[930,665],[930,735],[1435,735],[1435,690]],'Vo · nhánh riêng',1030,739,230);
arrow([[1565,642],[1640,642]]);box(1305,455,260,70,'Prefix token IDs\nTRAIN ref / VALID tự sinh','purple',21);arrow([[1435,525],[1435,595]]);
frame(40,810,875,160,'TRAIN · 113,287 ảnh / 566,747 caption events');
box(60,860,240,68,'GT caption prefix\nshift-right','orange',22);
box(345,860,230,68,'Cùng mạng CSA\nở phía trên','learn',22);
box(630,860,265,68,'Targets next tokens\nXE / CE loss','orange',22);
arrow([[300,894],[345,894]]);arrow([[575,894],[630,894]]);
arrow([[895,894],[932,894],[932,790],[690,790],[690,775]],'Backward: chỉ module T',960,776,340,C.blue,true);
text(60,935,830,28,'10 epochs · global32 · 17,711 updates/epoch · không SCST/RL',20,false,C.muted,'center');
frame(950,810,925,160,'VALID · 5000 ảnh sau mỗi epoch / không backward');
box(970,860,250,68,'BOS → beam5 captions\nkhông reference prefix','green',21);
box(1260,860,240,68,'PTB / CIDEr\n+ raw references','green',22);
box(1540,860,315,68,'best.bin: max CIDEr\ncheckpoint.bin: latest','green',22);
arrow([[1220,894],[1260,894]]);arrow([[1500,894],[1540,894]]);
text(970,935,885,28,'NLL: 25,010 refs, teacher forcing, eval only · TEST chưa dùng',20,false,C.muted,'center');
legend();footer(1);

page('02_encoder','Encoder CSA — hai stage object, một stage relation','Đối chiếu CSACaptioner.encode, Alignment.forward và Fusion.forward • To/T1/Tr là semantic prototypes, không phải UPT triplets');
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
text(55,920,1800,57,'N thay đổi theo ảnh, max87 trong cache full; empty detections → 1 full-frame fallback.\nROI dùng contextual CLIP patches; selection/attention chưa chứng minh mỗi từ caption gắn chính xác một box.',22,false,C.muted);
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
   if(!isFrame){addText({...a,x:a.x+9,y:a.y+9,w:a.w-18,h:a.h-18,color:C.ink,align:'center'});if(a.h>40&&['learn','frozen'].includes(a.kind))addText({x:a.x+a.w-22,y:a.y+2,w:19,h:19,t:a.kind==='learn'?'T':'F',size:13,bold:true,color:C.ink,align:'center'});}
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
