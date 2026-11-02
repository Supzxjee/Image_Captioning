// Editable paper figure. All network modules and data symbols are native shapes.
// No model imports, inference, training or generated model outputs.
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {createRequire} from 'node:module';
const R=process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES;
if(!R)throw new Error('Use the supplied presentation runtime');
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(R,'@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const require=createRequire(path.join(R,'@oai/artifact-tool/package.json'));
const {Canvas}=require('skia-canvas');const measure=new Canvas(1,1).getContext('2d');
const out=path.dirname(fileURLToPath(import.meta.url)),work='/workspace/csa-figure-v3';
await fs.mkdir(work,{recursive:true});
const W=1600,H=1040,FONT='Liberation Serif',INK='#232323',BLUE='#6a9ca9',GOLD='#cf8e3a';
const items=[];
function txt(x,y,w,h,t,size=19,italic=false,bold=false,align='center',color=INK){items.push({type:'text',x,y,w,h,t,size,italic,bold,align,color});}
function shape(x,y,w,h,t='',kind='module',size=19,learn=false,frozen=false){items.push({type:'shape',x,y,w,h,t,kind,size,learn,frozen});}
function frame(x,y,w,h,t,tw=350){items.push({type:'frame',x,y,w,h});txt(x+15,y+10,tw,26,t,19,true,true,'left');}
function wire(pts,label='',x=0,y=0,w=80){items.push({type:'wire',pts});if(label)txt(x,y,w,26,label,18,true);}
function marker(x,y,kind){items.push({type:'marker',x,y,kind});}
// Semantic preprocessing: compact band, with two visible bank stores.
frame(35,20,1530,210,'Offline semantic bank construction · TRAIN only',650);
shape(75,105,170,70,'TRAIN captions','data',20);
shape(285,105,175,70,'POS tagging\n+ lemmatization','module',19);
shape(500,105,175,70,'CLIP text encoder','module',19,false,true);
shape(715,105,145,70,'L2 normalization','module',18);
shape(955,85,210,105,'Noun bank\n5269 × 512','bank',20);
shape(1270,85,210,105,'Verb bank\n1594 × 512','bank',20);
wire([[245,140],[285,140]]);wire([[460,140],[500,140]]);wire([[675,140],[715,140]]);
wire([[860,140],[955,140]]);
wire([[900,140],[900,65],[1375,65],[1375,85]]);
// Learned aggregators remain outside the offline bank band.
shape(970,270,180,60,'Noun aggregator','module',19,true);
shape(1285,270,180,60,'Verb aggregator','module',19,true);
wire([[1060,190],[1060,270]]);wire([[1375,190],[1375,270]]);
txt(1090,235,180,24,'K = 150, d = 512',16,true);
// Offline visual branch. Photograph is an existing documented TRAIN example.
frame(290,365,615,270,'Offline visual feature extraction',470);
items.push({type:'photo',x:55,y:490,w:200,h:150,file:'assets/COCO_train2014_000000263974.jpg'});
txt(55,648,200,28,'Input image I',19,true);
shape(310,500,175,75,'CLIP vision encoder\nViT-B/16','module',19,false,true);
shape(535,500,140,75,'Patch tokens\n196 × 768','tokens',19);
shape(725,395,150,60,'Predicted boxes\nproducer cache','data',18);
shape(725,500,150,75,'ROI pooling\nN × 768 cache','module',19);
wire([[155,490],[155,425],[725,425]]);
wire([[255,565],[282,565],[282,538],[310,538]]);
wire([[485,538],[535,538]]);wire([[675,538],[725,538]]);wire([[800,455],[800,500]]);
txt(520,587,210,25,'14 × 14 · CLS removed',16,true);
shape(725,665,150,75,'Visual projection\n768 → 512','module',19,true);
wire([[800,575],[800,665]],'N × 768',810,627,83);
txt(300,678,360,52,'N varies per image (max 87)\nEmpty boxes → full-frame fallback',17,true);
// Three alignment stages. Semantics enter from above / right, visual from left.
frame(930,365,635,425,'CSA encoder',195);
shape(1010,420,360,75,'Object alignment 1\n(V₀, Tₒ) → (V₁, T₁)','module',21,true);
shape(1010,545,360,75,'Object alignment 2\n(V₁, T₁) → Vₒ','module',21,true);
shape(1010,670,360,75,'Relation alignment\n(Vₒ, Tᵣ) → Vᵣ','module',21,true);
wire([[875,702],[910,702],[910,458],[1010,458]],'V₀: N × 512',785,753,140);
wire([[1060,330],[1060,350],[1190,350],[1190,420]],'Tₒ',1205,380,45);
wire([[1375,330],[1535,330],[1535,708],[1370,708]],'Tᵣ',1480,485,45);
wire([[1190,495],[1190,545]],'V₁, T₁',1205,506,100);
wire([[1190,620],[1190,670]],'Vₒ',1205,631,55);
// Vo is retained independently, avoiding the Tr wire on the right side.
wire([[1010,582],[970,582],[970,813],[1040,813],[1040,835]],'Vₒ',974,766,45);
wire([[1190,745],[1190,835]],'Vᵣ',1205,800,50);
shape(945,835,425,75,'Parallel adaptive decoder (×3)','module',22,true);
shape(715,835,150,75,'Linear\n+ softmax','module',20,true);
shape(430,835,210,75,'Generated caption','data',21);
wire([[945,872],[865,872]]);wire([[715,872],[640,872]]);
txt(390,916,295,44,'VALID: beam 5 → CIDEr\nBest checkpoint selection',16,true);
txt(705,916,170,30,'TRAIN: next-token XE',16,true);
shape(1010,965,235,55,'Embedding + positions','module',19,true);
shape(1300,965,225,55,'GT captions (shift right)\nTRAIN only','data',18);
shape(685,965,245,55,'BOS / generated prefix\nVALID & inference','data',18);
wire([[1300,992],[1245,992]]);wire([[930,992],[1010,992]]);wire([[1127,965],[1127,910]]);
// Small figure caption and conventional frozen / learned markers.
txt(40,935,345,29,'CSA adaptation',20,true,false,'left');
txt(40,964,485,25,'Cached CLIP ROI + global semantic banks',16,true,false,'left');
marker(51,1007,'frozen');txt(66,994,95,26,'Frozen',16,false,false,'left');
marker(180,1007,'learn');txt(196,994,135,26,'Trainable',16,false,false,'left');

const deck=Presentation.create({slideSize:{width:W,height:H}});
const slide=deck.slides.add();slide.background.fill='#ffffff';
slide.speakerNotes.text='CSA adaptation, current forward model.py SHA256139edbb1b73caf3dadaf5c960b25542e282f88baf2ce9bdf141de922ceacbc09. X051, FULL_XE_10_PROTOCOL.md and full_xe_decode.py; architecture/recipe, not completed training results. Input photograph: COCO_train2014_000000263974.jpg, https://images.cocodataset.org/train2014/COCO_train2014_000000263974.jpg ; exact image and SHA already documented in RID_IMAGE_SEMANTIC_CHECK.json. Photo is an illustrative input, not a detection or caption result. Ground-truth training prefix and BOS/generated validation prefix are alternative modes. Validation has no backward. ROI patches contain global context. Two bank aggregators each have150learnedprototypes512D. Decoder has parallel object/relation cross-attention and separate vector gates; see v2 encoder/decoder zooms for internals.';
let index=0;
const esc=s=>s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
let svg=`<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}"><rect width="${W}" height="${H}" fill="white"/>`;
function nativeRect(x,y,w,h,fill='white',line=INK,round=false){slide.shapes.add({geometry:round?'roundRect':'rect',borderRadius:round?9:0,name:`shape-${index++}`,position:{left:x,top:y,width:w,height:h},fill,line:{fill:line,width:1,style:'solid'}});}
function nativeLine(x,y,u,v,color=INK,width=1){slide.shapes.add({geometry:'connector',kind:'straight',name:`line-${index++}`,position:{left:Math.min(x,u),top:Math.min(y,v),width:Math.abs(u-x),height:Math.abs(v-y),horizontalFlip:u<x,verticalFlip:v<y},line:{fill:color,width,style:'solid'}});}
function lineSvg(x,y,u,v,color=INK,width=1){return `<path d="M${x},${y}L${u},${v}" stroke="${color}" stroke-width="${width}" fill="none"/>`;}
function renderText(input){const a={...input};let lines=[];
 do {measure.font=`${a.italic?'italic ':''}${a.bold?'bold ':''}${a.size}px ${FONT}`;lines=[];
  for(const raw of a.t.split('\n')){let s='';for(const word of raw.split(' ')){if(s&&measure.measureText(s+' '+word).width>a.w-2){lines.push(s);s=word;}else s=s?s+' '+word:word;}lines.push(s);}
  if(lines.length*a.size*1.23>a.h&&a.size>14)a.size--;else break;
 }while(true);
 const s=slide.shapes.add({geometry:'textbox',name:`text-${index++}`,position:{left:a.x,top:a.y,width:a.w,height:a.h},fill:'none',line:{fill:'none',width:0}});
 s.text=lines.join('\n');s.text.style={typeface:FONT,fontSize:a.size,bold:!!a.bold,italic:!!a.italic,color:a.color||INK,alignment:a.align||'center',verticalAlignment:'middle',insets:0};
 const lh=a.size*1.23,y=a.y+(a.h-lines.length*lh)/2+a.size;
 for(let j=0;j<lines.length;j++)svg+=`<text x="${a.align==='left'?a.x:a.x+a.w/2}" y="${y+j*lh}" text-anchor="${a.align==='left'?'start':'middle'}" font-family="${FONT}" font-size="${a.size}" font-style="${a.italic?'italic':'normal'}" font-weight="${a.bold?'700':'400'}" fill="${a.color||INK}">${esc(lines[j])}</text>`;
}
function renderMarker(a){const {x,y}=a;
 if(a.kind==='frozen'){
  for(let k=0;k<3;k++){const angle=k*Math.PI/3,dx=6*Math.cos(angle),dy=6*Math.sin(angle);nativeLine(x-dx,y-dy,x+dx,y+dy,BLUE,1.4);svg+=lineSvg(x-dx,y-dy,x+dx,y+dy,BLUE,1.4);}
 }else{
  const d=`M${x},${y-9} C${x+1},${y-3} ${x+7},${y-2} ${x+5},${y+4} C${x+3},${y+9} ${x-5},${y+8} ${x-5},${y+2} C${x-5},${y-2} ${x-2},${y-3} ${x},${y-9} Z`;
  slide.shapes.add({geometry:'custom',name:`learn-marker-${index++}`,position:{left:x-6,top:y-10,width:12,height:19},fill:GOLD,line:{fill:'none',width:0},customPaths:[{width:12,height:19,commands:[{moveTo:{x:6,y:1}},{cubicBezTo:{x1:7,y1:7,x2:13,y2:8,x:11,y:14}},{cubicBezTo:{x1:9,y1:19,x2:1,y2:18,x:1,y:12}},{cubicBezTo:{x1:1,y1:8,x2:4,y2:7,x:6,y:1}},{close:{}}]}]});
  svg+=`<path d="${d}" fill="${GOLD}"/>`;
 }
}
for(const a of items){
 if(a.type==='text'){renderText(a);continue;}
 if(a.type==='marker'){renderMarker(a);continue;}
 if(a.type==='photo'){
  const blob=new Uint8Array(await fs.readFile(path.join(out,a.file)));
  slide.images.add({blob,contentType:'image/jpeg',fit:'contain',alt:'Illustrative COCO TRAIN input263974; not a model result',position:{left:a.x,top:a.y,width:a.w,height:a.h}});
  svg+=`<image x="${a.x}" y="${a.y}" width="${a.w}" height="${a.h}" preserveAspectRatio="xMidYMid meet" xlink:href="data:image/jpeg;base64,${Buffer.from(blob).toString('base64')}"/>`;continue;
 }
 if(a.type==='frame'){
  slide.shapes.add({geometry:'rect',name:`offline-frame-${index++}`,position:{left:a.x,top:a.y,width:a.w,height:a.h},fill:'none',line:{fill:INK,width:.95,style:'dashed'}});
  const pts=[[a.x,a.y],[a.x+a.w,a.y],[a.x+a.w,a.y+a.h],[a.x,a.y+a.h],[a.x,a.y]];
  for(let j=1;j<pts.length;j++){const [x,y]=pts[j-1],[u,v]=pts[j],d=Math.hypot(u-x,v-y),dx=(u-x)/d,dy=(v-y)/d;for(let q=0;q<d;q+=9){const end=Math.min(q+4,d);svg+=lineSvg(x+q*dx,y+q*dy,x+end*dx,y+end*dy,INK,.9);}}
  continue;
 }
 if(a.type==='shape'){
  if(a.kind==='bank'){
   slide.shapes.add({geometry:'can',name:`bank-${index++}`,position:{left:a.x,top:a.y,width:a.w,height:a.h},fill:'#ffffff',line:{fill:INK,width:1}});
   svg+=`<path d="M${a.x},${a.y+14} L${a.x},${a.y+a.h-14} C${a.x},${a.y+a.h+5} ${a.x+a.w},${a.y+a.h+5} ${a.x+a.w},${a.y+a.h-14} L${a.x+a.w},${a.y+14}" fill="white" stroke="${INK}"/><ellipse cx="${a.x+a.w/2}" cy="${a.y+14}" rx="${a.w/2}" ry="14" fill="white" stroke="${INK}"/>`;
  }else{
   if(a.kind==='tokens')for(let k=2;k>0;k--){nativeRect(a.x+k*5,a.y-k*5,a.w,a.h,'#ffffff','#777777');svg+=`<rect x="${a.x+k*5}" y="${a.y-k*5}" width="${a.w}" height="${a.h}" fill="white" stroke="#777777"/>`;}
   const round=a.kind==='module';nativeRect(a.x,a.y,a.w,a.h,'#ffffff',INK,round);svg+=`<rect x="${a.x}" y="${a.y}" width="${a.w}" height="${a.h}" rx="${round?9:0}" fill="white" stroke="${INK}" stroke-width="1"/>`;
  }
  renderText({...a,x:a.x+9,y:a.y+(a.kind==='bank'?23:8),w:a.w-18,h:a.h-(a.kind==='bank'?34:16),size:a.size,align:'center'});
  if(a.learn||a.frozen)renderMarker({x:a.x+a.w-11,y:a.y+(a.learn?11:1),kind:a.learn?'learn':'frozen'});
  continue;
 }
 if(a.type==='wire'){
  const pts=a.pts;
  for(let j=1;j<pts.length;j++){const [x,y]=pts[j-1],[u,v]=pts[j];slide.shapes.add({geometry:'connector',kind:'straight',name:`arrow-${index++}`,position:{left:Math.min(x,u),top:Math.min(y,v),width:Math.abs(u-x),height:Math.abs(v-y),horizontalFlip:u<x,verticalFlip:v<y},line:{fill:INK,width:1.3},tail:{type:j===pts.length-1?'triangle':'none',width:'sm',length:'sm'}});svg+=lineSvg(x,y,u,v,INK,1.3);}
  const [x,y]=pts.at(-1),[u,v]=pts.at(-2),d=Math.hypot(x-u,y-v),dx=(x-u)/d,dy=(y-v)/d;svg+=`<polygon points="${x},${y} ${x-7*dx+3*dy},${y-7*dy-3*dx} ${x-7*dx-3*dy},${y-7*dy+3*dx}" fill="${INK}"/>`;
 }
}
svg+='</svg>';
await fs.writeFile(path.join(out,'01_overview.svg'),svg);
await fs.writeFile(path.join(out,'01_overview.png'),new Uint8Array(await(await deck.export({slide,format:'png',scale:1.5})).arrayBuffer()));
await fs.writeFile(path.join(out,'diagram_layout.json'),JSON.stringify({width:W,height:H,items},null,2));
const candidate=path.join(work,'candidate.pptx');await(await PresentationFile.exportPptx(deck)).save(candidate);
const {finalizePresentation}=await import(pathToFileURL('/home/agent/.codex/skills/builtins/presentations/container_tools/artifact_tool_utils.mjs').href);
await finalizePresentation({workspaceDir:'/workspace',candidatePath:candidate,finalPath:path.join(out,'CSA_pipeline_editable.pptx'),pythonExecutable:process.env.CODEX_PRIMARY_RUNTIME_PYTHON,integrityValidatorPath:'/home/agent/.codex/skills/builtins/presentations/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:'/home/agent/.codex/skills/builtins/presentations/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu',`${W*9525},${H*9525}`,'--validate-bullet-geometry','--validate-heading-fit'],fontPolicy:{basis:'design',families:[FONT]},explicitTotalSlideCount:1,verifyArtifactToolImport:true,receiptPath:path.join(work,`validation-${Date.now()}.json`)});
const {spawnSync}=await import('node:child_process');
const pdf=spawnSync(process.env.CODEX_PRIMARY_RUNTIME_PYTHON,['-c',`import fitz,sys,pathlib
p=pathlib.Path(sys.argv[1]); s=fitz.open(str(p/'01_overview.svg'));d=fitz.open('pdf',s.convert_to_pdf());d.save(str(p/'CSA_pipeline_vector.pdf'))
`,out],{encoding:'utf8'});
if(pdf.status!==0)throw new Error(pdf.stderr);
console.log('Rendered one editable paper figure and matching PNG/SVG/vector PDF.');
