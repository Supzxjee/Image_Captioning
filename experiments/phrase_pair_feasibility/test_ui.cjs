// Isolated DOM logic test; no browser, localStorage or human labels are modified.
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const file=path.join(__dirname,'review','index.html'),html=fs.readFileSync(file,'utf8');
const manifestText=html.match(/<script id="manifest" type="application\/json">([\s\S]*?)<\/script>/)[1];
const code=html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
class Element{constructor(){this.children=[];this.value='';this.textContent='';this.events={};}append(...e){this.children.push(...e);}replaceChildren(...e){this.children=e;}setAttribute(k,v){this[k]=v;}addEventListener(k,v){this.events[k]=v;}click(){this.clicked=true;}focus(){this.focused=true;}}
const ids={};for(const id of ['manifest','notice','progress','jump','prev','next','counter','case','reviewer','export'])ids[id]=new Element();
ids.manifest.textContent=manifestText;const stored=new Map();let blob;
const context=vm.createContext({document:{getElementById:id=>ids[id],createElement:()=>new Element(),createElementNS:()=>new Element()},
 localStorage:{getItem:k=>stored.get(k)||null,setItem:(k,v)=>stored.set(k,v)},Blob,URL:{createObjectURL:b=>{blob=b;return 'fixture:blob';},revokeObjectURL:()=>{}},setTimeout:f=>f()});
vm.runInContext(code,context);
assert.equal(ids.jump.children.length,30);assert(ids.counter.textContent.startsWith('1/30'));
ids.export.onclick();assert(ids.notice.textContent.includes('Nhập tên'));
ids.reviewer.value='Synthetic UI fixture';ids.reviewer.oninput();ids.export.onclick();assert(ids.notice.textContent.includes('Chưa có'));
ids.next.onclick();assert(ids.counter.textContent.startsWith('2/30'));ids.prev.onclick();assert(ids.counter.textContent.startsWith('1/30'));
vm.runInContext("const a=answer(groups[0]);a.truth=[['supported','contradicted'],['contradicted','supported']];a.boxes=[['clear','clear'],['clear','clear']];a.detail_224=['yes','yes'];save();",context);
assert(ids.progress.textContent.includes('1/30'));ids.export.onclick();assert(blob);
blob.text().then(text=>{const report=JSON.parse(text);assert.equal(report.reviewer,'Synthetic UI fixture');assert.equal(report.reviewer_type,'human_self_reported');assert.equal(Object.keys(report.annotations).length,1);assert.equal(report.groups_sha256,JSON.parse(manifestText).groups_sha256);console.log('PASS: startup, blank export guards, navigation, progress, isolated JSON export');});
