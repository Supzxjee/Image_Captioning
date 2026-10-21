// Isolated DOM fixture: never modifies the user's browser or labels.
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const base=path.join(__dirname,'review','pair_source_blind');
const html=fs.readFileSync(path.join(base,'index.html'),'utf8');
const payload=html.match(/<script id="data" type="application\/json">([\s\S]*?)<\/script>/)[1];
const code=html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
const data=JSON.parse(payload),ids=data.tasks.flatMap(t=>t.candidates.map(c=>c.candidate_id));
assert.equal(data.tasks.length,16);assert.equal(new Set(ids).size,32);
assert(!payload.includes('FULLFRAME')&&!payload.includes('PAIRCROP'));
for(const task of data.tasks)assert(fs.existsSync(path.join(base,'images',task.event.filename)));
class Element{constructor(){this.children=[];this.value='';this.textContent='';}append(...e){this.children.push(...e);}setAttribute(k,v){this[k]=v;}click(){this.clicked=true;}}
const nodes={};for(const id of ['data','reviewer','progress','notice','cases','export'])nodes[id]=new Element();
nodes.data.textContent=payload;let blob;const storage=new Map();
const context=vm.createContext({document:{getElementById:id=>nodes[id],createElement:()=>new Element(),createElementNS:()=>new Element()},localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)},Blob,URL:{createObjectURL:b=>{blob=b;return 'fixture:blob'},revokeObjectURL:()=>{}},setTimeout:f=>f()});
vm.runInContext(code,context);
assert(nodes.progress.textContent.includes('0/32'));assert.equal(nodes.cases.children.length,16);
nodes.export.onclick();assert(nodes.notice.textContent.includes('Nhập tên'));
nodes.reviewer.value='Synthetic fixture';nodes.reviewer.oninput();nodes.export.onclick();assert(nodes.notice.textContent.includes('Chưa có'));
async function run(){
 vm.runInContext("state.annotations[ids[0]].status='correct_specific';save();",context);
 nodes.export.onclick();let result=JSON.parse(await blob.text());assert.equal(Object.keys(result.annotations).length,1);assert.equal(result.complete,false);
 vm.runInContext("for(const id of ids)state.annotations[id].status='unclear';save();",context);
 nodes.export.onclick();result=JSON.parse(await blob.text());assert.equal(result.complete,true);assert.equal(Object.keys(result.annotations).length,32);assert.equal(result.review_sha256,data.review_sha256);
 assert(Object.values(result.annotations).every(a=>a.status==='unclear'));
 vm.runInContext("state.annotations[ids[0]].status='invalid';save();",context);
 nodes.export.onclick();result=JSON.parse(await blob.text());assert.equal(result.complete,false);assert.equal(Object.keys(result.annotations).length,31);
 console.log('PASS: 16 image pairs, 32 blinded candidates, local assets, blank state, name/empty export guards, partial/complete export, unresolved labels and invalid-status filtering');
}
run().catch(e=>{console.error(e);process.exitCode=1;});
