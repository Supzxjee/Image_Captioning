// Isolated DOM test: no user browser/localStorage or human labels are modified.
const fs=require('fs'),path=require('path'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(path.join(__dirname,'review','assistant_review.html'),'utf8');
const manifest=html.match(/<script id="manifest" type="application\/json">([\s\S]*?)<\/script>/)[1];
const proposals=html.match(/<script id="assistant-proposals" type="application\/json">([\s\S]*?)<\/script>/)[1];
const code=html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
class Element{constructor(){this.children=[];this.value='';this.textContent='';this.events={};}append(...e){this.children.push(...e);}replaceChildren(...e){this.children=e;}setAttribute(k,v){this[k]=v;}addEventListener(k,v){this.events[k]=v;}click(){this.clicked=true;}focus(){}}
const ids={};for(const id of ['manifest','assistant-proposals','notice','progress','jump','prev','next','counter','case','reviewer','export'])ids[id]=new Element();
ids.manifest.textContent=manifest;ids['assistant-proposals'].textContent=proposals;
const stored=new Map();let blob;
const context=vm.createContext({document:{getElementById:id=>ids[id],createElement:()=>new Element(),createElementNS:()=>new Element()},localStorage:{getItem:k=>stored.get(k)||null,setItem:(k,v)=>stored.set(k,v)},Blob,URL:{createObjectURL:b=>{blob=b;return 'fixture:blob'},revokeObjectURL:()=>{}},setTimeout:f=>f()});
vm.runInContext(code,context);
assert(ids.progress.textContent.includes('0/21'));assert.equal(ids.jump.children.length,21);
ids.export.onclick();assert(ids.notice.textContent.includes('Nhập tên'));
ids.reviewer.value='Synthetic reviewer';ids.reviewer.oninput();
async function run(){
 ids.export.onclick();let r=JSON.parse(await blob.text());assert.equal(r.reviewer_type,'assistant_proposed_pending_human_verification');assert.equal(Object.values(r.annotations).filter(a=>a.confirmed_by_human).length,0);
 const response=ids.case.children[ids.case.children.length-1],label=response.children[response.children.length-1],checkbox=label.children[0];
 checkbox.checked=true;checkbox.onchange();assert(ids.progress.textContent.includes('1/21'));
 // Editing a previously confirmed answer must revoke confirmation.
 const table=response.children[1],select=table.children[1].children[1].children[0];select.value='unclear';select.events.change();assert(ids.progress.textContent.includes('0/21'));
 vm.runInContext('Object.values(state.annotations).forEach(a=>a.confirmed_by_human=true);save();',context);
 ids.export.onclick();r=JSON.parse(await blob.text());assert.equal(r.reviewer_type,'human_self_reported');assert.equal(r.training_approved,false);assert(r.assistant_proposal_sha256);assert.equal(r.label_origin,'assistant_draft_then_explicit_human_confirmation');
 console.log('PASS: AI provenance, no auto human labels, per-case confirmation, edit revocation, verified export');
}
run().catch(e=>{console.error(e);process.exitCode=1});
