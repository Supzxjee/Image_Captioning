// Synthetic DOM checks only. In-memory confirmations never alter real review files/state.
const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path'),assert=require('node:assert/strict');
class Element{
  constructor(tag,text=''){this.tag=tag;this.children=[];this.textContent=text;this.value='';}
  append(...nodes){this.children.push(...nodes);}
  click(){}
}
async function check(filename,draftExpected){
  const html=fs.readFileSync(path.join(__dirname,'review',filename),'utf8');
  const data=html.match(/<script id="audit-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
  const draft=html.match(/<script id="draft-data" type="application\/json">([\s\S]*?)<\/script>/)[1];
  const code=html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
  const nodes=Object.fromEntries(['root','reviewer','export','progress','notice'].map(k=>[k,new Element(k)]));
  nodes['audit-data']=new Element('script',data);nodes['draft-data']=new Element('script',draft);
  const storage=new Map();let blob;
  const context=vm.createContext({document:{getElementById:k=>nodes[k],createElement:t=>new Element(t),createTextNode:t=>new Element('text',t)},
    localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)},
    Option:class extends Element{constructor(text,value){super('option',text);this.value=value;}},
    structuredClone,Blob,URL:{createObjectURL:b=>{blob=b;return 'blob:synthetic';},revokeObjectURL:()=>{}},setTimeout:()=>{}});
  vm.runInContext(code,context);
  const all=[];function walk(n){all.push(n);for(const c of n.children||[])walk(c);}walk(nodes.root);
  assert.equal(nodes.root.children.length,16);
  assert.equal(all.filter(n=>n.tag==='select').length,96);
  assert.equal(all.filter(n=>n.tag==='input'&&n.type==='checkbox').length,32);
  const state=vm.runInContext('state',context);
  assert.equal(Object.keys(state.annotations).length,32);
  assert(Object.values(state.annotations).every(a=>a.confirmed_by_human===false));
  if(!draftExpected)assert(Object.values(state.annotations).every(a=>a.assertion===''));
  nodes.export.onclick();let exported=JSON.parse(await blob.text());
  assert.notEqual(exported.reviewer_type,'human_self_reported');assert.equal(exported.training_approved,false);
  // Simulate a complete user flow in the fake DOM, without saving anything externally.
  nodes.reviewer.value='Synthetic UI test';nodes.reviewer.oninput();
  for(const a of Object.values(state.annotations))Object.assign(a,{subject_box:'clear',object_box:'clear',assertion:'uncertain',confirmed_by_human:true});
  nodes.export.onclick();exported=JSON.parse(await blob.text());
  assert.equal(exported.reviewer_type,'human_self_reported');assert.equal(exported.training_approved,false);
  const select=all.find(n=>n.tag==='select');select.value='incorrect';select.onchange();
  nodes.export.onclick();exported=JSON.parse(await blob.text());
  assert.notEqual(exported.reviewer_type,'human_self_reported');
}
(async()=>{await check('index.html',false);await check('assistant_review.html',true);console.log('Review UI synthetic checks PASS:32records,blank/draft separation,explicit confirmation,edit reset,training remains unapproved.');})().catch(e=>{console.error(e);process.exitCode=1;});
