// Check browser-script syntax and annotation consistency rules without a browser.
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const html=fs.readFileSync(__dirname+'/panel_template.html','utf8');
const script=html.split('<script>')[1].split('</script>')[0];
new vm.Script(script);
const fields=script.slice(script.indexOf('const fields='),script.indexOf('const ids='));
const valid=script.slice(script.indexOf('function valid(a)'),script.indexOf('function save()'));
const ctx=vm.createContext({});vm.runInContext(fields+'\n'+valid+'\nthis.check=valid;this.finish=complete;',ctx);
let n=0;function test(a,want){assert.strictEqual(Boolean(ctx.check(a)),want);n++;}
test({},false);test(null,false);
for(const c of ['correct','wrong','uncertain'])test({q_claim:'yes',target_relation:'yes',relation_correctness:c},true);
test({q_claim:'no',target_relation:'no',relation_correctness:'not_applicable'},true);
test({q_claim:'no',target_relation:'yes',relation_correctness:'correct'},true);
test({q_claim:'uncertain',target_relation:'uncertain',relation_correctness:'uncertain'},true);
test({q_claim:'yes',target_relation:'no',relation_correctness:'not_applicable'},false);
test({q_claim:'no',target_relation:'no',relation_correctness:'correct'},false);
test({q_claim:'no',target_relation:'yes',relation_correctness:'not_applicable'},false);
test({q_claim:'no',target_relation:'uncertain',relation_correctness:'correct'},false);
test({q_claim:'bad',target_relation:'yes',relation_correctness:'correct'},false);
ctx.assisted=true;const sample={q_claim:'yes',target_relation:'yes',relation_correctness:'correct',confirmed_by_human:false};assert.strictEqual(ctx.finish(sample),false);sample.confirmed_by_human=true;assert.strictEqual(ctx.finish(sample),true);sample.q_claim='';assert.strictEqual(ctx.finish(sample),false);ctx.assisted=false;sample.q_claim='yes';sample.confirmed_by_human=false;assert.strictEqual(ctx.finish(sample),true);console.log('Panel script syntax PASS; '+n+' consistency cases and 4 confirmation-gating cases PASS. No browser UI execution.');
