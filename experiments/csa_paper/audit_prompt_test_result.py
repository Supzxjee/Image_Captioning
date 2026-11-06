"""Stored TEST metadata/count/selector audit; no models or official scorers."""
import json,hashlib,subprocess
from pathlib import Path
import numpy as np
from real_bank_data import read
from prompt_scst_rerank import fixed_rank
from prompt_test_data import donor_plan
from saved_quality_data import check_rows,chair_counts
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];cache=ROOT.parent/'.review_cache/prompt_TEST';checks=[];results={};vectors={};locks={}
def check(n,ok):
    if not ok:raise ValueError(n)
    checks.append(n)
prep=json.loads((ROOT.parent/'.review_cache/prompt_test_preparation/archive_text.json').read_text())['files'];promptrows=[json.loads(s) for s in prep['prompt_links.jsonl'].splitlines()];_,donors=donor_plan(promptrows)
for arm in ('CONTROL','PROMPT'):
    saved=json.loads((cache/(arm+'.json')).read_text());f=saved['files'];rd=lambda n:json.loads(f[n]);result=rd('result.json');lock=rd('test_lock.json');locks[arm]=lock
    check(arm+'/complete',result['status']=='PROMPT_TEST5000_COMPLETE' and rd('run_status.json')==dict(complete=True,arm=arm,training=False,test_tuning=False))
    check(arm+'/weights',result['checkpoint_sha256']==read(HERE/'PROMPT_SCST_RESULT.json')['arms'][arm]['checkpoint_sha256']==lock['checkpoint_sha256'])
    for name,key in [('inference_inputs.json','inference_sha256'),('test_references.json','references_sha256'),('donors.json','donors_sha256')]:check(arm+'/'+name,hashlib.sha256(f[name].encode()).hexdigest()==lock[key])
    check(arm+'/donorplan',rd('donors.json')==donors and all(r['coco_id']!=r['donor_coco_id'] for r in donors))
    check(arm+'/cache',lock['cache_files']==read(HERE/'PROMPT_TEST_PREPARATION_RESULT.json')['transport_inventory']['cache_files'])
    src=rd('source_receipt.json')
    for name,digest in src['sha256'].items():check(arm+'/source/'+name,hashlib.sha256(subprocess.check_output(['git','show',src['commit']+':'+name],cwd=ROOT)).hexdigest()==digest)
    refs=rd('test_references.json');ids={r['coco_id'] for r in refs};check(arm+'/ids',len(ids)==5000 and ids=={r['coco_id'] for r in promptrows})
    inputs=rd('inference_inputs.json');inputmap={r['coco_id']:r for r in inputs};scored={}
    for rank in (0,1):
        name=f'rerank_rank{rank}.jsonl';done=rd(f'rerank_worker{rank}.json');part=[json.loads(s) for s in f[name].splitlines()]
        check(arm+'/rerank'+str(rank),done['complete'] and done['images']==2500 and hashlib.sha256(f[name].encode()).hexdigest()==done['sha256'] and [r['coco_id'] for r in part]==[r['coco_id'] for r in inputs[rank::2]])
        scored.update({r['coco_id']:r for r in part})
    beams={}
    for mode in lock['modes']:
        rows=[]
        for rank in (0,1):
            name=f'{mode}_beams_rank{rank}.jsonl';done=rd(f'{mode}_decode_rank{rank}.json');part=[json.loads(s) for s in f[name].splitlines()]
            check(arm+'/'+mode+'/rank'+str(rank),done['complete'] and done['images']==2500 and done['optimizer_steps']==0 and done['checkpoint_sha256']==result['checkpoint_sha256'] and hashlib.sha256(f[name].encode()).hexdigest()==done['sha256'])
            rows.extend(part)
        check(arm+'/'+mode+'/beams',len(rows)==5000 and {r['coco_id'] for r in rows}==ids and all(len(r['candidates'])==5 for r in rows));beams[mode]={r['coco_id']:r for r in rows}
    for policy,point in result['metrics'].items():
        rows=rd(policy+'_captions.json');check_rows(rows,ids);ch=rd(policy+'_chair_output.json');m,c=chair_counts(ch,rows)
        check(arm+'/'+policy+'/counts',c==point['counts'] and all(abs(v-point[k])<1e-12 for k,v in m.items()))
        check(arm+'/'+policy+'/input',rd(policy+'_chair_input.json')==[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
        per={r['coco_id']:r['CIDEr'] for r in rd(policy+'_per_image.json')};check(arm+'/'+policy+'/cider',set(per)==ids and abs(np.mean(list(per.values()))-point['CIDEr'])<1e-12)
        if policy=='fixed_rerank':
            check(arm+'/selector',all(fixed_rank(beams['normal'][i],scored[i]['clipscore'],inputmap[i]['detected'])==scored[i]['selected_rank'] for i in ids))
            check(arm+'/selectedmembership',all(r['caption']==beams['normal'][r['coco_id']]['candidates'][scored[r['coco_id']]['selected_rank']]['caption'] for r in rows))
        else:check(arm+'/'+policy+'/rank0membership',all(r['caption']==beams[policy][r['coco_id']]['candidates'][0]['caption'] for r in rows))
        flags={r['image_id']:int(bool(r['mscoco_hallucinated_words'])) for r in ch['sentences']};vectors[arm,policy]=(per,flags)
    results[arm]=dict(archive_sha256=saved['archive_sha256'],metrics=result['metrics'],checkpoint_sha256=result['checkpoint_sha256'])
check('common TEST references/cache/donors',all(locks['CONTROL'][k]==locks['PROMPT'][k] for k in ('references_sha256','cache_files','donors_sha256')))
comparisons={}
for label,left,right in [('PROMPT_minus_CONTROL_raw',('CONTROL','normal'),('PROMPT','normal')),('PROMPT_minus_CONTROL_rerank',('CONTROL','fixed_rerank'),('PROMPT','fixed_rerank')),('normal_minus_zero',('PROMPT','zero'),('PROMPT','normal')),('normal_minus_shuffle',('PROMPT','shuffle'),('PROMPT','normal'))]:
    a,af=vectors[left];b,bf=vectors[right];ids=sorted(a);delta=np.array([b[i]-a[i] for i in ids]);flag=np.array([bf[i]-af[i] for i in ids]);rng=np.random.default_rng(42)
    draws=np.array([(delta[ix].mean()*100,flag[ix].mean()*100) for ix in (rng.integers(0,5000,5000) for _ in range(2000))])
    comparisons[label]=dict(CIDEr_points=float(delta.mean()*100),CHAIRs_pp=float(flag.mean()*100),CI95=np.quantile(draws,[.025,.975],axis=0).tolist())
report=dict(status='PROMPT_TEST_STORED_REVIEW_COMPLETE',checks=len(checks),arms=results,comparisons=comparisons,donor_slots=len(donors),unchanged_donor_texts=sum(r['unchanged_text'] for r in donors),limits=['One seed, fixed-IDF conditional bootstrap, prior TEST exposure.','C10/P9 initial XE training differs; zero/shuffle are OOD sensitivity, not a matched training ablation.','No author model/scorer/CLIP execution; raw image and binary checkpoint identity checks remain user-run evidence.'])
(HERE/'PROMPT_TEST_RESULT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print('PASS',len(checks));print(json.dumps(comparisons,indent=2))
