"""Stored-output audit only; no models, scorers or inference."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
from saved_quality_data import check_rows,chair_counts
from prompt_scst_data import SOURCES,validate_state
from prompt_scst_rerank import fixed_rank
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
cache=ROOT.parent/'.review_cache/prompt_scst_full1'
checks=[];summaries={};vectors={}
def check(name,ok):
    if not ok:raise ValueError(name)
    checks.append(name)
for arm,version in [('CONTROL',180),('PROMPT',60)]:
    stored=json.loads((cache/(arm+'.json')).read_text());f=stored['files'];read=lambda n:json.loads(f[n])
    result=read('result.json');state=read('progress.json');validate_state(state)
    check(arm+'/complete',read('run_status.json')['complete'] and state['updates']==14161 and state['image_events']==113287 and state['validated']==['original','end'] and not result['test_used'])
    check(arm+'/source_weights',result['source']==SOURCES[arm])
    receipt=read('checkpoint_receipt.json');check(arm+'/receipt',receipt['updates']==14161 and receipt['image_events']==113287)
    for n,h in state['validation_files'].items():check(arm+'/'+n,hashlib.sha256(f[n].encode()).hexdigest()==h)
    src=read('source_receipt.json')
    for n,h in src['sha256'].items():check(arm+'/source/'+n,hashlib.sha256(subprocess.check_output(['git','show',src['commit']+':'+n],cwd=ROOT)).hexdigest()==h)
    inputs=read('rerank_inputs.json');input_map={r['coco_id']:r for r in inputs};scored={r['coco_id']:r for rank in (0,1) for r in map(json.loads,f['rerank_scores_rank'+str(rank)+'.jsonl'].splitlines())}
    refs=read('validation_references.json');ids={r['coco_id'] for r in refs};check(arm+'/refs',len(ids)==5000 and sum(len(r['references']) for r in refs)==25010)
    for rank in (0,1):check(arm+'/guard'+str(rank),read('preflight_rank'+str(rank)+'.json')['passed'])
    for phase in ('original','end'):
        for policy,folder in [('decoder_rank0','evaluations'),('fixed_rerank','reranked')]:
            d=folder+'/'+phase+'/';rows=read(d+'val_rank0.json')+read(d+'val_rank1.json');check_rows(rows,ids)
            point=result['metrics'][phase][policy];ch=read(d+'chair_output.json');m,c=chair_counts(ch,rows)
            check(arm+'/'+d+'counts',c==point['counts'] and all(abs(v-point[k])<1e-12 for k,v in m.items()))
            check(arm+'/'+d+'chairinput',read(d+'chair_input.json')==[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
            per={r['coco_id']:r['CIDEr'] for r in read(d+'cider_per_image.json')};check(arm+'/'+d+'cider',set(per)==ids and abs(np.mean(list(per.values()))-point['CIDEr'])<1e-12)
            flags={r['image_id']:bool(r['mscoco_hallucinated_words']) for r in ch['sentences']}
            vectors[arm,phase,policy]=(per,flags)
            if policy=='fixed_rerank':
                selected=read(d+'selected_ranks.json');beams=read('evaluations/'+phase+'/candidates_rank0.json')+read('evaluations/'+phase+'/candidates_rank1.json');lookup={r['coco_id']:r for r in beams}
                check(arm+'/'+d+'selector_replay',all(fixed_rank(lookup[i],scored[i]['phases'][phase]['clipscore'],input_map[i]['detected'])==selected[str(i)]==scored[i]['phases'][phase]['selected_rank'] for i in ids))
                check(arm+'/'+d+'membership',all(r['caption']==lookup[r['coco_id']]['candidates'][selected[str(r['coco_id'])]]['caption'] for r in rows))
    check(arm+'/original_replay',abs(result['metrics']['original']['decoder_rank0']['CIDEr']-SOURCES[arm]['cider'])<1e-12)
    summaries[arm]=dict(save_version=version,archive_sha256=stored['archive_sha256'],checkpoint_sha256=receipt['sha256'],metrics=result['metrics'],worker_seconds=read('training_complete.json')['seconds'])
# Conditional paired bootstrap on fixed corpus-IDF per-image scores, not training-seed uncertainty.
paired={}
for policy in ('decoder_rank0','fixed_rerank'):
    a,af=vectors['CONTROL','end',policy];b,bf=vectors['PROMPT','end',policy];ids=sorted(a);delta=np.array([b[i]-a[i] for i in ids]);flags=np.array([int(bf[i])-int(af[i]) for i in ids]);rng=np.random.default_rng(42)
    draws=np.array([(delta[ix].mean()*100,flags[ix].mean()*100) for ix in (rng.integers(0,5000,5000) for _ in range(2000))])
    paired[policy]=dict(CIDEr_points=float(delta.mean()*100),CHAIRs_pp=float(flags.mean()*100),CI95=np.quantile(draws,[.025,.975],axis=0).tolist())
report=dict(status='PROMPT_SCST_STORED_OUTPUT_REVIEW_COMPLETE',checks=len(checks),arms=summaries,paired_prompt_minus_control=paired,limits=['One seed/reused VALID, conditional fixed-IDF bootstrap.','C10/P9 initial XE histories unequal; text-only causal effect not isolated.','No author model/scorer execution; binary checkpoint bytes absent review ZIP.','Selector membership and all20000 fixed choices replayed; CLIP forwards and official metrics not independently rerun.'])
(HERE/'PROMPT_SCST_RESULT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print('PASS',len(checks));print(json.dumps(paired,indent=2))
