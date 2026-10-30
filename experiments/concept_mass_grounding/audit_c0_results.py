"""Review C0 against cached, previously audited C/G; metadata/NumPy only."""
import hashlib,json,pathlib,subprocess
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
raw={a:json.loads((HERE/'review'/f'{a}_loaded.json').read_text(encoding='utf-8')) for a in ['C','G','C0']}
def get(a,n):return json.loads(raw[a]['text'][n])
s={a:get(a,'summary.json') for a in raw};checks=[]
def check(condition,label):
 assert condition,label
 checks.append(label)
a='C0';summary=s[a]
check(get(a,'execution_status.json')['success'],'C0 execution success')
check(summary['objective_weights']==dict(kl=.1,keep=0.,negative=0.),'effective zero keep and negative')
for name,text in raw[a]['text'].items():
 if name.startswith('source/'):
  expected=subprocess.check_output(['git','show',summary['source_commit']+':'+name[7:]],cwd=ROOT)
  check(text.encode()==expected,'source '+name)
  check(hashlib.sha256(expected).hexdigest()==summary['source_assets'][name[7:]],'declared source hash '+name)
allowed={'concept_data.py','prepare_concept.py','run_concept.py','evaluate_concept.py','test_concept_data.py'}
diff={pathlib.PurePosixPath(n).name for n in raw[a]['text'] if n.startswith('source/') and raw[a]['text'][n]!=raw['C']['text'][n]}
check(diff==allowed,'only predeclared source files changed versus C')
for name in ['metadata.json','object_targets.json','A_predictions.json','A_per_image_cider.json']:
 check(get(a,name)==get('C',name),'matched '+name)
for key in ['policy','initial_digest','frozen_digest','tensor_content_sha256','preparation_payload_sha256']:
 check(summary[key]==s['C'][key],'matched '+key)
for rank in [0,1]:
 w=get(a,f'worker_rank{rank}.json');f=get(a,f'preflight_rank{rank}.json')
 check(w['status']=='ENDPOINT_500_COMPLETE' and w['updates']==500,f'complete rank{rank}')
 check(w['initial_frozen_digest']==w['final_frozen_digest']==s['C']['frozen_digest'],f'frozen rank{rank}')
 check(w['initial_digest']==summary['initial_digest'] and w['final_digest']==summary['final_digest'],f'states rank{rank}')
 check(f['status']=='PREFLIGHT_PASS' and w['historical_A_caption_mismatches']==0,f'preflight and A rank{rank}')
 check(get(a,f'exposure_rank{rank}.json')==get('C',f'exposure_rank{rank}.json'),f'exposures rank{rank}')
 history=[json.loads(t) for t in raw[a]['text'][f'history_rank{rank}.jsonl'].splitlines()]
 check(history[-1]['step']==500,f'endpoint history rank{rank}')
 for row in history:
  check(row['objective_weights']==summary['objective_weights'],'logged weights')
  check(abs(row['loss']-(row['scst']+.1*row['kl']))<1e-5,'C0 loss formula')
  check(all(row['weighted_aux_gradient_norms_local'][n]==0 for n in ['keep','negative']),'zero weighted aux gradient in logged snapshots')
  check(np.isfinite([row[k] for k in ['loss','scst','kl','keep','negative','grad_norm']]).all(),'finite history')
ids=get(a,'exposure_rank0.json')+get(a,'exposure_rank1.json');check(len(ids)==len(set(ids))==4000,'4000 unique train exposures')
models={'A':('C0','A'),'C0':('C0','end'),'C':('C','end'),'G':('G','end')};vectors={};metrics={}
ids=sorted(x['coco_id'] for x in get('C0','A_per_image_cider.json'));check(len(ids)==len(set(ids))==1000,'1000 unique dev IDs')
for name,(arm,tag) in models.items():
 scores={x['coco_id']:x['CIDEr'] for x in get(arm,tag+'_per_image_cider.json')}
 chair={x['image_id']:x for x in get(arm,tag+'_chair_output.json')['sentences']}
 check(set(scores)==set(chair)==set(ids),name+' metric IDs')
 vectors[name]={'CIDEr':np.array([scores[i] for i in ids]),'CHAIRs':np.array([bool(chair[i]['mscoco_hallucinated_words']) for i in ids],dtype=float)}
 metrics[name]=s[arm]['metrics'][tag]
 check(abs(vectors[name]['CIDEr'].mean()-metrics[name]['CIDEr'])<1e-12,name+' mean CIDEr')
 check(int(vectors[name]['CHAIRs'].sum())==metrics[name]['counts']['hallucinated_captions'],name+' hallucinated captions')
 check(sum(len(v['mscoco_hallucinated_words']) for v in chair.values())==metrics[name]['counts']['hallucinated_mentions'],name+' hallucinated mentions')
rng=np.random.default_rng(20261029);boot=rng.integers(0,1000,size=(10000,1000));comparisons={}
for x,y in [('C0','A'),('C','C0'),('G','C0')]:
 comparisons[x+'-'+y]={}
 for metric in ['CIDEr','CHAIRs']:
  d=vectors[x][metric]-vectors[y][metric]
  comparisons[x+'-'+y][metric]=dict(delta=float(d.mean()),paired95_ci=np.quantile(d[boot].mean(1),[.025,.975]).tolist())
conditioning={}
for arm in raw:
 diag=get(arm,'class_conditioning.json');rows=list(diag['per_class'].values())
 conditioning[arm]=dict(classes=len(rows),separation_A=diag['macro_A_separation'],separation_end=diag['macro_end_separation'],supported_A=float(np.mean([v['supported']['A_mass'] for v in rows])),supported_end=float(np.mean([v['supported']['end_mass'] for v in rows])),weak_negative_A=float(np.mean([v['weak_negative']['A_mass'] for v in rows])),weak_negative_end=float(np.mean([v['weak_negative']['end_mass'] for v in rows])))
result=dict(date='2026-10-30',checks_passed=len(checks),zip_sha256={a:raw[a]['zip_sha256'] for a in raw},metrics=metrics,comparisons=comparisons,conditioning=conditioning,source_commits={a:s[a]['source_commit'] for a in raw},C0_checkpoint_sha256=summary['checkpoint_sha256'],decision='C0 retains the CIDEr gain without positive preservation; current pilot does not establish a separate benefit of keep or negative objectives. No hallucination reduction versus A demonstrated.',limitations=['One seed and reused dev1000; post-pilot C0; image bootstrap is not seed robustness.','No model/checkpoint/HDF5 execution or loading locally; reviewed text reports and pinned sources only.','C versus C0 interval including zero does not prove equivalence or preservation is universally harmful.'])
(HERE/'C0_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['metrics','zip_sha256']},indent=2))
