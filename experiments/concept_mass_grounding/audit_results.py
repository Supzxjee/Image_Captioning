"""Metadata-only review of cached C/G ZIP text. No model execution."""
import json,hashlib,pathlib,subprocess
import numpy as np
HERE=pathlib.Path(__file__).resolve().parent
ROOT=HERE.parents[1]
def load(a):return json.loads((HERE/'review'/f'{a}_loaded.json').read_text(encoding='utf-8'))
raw={a:load(a) for a in ['C','G']}
def get(a,n):return json.loads(raw[a]['text'][n])
s={a:get(a,'summary.json') for a in raw}
checks=[]
def check(x,n):
 assert x,n
 checks.append(n)
for a in raw:
 check(get(a,'execution_status.json')['success'],a+' success')
 for n,t in raw[a]['text'].items():
  if n.startswith('source/'):
   rel=n[7:];expected=subprocess.check_output(['git','show',s[a]['source_commit']+':'+rel],cwd=ROOT)
   check(t.encode()==expected,a+' source '+rel)
 for r in range(2):
  w=get(a,f'worker_rank{r}.json');f=get(a,f'preflight_rank{r}.json')
  check(w['updates']==500 and w['status']=='ENDPOINT_500_COMPLETE',a+f' rank{r} complete')
  check(w['initial_frozen_digest']==w['final_frozen_digest'],a+f' rank{r} frozen')
  check(w['historical_A_caption_mismatches']==0 and f['status']=='PREFLIGHT_PASS',a+f' rank{r} preflight/A')
  hist=[json.loads(t) for t in raw[a]['text'][f'history_rank{r}.jsonl'].splitlines()]
  check(hist[-1]['step']==500,a+f' rank{r} logged endpoint')
  for h in hist:
   want=h['scst']+.1*h['kl']+.1*h['keep']+(.1*h['negative'] if a=='G' else 0)
   check(abs(h['loss']-want)<1e-5 and np.isfinite(list(h[k] for k in ['loss','scst','kl','keep','negative','grad_norm'])).all(),a+f' loss {r}/{h["step"]}')
 ids=get(a,'exposure_rank0.json')+get(a,'exposure_rank1.json')
 check(len(ids)==len(set(ids))==4000,a+' 4000 unique train exposures')
 check(s[a]['global_negative_events']>0,a+' negative events exist')
 for metric in ['A','end']:
  v=get(a,f'{metric}_per_image_cider.json');check(len(v)==1000 and len({x['coco_id'] for x in v})==1000,a+' '+metric+' 1000 dev IDs')
  check(abs(np.mean([x['CIDEr'] for x in v])-s[a]['metrics'][metric]['CIDEr'])<1e-12,a+' '+metric+' CIDEr reproduced')
for n in ['metadata.json','object_targets.json','A_predictions.json','A_per_image_cider.json']:
 check(get('C',n)==get('G',n),'matched '+n)
for k in ['source_commit','policy','initial_digest','frozen_digest','tensor_content_sha256','preparation_payload_sha256']:
 check(s['C'][k]==s['G'][k],'matched '+k)
for r in range(2):check(get('C',f'exposure_rank{r}.json')==get('G',f'exposure_rank{r}.json'),'matched exposure rank'+str(r))
base=get('C','A_per_image_cider.json');ids=[x['coco_id'] for x in base]
models={'A':('C','A'),'C':('C','end'),'G':('G','end')}
vectors={}
for name,(arm,tag) in models.items():
 scores={x['coco_id']:x['CIDEr'] for x in get(arm,tag+'_per_image_cider.json')}
 chairs={x['image_id']:x['metrics']['CHAIRs'] for x in get(arm,tag+'_chair_output.json')['sentences']}
 check(set(scores)==set(chairs)==set(ids),name+' aligned scores/CHAIR')
 vectors[name]={'CIDEr':np.array([scores[i] for i in ids]),'CHAIRs':np.array([chairs[i] for i in ids])}
rng=np.random.default_rng(20261029)
boot=rng.integers(0,1000,size=(10000,1000))
comparisons={}
for x,y in [('C','A'),('G','A'),('G','C')]:
 comparisons[x+'-'+y]={}
 for metric in ['CIDEr','CHAIRs']:
  d=vectors[x][metric]-vectors[y][metric]
  comparisons[x+'-'+y][metric]={'delta':float(d.mean()),'paired95_ci':np.quantile(d[boot].mean(1),[.025,.975]).tolist()}
conditioning={}
for a in raw:
 d=get(a,'class_conditioning.json');p=list(d['per_class'].values())
 conditioning[a]={k:d[k] for k in d if k!='per_class'}
 conditioning[a].update(classes=len(p),macro_A_supported=float(np.mean([v['supported']['A_mass'] for v in p])),macro_end_supported=float(np.mean([v['supported']['end_mass'] for v in p])),macro_A_weak_negative=float(np.mean([v['weak_negative']['A_mass'] for v in p])),macro_end_weak_negative=float(np.mean([v['weak_negative']['end_mass'] for v in p])))
result={'date':'2026-10-29','zip_sha256':{a:raw[a]['zip_sha256'] for a in raw},'checks_passed':len(checks),'metrics':{n:s[a]['metrics'][t] for n,(a,t) in models.items()},'comparisons':comparisons,'conditioning':conditioning,'negative_events':{a:s[a]['global_negative_events'] for a in raw},'decision':'CIDEr gain versus A on reused dev; no demonstrated added benefit of G over C or hallucination reduction. Do not promote G as successful alignment contribution.','limitations':['One seed and reused dev1000; bootstrap is image-conditional, not seed robustness.','Weak annotation-conditioned negative labels are not verified absence.','Only review texts/hashes verified; checkpoint and HDF5 binaries not supplied or loaded.']}
(HERE/'RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['metrics','zip_sha256']},indent=2))
