"""Audit cached expanded-evaluation text and saved score/count vectors; no model."""
import hashlib,json,pathlib,subprocess
import numpy as np
from expanded_data import EVAL_POLICY,CHECKPOINTS,MANIFEST_SHA,rows_and_sentinels,pinned_metadata,validate_predictions
from expanded_metrics import paired_contrasts,gates
HERE=pathlib.Path(__file__).resolve().parent;ROOT=HERE.parents[1]
cache=json.loads((HERE/'review/expanded_loaded.json').read_text(encoding='utf-8'));text=cache['text']
def get(n):return json.loads(text[n])
s=get('summary.json');cfg=get('protocol.json');meta=get('metadata.json');prep=get('preparation.json');checks=[]
def check(v,label):
 assert v,label
 checks.append(label)
check(get('execution_status.json')['success'] and s['status']=='EXPANDED_EVALUATION_COMPLETE','complete archive')
check(s['count']==4000 and s['training'] is False and s['test_used'] is False,'4000 no training/test')
check(cfg==get('notebook_config.json'),'locked config unchanged')
check(s['policy']==cfg['policy']==EVAL_POLICY,'locked evaluation policy')
check(s['source_assets']==cfg['source_assets'],'source declarations aligned')
for n,h in s['source_assets'].items():
 raw=subprocess.check_output(['git','show',s['source_commit']+':'+n],cwd=ROOT)
 check(hashlib.sha256(raw).hexdigest()==h,'pinned Git asset '+n)
 if n.endswith('.py'):check(text['source/'+n].encode()==raw,'archived Python source '+n)
check(s['checkpoint_sha256']==prep['checkpoint_sha256']=={n:i['sha256'] for n,i in CHECKPOINTS.items()},'all three fixed checkpoint hashes')
check(prep['status']=='PREPARATION_COMPLETE' and prep['images']==4000 and prep['sentinels']==16,'preparation size/complete')
check(s['manifest_sha256']==prep['manifest_sha256']==MANIFEST_SHA,'fixed LF manifest identity')
check(hashlib.sha256(text['metadata.json'].encode()).hexdigest()==prep['payload_sha256']['metadata.json'],'metadata payload bytes')
manifest=json.loads((HERE/'EXPANDED_EVAL_MANIFEST.json').read_bytes());old=pinned_metadata(ROOT/'experiments/semantic_grounding_method/data/gain_metadata.json.gz');rows,sentinels=rows_and_sentinels(manifest,old)
check(meta['primary_indices']==list(range(4000)) and meta['sentinel_indices']==list(range(4000,4016)),'primary/sentinel indices fixed')
check(len(meta['rows'])==4016,'stage4016rows')
for actual,want in zip(meta['rows'],rows+sentinels):
 check(all(actual[k]==v for k,v in want.items()) and len(actual['references'])==5,'frozen row identity '+str(want['coco_id']))
for rank in [0,1]:
 w=get(f'worker_rank{rank}.json');check(w['status']=='EVALUATION_INFERENCE_COMPLETE' and w['images']==2000 and w['training'] is False,'worker complete '+str(rank))
 for name in ['A','C0','C']:
  info=w['models'][name];check(info['model_digest']==CHECKPOINTS[name]['digest'] and info['sentinel_mismatches']==0,'native tensors and parity '+name+str(rank))
  expected=[meta['rows'][i] for i in meta['primary_indices'][rank::2]]
  validate_predictions(get(f'{name}_rank{rank}.json'),expected);check(True,'rank prediction coverage '+name+str(rank))
per=get('per_image_metrics.json');check(per['columns']==['CIDEr','CHAIRs_flag','hallucinated_mentions','object_mentions','recalled_unique','reference_unique','words'],'per-image vector schema')
vectors={};metrics={}
for name in ['A','C0','C']:
 records=validate_predictions(get(name+'_predictions.json'),rows)
 check(records==sorted(get(name+'_rank0.json')+get(name+'_rank1.json'),key=lambda r:r['image_id']),'merged predictions '+name)
 ch=get(name+'_chair_output.json');sentences={r['image_id']:r for r in ch['sentences']}
 check(len(ch['sentences'])==len(sentences)==4000 and set(sentences)=={r['coco_id'] for r in rows},'CHAIR coverage '+name)
 points=per['models'][name];check([(r['image_id'],r['coco_id']) for r in points]==[(r['image_id'],r['coco_id']) for r in records],'score alignment '+name)
 v=np.asarray([r['values'] for r in points],dtype=float);check(v.shape==(4000,7) and np.isfinite(v).all(),'finite per-image metrics '+name)
 for record,point in zip(records,points):
  sentence=sentences[record['coco_id']];gen=sentence['mscoco_generated_words'];gt=sentence['mscoco_gt_words'];false=sentence['mscoco_hallucinated_words']
  check(sentence['caption']==record['caption'],'CHAIR raw caption '+name+str(record['image_id']))
  expected=[int(bool(false)),len(false),len(gen),len(set(gen)&set(gt)),len(set(gt)),len(record['caption'].split())]
  check(point['values'][1:]==expected,'independent CHAIR counts/length '+name+str(record['image_id']))
 counts=dict(zip(['hallucinated_captions','hallucinated_mentions','object_mentions','recalled_unique','reference_unique'],map(int,v[:,1:6].sum(0))))
 check(counts==s['metrics'][name]['counts'],'aggregate counts '+name)
 recomputed=dict(CIDEr=float(v[:,0].mean()),CHAIRs=counts['hallucinated_captions']/4000,CHAIRi=counts['hallucinated_mentions']/counts['object_mentions'],Recall=counts['recalled_unique']/counts['reference_unique'],mean_words=float(v[:,6].mean()))
 for key,value in recomputed.items():check(abs(value-s['metrics'][name][key])<1e-12,'aggregate '+name+'/'+key)
 for key in ['CHAIRs','CHAIRi','Recall']:check(abs(recomputed[key]-ch['overall_metrics'][key])<1e-12,'official CHAIR summary '+name+'/'+key)
 vectors[name]=v;metrics[name]=s['metrics'][name]
print('Identity/count/source audit passed; recomputing locked paired statistics from saved vectors.',flush=True)
contrasts=paired_contrasts(vectors,draws=20000,seed=20261030)
for pair,values in contrasts.items():
 for key,value in values.items():
  reported=s['contrasts'][pair][key]
  if key=='McNemar':
   check(all(value[k]==reported[k] for k in ['worse','better','discordant']) and abs(value['p_two_sided']-reported['p_two_sided'])<1e-12,'exact paired counts '+pair)
  else:check(abs(value['delta']-reported['delta'])<1e-12 and np.allclose(value['paired95_ci'],reported['paired95_ci'],rtol=0,atol=1e-12),'reproduced CI '+pair+'/'+key)
check(gates(contrasts)==s['decision'],'locked gate decision')
result=dict(date='2026-10-30',status='AUDITED_EXPANDED_EVALUATION_COMPLETE',zip_sha256=cache['zip_sha256'],source_commit=s['source_commit'],checks_passed=len(checks),count=4000,metrics=metrics,contrasts=contrasts,decision=s['decision'],manifest_sha256=MANIFEST_SHA,checkpoint_sha256=s['checkpoint_sha256'],timing=dict(preparation_seconds=prep['seconds'],worker_seconds=[get(f'worker_rank{r}.json')['seconds'] for r in [0,1]]),interpretation='C has higher CIDEr and fewer hallucinations as point estimates, but neither preservation-specific benefit nor dual-goal success passes the prelocked gate. Stop current method claim; no tuning on these4000images.',limits=['One seed, historically reused validation, post-pilot evaluation; no independent test.','Official CIDEr/other caption metrics not rerun locally; saved per-image CIDEr aggregates and paired intervals reproduced.','No model/checkpoint/HDF5 loading or neural execution locally; binary identities are user-run receipts and source checks.'])
(HERE/'EXPANDED_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(checks_passed=len(checks),contrasts=contrasts,decision=s['decision']),indent=2))
