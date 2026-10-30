"""Official metrics and locked paired contrasts, CPU only; no neural imports."""
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
from expanded_data import *

def exact_mcnemar(x,y):
 x=np.asarray(x,dtype=bool);y=np.asarray(y,dtype=bool)
 worse=int(np.sum(x&~y));better=int(np.sum(~x&y));n=worse+better
 tail=math.fsum(math.exp(math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)-n*math.log(2)) for k in range(min(worse,better)+1)) if n else .5
 return dict(worse=worse,better=better,discordant=n,p_two_sided=min(1.,2*tail))

def quantities(v):
 """Columns CIDEr, false caption, false mention, mention, recall hit, GT, words."""
 return dict(CIDEr=v[...,0].mean(-1),CHAIRs=v[...,1].mean(-1),CHAIRi=v[...,2].sum(-1)/np.maximum(1,v[...,3].sum(-1)),Recall=v[...,4].sum(-1)/np.maximum(1,v[...,5].sum(-1)),mean_words=v[...,6].mean(-1))

def paired_contrasts(vectors,draws=20000,seed=20261030):
 pairs=[('C','C0'),('C','A'),('C0','A')];n=len(vectors['A'])
 if any(v.shape!=(n,7) or not np.isfinite(v).all() for v in vectors.values()):raise ValueError('Malformed per-image metric vectors')
 rng=np.random.default_rng(seed);samples={x+'-'+y:{k:[] for k in quantities(vectors[x])} for x,y in pairs}
 for first in range(0,draws,200):
  ix=rng.integers(0,n,size=(min(200,draws-first),n));q={k:quantities(v[ix]) for k,v in vectors.items()}
  for x,y in pairs:
   for metric in samples[x+'-'+y]:samples[x+'-'+y][metric].extend((q[x][metric]-q[y][metric]).tolist())
 result={}
 for x,y in pairs:
  key=x+'-'+y;qx,qy=quantities(vectors[x]),quantities(vectors[y])
  result[key]={k:dict(delta=float(qx[k]-qy[k]),paired95_ci=np.quantile(samples[key][k],[.025,.975]).tolist()) for k in qx}
  result[key]['McNemar']=exact_mcnemar(vectors[x][:,1],vectors[y][:,1])
 return result

def gates(contrasts):
 ca=contrasts['C-A'];cc=contrasts['C-C0']
 conditions=dict(cider_C_A=ca['CIDEr']['paired95_ci'][0]>0,
  hallucination_C_A=ca['CHAIRs']['paired95_ci'][1]<0 and ca['McNemar']['p_two_sided']<.05,
  hallucination_C_C0=cc['CHAIRs']['paired95_ci'][1]<0 and cc['McNemar']['p_two_sided']<.05,
  cider_cost_C_C0=cc['CIDEr']['paired95_ci'][0]>-EVAL_POLICY['cider_margin'],
  recall_C_A=ca['Recall']['paired95_ci'][0]>-EVAL_POLICY['recall_margin'],
  recall_C_C0=cc['Recall']['paired95_ci'][0]>-EVAL_POLICY['recall_margin'])
 return dict(conditions=conditions,preservation_relative=all(conditions[k] for k in ['hallucination_C_C0','cider_cost_C_C0','recall_C_C0']),
  dual_goal_vs_A=all(conditions[k] for k in ['cider_C_A','hallucination_C_A','recall_C_A']),joint_pass=all(conditions.values()))

def evaluate(cfg):
 from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
 from pycocoevalcap.cider.cider import Cider
 from pycocoevalcap.bleu.bleu import Bleu
 from pycocoevalcap.meteor.meteor import Meteor
 from pycocoevalcap.rouge.rouge import Rouge
 out=Path(cfg['output']);meta=json.loads((out/'metadata.json').read_bytes());rows=[meta['rows'][i] for i in meta['primary_indices']]
 if len(rows)!=4000 or sha(Path(cfg['chair'])/'chair.pkl')!=CHAIR_SHA:raise ValueError('Changed cohort/evaluator')
 workers=[json.loads((out/f'worker_rank{rank}.json').read_bytes()) for rank in [0,1]]
 for w in workers:
  if w['status']!='EVALUATION_INFERENCE_COMPLETE' or w['images']!=2000 or set(w['models'])!=set(CHECKPOINTS):raise ValueError('Incomplete workers')
  for name,v in w['models'].items():
   if v['model_digest']!=CHECKPOINTS[name]['digest'] or v['sentinel_mismatches']!=0:raise ValueError('Inference parity/digest failure')
 expected={r['image_id']:r for r in rows};ptb=PTBTokenizer();refs=ptb.tokenize({i:[dict(caption=s) for s in r['references']] for i,r in sorted(expected.items())})
 metrics={};vectors={};per_image={};meteor=Meteor()
 try:
  for name in EVAL_POLICY['models']:
   records=validate_predictions(sum([json.loads((out/f'{name}_rank{rank}.json').read_bytes()) for rank in [0,1]],[]),rows)
   save(out/f'{name}_predictions.json',records);hyp=ptb.tokenize({r['image_id']:[dict(caption=r['caption'])] for r in records})
   cider,per=Cider().compute_score(refs,hyp);bleu,_=Bleu(4).compute_score(refs,hyp);met,_=meteor.compute_score(refs,hyp);rouge,_=Rouge().compute_score(refs,hyp)
   save(out/f'{name}_chair_input.json',[dict(image_id=r['coco_id'],caption=r['caption']) for r in records])
   subprocess.run([sys.executable,'-u',str(Path(cfg['chair'])/'chair.py'),'--cap_file',str(out/f'{name}_chair_input.json'),'--cache',str(Path(cfg['chair'])/'chair.pkl'),'--save_path',str(out/f'{name}_chair_output.json')],cwd=cfg['chair'],check=True)
   ch=json.loads((out/f'{name}_chair_output.json').read_bytes());sentences=ch['sentences']
   if len(sentences)!=4000 or {s['image_id'] for s in sentences}!={r['coco_id'] for r in rows}:raise ValueError('CHAIR IDs changed')
   indexed={s['image_id']:s for s in sentences};v=[];per_image[name]=[]
   for r,score in zip(records,per):
    sentence=indexed[r['coco_id']];generated=sentence['mscoco_generated_words'];gt=sentence['mscoco_gt_words'];false=sentence['mscoco_hallucinated_words']
    values=[float(score),int(bool(false)),len(false),len(generated),len(set(generated)&set(gt)),len(set(gt)),len(r['caption'].split())]
    v.append(values);per_image[name].append(dict(image_id=r['image_id'],coco_id=r['coco_id'],values=values))
   vectors[name]=np.asarray(v,dtype='float64');q={k:float(v) for k,v in quantities(vectors[name]).items()}
   if abs(q['CIDEr']-cider)>1e-12 or any(abs(q[k]-ch['overall_metrics'][k])>1e-12 for k in ['CHAIRs','CHAIRi','Recall']):raise ValueError('Metric aggregation mismatch')
   metrics[name]=dict(q,Bleu_1=bleu[0],Bleu_2=bleu[1],Bleu_3=bleu[2],Bleu_4=bleu[3],METEOR=met,ROUGE_L=rouge,
    counts=dict(zip(['hallucinated_captions','hallucinated_mentions','object_mentions','recalled_unique','reference_unique'],map(int,vectors[name][:,1:6].sum(0)))))
 finally:
  # pycocoevalcap owns the METEOR process; release it when this phase exits.
  del meteor
 if not all([r['coco_id'] for r in per_image[name]]==[r['coco_id'] for r in per_image['A']] for name in per_image):raise ValueError('Paired score alignment differs')
 save(out/'per_image_metrics.json',dict(columns=['CIDEr','CHAIRs_flag','hallucinated_mentions','object_mentions','recalled_unique','reference_unique','words'],models=per_image))
 contrasts=paired_contrasts(vectors,EVAL_POLICY['bootstrap_draws'],EVAL_POLICY['bootstrap_seed']);decision=gates(contrasts)
 summary=dict(status='EXPANDED_EVALUATION_COMPLETE',count=4000,policy=EVAL_POLICY,metrics=metrics,contrasts=contrasts,decision=decision,
  checkpoint_sha256={n:i['sha256'] for n,i in CHECKPOINTS.items()},manifest_sha256=MANIFEST_SHA,source_commit=cfg['source_commit'],source_assets=cfg['source_assets'],training=False,test_used=False,
  limits=['Historically reused validation; one training seed; post-pilot evaluation.','Marginal image-bootstrap intervals are not simultaneous bands or seed robustness.','CIDEr uses fixed4000IDF; no pooled5000 report; practical margins selected after pilot, before4000outputs.','Evaluation does not establish novelty or physical grounding; no tuning on this cohort.'])
 save(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
