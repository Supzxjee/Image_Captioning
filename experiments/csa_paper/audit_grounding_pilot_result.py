"""Audit cached review text and paired NumPy resamples; no model/scorer/ZIP reread."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from saved_quality_data import chair_counts,check_rows

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def run():
    checks=0;arms={}
    def check(condition):
        nonlocal checks
        if not condition:raise AssertionError('Audit failed at '+str(checks+1))
        checks+=1
    for arm in ('XE','GROUND'):
        bundle=json.loads((HERE/'review'/('grounding_pilot_'+arm+'_loaded.json')).read_text(encoding='utf-8'))
        text={Path(n).name:t for n,t in bundle['text'].items()};d={n:json.loads(t) for n,t in text.items() if n.endswith('.json')}
        result=d['result.json'];m=result['metrics'];p=d['progress.json'];identity=d['identity.json'];source=d['source_receipt.json']
        check(result['status']=='CSA_GROUNDING_PILOT_COMPLETE' and result['arm']==arm)
        check(result['updates']==2000 and result['caption_events']==64000 and result['validation_images']==5000)
        check(d['run_status.json']['complete'] and not result['test_used'] and not result['labels_human_verified'])
        check('failure.txt' not in text and 'DDP unequal/empty grounding denominators gradient parity PASS' in text['notebook.log'])
        for name,digest in source['sha256'].items():
            check(hashlib.sha256(subprocess.check_output(['git','show',source['commit']+':experiments/csa_paper/'+name],cwd=ROOT)).hexdigest()==digest)
        check(identity['source']==source and identity['source_checkpoint']=='272f843a21b3a63c61323f748ed916fa3ae0cf2777549061439c07b410006950')
        check(identity['weight']==(0 if arm=='XE' else .1) and identity['global_batch']==32 and identity['lr']==1e-5 and identity['seed']==42)
        receipt=d['checkpoint_receipt.json'];check(receipt['updates']==2000 and receipt['caption_events']==64000 and receipt['arm']==arm)
        refs=d['validation_references.json'];expected={r['coco_id'] for r in refs}
        check(len(refs)==len(expected)==5000)
        rows=d['val_rank0.json']+d['val_rank1.json'];check_rows(rows,expected)
        point,counts=chair_counts(d['chair_output.json'],rows)
        check(counts==m['counts'])
        for k,v in point.items():check(abs(v-m[k])<1e-12)
        lengths=[len(r['caption'].split()) for r in rows]
        check(abs(np.mean(lengths)-m['mean_words'])<1e-12 and sum(n<=2 for n in lengths)==m['short_le2'])
        workers=[d[f'worker{r}.json'] for r in (0,1)]
        check(all(w['complete'] and w['images']==2500 and w['updates']==2000 and w['arm']==arm for w in workers))
        check(sum(w['nll_tokens'] for w in workers)==286871)
        check(abs(sum(w['nll_sum'] for w in workers)/286871-m['NLL'])<1e-12)
        check(p['updates']==2000 and p['caption_events']==64000 and p['tokens']==734154 and p['ground_mentions']==67734)
        check(abs(p['nll_sum']/p['tokens']-m['train_NLL'])<1e-12 and abs(p['ground_sum']/p['ground_mentions']-m['train_ground'])<1e-12)
        check(d['instrumentation_parity.json']['passed'])
        cider={r['coco_id']:r['CIDEr'] for r in d['cider_per_image.json']}
        check(len(cider)==5000 and set(cider)==expected and abs(np.mean(list(cider.values()))-m['CIDEr'])<1e-12)
        sentences={r['image_id']:r for r in d['chair_output.json']['sentences']}
        # Per-image numerators/denominators, preserving ratio metrics under resampling.
        stats={}
        for i in sorted(expected):
            r=sentences[i];generated=r['mscoco_generated_words'];gt=r['mscoco_gt_words'];bad=r['mscoco_hallucinated_words']
            stats[i]=[cider[i],int(bool(bad)),len(bad),len(generated),len(set(generated)&set(gt)),len(set(gt))]
        arms[arm]=dict(bundle=bundle,text=text,data=d,stats=stats,rows={r['coco_id']:r for r in rows})
    a,b=arms['XE'],arms['GROUND'];ia=dict(a['data']['identity.json']);ib=dict(b['data']['identity.json'])
    for k in ('arm','weight'):ia.pop(k);ib.pop(k)
    check(ia==ib)
    for name in ('validation_references.json','vocabulary.json'):check(a['text'][name]==b['text'][name])
    check(a['data']['pilot_stage.json']==b['data']['pilot_stage.json'])
    ids=sorted(a['stats']);x=np.asarray([a['stats'][i] for i in ids],np.float64);y=np.asarray([b['stats'][i] for i in ids],np.float64)
    check(np.array_equal(x[:,5],y[:,5]))
    rng=np.random.default_rng(20261104);draws=[]
    def measures(v):
        return np.stack((v[...,0]/5000,v[...,1]/5000,v[...,2]/np.maximum(1,v[...,3]),v[...,4]/np.maximum(1,v[...,5])),axis=-1)
    for start in range(0,10000,100):
        indices=rng.integers(0,5000,size=(100,5000));draws.append(measures(y[indices].sum(1))-measures(x[indices].sum(1)))
    draws=np.concatenate(draws);names=['CIDEr','CHAIRs','CHAIRi','Recall'];ci={k:np.quantile(draws[:,j],[.025,.975]).tolist() for j,k in enumerate(names)}
    changed=[i for i in ids if a['rows'][i]['caption']!=b['rows'][i]['caption']]
    repaired=[i for j,i in enumerate(ids) if x[j,1] and not y[j,1]];introduced=[i for j,i in enumerate(ids) if y[j,1] and not x[j,1]]
    metrics_a=a['data']['result.json']['metrics'];metrics_b=b['data']['result.json']['metrics']
    delta={k:metrics_b[k]-v for k,v in metrics_a.items() if k!='counts'}
    gates=dict(CIDEr=delta['CIDEr']>=.01,CHAIRs=delta['CHAIRs']<=-.003,CHAIRi=delta['CHAIRi']<=0,
               Recall=delta['Recall']>=-.005,short_le2=(metrics_b['short_le2']-metrics_a['short_le2'])/5000<=.01)
    examples=[]
    for group,population in [('repaired',repaired),('introduced',introduced)]:
        for i in population[:10]:examples.append(dict(kind=group,coco_id=i,XE=a['rows'][i]['caption'],GROUND=b['rows'][i]['caption']))
    result=dict(status='MATCHED_PILOT_COMPLETE_BELOW_PREDECLARED_GAIN_GATES',checks=checks,
        zip_sha256={k:v['bundle']['zip_sha256'] for k,v in arms.items()},source_commit=a['data']['source_receipt.json']['commit'],
        metrics=dict(XE=metrics_a,GROUND=metrics_b),delta=delta,paired95CI=ci,bootstrap=dict(resamples=10000,seed=20261104,unit='image',ratio_metrics_recomputed=True),
        changed_captions=len(changed),repaired_captions=len(repaired),introduced_captions=len(introduced),screening_gates=gates,
        checkpoint_sha256={k:v['data']['checkpoint_receipt.json']['sha256'] for k,v in arms.items()},
        worker_seconds={k:[v['data'][f'worker{r}.json']['seconds'] for r in (0,1)] for k,v in arms.items()},examples=examples,
        test_used=False,labels_human_verified=False,local_model_execution=False,
        decision='Do not claim grounding gain or scale solely from positive point estimates. No automatic more updates/lambda sweep/TEST; discuss bounded next step.',
        limitations='One seed, reused VALID, weak labels. Full checkpoint/stage bytes absent ZIP; runtime source/receipts support identity, not local independent binary verification. CHAIR only object hallucination, not relation truth.')
    target=HERE/'GROUNDING_PILOT_RESULT.json';target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    out=HERE/'grounding_pilot_review_text';out.mkdir(exist_ok=True)
    for arm,v in arms.items():
        sub=out/arm;sub.mkdir(exist_ok=True)
        for name in ('source_receipt.json','run_status.json','identity.json','pilot_stage.json','progress.json','checkpoint_receipt.json','worker0.json','worker1.json','instrumentation_parity.json','result.json'):
            (sub/name).write_text(v['text'][name],encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('examples',)},indent=2))


if __name__=='__main__':run()
