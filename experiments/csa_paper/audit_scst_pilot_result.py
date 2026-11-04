"""Audit saved review text and paired NumPy intervals; no Torch/scorer/ZIP reread."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import numpy as np
from saved_quality_data import chair_counts,check_rows

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def run(updates=500):
    if updates not in (500,2500):raise ValueError('Only approved500/2500 study')
    prefix='scst_' if updates==500 else 'scst2500_'
    checks=0;arms={}
    def check(value):
        nonlocal checks
        if not value:raise AssertionError('Audit check failed:'+str(checks+1))
        checks+=1
    for arm,folder in [('CE',prefix+'ce_review_text'),('OBJECT',prefix+'object_review_text')]:
        root=HERE/folder;d={p.name:json.loads(p.read_text(encoding='utf-8')) for p in root.glob('*.json')}
        source=d['source_receipt.json'];identity=d['identity.json'];stage=d['pilot_stage.json'];result=d['result.json'];m=result['metrics']
        log=(root/'notebook.log').read_text(encoding='utf-8')
        check(d['saved_text_receipt.json']['zip_crc_pass'] and d['run_status.json']['complete'])
        check(result['status']=='CSA_SCST_PILOT_COMPLETE' and result['updates']==updates and result['image_events']==updates*8)
        check(not result['test_used'] and result['validation_images']==5000)
        check('Ran 5 tests' in log and 'Ran 3 tests' in log and 'SCST two-GPU DDP gradient scaling PASS' in log)
        if updates==2500:check('Ran 4 tests' in log)
        check('Traceback' not in log and 'FAILED' not in log)
        for name,h in source['sha256'].items():
            check(hashlib.sha256(subprocess.check_output(['git','show',source['commit']+':'+name],cwd=ROOT)).hexdigest()==h)
        check(identity['source']==source and identity['stage']==stage)
        check(identity['checkpoint_sha256']=='e02593ea079bdb563a1afe0c5898abdbf81297aa5e688b3b0df839cb820e5b25')
        check(identity['object_weight']==(0 if arm=='CE' else .5) and identity['scst_weight']==.05)
        check(identity['lr']==5e-6 and identity['global_batch']==8 and not identity['dropout'] and identity['seed']==42)
        check(identity['updates']==updates and d['run_status.json']['updates_requested']==updates)
        check(stage['train_images']==113287 and stage['pool_images']==10000 and stage['unique_exposed']==min(10000,updates*8))
        check(len(stage['pool_ids'])==len(set(stage['pool_ids']))==10000)
        ids=np.concatenate([np.random.default_rng(42+j).permutation(stage['pool_ids']) for j in range((updates*8+9999)//10000)])[:updates*8].reshape(updates,8)
        if updates==2500:
            check(stage['image_events']==20000 and stage['image_exposure_histogram']=={'2':10000})
            check(len(np.unique(ids))==10000 and (np.unique(ids,return_counts=True)[1]==2).all())
        buf=io.BytesIO();np.save(buf,ids)
        check(hashlib.sha256(buf.getvalue()).hexdigest()==stage['files_sha256']['coco_ids.npy'])
        for name in ('validation_references.json','vocabulary.json'):
            check(hashlib.sha256((root/name).read_bytes()).hexdigest()==stage['files_sha256'][name])
        refs=d['validation_references.json'];expected={r['coco_id'] for r in refs};check(len(refs)==len(expected)==5000)
        check(not expected & set(stage['pool_ids']))
        rows=d['val_rank0.json']+d['val_rank1.json'];check_rows(rows,expected)
        check(not {r['coco_id'] for r in d['val_rank0.json']} & {r['coco_id'] for r in d['val_rank1.json']})
        vocab=d['vocabulary.json']
        for r in rows:
            content=r['ids'][1:];content=content[:content.index(2)] if 2 in content else content
            if r['caption']!=' '.join(vocab[i] for i in content if i>1):raise ValueError('Token-caption mismatch')
        check(True)
        metric,counts=chair_counts(d['chair_output.json'],rows);check(counts==m['counts'])
        for k,v in metric.items():check(abs(v-m[k])<1e-12)
        lengths=[len(r['caption'].split()) for r in rows]
        check(abs(np.mean(lengths)-m['mean_words'])<1e-12 and sum(n<=2 for n in lengths)==m['short_le2'])
        workers=[d[f'worker{r}.json'] for r in (0,1)]
        check(all(w['complete'] and w['updates']==updates and w['image_events']==updates*8 and w['images']==2500 for w in workers))
        check(sum(w['nll_tokens'] for w in workers)==286871)
        check(abs(sum(w['nll_sum'] for w in workers)/286871-m['NLL'])<1e-12)
        check(d['progress.json']=={'updates':updates,'image_events':updates*8})
        check(d['checkpoint_receipt.json']['updates']==updates and d['checkpoint_receipt.json']['arm']==result['arm'])
        check(d['metric_preflight.json']['passed'])
        for rank in (0,1):check(d[f'policy_replay_rank{rank}.json']['passed'] and d[f'policy_replay_rank{rank}.json']['max_abs']<=2e-5)
        training=[json.loads(s) for s in (root/'training.jsonl').read_text(encoding='utf-8').splitlines()]
        check([r['updates'] for r in training]==list(range(25,updates+1,25)))
        check(all(r['image_events']==r['updates']*8 and all(np.isfinite(v) for v in r.values()) for r in training))
        cider={r['coco_id']:r['CIDEr'] for r in d['cider_per_image.json']}
        check(len(cider)==5000 and set(cider)==expected and abs(np.mean(list(cider.values()))-m['CIDEr'])<1e-12)
        sentences={r['image_id']:r for r in d['chair_output.json']['sentences']}
        arms[arm]=dict(data=d,rows={r['coco_id']:r for r in rows},sentences=sentences,cider=cider)
    a,b=arms['CE'],arms['OBJECT'];ia=dict(a['data']['identity.json']);ib=dict(b['data']['identity.json'])
    for k in ('arm','object_weight'):ia.pop(k);ib.pop(k)
    check(ia==ib)
    check(a['data']['gpu_environment.json']==b['data']['gpu_environment.json'])
    old=json.loads((HERE/'review/xe15_loaded.json').read_text(encoding='utf-8'))['entries']
    check(json.loads(old['validation_references.json'])==a['data']['validation_references.json'])
    old_rows=json.loads(old['val_epoch15_rank0.json'])+json.loads(old['val_epoch15_rank1.json'])
    basechair=json.loads(old['quality_epoch15.chair_output.json']);basepoint,counts=chair_counts(basechair,old_rows)
    base=json.loads(old['quality_result.json'])['metrics']['15']
    check(base['counts']==counts)
    for k,v in basepoint.items():check(abs(base[k]-v)<1e-12)
    base['NLL']=json.loads(old['metrics_epoch15.json'])['val_nll']
    arms['CSA15']=dict(rows={r['coco_id']:r for r in old_rows},sentences={r['image_id']:r for r in basechair['sentences']})
    historical500=None
    if updates==2500:
        historical500=json.loads((HERE/'SCST_PILOT_RESULT.json').read_text(encoding='utf-8'))
        for arm in ('CE','OBJECT'):
            folder=HERE/('scst_'+arm.lower()+'_review_text')
            refs=json.loads((folder/'validation_references.json').read_text(encoding='utf-8'))
            check(refs==a['data']['validation_references.json'])
            oldrows=json.loads((folder/'val_rank0.json').read_text(encoding='utf-8'))+json.loads((folder/'val_rank1.json').read_text(encoding='utf-8'))
            oldchair=json.loads((folder/'chair_output.json').read_text(encoding='utf-8'))
            arms[arm+'500']=dict(rows={r['coco_id']:r for r in oldrows},sentences={r['image_id']:r for r in oldchair['sentences']},
                cider={r['coco_id']:r['CIDEr'] for r in json.loads((folder/'cider_per_image.json').read_text(encoding='utf-8'))})
    keys=sorted(a['rows']);comparisons={}
    def stats(arm):
        result=[]
        for i in keys:
            s=arms[arm]['sentences'][i];words=s['mscoco_generated_words'];gt=set(s['mscoco_gt_words']);bad=s['mscoco_hallucinated_words']
            result.append([arms[arm].get('cider',{}).get(i,0),int(bool(bad)),len(bad),len(words),len(set(words)&gt),len(gt)])
        return np.asarray(result,np.float64)
    def measure(v):return np.stack((v[...,0]/5000,v[...,1]/5000,v[...,2]/np.maximum(1,v[...,3]),v[...,4]/np.maximum(1,v[...,5])),axis=-1)
    allstats={k:stats(k) for k in arms}
    pairs=[('CE','OBJECT'),('CSA15','CE'),('CSA15','OBJECT')]+([('CE500','CE'),('OBJECT500','OBJECT')] if updates==2500 else [])
    for first,last in pairs:
        x,y=allstats[first],allstats[last];check(np.array_equal(x[:,5],y[:,5]))
        rng=np.random.default_rng(20261104);draws=[]
        for _ in range(100):
            ix=rng.integers(0,5000,size=(100,5000));draws.append(measure(y[ix].sum(1))-measure(x[ix].sum(1)))
        ci=np.quantile(np.concatenate(draws),[.025,.975],axis=0)
        intervals={n:ci[:,j].tolist() for j,n in enumerate(('CIDEr','CHAIRs','CHAIRi','Recall')) if first!='CSA15' or n!='CIDEr'}
        repaired=[i for j,i in enumerate(keys) if x[j,1] and not y[j,1]]
        introduced=[i for j,i in enumerate(keys) if y[j,1] and not x[j,1]]
        examples=[]
        for kind,pop in [('repaired',repaired),('introduced',introduced)]:
            for i in pop[:5]:examples.append(dict(kind=kind,coco_id=i,before=arms[first]['rows'][i]['caption'],after=arms[last]['rows'][i]['caption']))
        comparisons[first+'->'+last]=dict(paired95CI=intervals,
            changed=sum(arms[first]['rows'][i]['caption']!=arms[last]['rows'][i]['caption'] for i in keys),
            repairs=len(repaired),new_hallucinations=len(introduced),examples=examples)
    result=dict(status='CSA_SCST_PILOT_REVIEW_COMPLETE',updates=updates,metadata_checks=checks,
        zip_sha256={k:arms[k]['data']['saved_text_receipt.json']['zip_sha256'] for k in ('CE','OBJECT')},
        metrics=dict(CSA15=base,**{k:arms[k]['data']['result.json']['metrics'] for k in ('CE','OBJECT')}),
        comparisons=comparisons,checkpoint_sha256={k:arms[k]['data']['checkpoint_receipt.json']['sha256'] for k in ('CE','OBJECT')},
        worker_seconds={k:[arms[k]['data'][f'worker{r}.json']['seconds'] for r in (0,1)] for k in ('CE','OBJECT')},
        bootstrap=dict(resamples=10000,seed=20261104,unit='image',ratio_metrics_recomputed=True),
        local_model_execution=False,test_used=False,
        historical500=historical500['metrics'] if historical500 else None,
        limitations='One seed, fixed short pilot, reused VALID; label/stage/checkpoint binaries absent from ZIP. Stored source/counts checked, no independent model/scorer replay. Baseline CIDEr per-image unavailable: no baseline CIDEr CI. CHAIR object-level only.')
    target='SCST_PILOT_RESULT.json' if updates==500 else 'SCST_2500_RESULT.json'
    (HERE/target).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('comparisons',)},indent=2))
    print(json.dumps({k:{n:v for n,v in c.items() if n!='examples'} for k,c in comparisons.items()},indent=2))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--updates',type=int,choices=(500,2500),default=500)
    run(p.parse_args().updates)
