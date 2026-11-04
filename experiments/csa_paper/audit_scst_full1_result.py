"""Audit cached full1 review text/counts with NumPy only, no model/scorer/ZIP reread."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import numpy as np
from real_bank_data import read,sha
from saved_quality_data import chair_counts,check_rows
from scst_full_data import full_schedule,selected_name,validate_state,PHASE_STEPS,image_events

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def run():
    checks=0;arms={}
    def check(v):
        nonlocal checks
        if not v:raise ValueError('Audit check failed:'+str(checks+1))
        checks+=1
    for arm in ('CE','OBJECT'):
        folder=HERE/('scst_full1_'+arm.lower()+'_review_text')
        d={p.name:read(p) for p in folder.glob('*.json')};result=d['result.json'];stage=d['pilot_stage.json'];identity=d['identity.json']
        check(d['saved_text_receipt.json']['zip_crc_pass'] and d['run_status.json']['complete'])
        check(result['status']=='CSA_SCST_FULL1_COMPLETE' and result['updates']==14161 and result['image_events']==113287 and not result['test_used'])
        source=d['source_receipt.json']
        for name,h in source['sha256'].items():check(hashlib.sha256(subprocess.check_output(['git','show',source['commit']+':'+name],cwd=ROOT)).hexdigest()==h)
        check(identity['source']==source and identity['stage']==stage and identity['schema']=='csa15_scst_full1_v1')
        check(identity['object_weight']==(0 if arm=='CE' else .5) and identity['scst_weight']==.05 and identity['lr']==5e-6)
        check(identity['checkpoint_sha256']=='e02593ea079bdb563a1afe0c5898abdbf81297aa5e688b3b0df839cb820e5b25')
        check(identity['updates']==14161 and identity['global_batch']==8 and not identity['dropout'])
        check(stage['pool_images']==stage['train_images']==stage['unique_exposed']==113287)
        check(stage['last_batch_images']==7 and stage['image_exposure_histogram']=={'1':113287})
        _,ids=full_schedule(stage['pool_ids']);buf=io.BytesIO();np.save(buf,ids)
        check(hashlib.sha256(buf.getvalue()).hexdigest()==stage['files_sha256']['coco_ids.npy'])
        refs=d['validation_references.json'];expected={r['coco_id'] for r in refs}
        check(len(refs)==len(expected)==5000 and not expected & set(stage['pool_ids']))
        for n in ('validation_references.json','vocabulary.json'):check(sha(folder/n)==stage['files_sha256'][n])
        state=d['progress.json'];validate_state(state);check(state['updates']==14161 and set(state['validated'])==set(PHASE_STEPS))
        for name,h in state['validation_files'].items():check(sha(folder/name)==h)
        cr=d['checkpoint_receipt.json'];check(cr['updates']==14161 and cr['image_events']==113287 and sha(folder/'training.jsonl')==cr['training_log_sha256'])
        check(d['mid_receipt.json']['updates']==7081 and d['mid_receipt.json']['image_events']==56648)
        log=(folder/'notebook.log').read_text(encoding='utf-8')
        check('FULL1 uneven4/3 CEtoken+SCSTimage DDP gradient parity PASS' in log and 'SCST two-GPU DDP gradient scaling PASS' in log)
        check(all(f'Ran {n} tests' in log for n in (5,4,6,3)) and 'Traceback' not in log and 'FAILED' not in log)
        history=[json.loads(s) for s in (folder/'training.jsonl').read_text(encoding='utf-8').splitlines()]
        check([r['updates'] for r in history]==list(range(25,14162,25))+[14161])
        check(all(r['image_events']==image_events(r['updates']) for r in history) and history[-1]['image_events']==113287)
        check(all(all(np.isfinite(v) for v in r.values()) for r in history))
        for rank in (0,1):check(d[f'policy_replay_rank{rank}.json']['passed'])
        phases={}
        for name,step in PHASE_STEPS.items():
            root=folder/'evaluations'/name
            rows=read(root/'val_rank0.json')+read(root/'val_rank1.json');check_rows(rows,expected)
            check(not {r['coco_id'] for r in read(root/'val_rank0.json')} & {r['coco_id'] for r in read(root/'val_rank1.json')})
            metric,counts=chair_counts(read(root/'chair_output.json'),rows);m=result['metrics'][name]
            check(counts==m['counts'])
            for k,v in metric.items():check(abs(v-m[k])<1e-12)
            workers=[read(root/f'worker{r}.json') for r in (0,1)]
            check(all(w['complete'] and w['updates']==step and w['image_events']==image_events(step) and w['images']==2500 for w in workers))
            check(sum(w['nll_tokens'] for w in workers)==286871 and abs(sum(w['nll_sum'] for w in workers)/286871-m['NLL'])<1e-12)
            lengths=[len(r['caption'].split()) for r in rows]
            check(abs(np.mean(lengths)-m['mean_words'])<1e-12 and sum(n<=2 for n in lengths)==m['short_le2'])
            for row in rows:
                content=row['ids'][1:];content=content[:content.index(2)] if 2 in content else content
                if row['caption']!=' '.join(d['vocabulary.json'][i] for i in content if i>1):raise ValueError('Caption/token mismatch')
            check(True)
            cider={r['coco_id']:r['CIDEr'] for r in read(root/'cider_per_image.json')}
            check(len(cider)==5000 and set(cider)==expected and abs(np.mean(list(cider.values()))-m['CIDEr'])<1e-12)
            check(read(root/'result.json')['metrics']==m and sha(root/'result.json')==d['locked_selection.json']['metrics_sha256'][name])
            sentences={r['image_id']:r for r in read(root/'chair_output.json')['sentences']}
            stats=[]
            for i in sorted(expected):
                s=sentences[i];words=s['mscoco_generated_words'];gt=set(s['mscoco_gt_words']);bad=s['mscoco_hallucinated_words']
                stats.append([cider[i],int(bool(bad)),len(bad),len(words),len(set(words)&gt),len(gt)])
            phases[name]=dict(stats=np.asarray(stats),rows={r['coco_id']:r['caption'] for r in rows},workers=workers)
        selected,eligible=selected_name(result['metrics'])
        check(d['locked_selection.json']==result['selection'] and selected=='end' and eligible==result['selection']['eligible_names'])
        check(cr['sha256']==d['best_receipt.json']['sha256']==result['selection']['checkpoint_sha256'])
        arms[arm]=dict(data=d,phases=phases)
    a,b=arms['CE'],arms['OBJECT'];ia=dict(a['data']['identity.json']);ib=dict(b['data']['identity.json'])
    for k in ('arm','object_weight'):ia.pop(k);ib.pop(k)
    check(ia==ib)
    check(a['data']['gpu_environment.json']==b['data']['gpu_environment.json'])
    check(a['phases']['original']['rows']==b['phases']['original']['rows'])
    check(a['data']['result.json']['metrics']['original']==b['data']['result.json']['metrics']['original'])
    comparisons={}
    pairs=[('CE','end','OBJECT','end'),('CE','original','CE','end'),('OBJECT','original','OBJECT','end'),('CE','mid','CE','end'),('OBJECT','mid','OBJECT','end')]
    def measure(v):return np.stack((v[...,0]/5000,v[...,1]/5000,v[...,2]/np.maximum(1,v[...,3]),v[...,4]/np.maximum(1,v[...,5])),axis=-1)
    for first,fp,last,lp in pairs:
        x,y=arms[first]['phases'][fp]['stats'],arms[last]['phases'][lp]['stats'];check(np.array_equal(x[:,5],y[:,5]))
        rng=np.random.default_rng(20261105);draws=[]
        for _ in range(100):
            ix=rng.integers(0,5000,(100,5000));draws.append(measure(y[ix].sum(1))-measure(x[ix].sum(1)))
        ci=np.quantile(np.concatenate(draws),[.025,.975],axis=0)
        repaired=int(((x[:,1]>0)&(y[:,1]==0)).sum());introduced=int(((x[:,1]==0)&(y[:,1]>0)).sum())
        comparisons[f'{first}_{fp}->{last}_{lp}']=dict(delta=dict(zip(('CIDEr','CHAIRs','CHAIRi','Recall'),(measure(y.sum(0))-measure(x.sum(0))).tolist())),
            paired95CI={n:ci[:,j].tolist() for j,n in enumerate(('CIDEr','CHAIRs','CHAIRi','Recall'))},repairs=repaired,new_flags=introduced,
            changed=sum(arms[first]['phases'][fp]['rows'][i]!=arms[last]['phases'][lp]['rows'][i] for i in a['phases']['original']['rows']))
    output=dict(status='CSA_SCST_FULL1_REVIEW_COMPLETE',metadata_checks=checks,
        zip_sha256={k:v['data']['saved_text_receipt.json']['zip_sha256'] for k,v in arms.items()},
        metrics={k:v['data']['result.json']['metrics'] for k,v in arms.items()},comparisons=comparisons,
        selected={k:v['data']['locked_selection.json'] for k,v in arms.items()},user_reported_save_versions={'CE':161,'OBJECT':55},
        worker_seconds={k:[w['seconds'] for w in v['phases']['end']['workers']] for k,v in arms.items()},
        bootstrap=dict(resamples=10000,seed=20261105,unit='image',ratio_metrics_recomputed=True),
        test_used=False,local_model_or_scorer_execution=False,
        limitations='One seed/reusedVALID and validation-based checkpoint selection; conditional unadjusted intervals omit seed/selection variability. Binaries/target payloads absent ZIP, runtime receipts are not independent local binary verification. CHAIR measures COCO object classes, not relation correctness.')
    (HERE/'SCST_FULL1_RESULT.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(checks=checks,comparisons=comparisons,selected={k:v['selected_name'] for k,v in output['selected'].items()}),indent=2))


if __name__=='__main__':run()
