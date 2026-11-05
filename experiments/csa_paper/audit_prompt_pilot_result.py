"""Saved text/count/source and paired NumPy analysis only; no model/scorer or ZIP replay."""
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from real_bank_data import read,save,sha
from saved_quality_data import chair_counts,check_rows
from prompt_pilot_data import CHECKPOINT_SHA,CACHE_RECEIPT_SHA,MANIFEST_SHA

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def paired(a,b):
    ids=sorted(a['rows']);d=np.array([b['cider'][i]-a['cider'][i] for i in ids])
    h=np.array([bool(b['chair'][i]['mscoco_hallucinated_words'])-int(bool(a['chair'][i]['mscoco_hallucinated_words'])) for i in ids])
    rng=np.random.default_rng(42);boot=[];hb=[]
    for _ in range(20):
        take=rng.integers(len(ids),size=(100,len(ids)));boot.extend(d[take].mean(1));hb.extend(h[take].mean(1))
    repair=int((h==-1).sum());new=int((h==1).sum());n=repair+new
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(repair,new)+1))/2**n) if n else 1.
    return dict(CIDEr_delta=float(d.mean()),CIDEr_paired95CI=np.quantile(boot,[.025,.975]).tolist(),
        CHAIRs_delta=float(h.mean()),CHAIRs_paired95CI=np.quantile(hb,[.025,.975]).tolist(),
        repaired_captions=repair,new_hallucinated_captions=new,McNemar_exact_p=p,
        changed_captions=sum(a['rows'][i]['caption']!=b['rows'][i]['caption'] for i in ids))


def run():
    checks=0;arms={};metrics={};phases={};runtime={};sources={};weights={}
    def check(condition):
        nonlocal checks
        if not condition:raise AssertionError('Stored prompt-pilot check failed:'+str(checks+1))
        checks+=1
    for arm in ('CONTROL','PROMPT'):
        root=HERE/f'prompt_pilot_{arm.lower()}_review_text';archive=read(root/'saved_text_receipt.json')
        check(archive['crc_verified'])
        for n,v in archive['files'].items():check(sha(root/n)==v['sha256'])
        result=read(root/'result.json');status=read(root/'run_status.json');identity=read(root/'identity.json');stage=read(root/'pilot_stage.json');source=read(root/'source_receipt.json')
        check(result['status']=='PROMPT_PILOT500_COMPLETE' and result['arm']==arm and result['updates']==500 and result['image_events']==4000 and result['validation_images']==5000)
        check(status['complete'] and status['updates_requested']==500 and not status['test_used'] and not status['rerank_measured'] and not result['rerank_measured'])
        check(not result['test_used'] and identity['checkpoint_sha256']==stage['checkpoint_sha256']==CHECKPOINT_SHA)
        check(stage['status']=='PROMPT_PILOT_STAGE_COMPLETE' and stage['cache_receipt_sha256']==CACHE_RECEIPT_SHA)
        check(identity['stage']==stage and identity['arm']==arm and identity['lr']==5e-6 and identity['scst_weight']==.05 and identity['object_weight']==0 and not identity['dropout'])
        check(identity['seed']==42 and identity['policy_seed']==1234 and source['commit']=='fc54bdea66479e66c8e7662831be167325ffef8e' and len(source['sha256'])==25)
        for n,h in source['sha256'].items():check(hashlib.sha256(subprocess.check_output(['git','show',source['commit']+':'+n],cwd=ROOT)).hexdigest()==h)
        check(stage['updates']==500 and stage['image_events']==4000 and stage['unique_exposed']==4000 and stage['global_batch']==8 and not stage['test_used'])
        pool=stage['pool_ids'];check(len(pool)==len(set(pool))==10000)
        ids=np.random.default_rng(42).permutation(pool)[:4000].reshape(500,8);buf=io.BytesIO();np.save(buf,ids)
        check(hashlib.sha256(buf.getvalue()).hexdigest()==stage['files_sha256']['coco_ids.npy'])
        for n in ('validation_references.json','vocabulary.json'):check(sha(root/n)==stage['files_sha256'][n])
        refs=read(root/'validation_references.json');expected={r['coco_id'] for r in refs};vocab=read(root/'vocabulary.json')
        check(len(refs)==len(expected)==5000 and not expected&set(pool));check(read(root/'metric_preflight.json')['passed'])
        log=(root/'notebook.log').read_text(encoding='utf-8')
        check('Traceback' not in log and 'FAILED' not in log and 'Ran 5 tests' in log and 'Ran 3 tests' in log and 'SCST two-GPU DDP gradient scaling PASS' in log)
        training=[json.loads(line) for line in (root/'training.jsonl').read_text(encoding='utf-8').splitlines()]
        check([r['updates'] for r in training]==list(range(25,501,25)))
        check(all(r['image_events']==r['updates']*8 and all(np.isfinite(v) for k,v in r.items() if isinstance(v,(float,int))) for r in training))
        endpoint={'updates':500,'image_events':4000,'validated':['original','end']}
        check(read(root/'progress.json')==endpoint)
        for rank in (0,1):
            done=read(root/f'training_complete_rank{rank}.json');pre=read(root/f'actual_preflight_rank{rank}.json')
            check(done['complete'] and done['arm']==arm and all(done[k]==v for k,v in endpoint.items()))
            check(pre['passed'] and pre['arm']==arm and pre['checkpoint_sha256']==CHECKPOINT_SHA and pre['smoke_updates_discarded'])
            check(pre['closed_memory_exact'] and pre['empty_memory_exact'] and pre['policy_replay_max_abs']<=2e-5)
            check(len(pre['gradient_reports'])==2 and all(np.isfinite(r['grad_norm']) for r in pre['gradient_reports']))
            if arm=='PROMPT':
                for r in pre['gradient_reports']:
                    check(set(r['adapters'])=={'object_prompt','relation_prompt'})
                    check(all(g['output_grad_l1']>0 and (r['step']==1 or g['upstream_grad_l1']>0) for g in r['adapters'].values()))
        for phase,step in [('original',0),('end',500)]:
            d=root/'evaluations'/phase;point=result['metrics'][phase];r0=read(d/'val_rank0.json');r1=read(d/'val_rank1.json');rows=r0+r1;check_rows(rows,expected)
            check(not {r['coco_id'] for r in r0}&{r['coco_id'] for r in r1})
            workers=[read(d/f'worker{r}.json') for r in (0,1)]
            check(all(w['complete'] and w['updates']==step and w['image_events']==step*8 and w['images']==2500 and w['arm']==arm and w['phase']==phase for w in workers))
            check(sum(w['nll_tokens'] for w in workers)==286871)
            check(abs(sum(w['nll_sum'] for w in workers)/286871-point['NLL'])<1e-12)
            packet=read(d/'chair_output.json');metric,counts=chair_counts(packet,rows)
            check(counts==point['counts'] and all(abs(v-point[k])<1e-12 for k,v in metric.items()))
            check(abs(np.mean([len(r['caption'].split()) for r in rows])-point['mean_words'])<1e-12)
            check(sum(len(r['caption'].split())<=2 for r in rows)==point['short_le2'])
            candidates=read(d/'candidates_rank0.json')+read(d/'candidates_rank1.json');lookup={r['coco_id']:r for r in rows}
            check(len(candidates)==len({r['coco_id'] for r in candidates})==5000 and {r['coco_id'] for r in candidates}==expected)
            check(all(len(r['candidates'])==5 and r['candidates'][0]['caption']==lookup[r['coco_id']]['caption'] for r in candidates))
            for row in candidates:
                scores=[]
                for rank,r in enumerate(row['candidates']):
                    tokens=r['ids'];check(tokens[0]==1 and len(tokens)<=21 and r['rank']==rank and np.isfinite(r['score']))
                    content=tokens[1:];content=content[:content.index(2)] if 2 in content else content
                    check(r['caption']==' '.join(vocab[t] for t in content if t>1));scores.append(r['score'])
                check(all(a>=b for a,b in zip(scores,scores[1:])))
            per=read(d/'cider_per_image.json');cider={r['coco_id']:r['CIDEr'] for r in per}
            check(len(per)==len(cider)==5000 and set(cider)==expected and abs(np.mean(list(cider.values()))-point['CIDEr'])<1e-12)
            check(read(d/'result.json')==point)
            phases[arm+'_'+phase]=dict(rows=lookup,cider=cider,chair={r['image_id']:r for r in packet['sentences']})
        cr=read(root/'checkpoint_receipt.json');check(cr['arm']==arm and all(cr[k]==v for k,v in endpoint.items()))
        arms[arm]=dict(stage=stage,identity=identity,refs=refs,source=source);metrics[arm]=result['metrics'];weights[arm]=cr['sha256']
        sources[arm]=archive['zip_sha256'];runtime[arm]=dict(worker_seconds=[read(root/f'training_complete_rank{r}.json')['seconds'] for r in (0,1)],
            recorded_25_to_500_update_interval_seconds=training[-1]['seconds']-training[0]['seconds'],torch_versions=[read(root/f'actual_preflight_rank{r}.json')['torch_version'] for r in (0,1)])
    a,b=arms['CONTROL'],arms['PROMPT'];check(a['refs']==b['refs'] and a['source']==b['source'])
    for k in ('files_sha256','source_stage','pool_ids','validation','checkpoint_sha256','dataset_json_sha256','model_sha256'):check(a['stage'][k]==b['stage'][k])
    check(metrics['CONTROL']['original']==metrics['PROMPT']['original'])
    check(phases['CONTROL_original']['rows']==phases['PROMPT_original']['rows'])
    check(phases['CONTROL_original']['cider']==phases['PROMPT_original']['cider'])
    probe_root=HERE/'prompt_pilot_prompt_review_text';probe=read(probe_root/'probe_rank0.json')['rows']+read(probe_root/'probe_rank1.json')['rows']
    check(len(probe)==len({r['coco_id'] for r in probe})==64 and sorted(r['coco_id'] for r in probe)==sorted(phases['PROMPT_end']['rows'])[:64])
    changes={m:sum(r['modes'][m]['caption']!=r['modes']['normal']['caption'] for r in probe) for m in ('zero','shuffle','off')}
    check(changes==read(probe_root/'probe_summary.json')['caption_changes'])
    detail={}
    for m in ('zero','shuffle','off'):
        tokens=sum(r['target_tokens'] for r in probe)
        diffs=[r['modes'][m]['logit_max_abs_from_normal'] for r in probe]
        check(all(np.isfinite(diffs)) and all(r['modes']['normal']['logit_max_abs_from_normal']==0 for r in probe))
        delta=sum(r['modes'][m]['nll_sum']-r['modes']['normal']['nll_sum'] for r in probe)/tokens
        detail[m]=dict(changed_captions=changes[m],max_abs_logit_difference=max(diffs),mean_max_abs_logit_difference=float(np.mean(diffs)),
            rows_with_nonzero_logit_difference=sum(x>0 for x in diffs),token_weighted_delta_NLL=delta)
    comparisons={name:paired(phases[left],phases[right]) for name,left,right in [
        ('PROMPT_minus_CONTROL','CONTROL_end','PROMPT_end'),('CONTROL_minus_original','CONTROL_original','CONTROL_end'),('PROMPT_minus_original','CONTROL_original','PROMPT_end')]}
    result=dict(status='PROMPT_PILOT500_STORED_REVIEW_PASS',checks=checks,source=a['source']['commit'],source_files=25,
        zip_sha256=sources,user_reported_sources=dict(CONTROL='ducanh2403/2-vi-cai-kia-qua-nhieu/172',PROMPT='ducanh3/notebook7bc83c4c76/56'),
        checkpoint_sha256=weights,initial_checkpoint_sha256=CHECKPOINT_SHA,cache_receipt_sha256=CACHE_RECEIPT_SHA,
        metrics=metrics,comparisons=comparisons,probe=dict(images=64,selection='first64sortedVALIDIDs_not_random_or_relation_stratified',modes=detail),runtime=runtime,
        updates=500,image_events=4000,validation_images=5000,test_used=False,rerank_measured=False,local_models_or_scorers_executed=False,
        limitations=['Single seed; reused VALID,4000TRAIN exposures, no TEST or fixed rerank yet.',
            'Binaries/staged arrays/training references absent review; actual hash checks are user-run evidence. Agent checked saved text/count/source only, did not independently rerun metric scorers.',
            '64probes are sorted-ID convenience subset, not representative or relation-stratified; numerical differences do not prove semantic grounding.',
            'Paired2000-image bootstrap95CI uses stored per-image scores; no correction for model selection/multiple historical trials.',
            'Architecture has additional trainable visual+geometry parameters; no isolated text-contribution control. No verified relation accuracy.'],
        decision='No meaningful joint rank0 gain: do not automatically expand training. Keep161; finish preplanned same fixed rerank of existing candidates if continuing, no resampling.')
    save(HERE/'PROMPT_PILOT_RESULT.json',result)
    print(json.dumps(dict(status=result['status'],checks=checks,comparisons=comparisons,probe=result['probe'],runtime=runtime),indent=2))


if __name__=='__main__':run()
