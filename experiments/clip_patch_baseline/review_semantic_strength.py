"""Cached-text arithmetic/source audit only; never load Torch or rerun scorers."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from full_data import validate_state,learning_rate
from saved_quality_data import chair_counts

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def audit():
    cache=ROOT.parent/'.review_cache/semantic_strength_xe10_v2/archive_text.json'
    archived=json.loads(cache.read_text(encoding='utf-8'));texts=archived['files'];get=lambda n:json.loads(texts[n]);checks=[]
    def check(name,value):
        if not value:raise AssertionError(name)
        checks.append(name)
    old=json.loads((ROOT.parent/'.review_cache/clip_patch_bank_prompt_xe10_v1/archive_text.json').read_text(encoding='utf-8'))['texts'];baseline=lambda n:json.loads(old[n])
    result=get('result.json');state=get('progress.json');identity=get('identity.json');stage=get('stage_summary.json');source=get('source_receipt.json');cfg=get('protocol.json')
    validate_state(state)
    check('complete fixed10 horizon',get('run_status.json')['complete'] and result['status']=='CLIP_PATCH_FULL_XE10_COMPLETE' and state['epoch']==11 and state['updates']==177110 and state['caption_events']==5667470)
    check('no TEST/SCST/reranking',all(result[k] is False for k in ('test_used','SCST','reranking')) and not stage['test_features_used'])
    check('fresh semantic method',result['method']==identity['method']=='bank_prompt_strength_v1' and cfg['semantic_strength_gate'] and cfg['parent_wall_hours']==11.5 and cfg['semantic'] and cfg['resume_root']=='')
    check('exact source30',source['commit']=='cc6a20ea7f7f3263a7a637100ab1166dd6ff6a1b' and len(source['sha256'])==30)
    for n,h in source['sha256'].items():check('source '+n,hashlib.sha256(subprocess.check_output(['git','show',source['commit']+':'+n],cwd=ROOT)).hexdigest()==h)
    check('source identity and stage seals',identity['source']==source and identity['stage_files']==stage['files_sha256'])
    prior=baseline('stage_summary.json')
    check('same baseline data/features/references/vocab bytes',all(stage['files_sha256'][k]==h for k,h in prior['files_sha256'].items()) and stage['feature_profile']==prior['feature_profile'])
    check('same optimizer schedule/exposure policy',all(identity[k]==baseline('identity.json')[k] for k in ('epochs','global_batch','lr','dropout','clip1','seed')))
    sem=get('semantic_stage.json');check('semantic stage exact summary',stage['semantic']==sem and sem['TRAIN_VALID_only'] and not sem['test_used'])
    check('exact prompt175',sem['prompt_cache']==json.loads((ROOT/'experiments/csa_paper/FULL_PROMPT_FRESH_V175_RESULT.json').read_text())['transport_inventory']['cache_files'])
    check('exact bank130',sem['bank_cache']==json.loads((ROOT/'experiments/csa_paper/SAVE_BANK_V130_PACKAGE.json').read_text())['files'])
    check('semantic coverage',sem['counts']['train']['images']==113287 and sem['counts']['val']['images']==5000 and sem['counts']['train']['objects']==335723 and sem['counts']['train']['relations']==54574)
    for rank in (0,1):
        p=get(f'preflight_rank{rank}.json')
        check(f'GPU guards rank{rank}',p['passed'] and all(p[k] for k in ('causal','memory_conditioning','global_token_DDP_gradient_parity','beam_score_rescore')) and p['discarded_benchmark_updates']==64)
        check(f'core initialization rank{rank}',p['semantic']['core_initial_sha256']=='5c580d0dfbd0ed828b7697d9ead7575dc7f069e8e3ca51acc776e6b23d9c06bd' and p['initial_tensor_sha256']==identity['initial_sha256'] and all(p['semantic'][k] for k in ('initial_memory_equal_baseline','empty_slot_finite_identity','nonzero_adapter_mask_locality')))
    refids={r['coco_id'] for r in baseline('evaluations/epoch_10/val_rank0.json')+baseline('evaluations/epoch_10/val_rank1.json')}
    curve=[]
    for point in state['history']:
        epoch=point['epoch'];prefix=f'evaluations/epoch_{epoch:02d}/';metrics=get(prefix+'result.json')
        for name,h in point['files_sha256'].items():check(f'epoch{epoch} seal '+name,hashlib.sha256(texts[prefix+name].encode()).hexdigest()==h)
        rows=get(prefix+'val_rank0.json')+get(prefix+'val_rank1.json');per=get(prefix+'cider_per_image.json');workers=[get(prefix+f'worker{r}.json') for r in (0,1)]
        check(f'epoch{epoch} VALID IDs and token bounds',len(rows)==5000 and len({r['coco_id'] for r in rows})==5000 and {r['coco_id'] for r in rows}==refids and all(r['ids'][0]==1 and len(r['ids'])<=21 and all(type(t)==int and 0<=t<10000 for t in r['ids']) and math.isfinite(r['score']) and r['caption'].strip() for r in rows))
        check(f'epoch{epoch} saved CIDEr arithmetic',len(per)==5000 and {r['coco_id'] for r in per}==refids and math.isclose(np.mean([r['CIDEr'] for r in per]),metrics['CIDEr'],abs_tol=1e-12))
        check(f'epoch{epoch} weighted NLL',all(w['complete'] and w['images']==2500 for w in workers) and math.isclose(sum(w['NLL_sum'] for w in workers)/sum(w['NLL_tokens'] for w in workers),metrics['NLL'],abs_tol=1e-12))
        check(f'epoch{epoch} history replay',all(point[k]==metrics[k] for k in ('epoch','CIDEr','NLL','TRAIN_online_NLL')));curve.append(metrics)
    check('best selected by locked CIDEr',max(curve,key=lambda r:r['CIDEr'])['epoch']==state['best_epoch']==result['best_epoch']==8)
    for epoch in (8,10):
        prefix=f'quality/epoch_{epoch:02d}/';q=get(prefix+'result.json');rows=get(f'evaluations/epoch_{epoch:02d}/val_rank0.json')+get(f'evaluations/epoch_{epoch:02d}/val_rank1.json')
        metric,counts=chair_counts(get(prefix+'chair_output.json'),rows)
        check(f'quality{epoch} score/count replay',q==result['metrics_by_epoch'][str(epoch)] and counts==q['counts'] and all(metric[k]==q[k] for k in metric))
        check(f'quality{epoch} caption equality',get(prefix+'chair_input.json')==[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
        check(f'quality{epoch} length/NLL/CIDEr',q['mean_words']==np.mean([len(r['caption'].split()) for r in rows]) and q['short_le2']==sum(len(r['caption'].split())<=2 for r in rows)==0 and q['NLL']==curve[epoch-1]['NLL'] and q['CIDEr']==curve[epoch-1]['CIDEr'])
    check('receipts parity',all(get(n)==result['receipts'][n] for n in ('best_receipt.json','checkpoint_receipt.json')) and get('best_receipt.json')['epoch']==8 and get('checkpoint_receipt.json')['updates']==177110)
    check('training log SHA',hashlib.sha256(texts['training.jsonl'].encode()).hexdigest()==get('checkpoint_receipt.json')['training_log_sha256'])
    log=[json.loads(s) for s in texts['training.jsonl'].splitlines()]
    check('training log monotonic to endpoint',log[-1]['updates']==177110 and all(a['updates']<b['updates'] and a['seconds']<b['seconds'] for a,b in zip(log,log[1:])))
    for row in log:check('LR/exposure '+str(row['updates']),row['updates']==(row['epoch']-1)*17711+row['batch_cursor'] and row['caption_events']==(row['epoch']-1)*566747+min(row['batch_cursor']*32,566747) and math.isclose(row['lr'],learning_rate(row['updates']),abs_tol=1e-15) and math.isfinite(row['NLL']) and math.isfinite(row['grad_norm']))
    check('completion state replay',all(get('training_complete.json')[k]==state[k] for k in state))
    base=baseline('result.json')['metrics_by_epoch']['9'];best=result['metrics_by_epoch']['8'];end=result['metrics_by_epoch']['10']
    delta=lambda q:{k:q[k]-base[k] for k in ('CIDEr','BLEU_1','BLEU_4','ROUGE_L','CHAIRs','CHAIRi','Recall','NLL')}
    oldper={r['coco_id']:r['CIDEr'] for r in baseline('quality/epoch_09/cider_per_image.json')};newper={r['coco_id']:r['CIDEr'] for r in get('quality/epoch_08/cider_per_image.json')}
    oldchair={r['image_id']:r for r in baseline('quality/epoch_09/chair_output.json')['sentences']};newchair={r['image_id']:r for r in get('quality/epoch_08/chair_output.json')['sentences']}
    check('same CHAIR ground truth by image',all(set(oldchair[i]['mscoco_gt_words'])==set(newchair[i]['mscoco_gt_words']) for i in refids))
    ids=sorted(refids);values=np.array([[newper[i]-oldper[i],int(bool(newchair[i]['mscoco_hallucinated_words']))-int(bool(oldchair[i]['mscoco_hallucinated_words']))] for i in ids])
    rng=np.random.default_rng(42);samples=np.concatenate([values[rng.integers(0,5000,(100,5000))].mean(1) for _ in range(20)])
    ci=(np.quantile(samples,[.025,.975],axis=0)*100).T.tolist()
    repaired=sum(bool(oldchair[i]['mscoco_hallucinated_words']) and not newchair[i]['mscoco_hallucinated_words'] for i in ids);introduced=sum(not oldchair[i]['mscoco_hallucinated_words'] and bool(newchair[i]['mscoco_hallucinated_words']) for i in ids)
    from memory_fusion import merge_gate_stats
    for epoch in range(1,11):
        point=get(f'evaluations/epoch_{epoch:02d}/result.json')['semantic_strength']
        accum=[get(f'evaluations/epoch_{epoch:02d}/worker{r}.json')['semantic_strength_accumulator'] for r in (0,1)]
        check(f'gate{epoch} allVALID count/bounds',all(a['count']==490000 and 0<=a['minimum']<=a['maximum']<=1 and 0<=a['total']<=a['count'] and a['below_point1']+a['above_point9']<=a['count'] for a in accum))
        check(f'gate{epoch} weighted aggregate replay',point==merge_gate_stats(accum) and point['patches']==980000 and point==state['history'][epoch-1]['semantic_strength'])
    for epoch in (8,10):check(f'gate{epoch} quality replay',result['metrics_by_epoch'][str(epoch)]['semantic_strength']==state['history'][epoch-1]['semantic_strength'])
    check('strength model GPU endpoint/size',all(get(f'preflight_rank{r}.json')['semantic']['nonzero_strength_endpoints_passed'] and get(f'preflight_rank{r}.json')['semantic']['trainable_parameters']==34801913 for r in (0,1)))
    plain=json.loads((HERE/'LOCAL_V3_RESULT.json').read_text(encoding='utf-8'))['metrics']
    old_end=baseline('result.json')['metrics_by_epoch']['10']
    summary=dict(status='SEMANTIC_STRENGTH_FULL10_REVIEWED_COMPLETE',save_version=12,notebook='kltnnnnnff',notebook_inherited_from_session=True,owner=None,source_commit=source['commit'],archive_sha256=archived['archive_sha256'],checks_passed=len(checks),best_epoch=8,
        best=best,end=end,old_bank_prompt_best=base,old_bank_prompt_end=old_end,plain199_VALID=plain,delta_best_vs_old=delta(best),delta_best_vs_plain={k:best[k]-plain[k] for k in ('CIDEr','CHAIRs','CHAIRi','NLL')},
        delta_endpoint_vs_old={k:end[k]-old_end[k] for k in ('CIDEr','CHAIRs','CHAIRi','NLL')},epoch_curve=curve,receipts=result['receipts'],epochs=10,updates=177110,caption_events=5667470,
        stage_seconds=stage['local_stage']['seconds'],semantic_stage_seconds=sem['seconds'],worker_seconds=get('training_complete.json')['worker_seconds'],old_bank_worker_seconds=baseline('training_complete.json')['worker_seconds'],
        trainable_parameters=34801913,old_bank_parameters=34800888,extra_parameters=1025,
        paired_best=dict(CHAIR_flags_removed=repaired,CHAIR_flags_introduced=introduced,CIDEr_delta_CI95_points=ci[0],CHAIRs_delta_CI95_pp=ci[1],bootstrap_samples=2000,seed=42,limits='Conditional on saved captions, one seed, reused VALID and selected checkpoints; no seed or selection correction.'),
        test_used=False,SCST=False,reranking=False,author_limits=['No local Torch/model/scorer or binary checkpoint/feature reload. CIDEr saved-score arithmetic and CHAIR counts only; BLEU/ROUGE runtime-reported.',
        'Gated best8 vsoldbest9 follow same maxVALIDCIDEr within10epoch policy; selected weights have different exposure. End10 comparison also reported.',
        'Alpha is learned strength, not calibrated confidence, direct localization proof or evidence of prompt-specific contribution.',
        'No TEST/METEOR/SPICE for strength-gated variant. Extra gate did not increase VALID CIDEr over previous bank+prompt.'])
    (HERE/'SEMANTIC_STRENGTH_RESULT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (HERE/'SEMANTIC_STRENGTH_REVIEW_CHECKS.json').write_text(json.dumps(dict(checks_passed=len(checks),checks=checks,model_scorer_executed_locally=False),indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('epoch_curve','author_limits')},indent=2))

if __name__=='__main__':audit()
