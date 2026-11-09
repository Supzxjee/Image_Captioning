"""Audit cached V3 review text; no model or metric recomputation."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from full_data import validate_state,learning_rate
from saved_quality_data import chair_counts

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def audit(cache):
    c=json.loads(Path(cache).read_text(encoding='utf-8'));files=c['files'];get=lambda n:json.loads(files[n]);checks=[]
    def check(name,ok):
        if not ok:raise AssertionError(name)
        checks.append(name)
    result=get('result.json');state=get('progress.json');done=get('training_complete.json');identity=get('identity.json');source=get('source_receipt.json');stage=get('stage_summary.json');cfg=get('protocol.json')
    check('success status',result['status']=='CLIP_PATCH_FULL_XE10_COMPLETE' and get('run_status.json')['complete'])
    validate_state(state);check('complete horizon and caption events',state['epoch']==11 and state['updates']==177110 and state['caption_events']==5667470 and len(state['history'])==10)
    check('TRAIN VALID scope',result['train_images']==113287 and result['validation_images']==5000 and not result['test_used'] and not result['SCST'] and not result['reranking'])
    check('source exact V3',source['commit']=='9aaddf5694f72373b8ab2482fafe7d7992abbe5e' and len(source['sha256'])==18)
    for name,h in source['sha256'].items():check('source '+name,hashlib.sha256(subprocess.check_output(['git','show',source['commit']+':'+name],cwd=ROOT)).hexdigest()==h)
    check('identity source and raw package',identity['source']==source and identity['raw_sources']==stage['sources'] and identity['stage_files']==stage['files_sha256'])
    check('identity fixed recipe',identity['epochs']==10 and identity['global_batch']==32 and identity['seed']==42 and identity['test_used'] is False and cfg['resume_root']=='')
    check('stage complete exact payload',stage['feature_storage']=='local_fp16_memmap_v2' and stage['local_stage']=={k:v for k,v in get('local_stage_progress.json').items() if k!='status'} and stage['local_stage']['images']==118287 and stage['local_stage']['smoke_raw_rows_replayed']==256 and stage['local_stage']['all_copy_blocks_byte_equal'] and not stage['test_features_used'])
    check('unchanged initialization and passed guards',all(get(f'preflight_rank{r}.json')['passed'] and get(f'preflight_rank{r}.json')['initial_tensor_sha256']==identity['initial_sha256'] and get(f'preflight_rank{r}.json')['discarded_benchmark_updates']==64 for r in (0,1)))
    ref_ids={int(i) for i in json.loads(gzip.decompress((ROOT/'experiments/csa_paper/prompt_scratch_valid5000_saved.json.gz').read_bytes()))['reference_ptb']}
    check('historical VALID cohort5000',len(ref_ids)==5000)
    curve=[]
    for epoch,point in enumerate(state['history'],1):
        base=f'evaluations/epoch_{epoch:02d}/';metrics=get(base+'result.json')
        for n,h in point['files_sha256'].items():check(f'epoch{epoch} sealed '+n,hashlib.sha256(files[base+n].encode('utf-8')).hexdigest()==h)
        a=get(base+'val_rank0.json');b=get(base+'val_rank1.json');rows=a+b
        check(f'epoch{epoch} complete/disjoint VALID IDs',len(a)==len(b)==2500 and len({r['coco_id'] for r in rows})==5000 and {r['coco_id'] for r in rows}==ref_ids)
        check(f'epoch{epoch} caption/beam content',all(isinstance(r['caption'],str) and r['caption'].strip() and math.isfinite(r['score']) and r['ids'][0]==1 and len(r['ids'])<=21 and all(type(t)==int and 0<=t<10000 for t in r['ids']) for r in rows))
        per=get(base+'cider_per_image.json');check(f'epoch{epoch} perCIDEr/point parity',len(per)==5000 and {r['coco_id'] for r in per}==ref_ids and math.isclose(sum(r['CIDEr'] for r in per)/5000,metrics['CIDEr'],abs_tol=1e-10))
        workers=[get(base+f'worker{r}.json') for r in (0,1)]
        check(f'epoch{epoch} weighted NLL',all(w['complete'] and w['images']==2500 for w in workers) and math.isclose(sum(w['NLL_sum'] for w in workers)/sum(w['NLL_tokens'] for w in workers),metrics['NLL'],abs_tol=1e-12))
        check(f'epoch{epoch} metrics history',all(point[k]==metrics[k] for k in ('epoch','CIDEr','NLL','TRAIN_online_NLL')))
        curve.append(metrics)
    check('best epoch fixed VALID rule',max(curve,key=lambda r:r['CIDEr'])['epoch']==result['best_epoch']==state['best_epoch']==10)
    q=get('quality/epoch_10/result.json');rows=get('evaluations/epoch_10/val_rank0.json')+get('evaluations/epoch_10/val_rank1.json')
    check('saved quality exact point',q==result['metrics_by_epoch']['10'] and q['CIDEr']==curve[-1]['CIDEr'] and q['NLL']==curve[-1]['NLL'])
    check('quality saved caption/CHAIR input',get('quality/epoch_10/chair_input.json')==[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
    chair,counts=chair_counts(get('quality/epoch_10/chair_output.json'),rows)
    check('CHAIR counts and scores replay',counts==q['counts'] and all(chair[k]==q[k] for k in chair))
    check('length and empty caption counts',sum(len(r['caption'].split()) for r in rows)/5000==q['mean_words'] and sum(len(r['caption'].split())<=2 for r in rows)==q['short_le2']==0)
    for key in ('best_receipt.json','checkpoint_receipt.json'):check(key+' nested replay',get(key)==result['receipts'][key])
    check('best/end same weight receipt',get('best_receipt.json')['sha256']==get('checkpoint_receipt.json')['sha256']=='5061e305d98edf54930a200090a41c983c354d1ba2dfcf700b715cf20a5b80b7')
    check('training log seal',hashlib.sha256(files['training.jsonl'].encode('utf-8')).hexdigest()==get('checkpoint_receipt.json')['training_log_sha256'])
    log=[json.loads(line) for line in files['training.jsonl'].splitlines()]
    check('logged updates monotonic/end',all(a['updates']<b['updates'] and a['seconds']<b['seconds'] for a,b in zip(log,log[1:])) and log[-1]['updates']==177110)
    for r in log:check('training cursor/lr/finite '+str(r['updates']),r['updates']==(r['epoch']-1)*17711+r['batch_cursor'] and r['caption_events']==(r['epoch']-1)*566747+min(r['batch_cursor']*32,566747) and math.isclose(r['lr'],learning_rate(r['updates']),abs_tol=1e-15) and math.isfinite(r['NLL']) and math.isfinite(r['grad_norm']))
    check('worker complete state replay',all(done[k]==state[k] for k in state))
    summary=dict(status='FULL_LOCAL_V3_REVIEWED_COMPLETE',source=dict(owner='ducanh2403',notebook='2-vi-cai-kia-qua-nhieu',save_version=199,owner_inherited_from_session=True,commit=source['commit']),archive_sha256=c['archive_sha256'],checks_passed=len(checks),metrics=q,epoch_curve=curve,train_images=113287,validation_images=5000,epochs=10,updates=177110,caption_events=5667470,best_epoch=10,checkpoint_sha256=get('best_receipt.json')['sha256'],stage_seconds=stage['local_stage']['seconds'],worker_seconds=done['worker_seconds'],timing_scope='worker includes train/validation/checkpoints; stage excludes setup, preflight and final scoring; no exact billed/GPU-hours claim',preflight_median_step_seconds=max(get(f'preflight_rank{r}.json')['full_IO_step_median_seconds'] for r in (0,1)),main_log_last_update_seconds=log[-1]['seconds'],test_used=False,SCST=False,reranking=False,metrics_not_measured=result['metrics_not_measured'],author_limits=['Cached metadata/seals/counts/arithmetic audited, not checkpoint weights or features reloaded.','Official scorers not rerun locally; BLEU/ROUGE remain runtime-reported, CIDEr average and CHAIR counts replayed.','One seed on reused VALID, no independent TEST result or isolated prompt/bank effect.'])
    (HERE/'LOCAL_V3_RESULT.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    (HERE/'LOCAL_V3_REVIEW_CHECKS.json').write_text(json.dumps(dict(checks_passed=len(checks),checks=checks,model_scorer_executed_locally=False),indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='epoch_curve'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);audit(p.parse_args().cache)
