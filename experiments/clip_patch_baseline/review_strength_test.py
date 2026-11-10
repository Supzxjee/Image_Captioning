"""Audit archived TEST199 text/counts/source; never rerun neural or scorers."""
import argparse
import gzip
import hashlib
import json
import math
import numpy as np
from pathlib import Path
import subprocess
from saved_quality_data import check_rows,chair_counts
from patch_beam import text_from_ids
from strength_test_data import validate_lock,CHECKPOINT_SHA,runtime_policy

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def audit(path):
    cache=json.loads(Path(path).read_text(encoding='utf-8'));files=cache['files'];get=lambda name:json.loads(files[name]);checks=[]
    def check(name,ok):
        if not ok:raise AssertionError(name)
        checks.append(name)
    result=get('result.json');locked=validate_lock(get('test_lock.json'));stage=get('stage_receipt.json');source=get('source_receipt.json');sealed=get('scoring_lock.json');words=get('vocabulary.json')
    check('complete nontraining no tuning',result['status']=='CLIP_PATCH_STRENGTH_TEST5000_COMPLETE' and get('run_status.json')['complete'] and not result['training'] and not result['reranking'] and not result['test_tuning'])
    check('same best8 fixeddecoder',result['checkpoint_sha256']==CHECKPOINT_SHA==get('checkpoint_source.json')['sha256'] and result['epoch']==8 and result['beam_width']==5 and result['max_new_tokens']==20)
    check('actual exact12 API origin',get('checkpoint_source.json')['version']==locked['source_version']==result['source_version']=='ducanh2403/kltnnnnnff/12')
    check('source version/pins',source['commit']=='29fa3bd7b0353941eb40038249a13b4e832c36f1' and len(source['sha256'])==28)
    for name,h in source['sha256'].items():check('source '+name,hashlib.sha256(subprocess.check_output(['git','show',source['commit']+':'+name],cwd=ROOT)).hexdigest()==h)
    check('stage receipt seal',hashlib.sha256(files['stage_receipt.json'].encode()).hexdigest()==locked['stage_receipt_sha256'])
    check('TEST only coverage/profile',stage['images']==5000 and not stage['TRAIN_VALID_features_read'] and sum(s['selected_TEST_rows'] for s in stage['sources'])==5000 and stage['feature_profile']==json.loads((HERE/'SMOKE_SOURCE.json').read_text())['feature_profile'])
    for n,h in stage['output_files_sha256'].items():check('stage output '+n,hashlib.sha256(files[n].encode()).hexdigest()==h)
    for n,h in sealed['files_sha256'].items():check('scoring input '+n,hashlib.sha256(files[n].encode()).hexdigest()==h)
    rows=get('captions.json');ids=locked['expected_test_ids'];check_rows(rows,ids);workers=[get(f'worker{r}.json') for r in (0,1)];rankrows=[]
    for rank,w in enumerate(workers):
        name=f'captions_rank{rank}.json';one=get(name);rankrows+=one
        check(f'rank{rank} complete/captionseal',w['complete'] and w['images']==len(one)==2500 and hashlib.sha256(files[name].encode()).hexdigest()==w['caption_sha256'] and w['checkpoint_sha256']==CHECKPOINT_SHA and not w['training'] and w['first_GPU_beam_path_check_passed'])
        policy=runtime_policy(w['runtime_policy']['training_torch'],w['runtime_policy']['inference_torch'])
        check(f'rank{rank} runtime provenance',policy==w['runtime_policy']==get(f'runtime_policy_rank{rank}.json')==result['runtime_policy']==get('checkpoint_runtime_preflight.json')['runtime_policy'])
    check('ranks disjoint and combinedexact',sorted(rankrows,key=lambda r:r['coco_id'])==rows and len({r['coco_id'] for r in rankrows})==5000)
    check('unchanged TRAIN vocab',len(words)==10000 and hashlib.sha256(files['vocabulary.json'].encode()).hexdigest()=='cc5f48afa9e471c7a8fd9e30ac58df2e47a54fbf4c0e86b6e0e340006b1601fd')
    for r in rows:
        seq=r['ids'];check('caption/token '+str(r['coco_id']),r['caption']==text_from_ids(seq,words) and seq[0]==1 and len(seq)<=21 and all(type(x)==int and 0<=x<10000 for x in seq) and math.isfinite(r['score']) and r['score']<=0)
    refs=get('test_references.json');snapshot=json.loads(gzip.decompress((ROOT/'experiments/csa_paper/plain_csa10_test_snapshot.json.gz').read_bytes()))
    check('same historical TEST raw references',len(refs)==5000 and {r['coco_id']:r['references'] for r in refs}=={r['coco_id']:r['references'] for r in snapshot})
    check('PTB5000cohort',len(get('reference_ptb.json'))==5000 and {int(i) for i in get('reference_ptb.json')}==set(ids)=={int(i) for i in get('caption_ptb.json')})
    metrics={}
    for mode in ('basic','meteor','spice','chair'):
        part=get(mode+'.json');check(mode+' input seal/complete',part['images']==5000 and part['input_sha256']==sealed['files_sha256']);metrics.update(part['metrics'])
    check('all eightmetrics finite',{'BLEU_1','BLEU_4','METEOR','ROUGE_L','CIDEr','SPICE','CHAIRs','CHAIRi','Recall'}==set(metrics) and all(math.isfinite(x) for x in metrics.values()))
    check('point phase/result parity',all(result['metrics'][k]==v for k,v in metrics.items()))
    for file,key,mean in [('cider_per_image.json','CIDEr',True),('spice_per_image.json','SPICE',True),('meteor_per_image.json','METEOR',False)]:
        per=get(file);vkey='CIDEr' if key=='CIDEr' else 'score';check(file+' finite/coverage',len(per)==5000 and {r['coco_id'] for r in per}==set(ids) and all(math.isfinite(r[vkey]) for r in per))
        if mean:check(file+' mean parity',math.isclose(sum(r[vkey] for r in per)/5000,metrics[key],abs_tol=1e-10))
    check('CHAIR exact inputs',get('chair_input.json')==[dict(image_id=r['coco_id'],caption=r['caption']) for r in rows])
    point,counts=chair_counts(get('chair_output.json'),rows);check('CHAIR counts aggregate',counts==result['chair_counts']==get('chair_counts.json') and all(metrics[k]==v for k,v in point.items()))
    check('NLL weighted aggregation',math.isclose(result['metrics']['NLL'],sum(w['NLL_sum'] for w in workers)/sum(w['NLL_tokens'] for w in workers),abs_tol=1e-12))
    check('display scaling',result['display_x100']=={k:100*v for k,v in metrics.items()})
    check('metric environment/smoke',get('environment.json')==result['environment'] and result['environment']['pycocoevalcap']=='1.2' and get('smoke.json')['passed'])
    baseline_cache=json.loads((ROOT.parent/'.review_cache/clip_patch_test199_v2/archive_text.json').read_text(encoding='utf-8'))['files'];base=lambda n:json.loads(baseline_cache[n]);old_result=base('result.json');old_metrics=old_result['metrics']
    check('identical TEST raw references and PTB',get('test_references.json')==base('test_references.json') and get('reference_ptb.json')==base('reference_ptb.json'))
    check('identical baseline TEST patch/caption/vocab bytes',all(stage['stage_files_sha256'][k]==h for k,h in base('stage_receipt.json')['stage_files_sha256'].items()) and words==base('vocabulary.json'))
    check('same evaluator source and assets',result['environment']['evaluator_files']==old_result['environment']['evaluator_files'] and result['environment']['java_archive_sha256']==old_result['environment']['java_archive_sha256'])
    expected181=json.loads((ROOT/'experiments/csa_paper/PROMPT_TEST_PREPARATION_RESULT.json').read_text())['transport_inventory']['cache_files']
    check('exact TEST181 frozen semantic source',all(expected181[n]==h for n,h in stage['prompt181_files_sha256'].items()))
    semantic=get('semantic_test_stage.json');check('semantic TEST scope',semantic['images']==5000 and semantic['objects']==14593 and semantic['relations']==2228 and semantic['text_vectors']==258 and not semantic['old_ROI_features_used'] and not semantic['old_vocabulary_used'])
    new_per={r['coco_id']:r['CIDEr'] for r in get('cider_per_image.json')};old_per={r['coco_id']:r['CIDEr'] for r in base('cider_per_image.json')}
    new_chair={r['image_id']:r for r in get('chair_output.json')['sentences']};old_chair={r['image_id']:r for r in base('chair_output.json')['sentences']}
    check('same per-image CHAIR ground truth',all(set(new_chair[i]['mscoco_gt_words'])==set(old_chair[i]['mscoco_gt_words']) for i in ids))
    removed=sum(bool(old_chair[i]['mscoco_hallucinated_words']) and not new_chair[i]['mscoco_hallucinated_words'] for i in ids)
    added=sum(not old_chair[i]['mscoco_hallucinated_words'] and bool(new_chair[i]['mscoco_hallucinated_words']) for i in ids)
    values=np.array([[new_per[i]-old_per[i],int(bool(new_chair[i]['mscoco_hallucinated_words']))-int(bool(old_chair[i]['mscoco_hallucinated_words']))] for i in ids])
    rng=np.random.default_rng(42);samples=np.concatenate([values[rng.integers(0,5000,(100,5000))].mean(1) for _ in range(20)]);ci=(np.quantile(samples,[.025,.975],axis=0)*100).T.tolist()
    summary=dict(status='SEMANTIC_STRENGTH_TEST_REVIEWED_COMPLETE',evaluation_source=dict(notebook_slug="kltnnnnnff",save_version=14,owner="ducanh2403",version_user_confirmed=True,notebook_inherited_from_session=True),checkpoint_source_version=result['source_version'],checkpoint_sha256=CHECKPOINT_SHA,
        source_commit=source['commit'],archive_sha256=cache['archive_sha256'],checks_passed=len(checks),images=5000,epoch=8,metrics=result['metrics'],display_x100=result['display_x100'],chair_counts=counts,
        baseline199=dict(metrics=old_metrics,chair_counts=old_result['chair_counts']),delta_raw={k:result['metrics'][k]-v for k,v in old_metrics.items()},
        delta_display={k:100*(result['metrics'][k]-v) for k,v in old_metrics.items() if k!='NLL'},
        paired=dict(CHAIR_flags_removed=removed,CHAIR_flags_introduced=added,CIDEr_delta_CI95_points=ci[0],CHAIRs_delta_CI95_pp=ci[1],samples=2000,seed=42,
            limits='Saved-caption conditional bootstrap, fixed reference IDF; no seed variation, multiple-testing or selection correction; not a formal equivalence/noninferiority test.'),
        runtime_policy=result['runtime_policy'],mean_words=sum(len(r['caption'].split()) for r in rows)/5000,empty_captions=sum(not r['caption'].strip() for r in rows),short_le2=sum(len(r['caption'].split())<=2 for r in rows),
        stage_seconds=stage['seconds'],inference_worker_seconds=max(w['seconds'] for w in workers),same_historical_test_refs=True,training=False,reranking=False,test_tuning=False,prior_project_test_exposure=True,
        limits=['No local Torch/model/scorer/checkpoint/feature load; text/seal/arithmetic audit only.','BLEU/METEOR/ROUGE runtime-reported; CIDEr/SPICE saved per-image means and CHAIR counts replayed, no official scorer rerun.',
        'Combined system vs plain baseline does not isolate M2/M3/text/masks; strength comparison adds 1025 parameters.','CIDEr point improves slightly; conditional interval and declines in METEOR/ROUGE/SPICE and recall must be reported.','One seed; prior project TEST exposure. Model selection was fixed on VALID before this TEST.'])
    previous_files=json.loads((ROOT.parent/'.review_cache/bank_prompt_test5000_v1/archive_text.json').read_text(encoding='utf-8'))['files']
    previous=lambda n:json.loads(previous_files[n])
    prev=previous('result.json')
    check('same prior bankprompt inputs',all(stage['stage_files_sha256'][k]==h for k,h in previous('stage_receipt.json')['stage_files_sha256'].items()) and refs==previous('test_references.json') and get('reference_ptb.json')==previous('reference_ptb.json') and stage['prompt181_files_sha256']==previous('stage_receipt.json')['prompt181_files_sha256'])
    check('same prior bankprompt evaluators',result['environment']['evaluator_files']==prev['environment']['evaluator_files'] and result['environment']['java_archive_sha256']==prev['environment']['java_archive_sha256'])
    prev_per={r['coco_id']:r['CIDEr'] for r in previous('cider_per_image.json')}
    prev_chair={r['image_id']:r for r in previous('chair_output.json')['sentences']}
    check('same prior bankprompt CHAIR groundtruth',all(set(new_chair[i]['mscoco_gt_words'])==set(prev_chair[i]['mscoco_gt_words']) for i in ids))
    differences=np.array([[new_per[i]-prev_per[i],int(bool(new_chair[i]['mscoco_hallucinated_words']))-int(bool(prev_chair[i]['mscoco_hallucinated_words']))] for i in ids])
    rng=np.random.default_rng(42)
    draws=np.concatenate([differences[rng.integers(0,5000,(100,5000))].mean(1) for _ in range(20)])
    interval=(np.quantile(draws,[.025,.975],axis=0)*100).T.tolist()
    summary['previous_bank_prompt']=dict(metrics=prev['metrics'],chair_counts=prev['chair_counts'],delta_display={k:100*(result['metrics'][k]-v) for k,v in prev['metrics'].items() if k!='NLL'},paired=dict(CIDEr_delta_CI95_points=interval[0],CHAIRs_delta_CI95_pp=interval[1],CHAIR_flags_removed=sum(bool(prev_chair[i]['mscoco_hallucinated_words']) and not new_chair[i]['mscoco_hallucinated_words'] for i in ids),CHAIR_flags_introduced=sum(not prev_chair[i]['mscoco_hallucinated_words'] and bool(new_chair[i]['mscoco_hallucinated_words']) for i in ids)))
    summary['checks_passed']=len(checks)
    (HERE/'SEMANTIC_STRENGTH_TEST_RESULT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (HERE/'SEMANTIC_STRENGTH_TEST_REVIEW_CHECKS.json').write_text(json.dumps(dict(checks_passed=len(checks),checks=checks,models_scorers_executed_locally=False),indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='limits'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);audit(p.parse_args().cache)
