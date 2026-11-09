"""Audit archived TEST199 text/counts/source; never rerun neural or scorers."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
from saved_quality_data import check_rows,chair_counts
from patch_beam import text_from_ids
from test199_data import validate_lock,CHECKPOINT_SHA,runtime_policy

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

def audit(path):
    cache=json.loads(Path(path).read_text(encoding='utf-8'));files=cache['files'];get=lambda name:json.loads(files[name]);checks=[]
    def check(name,ok):
        if not ok:raise AssertionError(name)
        checks.append(name)
    result=get('result.json');locked=validate_lock(get('test_lock.json'));stage=get('stage_receipt.json');source=get('source_receipt.json');sealed=get('scoring_lock.json');words=get('vocabulary.json')
    check('complete nontraining no tuning',result['status']=='CLIP_PATCH_199_TEST5000_COMPLETE' and get('run_status.json')['complete'] and not result['training'] and not result['reranking'] and not result['test_tuning'])
    check('same199 epoch10 fixeddecoder',result['checkpoint_sha256']==CHECKPOINT_SHA==get('checkpoint_source.json')['sha256'] and result['epoch']==10 and result['beam_width']==5 and result['max_new_tokens']==20)
    check('actual exact199 API origin',get('checkpoint_source.json')['version']==locked['source_version']==result['source_version']=='ducanh2403/2-vi-cai-kia-qua-nhieu/199')
    check('source version/pins',source['commit']=='fc17e9984b37bad99e1c21a1c06337d80ebf16ec' and len(source['sha256'])==17)
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
    old=json.loads((ROOT/'experiments/csa_paper/PROMPT_TEST_EXTRA_INPUTS.json').read_text(encoding='utf-8'));oldbase=json.loads((ROOT/'experiments/csa_paper/PLAIN_CSA10_EXTRA_INPUTS.json').read_text(encoding='utf-8'))
    historical={}
    for arm in ('CONTROL','PROMPT'):
        spec=old['arms'][arm];check('same '+arm+' raw reference SHA',spec['payload_sha256']['test_references.json']==hashlib.sha256(files['test_references.json'].encode()).hexdigest())
        historical['old_'+arm+'_SCST_rerank']=spec['metrics']['fixed_rerank']
    historical['old_bank_XE10']=oldbase['arms']['BASELINES']['metrics']['CSA10'];historical['old_plain_ROI_XE10']=oldbase['arms']['BASELINES']['metrics']['NoCSA10']
    comparisons={k:dict(CIDEr_delta_points=100*(metrics['CIDEr']-v['CIDEr']),CHAIRs_delta_pp=100*(metrics['CHAIRs']-v['CHAIRs']),CHAIRi_delta_pp=100*(metrics['CHAIRi']-v['CHAIRi'])) for k,v in historical.items()}
    summary=dict(status='TEST199_V2_REVIEWED_COMPLETE',evaluation_source=dict(notebook_slug='kltnnnnnff',save_version=8,owner=None,user_confirmed=True),checkpoint_source_version=result['source_version'],checkpoint_sha256=CHECKPOINT_SHA,source_commit=source['commit'],archive_sha256=cache['archive_sha256'],checks_passed=len(checks),images=5000,metrics=result['metrics'],display_x100=result['display_x100'],chair_counts=counts,runtime_policy=result['runtime_policy'],mean_words=sum(len(r['caption'].split()) for r in rows)/5000,empty_captions=sum(not r['caption'].strip() for r in rows),short_le2=sum(len(r['caption'].split())<=2 for r in rows),stage_seconds=stage['seconds'],inference_worker_seconds=max(w['seconds'] for w in workers),timing_scope='worker includes checkpoint/stage verification, beam generation and teacher NLL, excludes setup/staging/CPUmetrics; no exact totalbilling claim',same_historical_test_refs=True,historical_system_points=historical,descriptive_system_comparisons=comparisons,limits=['No local model/scorer/checkpoint/feature payload execution; stored source/seal/counts/arithmetic audit only.','BLEU/METEOR/ROUGE runtime-reported; CIDEr/SPICE perimage means and CHAIR counts replayed, no official scorer rerun.','CUDAcu128 training/cu130 inference, actual firstbatch pathguard passed; bitwise crossbuild equivalence unproven.','One seed, prior project TESTexposure; no tuning this run. Historical architectures/training/selection differ, not isolated bank/prompt effect.'])
    (HERE/'TEST199_RESULT.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');(HERE/'TEST199_REVIEW_CHECKS.json').write_text(json.dumps(dict(checks_passed=len(checks),checks=checks,models_scorers_executed_locally=False),indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in summary.items() if k not in ('historical_system_points','limits')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);audit(p.parse_args().cache)
