"""Metadata-only audit; no neural libraries, feature arrays or checkpoint loads."""
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import numpy as np

HERE=Path(__file__).resolve().parent

def audit():
    root=HERE/'full_benchmark_review_text';read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    result=read('result.json');source=read('source_receipt.json');stage=read('stage_summary.json');steps=read('steps.json');checks=0
    def check(condition,message):
        nonlocal checks
        if not condition:raise AssertionError(message)
        checks+=1
    for name,h in source['sha256'].items():
        blob=subprocess.check_output(['git','show',source['commit']+':experiments/csa_paper/'+name],cwd=HERE.parents[1])
        check(hashlib.sha256(blob).hexdigest()==h,'source '+name)
    check(result['status']=='FULL_CACHE_XE_BENCHMARK_COMPLETE','complete')
    check(read('run_status.json')==dict(complete=True,full_training_started=False,checkpoint_saved=False),'runstatus')
    check(result['bank_package']==json.loads((HERE/'SAVE_BANK_V130_PACKAGE.json').read_text(encoding='utf-8')),'bank identity')
    package=json.loads((HERE/'SAVE_FULL_ROI_V134_PACKAGE.json').read_text(encoding='utf-8'))
    check(stage['source_package']==package,'fullROI identity')
    check(result['full_cache_manifest_sha256']==package['files']['manifest.json'],'manifest identity')
    check(stage['status']=='FULL_CACHE_STAGE_COMPLETE' and stage['metadata']==dict(unique_images=118287,groups=dict(train=113287,val=5000),regions=833473,caption_events=591757,test_images=0),'stage counts')
    check(stage['shapes']==dict(features=[833473,768],offsets=[118288],ids=[118287],sequences=[591757,54],caption_image=[591757],group=[118287]),'stage shapes')
    check(set(stage['files_sha256'])=={n+'.npy' for n in stage['shapes']},'stage members')
    for h in stage['files_sha256'].values():check(bool(re.fullmatch('[0-9a-f]{64}',h)),'stageSHAformat')
    check(stage['train_caption_events']==result['train_caption_events']==566747 and stage['val_caption_events']==result['val_caption_events']==25010,'event counts')
    check((result['gpus_used'],result['effective_batch'],result['microbatch'],result['accumulation'])==(1,32,4,8),'batch/GPU')
    check(result['actual_optimizer_updates']==40 and result['warmup_updates']==10 and result['timed_updates']==30,'update counts')
    check(len(steps)==40 and [x['step'] for x in steps]==list(range(1,41)),'step records')
    all_events=[]
    for step in steps:
        check(step['timed']==(step['step']>10),'timed mask')
        check(step['seconds']>0 and math.isfinite(step['seconds']) and math.isfinite(step['nll']) and math.isfinite(step['grad_norm']) and step['grad_norm']>0,'finite step')
        check(len(step['event_indices'])==32 and all(type(i)==int and 0<=i<566747 for i in step['event_indices']),'TRAIN event range')
        all_events.extend(step['event_indices'])
        check(len(step['micro_shapes'])==8 and all(s['batch']==4 and 1<=s['regions']<=87 and 1<=s['prefix_length']<=53 for s in step['micro_shapes']),'micro shapes')
        check(32<=step['target_tokens']<=sum(s['batch']*s['prefix_length'] for s in step['micro_shapes']),'target count bounds')
    check(len(set(all_events))==1280,'unique benchmark events')
    timings=np.array([x['seconds'] for x in steps[10:]])
    for key,value in [('median',np.median(timings)),('p90',np.quantile(timings,.9)),('mean',timings.mean())]:
        check(math.isclose(result['train_step_seconds'][key],float(value),rel_tol=1e-12),'timing '+key)
    check(result['stress']==read('stress.json') and result['stress']['optimizer_steps']==0,'stress noopt')
    check(result['stress']['micro_shapes'][0]['regions']==87 and result['stress']['micro_shapes'][0]['prefix_length']==50,'actual stress shape')
    check(result['bank_buffers_unchanged'] and not result['checkpoint_saved'] and not result['quality_metrics'] and not result['grounding_accuracy'],'scope/frozen')
    check(read('runtime.json')['used_gpu']==0 and read('runtime.json')['precision']=='FP32,noAMP,noTF32','runtime')
    updates=math.ceil(566747/32);median=float(np.median(timings));projection=result['projection']
    check(result['updates_per_full_epoch']==updates==17711,'full epoch updates')
    check(math.isclose(projection['train_only_hours_1epoch'],median*updates/3600,rel_tol=1e-12),'epoch projection')
    check(math.isclose(projection['train_only_hours_15epochs'],median*updates*15/3600,rel_tol=1e-12),'15epoch projection')
    check(math.isclose(projection['p90_train_only_hours_15epochs'],float(np.quantile(timings,.9))*updates*15/3600,rel_tol=1e-12),'p90 projection')
    teacher=result['teacher_probe'];beam=result['beam_probe']
    check(teacher['images']==32 and teacher['caption_events']==160 and len(beam['outputs'])==beam['images']==32,'val probe counts')
    check(math.isclose(projection['teacher_seconds_per_full_val'],teacher['seconds']/160*25010,rel_tol=1e-12),'val teacher projection')
    check(math.isclose(projection['beam_seconds_per_full_val'],beam['seconds']/32*5000,rel_tol=1e-12),'beam projection')
    check(len({x['coco_id'] for x in beam['outputs']})==32 and all(x['tokens']==0 and x['eos'] for x in beam['outputs']),'empty best-beam outputs')
    check(stage['seconds']==result['stage_seconds'] and result['worker_seconds']>stage['seconds'],'worker duration')
    check(not (root/'failure.txt').exists(),'no failure')
    facts=dict(status='REVIEWED_FULL_CACHE_BENCHMARK_COMPLETE',zip_sha256='ae4bb5a8b371729be1f26e95d27f84ed7a5fe8bcca9fb929bcee595609f95501',
        source_commit=source['commit'],metadata_assertions=checks,git_hashes_verified=11,input_array_validation_passed_by_user_worker=True,
        stage_arrays_independently_available=False,device=read('runtime.json')['device'],gpus_used=1,precision='FP32',effective_batch=32,microbatch=4,accumulation=8,
        median_step_seconds=median,p90_step_seconds=float(np.quantile(timings,.9)),updates_per_epoch=updates,
        train_only_hours_per_epoch=projection['train_only_hours_1epoch'],train_only_hours_15epochs=projection['train_only_hours_15epochs'],
        scenario_hours_15epochs_including_probed_val=projection['train_only_hours_15epochs']+15*(projection['teacher_seconds_per_full_val']+projection['beam_seconds_per_full_val'])/3600,
        training_memory=result['training_memory'],stage_seconds=result['stage_seconds'],worker_seconds=result['worker_seconds'],
        beam_probe_empty_captions=32,beam_executed_decoder_steps_recorded=False,checkpoint_saved=False,local_model_execution=False,
        recommendation='Do not schedule15epochs under30h from thisconfiguration. Measure microbatch16/32 with effectivebatch32 and sameFP32/XE before addingDDP/AMP. Not yet a new authorizedrun.')
    (HERE/'FULL_BENCHMARK_RESULT.json').write_text(json.dumps(facts,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(facts,indent=2))

if __name__=='__main__':audit()
