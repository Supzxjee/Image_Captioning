"""Review metadata only. Never load Torch, features or checkpoints locally."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np

HERE=Path(__file__).resolve().parent

def audit():
    root=HERE/'ddp_sharing_review_text';read=lambda n:json.loads((root/n).read_text(encoding='utf-8'))
    result=read('result.json');source=read('source_receipt.json');checks=0
    def check(condition,message):
        nonlocal checks
        if not condition:raise AssertionError(message)
        checks+=1
    for n,h in source['sha256'].items():
        raw=subprocess.check_output(['git','show',source['commit']+':experiments/csa_paper/'+n],cwd=HERE.parents[1])
        check(hashlib.sha256(raw).hexdigest()==h,'source '+n)
    check(result['status']=='DDP_SHARING_BENCHMARK_COMPLETE','completion')
    check(read('run_status.json')==dict(complete=True,full_training_started=False,checkpoint_saved=False),'runstatus')
    check(read('order_rank0.json')==read('order_rank1.json'),'same rank orders')
    order=read('order_rank0.json')
    check(len(order['first1280'])==len(set(order['first1280']))==1280,'unique benchmark events')
    check(all(0<=i<566747 for i in order['first1280']+order['last27']) and len(order['last27'])==27,'TRAIN range/tail')
    package=json.loads((HERE/'SAVE_BANK_V130_PACKAGE.json').read_text(encoding='utf-8'))
    check(result['bank_package']==read('bank_identity.json')==package,'bank identity')
    stage=read('stage_summary.json');full_package=json.loads((HERE/'SAVE_FULL_ROI_V134_PACKAGE.json').read_text(encoding='utf-8'))
    check(stage['source_package']==full_package and result['full_cache_manifest_sha256']==full_package['files']['manifest.json'],'full cache identity')
    previous=json.loads((HERE/'full_benchmark_review_text/stage_summary.json').read_text(encoding='utf-8'))
    check(stage['files_sha256']==previous['files_sha256'] and stage['metadata']==previous['metadata'],'same staged data digests')
    preflight=read('preflight.json');check(len(preflight)==result['preflight_cases']==6,'six preflights')
    check({(p['shared'],p['case']) for p in preflight}=={(s,c) for s in (False,True) for c in ('batch32','tail27','stress32')},'preflight combinations')
    max_loss=max_rel=max_abs=0.
    for p in preflight:
        loss_error=abs(p['global_nll']-p['reference_nll']);max_loss=max(max_loss,loss_error)
        check(loss_error<=5e-5,'NLL tolerance')
        check(p['caption_events_global']==(27 if p['case']=='tail27' else 32) and p['local_captions']==16,'caption counts')
        check(p['global_tokens']>p['local_tokens']>0,'unequal token scaling')
        check(len(p['gradients'])==5,'gradient probes')
        for g in p['gradients'].values():
            check(all(math.isfinite(x) for x in g.values()) and g['reference_norm']>1e-12,'finite connected gradient')
            check(g['relative_l2']<=.005 and g['max_abs']<=3e-5,'gradient tolerance')
            max_rel=max(max_rel,g['relative_l2']);max_abs=max(max_abs,g['max_abs'])
    modes={m['mode']:m for m in result['modes']}
    check(set(modes)=={'ordinary','shared'},'both modes')
    histories={}
    for name,mode in modes.items():
        check(mode==read(name+'_result.json'),'standalone mode result')
        check((mode['world_size'],mode['global_batch'],mode['microbatch_per_rank'],mode['accumulation'])==(2,32,16,1),'batch geometry')
        check(mode['warmup_updates']==10 and mode['timed_updates']==30 and mode['precision']=='FP32','timing scope')
        steps=read(name+'_steps.json');histories[name]=steps
        check(len(steps)==40 and [s['step'] for s in steps]==list(range(1,41)),'40step sequence')
        for step in steps:
            check(step['timed']==(step['step']>10),'timed mask')
            check(step['caption_events_global']==32 and step['local_captions']==16,'full caption batch')
            check(step['encoded_images_global']==(32 if name=='ordinary' else 8),'encoded image rows')
            check(step['global_tokens']>step['local_tokens']>0,'token counts')
            check(step['seconds']>0 and all(math.isfinite(step[k]) for k in ('seconds','global_nll','grad_norm')),'finite step')
        times=np.array([s['seconds'] for s in steps[10:]])
        for key,value in [('median_seconds',np.median(times)),('p90_seconds',np.quantile(times,.9)),('mean_seconds',times.mean())]:
            check(math.isclose(mode[key],float(value),rel_tol=1e-12),'timing '+key)
        check(math.isclose(mode['train_only_hours_per_epoch'],mode['median_seconds']*17711/3600,rel_tol=1e-12),'epoch projection')
        check(math.isclose(mode['train_only_hours_15epochs'],mode['median_seconds']*17711*15/3600,rel_tol=1e-12),'15epoch projection')
        check({p['rank'] for p in mode['per_rank']}=={0,1} and mode['bank_buffers_unchanged'],'ranks/frozen')
        for p in mode['per_rank']:check(p['local_captions']==16 and p['peak_reserved_mib']>=p['peak_allocated_mib']>0,'GPU memory')
    for a,b in zip(histories['ordinary'],histories['shared']):
        check(a['global_tokens']==b['global_tokens'] and a['regions']==b['regions'] and a['prefix_length']==b['prefix_length'],'matched targets/padding')
    ordinary,shared=modes['ordinary'],modes['shared'];ratio=ordinary['median_seconds']/shared['median_seconds']
    check(math.isclose(ratio,result['speed_ratio_ordinary_over_shared'],rel_tol=1e-12),'speed ratio')
    check(not result['checkpoint_saved'] and not result['quality_metrics'] and not result['full_training_started'],'no fulltraining/quality')
    facts=dict(status='REVIEWED_DDP_SHARING_BENCHMARK_COMPLETE',zip_sha256='4a14542622cfb5f09131a1c6310a185f22de0e578375c0e7d385ec0b6ffb984c',
        source_commit=source['commit'],metadata_assertions=checks,git_hashes_verified=12,preflight_cases=6,
        max_preflight_loss_abs_error=max_loss,max_gradient_relative_l2_error=max_rel,max_gradient_abs_error=max_abs,
        ordinary_median_seconds=ordinary['median_seconds'],shared_median_seconds=shared['median_seconds'],
        sharing_throughput_gain_percent=(ratio-1)*100,sharing_time_reduction_percent=(1-1/ratio)*100,
        ordinary_train_only_hours_15epochs=ordinary['train_only_hours_15epochs'],shared_train_only_hours_15epochs=shared['train_only_hours_15epochs'],
        sharing_saved_train_minutes_15epochs=(ordinary['train_only_hours_15epochs']-shared['train_only_hours_15epochs'])*60,
        ordinary_encoder_rows_per_update=32,shared_encoder_rows_per_update=8,
        local_neural_execution=False,actual_feature_arrays_available_locally=False,quality_equivalence_measured=False,
        recommendation='TwoGPU ordinary is a conservative first CSA baseline (~7.77h train-only); sharing saves~25.5min and changesencoderdropoutcorrelation. Eitherlongrun needs explicit schedule/validation/resume/budget approval.')
    (HERE/'DDP_SHARING_RESULT.json').write_text(json.dumps(facts,indent=2)+'\n',encoding='utf-8');print(json.dumps(facts,indent=2))

if __name__=='__main__':audit()
