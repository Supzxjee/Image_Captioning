"""Replay archived sharing metrics/source/NumPy event plans, never neural/scorer runs."""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from csa_sharing_data import benchmark_orders, image_plan, local_events

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def audit(path):
    cache=json.loads(Path(path).read_text(encoding='utf-8'));files=cache['files'];get=lambda n:json.loads(files[n]);checks=[]
    def check(n,v):
        if not v:raise AssertionError(n)
        checks.append(n)
    r=get('result.json');source=get('source_receipt.json');status=get('run_status.json')
    check('complete benchmark only',r['status']=='CSA_FEATURE_SHARING_ACCELERATE_BENCHMARK_COMPLETE' and status['complete'] and not r['quality_evaluated'] and not r['full_training_complete'] and not r['checkpoint_saved'] and not r['test_used'] and not r['validation_used'])
    check('fixed recipe',r['world_size']==2 and r['global_batch']==32 and r['precision']=='FP32' and r['framework']=='Accelerate' and r['accelerate_version']=='1.15.0' and not r['prepared_dataloader'] and r['total_optimizer_updates']==320)
    check('source pins',source['commit']=='fa9abcc2b1cc862c44dc7ad1233fea1b8247e197' and len(source['sha256'])==31)
    for n,h in source['sha256'].items():check('source '+n,hashlib.sha256(subprocess.check_output(['git','show',source['commit']+':'+n],cwd=ROOT)).hexdigest()==h)
    for rank in (0,1):
        rt=get(f'accelerate_runtime_rank{rank}.json');cu=get(f'cuda_runtime_rank{rank}.json')
        check(f'rank{rank} actual runtime',rt['accelerate']=='1.15.0' and rt['mixed_precision']=='no' and rt['gradient_accumulation_steps']==1 and rt['processes']==2 and rt['distributed_type']=='DistributedType.MULTI_GPU' and rt['manual_caption_sharding'] and not rt['prepared_dataloader'] and cu['cuda_available'] and cu['device_count']==2 and cu['nccl_available'])
    previous=json.loads((ROOT.parent/'.review_cache/csa_structure_smoke_v1/archive_text.json').read_text(encoding='utf-8'))['files']
    old=lambda n:json.loads(previous[n]);stage=get('stage_summary.json')
    check('same selected raw cache/profile/tokenized captions',all(stage[k]==old('stage_summary.json')[k] for k in ('feature_profile','files_sha256','raw_selected_row_sha256')))
    check('same semantic payload/provenance',get('semantic_stage.json')==old('semantic_stage.json'))
    mapping=np.repeat(np.arange(256,dtype=np.int32),5);buffer=io.BytesIO();np.save(buffer,mapping)
    check('reconstructed caption-image NPY seal',hashlib.sha256(buffer.getvalue()).hexdigest()==stage['files_sha256']['caption_image.npy'])
    orders=benchmark_orders(mapping);parity=get('parity.json')
    check('six numerical cases',len(parity)==r['parity_cases']==6 and [(p['case'],p['shared']) for p in parity]==[(case,shared) for case in ('grouped32','tail27','random32') for shared in (False,True)])
    cases={'grouped32':orders['grouped'][:32],'tail27':orders['grouped'][-27:],'random32':orders['random'][:32]}
    diagnostic_count=0
    for p in parity:
        key=p['case']+'/'+str(p['shared']);events=cases[p['case']]
        check(key+' pass/forward normalization',p['passed'] and p['error'] is None and abs(p['global_NLL']-p['reference_NLL'])<=5e-5 and abs(p['reference_micro_NLL']-p['reference_NLL'])<=5e-5 and p['captions']==len(events))
        check(key+' actual encoder rows',p['encoded_images_global']==sum(len(image_plan(mapping,local_events(events,rank),p['shared'])[0]) for rank in (0,1)))
        ds=p['gradient_diagnostics'];diagnostic_count+=len(ds)
        for name,d in ds.items():
            norm=d['reference_norm'];absolute=d['max_abs_error'];limit=3e-7 if norm<=1e-7 else 3e-7+.005*d['reference_max_abs']
            if not (d['passed'] and math.isfinite(d['relative_l2']) and absolute<=limit and (norm<=1e-7 or d['relative_l2']<=.005) and math.isclose(limit,d['max_abs_limit'],rel_tol=1e-12) and math.isclose(abs(d['actual_at_max']-d['reference_at_max']),absolute,abs_tol=1e-15)):
                raise AssertionError(key+' per-parameter diagnostic '+name)
        check(key+' all gradient scalar diagnostics replay',bool(ds))
        check(key+' maximum errors replay',p['max_relative_gradient_error']==max(d['relative_l2'] for d in ds.values() if d['reference_norm']>1e-7) and p['max_absolute_gradient_error']==max(d['max_abs_error'] for d in ds.values()))
    passes=get('pass_summaries.json');check('eight passes and result parity',passes==r['passes'] and len(passes)==8)
    by_order={}
    for order_name,order in orders.items():
        mode_times={'ordinary':[],'shared':[]};mode_peaks={'ordinary':[],'shared':[]};encoder={'ordinary':[],'shared':[]};target_tokens=None
        for pass_id,mode in enumerate(('ordinary','shared','shared','ordinary')):
            label=f'{order_name}_{pass_id}_{mode}';history=get(label+'_steps.json');summary=next(s for s in passes if s['order']==order_name and s['pass_id']==pass_id)
            check(label+' complete and finite',len(history)==40 and [h['step'] for h in history]==list(range(1,41)) and all(h['timed']==(i>=10) and h['caption_events_global']==32 and h['target_tokens']>0 and math.isfinite(h['NLL']) and h['NLL']>0 and math.isfinite(h['seconds']) and h['seconds']>0 and math.isfinite(h['grad_norm']) for i,h in enumerate(history)))
            encoded=[sum(len(image_plan(mapping,local_events(ev,rank),mode=='shared')[0]) for rank in (0,1)) for ev in order.reshape(40,32)]
            check(label+' exact per-step encoder rows',[h['encoded_images_global'] for h in history]==encoded)
            tokens=[h['target_tokens'] for h in history]
            if target_tokens is None:target_tokens=tokens
            check(label+' same caption-token denominators',tokens==target_tokens)
            times=[h['seconds'] for h in history[10:]];rows=[h['encoded_images_global'] for h in history[10:]]
            check(label+' summary arithmetic',math.isclose(summary['median_seconds'],float(np.median(times)),rel_tol=1e-12) and math.isclose(summary['p90_seconds'],float(np.quantile(times,.9)),rel_tol=1e-12) and summary['mean_encoded_images_global']==float(np.mean(rows)) and summary['updates']==40 and summary['warmup']==10 and summary['timed_updates']==30)
            mode_times[mode]+=times;mode_peaks[mode].append(summary['peak_allocated_GiB']);encoder[mode]+=rows
        modes={}
        for mode in mode_times:
            m=dict(median_seconds=float(np.median(mode_times[mode])),p90_seconds=float(np.quantile(mode_times[mode],.9)),peak_allocated_GiB=max(mode_peaks[mode]),mean_encoded_images_global=float(np.mean(encoder[mode])))
            check(order_name+'/'+mode+' pooled arithmetic',m==r['by_order'][order_name]['modes'][mode]);modes[mode]=m
        speed=modes['ordinary']['median_seconds']/modes['shared']['median_seconds']
        check(order_name+' speed ratio replay',speed==r['by_order'][order_name]['speedup_ordinary_over_shared'])
        by_order[order_name]=dict(modes=modes,speedup=speed,update_time_reduction_percent=100*(1-1/speed),peak_memory_reduction_percent=100*(1-modes['shared']['peak_allocated_GiB']/modes['ordinary']['peak_allocated_GiB']))
    summary=dict(status='CSA_SHARING_ACCELERATE_V3_REVIEWED_COMPLETE',evaluation_source=dict(notebook_slug='kltnnnnnff',save_version=None,version_user_confirmed=False),archive_sha256=cache['archive_sha256'],source_commit=source['commit'],checks_passed=len(checks),per_parameter_diagnostics_replayed=diagnostic_count,
        parity_cases=6,max_relative_gradient_error=max(p['max_relative_gradient_error'] for p in parity),max_absolute_gradient_error=max(p['max_absolute_gradient_error'] for p in parity),by_order=by_order,
        grouped_shared_vs_random_ordinary_speedup=by_order['random']['modes']['ordinary']['median_seconds']/by_order['grouped']['modes']['shared']['median_seconds'],
        train_only_epoch_hours_arithmetic=by_order['grouped']['modes']['shared']['median_seconds']*17711/3600,train_only10_hours_arithmetic=by_order['grouped']['modes']['shared']['median_seconds']*177110/3600,
        no_native_DDP_vs_Accelerate_comparison=True,quality_evaluated=False,full_training_complete=False,checkpoint_saved=False,test_used=False,validation_used=False,
        limits=['Text/source/seal/NumPy arithmetic audit; no local Torch/Accelerate/model/scorer/Kaggle API. Raw gradient and feature tensors not in ZIP; diagnostics replay from saved scalars only.',
        'Tiny TRAIN256 pool and short benchmark, not full-data speed or end-to-end ETA/billing. No new validated caption metric.',
        'Grouping changes sampler and shared encoder dropout correlations; eval parity does not prove equal training trajectory or final CIDEr/CHAIR.',
        'All modes here use Accelerate: speed difference is sharing within sampler, not evidence Accelerate beats native DDP.'])
    (HERE/'CSA_SHARING_ACCELERATE_V3_RESULT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (HERE/'CSA_SHARING_ACCELERATE_V3_REVIEW_CHECKS.json').write_text(json.dumps(dict(checks_passed=len(checks),checks=checks,per_parameter_diagnostics_replayed=diagnostic_count,models_run_locally=False),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);audit(p.parse_args().cache)
