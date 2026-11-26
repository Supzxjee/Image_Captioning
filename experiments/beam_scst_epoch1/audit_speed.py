"""Recalculate saved timing statistics; no Torch, checkpoint or scorer execution."""
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent


def audit():
    cache=HERE/'speed_review_cache'
    read=lambda name:json.loads((cache/name).read_text(encoding='utf-8'))
    manifest=json.loads((HERE/'SPEED_SOURCE_MANIFEST.json').read_text())
    receipt=read('source_receipt.json')['sha256'];assert receipt==manifest['sha256']
    for name,digest in receipt.items():assert hashlib.sha256((cache/'source'/name).read_bytes()).hexdigest()==digest,name
    status=read('run_status.json');assert status['complete'] and status['all_optimizer_updates_discarded']
    assert not status['production_training'] and not status['test_used'] and not status['valid_evaluated']
    result=read('result.json');assert result['status']=='SCST_SPEED_PILOT_COMPLETE'
    parent=read('checkpoint_source.json');assert parent['sha256']==manifest['full_checkpoint_sha256']
    ranks=[read(f'probe_rank{i}.json') for i in (0,1)]
    positions=[64,96,40000,40032,80000,80032,113280]
    variants=['reference','candidate','candidate','reference']
    for rank,r in enumerate(ranks):
        guard=read(f'equivalence_rank{rank}.json')
        assert guard['passed'] and guard['exact_data_paths_logps_loss_and_RNG'] and guard['nonfinite_rejected']
        assert guard['full_source_sha256']==parent['sha256'] and guard['no_weights_saved']
        assert r['complete'] and r['discarded_updates']==38 and not r['training_checkpoint_saved']
        assert r['checkpoint_sha256']==parent['sha256']
        assert read(f'timing_rank{rank}.json')==dict(blocks=r['blocks'],complete=True)
        assert [b['variant'] for b in r['blocks']]==variants
        for b in r['blocks']:
            assert [x['position'] for x in b['measurements']]==positions
            assert all(x['real_images']==(7 if x['position']==113280 else 32) for x in b['measurements'])
            for x in b['measurements']:
                assert x['wall_seconds']>0 and math.isfinite(x['SCST']) and math.isfinite(x['mean_reward'])
                for family in ('GPU_event_spans_seconds','CPU_wall_seconds'):
                    assert all(math.isfinite(v) and v>=0 for v in x[family].values())
    times={v:[] for v in variants};phases={v:{} for v in variants};by_position={};samples=[]
    for b in range(4):
        variant=variants[b]
        for j,position in enumerate(positions):
            values=[r['blocks'][b]['measurements'][j] for r in ranks]
            if position==113280:continue
            seconds=max(x['wall_seconds'] for x in values)
            times[variant].append(seconds);by_position.setdefault(position,{}).setdefault(variant,[]).append(seconds)
            samples.append(dict(block=b,position=position,variant=variant,seconds=seconds))
            for family in ('GPU_event_spans_seconds','CPU_wall_seconds'):
                for key in values[0][family]:
                    phases[variant].setdefault(family+'/'+key,[]).append(max(x[family][key] for x in values))
    assert samples==result['measured_samples']
    summary={v:dict(n=len(x),mean=statistics.mean(x),median=statistics.median(x),min=min(x),max=max(x)) for v,x in times.items()}
    for variant in summary:
        assert math.isclose(summary[variant]['median'],result['median_global32_update_seconds'][variant],abs_tol=1e-12)
    gains={k:dict(reference=summary['reference'][k],candidate=summary['candidate'][k],
        time_reduction_percent=100*(1-summary['candidate'][k]/summary['reference'][k]),
        throughput_ratio=summary['reference'][k]/summary['candidate'][k],
        training_only_saved_seconds_projection=(summary['reference'][k]-summary['candidate'][k])*3541) for k in ('mean','median')}
    assert math.isclose(gains['median']['throughput_ratio'],result['ratio_reference_over_candidate'],abs_tol=1e-12)
    matched={str(p):{v:statistics.mean(x) for v,x in d.items()} for p,d in by_position.items()}
    max_loss_difference=0.
    for r in ranks:
        for j in range(7):
            for b in (1,2,3):
                assert r['blocks'][b]['measurements'][j]['mean_reward']==r['blocks'][0]['measurements'][j]['mean_reward']
                difference=abs(r['blocks'][b]['measurements'][j]['SCST']-r['blocks'][0]['measurements'][j]['SCST'])
                max_loss_difference=max(max_loss_difference,difference)
                assert difference<1e-6
    report=dict(status='SPEED_PILOT_AUDITED',date='2026-11-26',output_version=None,source_checkpoint=parent,
        timing_summary=summary,gains=gains,matched_position_mean=matched,
        phase_average={v:{k:statistics.mean(x) for k,x in d.items()} for v,d in phases.items()},
        peak_allocated_GiB=max(r['peak_allocated_bytes'] for r in ranks)/1024**3,
        verification=dict(source_SHA_files=42,two_rank_equivalence_guard_PASS=True,ABBA_samples_recomputed=24,
            tail_samples_excluded_from_global32_projection=True,mean_rewards_equal_across_blocks=True,
            timed_SCST_differences_below=1e-6,max_timed_SCST_difference=max_loss_difference,
            discarded_optimizer_steps=38,production_checkpoint_saved=False),
        recommendation='Use the guarded candidate for the next bounded epoch; modest gain, do not claim 1–2h epoch or new accuracy gain.',
        limits=['Same-session short warm-cache sample; phases overlap, GPU spans are not isolated kernel totals.',
            'One-update guard checks exact data/beam/logp/loss/RNG; gradient/parameter comparison uses tolerance, not full-epoch equivalence.',
            'No new caption-quality result, production weights or actual full-epoch timing; projections exclude setup/evaluation/quota billing.',
            'Host audit reads cached text/metadata only; no Torch/model/checkpoint/scorer run.'])
    (HERE/'SPEED_RESULT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(gains=gains,phases=report['phase_average'],peak_GiB=report['peak_allocated_GiB']),indent=2))


if __name__=='__main__':audit()
