"""Audit cached user-run reports using metadata/NumPy only; never run a model."""
import json
from pathlib import Path
import numpy as np
from gain_data import BASE_SHA,POLICY,load_metadata,save

HERE=Path(__file__).resolve().parent


def main():
    loaded={a:json.loads((HERE/f'review/{a}_loaded.json').read_text(encoding='utf-8')) for a in 'CHT'}
    read=lambda a,n:json.loads(loaded[a]['texts'][n])
    summaries={a:read(a,'summary.json') for a in 'CHT'}
    meta=load_metadata(HERE/'data/gain_metadata.json.gz');expected={r['image_id']:r for r in meta['groups']['val']}
    predictions={};cider={};diagnostics={};workers={}
    initial=read('C','worker_rank0.json')['initial_head_sha256'];runtime=read('C','runtime_rank0.json')
    baseline={r['image_id']:r['caption'] for r in meta['baseline']}
    for arm,s in summaries.items():
        assert s['status']=='COMPLETE_WAIT_MATCHED_CHT' and s['arm']==arm and s['updates']==500 and not s['test_used']
        assert s['policy']==POLICY and loaded[arm]['archive_source_sha256']==s['sources']
        for field in ('sources','preparation_receipt_sha256','policy'):assert s[field]==summaries['C'][field]
        workers[arm]=[read(arm,f'worker_rank{r}.json') for r in (0,1)]
        histories=[]
        for rank,w in enumerate(workers[arm]):
            assert w['status']=='COMPLETE' and w['updates']==500 and w['initial_head_sha256']==initial
            assert w['initial_base_sha256']==w['final_base_sha256']==BASE_SHA
            assert w['final_head_sha256']==workers[arm][0]['final_head_sha256']
            assert read(arm,f'runtime_rank{rank}.json')==runtime
            pre=read(arm,f'preflight_rank{rank}.json');assert pre['status']=='PASS' and all(v is True for k,v in pre.items() if k!='status')
            history=[json.loads(line) for line in loaded[arm]['texts'][f'history_rank{rank}.jsonl'].splitlines()]
            assert [x['step'] for x in history]==list(range(1,501))
            for row in history:
                assert row['epoch']==(row['step']-1)//250
                assert np.isfinite([row[k] for k in ('loss','scst','kl','aux','sample_reward','greedy_reward','grad_norm','seconds')]).all()
                assert np.isclose(row['loss'],row['scst']+.1*row['kl']+.1*row['aux'],atol=2e-5,rtol=1e-5)
                for r in row['auxiliary']:
                    assert np.isfinite(list(r.values())).all()
                    assert np.isclose(r['delta'],r['cg']+r['cu'],atol=1e-6)
                    assert np.isclose(r['cancel'],max(abs(r['cg'])+abs(r['cu'])-2*max(r['delta'],0),0),atol=1e-6)
                    assert np.isclose(r['direction'],max(.8*max(r['delta'],0)-r['cg'],0),atol=1e-6)
                    assert r['residual']<=1e-5+1e-4*abs(r['delta'])
            histories.append(history)
        for i in range(500):
            count=sum(len(h[i]['auxiliary']) for h in histories)
            for h in histories:
                values=h[i]['auxiliary'];aux=sum(r['gain']+r['cancel']+(r['rank'] if arm=='H' else r['direction'] if arm=='T' else 0) for r in values)*2/max(count,1)
                assert np.isclose(h[i]['aux'],aux,atol=2e-5,rtol=1e-5)
            assert histories[0][i]['grad_norm']==histories[1][i]['grad_norm']
        tail=[r for h in histories for x in h[-100:] for r in x['auxiliary']]
        positive=[r for r in tail if r['delta']>1e-6]
        diagnostics[arm]=dict(scope='Matched update401–500 training minibatches, not held-out or final fixed-model evaluation.',
            groups=len(tail),positive_delta_groups=len(positive),mean_delta=float(np.mean([r['delta'] for r in tail])),
            mean_cg=float(np.mean([r['cg'] for r in tail])),mean_direction_penalty=float(np.mean([r['direction'] for r in tail])),
            positive_delta_support_share_median=float(np.median([r['cg']/r['delta'] for r in positive])),
            positive_delta_meeting_80percent_support=sum(r['cg']>=.8*r['delta'] for r in positive),
            maximum_recorded_completeness_residual=max(r['residual'] for h in histories for x in h for r in x['auxiliary']))
        for condition in ('A','end'):
            key='A' if condition=='A' else arm
            rows=read(arm,f'{condition}_predictions.json');per=read(arm,f'{condition}_per_image_cider.json')
            assert len(rows)==len(per)==1000 and {r['image_id'] for r in rows}==set(expected)=={r['image_id'] for r in per}
            assert all(r['coco_id']==expected[r['image_id']]['coco_id'] for r in rows)
            pred={r['image_id']:r['caption'] for r in rows};values={r['image_id']:r['CIDEr'] for r in per}
            assert abs(np.mean(list(values.values()))-s['metrics'][condition]['CIDEr'])<1e-12
            if condition=='A':assert pred==baseline
            predictions[key]=pred;cider[key]=values
            chair=read(arm,f'{condition}_chair_output.json')['sentences']
            assert len(chair)==1000 and {r['image_id'] for r in chair}=={r['coco_id'] for r in expected.values()}
            counts=dict(hallucinated_captions=sum(bool(r['mscoco_hallucinated_words']) for r in chair),hallucinated_mentions=sum(len(r['mscoco_hallucinated_words']) for r in chair),
                object_mentions=sum(len(r['mscoco_generated_words']) for r in chair),recalled_unique=sum(len(set(r['mscoco_generated_words'])&set(r['mscoco_gt_words'])) for r in chair),reference_unique=sum(len(set(r['mscoco_gt_words'])) for r in chair))
            assert counts==s['metrics'][condition]['counts']
    pairs=[('C','A'),('H','A'),('T','A'),('T','C'),('T','H'),('H','C')]
    ids=sorted(expected);delta={f'{a}-{b}':np.array([cider[a][i]-cider[b][i] for i in ids]) for a,b in pairs}
    rng=np.random.default_rng(42);boot={k:[] for k in delta}
    for _ in range(20):
        indices=rng.integers(0,1000,size=(500,1000))
        for k,d in delta.items():boot[k].extend(d[indices].mean(1).tolist())
    comparisons={k:dict(delta=float(v.mean()),ci95=np.quantile(boot[k],[.025,.975]).tolist(),
        images_higher=int((v>0).sum()),images_lower=int((v<0).sum()),images_equal=int((v==0).sum())) for k,v in delta.items()}
    changed={f'{a}-{b}':sum(predictions[a][i]!=predictions[b][i] for i in ids) for a,b in pairs}
    metrics=dict(A=summaries['C']['metrics']['A'],**{a:s['metrics']['end'] for a,s in summaries.items()})
    t=metrics['T'];a=metrics['A']
    guards=dict(cider_gain_at_least_point01=comparisons['T-A']['delta']>=.01,
        ci_T_A_positive=comparisons['T-A']['ci95'][0]>0,ci_T_C_positive=comparisons['T-C']['ci95'][0]>0,
        ci_T_H_positive=comparisons['T-H']['ci95'][0]>0,point_T_H_positive=comparisons['T-H']['delta']>0,
        chair_s_not_worse=t['chair']['CHAIRs']<=a['chair']['CHAIRs'],chair_i_not_worse=t['chair']['CHAIRi']<=a['chair']['CHAIRi'],recall_guard=t['chair']['Recall']>=a['chair']['Recall']-.005)
    result=dict(date='2026-10-28',status='COMPLETED_QUALITY_NO_GO' if not all(guards.values()) else 'EXPLORATORY_CONTINUATION_GATE_PASS',
        zip_sha256={a:x['zip_sha256'] for a,x in loaded.items()},preparation_receipt_sha256=summaries['C']['preparation_receipt_sha256'],
        audit=dict(source_payloads=14,updates_per_arm=500,ranks=2,common_initial_head_sha256=initial,base_unchanged_reported=True,
            baseline_captions_exact=True,runtime_exact=True,loss_and_auxiliary_scaling_checked=True,chair_counts_reaggregated=True),
        metrics=metrics,comparisons=comparisons,changed_captions=changed,guards=guards,training_diagnostics=diagnostics,
        bootstrap=dict(resamples=10000,seed=42,unit='paired development image',ci='percentile95',shared_draws=True),
        limits=['One training seed and repeatedly reused development1000; image bootstrap does not measure seed variance or selection bias.',
            'Training attribution diagnostics are not held-out grounding/factuality evidence; gradient dominance cannot be inferred from scalar loss size.',
            'H-A and H-C are secondary exploratory contrasts; percentile intervals are not multiplicity-adjusted.',
            'Small reviews do not contain physical checkpoint files; state identity is reported by verified-source workers, not independently rehashed locally.'],
        recommendation='Retain A. Do not scale/sweep/addseed to rescue T recipe. H point gain is descriptive, not a confirmed new best or a novel method.')
    save(HERE/'RESULT.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
