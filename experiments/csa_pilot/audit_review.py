"""Audit saved CSA/A artifacts and paired scores; no model or metric reruns."""
import ast
import base64
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile
import zlib
import numpy as np
from phase_policy import contract,schedule,ADAM

HERE=Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load(path):
    archive_sha=sha(path.read_bytes())
    folder=HERE/'review'
    folder.mkdir(exist_ok=True)
    cache=folder/'loaded_CSA.json'
    if cache.is_file():
        old=json.loads(cache.read_text(encoding='utf-8'))
        if old['archive_sha256']==archive_sha:
            return old
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))
        raw={n:z.read(n) for n in z.namelist()}
    result=dict(archive_sha256=archive_sha,archive_bytes=path.stat().st_size,
                json={n:json.loads(v) for n,v in raw.items() if n.endswith('.json')},
                hashes={n:sha(v) for n,v in raw.items()},
                sources={n:v.decode('utf-8') for n,v in raw.items() if n.endswith('.py')},
                log=raw['notebook.log'].decode('utf-8',errors='replace'))
    cache.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8')
    return result


def index(rows,key):
    result={r['image_id']:r[key] for r in rows}
    assert len(result)==len(rows)==1000
    return result


def paired(left,right,ids):
    delta=np.asarray([left[i]-right[i] for i in ids],dtype=float)
    rng=np.random.default_rng(42)
    boot=np.concatenate([delta[rng.integers(0,len(ids),size=(500,len(ids)))].mean(axis=1) for _ in range(40)])
    return dict(delta=float(delta.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),n=len(ids),
                bootstrap_resamples=20000,seed=42,
                scope='Paired reused-development images; not seed variance or correction for repeated development use.')


def audit(path):
    saved=load(path)
    d=saved['json']
    cfg,summary=d['protocol.json'],d['run_summary.json']
    assert saved['hashes']['protocol.json']==summary['protocol_sha256']
    assert cfg['contract_sha256']==contract(cfg)
    nb=json.loads((HERE/'CSA_TWO_PHASE_PILOT_2GPU.ipynb').read_text(encoding='utf-8'))
    tree=ast.parse(''.join(nb['cells'][2]['source']))
    packed=next(n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call)
                and isinstance(n.func,ast.Attribute) and n.func.attr=='b64decode')
    sources=json.loads(zlib.decompress(base64.b64decode(packed)))
    assert set(sources)==set(cfg['implementation_sha256'])
    for name,source in sources.items():
        assert saved['hashes']['source/'+name]==cfg['implementation_sha256'][name]
        assert saved['sources']['source/'+name]==source,name
    assert saved['hashes']['reference/csa_net.py']=='d6bc6d8072c03cb7d42bc8ffcc48e48cefb03b1b128e081b608902d845b6ded1'
    assert d['groups.json']==cfg['groups']
    assert saved['hashes']['train_object_targets.json']==cfg['object_targets_sha256']
    assert not cfg['resume_phase1'] and not summary['test_used']
    a=json.loads((HERE.parent/'vg_region_caption/review/loaded_A.json').read_text(encoding='utf-8'))
    ad=a['json']
    ap=ad['protocol.json']
    assert cfg['groups']==ad['groups.json'] and cfg['cohort_sha256']==ap['cohort_sha256']
    assert d['train_object_targets.json']==ad['train_object_targets.json']
    assert cfg['input_sha256']['checkpoint']==ap['source_checkpoint_sha256']
    for key in ('checkpoint','legacy_prompt','visual_cache','dataset_json','detections'):
        assert cfg[key]==ap[key],key
    # Historical A predates a logging-only hook; verify the baseline module files are identical.
    files=['captioning/models.py','captioning/tokenizer.py','captioning/checkpoints.py',
           'captioning/scst.py','captioning/object_reward.py','captioning/object_consistency.py','captioning/visual_cache.py']
    diff=subprocess.check_output(['git','diff','--name-only',ap['project_commit'],cfg['project_commit'],'--',*files],cwd=HERE.parents[1],text=True)
    assert not diff.strip(),diff
    for name in ('pilot_data.py','pilot_model.py','hidden_data.py','captioning/scst_runtime.py','hidden_metric_smoke.py'):
        assert saved['hashes']['source/'+name]==a['source_hashes']['source/'+name],name
    hook="        if hasattr(model, 'training_diagnostics'):\n            record.update(model.training_diagnostics())\n"
    assert saved['sources']['source/run_pilot.py'].replace(hook,'')==a['source_loop']
    assert summary['initial_base_sha256']==ad['run_summary.json']['initial_base_sha256']
    assert summary['evaluation_reference_tokens_sha256']==ad['run_summary.json']['evaluation_reference_tokens_sha256']
    for name in ('origin_predictions.json','origin_per_image_cider.json','origin_chair_output.json'):
        assert d[name]==ad[name],name
    histories={}
    runtimediff={}
    for phase in (1,2):
        workers=[d[f'phase{phase}_complete_rank{r}.json'] for r in (0,1)]
        for key in ('initial_base_sha256','initial_full_sha256','final_base_sha256','final_full_sha256','contract_sha256'):
            assert workers[0][key]==workers[1][key],(phase,key)
        for rank,w in enumerate(workers):
            assert w['status']=='COMPLETE' and w['updates']==1250 and w['phase']==phase
            assert w['banks_unchanged'] and w['optimizer_started_empty'] and not w['test_used']
            assert w['contract_sha256']==cfg['contract_sha256']
            assert w['trainable_core_parameters']==23161644
            assert w['trainable_base_parameters']==(0 if phase==1 else 33520400)
            assert w['baseline_unchanged']==(phase==1)
            if phase==1:
                assert w['initial_base_sha256']==w['final_base_sha256']==summary['initial_base_sha256']
            else:
                assert w['initial_full_sha256']==d[f'phase1_complete_rank{rank}.json']['final_full_sha256']
                assert w['final_full_sha256']==summary['final_full_sha256']
                transition=d[f'phase2_transition_rank{rank}.json']
                assert all(transition[k] for k in ('baseline_unchanged','phase1_final_equals_phase2_initial','optimizer_reset','banks_unchanged'))
                assert transition['rank_seed']==42+rank and transition['sampler_epoch']==0
            order=d[f'phase{phase}_order_rank{rank}.json']
            assert len(order)==len(set(order))==5000
            assert order==d[f'phase1_order_rank{rank}.json']
            start=d[f'phase{phase}_training_start_rank{rank}.json']
            assert start['rank_seed']==42+rank and start['sampler_seed']==42 and start['sampler_epoch']==0 and start['optimizer_states']==0
            ident=lambda v:sha(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
            assert start['sampler_order_sha256']==ident(order)
            if phase==2:
                assert start['cpu_rng_sha256']==d[f'phase1_training_start_rank{rank}.json']['cpu_rng_sha256']
                assert start['cuda_rng_sha256']==d[f'phase1_training_start_rank{rank}.json']['cuda_rng_sha256']
            options=d[f'phase{phase}_optimizer_rank{rank}.json']
            assert len(options)==phase
            for k,v in ADAM.items():
                assert all(g[k]==v for g in options),(phase,k)
            if phase==2:
                assert options[0]==ad[f'optimizer_rank{rank}.json'][0]
                assert options[1]['lr']==2e-5
            else:
                assert math.isclose(options[0]['lr'],1e-6,rel_tol=1e-12)
            runtime=d[f'phase{phase}_runtime_rank{rank}.json']
            aruntime=ad[f'runtime_rank{rank}.json']
            runtimediff[f'phase{phase}_rank{rank}']=[k for k in set(runtime)|set(aruntime) if runtime.get(k)!=aruntime.get(k)]
            assert not runtimediff[f'phase{phase}_rank{rank}']
            h=d[f'phase{phase}_history_rank{rank}.json']
            histories[phase,rank]=h
            assert [r['step'] for r in h]==list(range(1,1251))
            for r in h:
                assert all(math.isfinite(v) for v in r.values() if isinstance(v,(float,int)))
                policy=schedule(phase,r['step'])
                assert all(math.isclose(r[k],v,abs_tol=1e-14,rel_tol=1e-12) for k,v in policy.items())
                assert abs(r['loss']-(r['ce']+r['scst_weight']*r['scst']))<2e-6
                assert r['core_grad_norm']>0 and r['base_grad_norm']>=0
                if phase==1:
                    assert r['base_grad_norm']==0
                assert abs(r['clip_coefficient']-min(1.,1./(r['global_grad_norm']+1e-6)))<1e-12
    train_ids={r['image_id'] for r in cfg['groups']['train']}
    for phase in (1,2):
        orders=[set(d[f'phase{phase}_order_rank{r}.json']) for r in (0,1)]
        assert not orders[0]&orders[1] and orders[0]|orders[1]==train_ids
    assert d['phase2_transition_rank0.json']['phase1_checkpoint_sha256']==d['phase2_transition_rank1.json']['phase1_checkpoint_sha256']
    metrics={'A':ad['run_summary.json']['metrics']['on']}
    scores={'A':index(ad['on_per_image_cider.json'],'CIDEr')}
    captions={'A':index(ad['on_predictions.json'],'caption')}
    ids=sorted(scores['A'])
    for label,condition in [('CSA_on','on'),('CSA_off','off'),('original','origin')]:
        m=summary['metrics'][condition]
        metrics[label]=m
        scores[label]=index(d[f'{condition}_per_image_cider.json'],'CIDEr')
        captions[label]=index(d[f'{condition}_predictions.json'],'caption')
        assert sorted(scores[label])==sorted(captions[label])==ids
        assert abs(np.mean(list(scores[label].values()))-m['CIDEr'])<1e-12
        rankrows=sum((d[f'{condition}_rank{r}.json'] for r in (0,1)),[])
        assert index(rankrows,'caption')==captions[label]
        sentences=d[f'{condition}_chair_output.json']['sentences']
        assert len(sentences)==1000 and {s['image_id'] for s in sentences}=={r['coco_id'] for r in cfg['groups']['val']}
        c=dict(images=1000,hallucinated_captions=sum(bool(s['mscoco_hallucinated_words']) for s in sentences),
               hallucinated_mentions=sum(len(s['mscoco_hallucinated_words']) for s in sentences),
               object_mentions=sum(len(s['mscoco_generated_words']) for s in sentences),
               recalled_unique_objects=sum(len(set(s['mscoco_generated_words'])&set(s['mscoco_gt_words'])) for s in sentences),
               reference_unique_objects=sum(len(set(s['mscoco_gt_words'])) for s in sentences))
        assert c==m['counts']
        for k,v in dict(CHAIRs=c['hallucinated_captions']/1000,CHAIRi=c['hallucinated_mentions']/c['object_mentions'],
                        Recall=c['recalled_unique_objects']/c['reference_unique_objects']).items():
            assert abs(v-m['chair'][k])<1e-12
    comparisons={l+'_minus_'+r:paired(scores[l],scores[r],ids) for l,r in
                 [('CSA_on','A'),('CSA_off','A'),('CSA_on','CSA_off'),('CSA_on','original'),('CSA_off','original')]}
    checks={}
    for label in ('CSA_on','CSA_off'):
        m,a_metric=metrics[label],metrics['A']
        comparison=comparisons[label+'_minus_A']
        checks[label]=dict(cider_gain_at_least_001=comparison['delta']>=.01,paired_ci_lower_positive=comparison['ci95'][0]>0,
                          beats_original=m['CIDEr']>metrics['original']['CIDEr'],chair_s_not_increased=m['chair']['CHAIRs']<=a_metric['chair']['CHAIRs'],
                          chair_i_not_increased=m['chair']['CHAIRi']<=a_metric['chair']['CHAIRi'],recall_drop_at_most_001=m['chair']['Recall']>=a_metric['chair']['Recall']-.01)
    changes={l+'_vs_'+r:sum(captions[l][i]!=captions[r][i] for i in ids) for l,r in [('CSA_on','CSA_off'),('CSA_on','A'),('CSA_off','A')]}
    assert changes['CSA_on_vs_CSA_off']==summary['sensitivity']['off_changed_captions']
    all_history=[r for h in histories.values() for r in h]
    assert 'Traceback (most recent call last)' not in saved['log']
    decision='NUMERIC_GATE_PASS_WAIT_BLIND_REVIEW' if all(checks['CSA_on'].values()) else 'STOP_CURRENT_CSA_RECIPE_NO_FULL_TRAIN'
    return dict(status='CSA_TWO_PHASE_SAVED_ARTIFACT_AUDIT_PASS',decision=decision,date='2026-10-22',
                archive_sha256=saved['archive_sha256'],archive_bytes=saved['archive_bytes'],metrics=metrics,comparisons=comparisons,
                gate_checks=checks,caption_changes=changes,source_files_verified=len(sources),runtime_differences_vs_A=runtimediff,
                phase1_baseline_frozen_pass=True,phase_transition_pass=True,sampler_reset_pass=True,optimizer_reset_pass=True,
                identical_original_predictions_scores_chair_vs_A=True,
                timing=dict(preparation_seconds=cfg['preparation_seconds'],
                            phase1_training_seconds=max(d[f'phase1_complete_rank{r}.json']['training_seconds'] for r in (0,1)),
                            phase2_training_seconds=max(d[f'phase2_complete_rank{r}.json']['training_seconds'] for r in (0,1)),
                            scoring_seconds=summary['scoring_seconds']),
                peak_allocated_gib=max(r['peak_allocated_bytes'] for r in all_history)/2**30,
                peak_reserved_gib=max(r['peak_reserved_bytes'] for r in all_history)/2**30,
                clipping={str(p):dict(active_rank_steps=sum(r['clip_coefficient']<1 for k,h in histories.items() if k[0]==p for r in h),
                                    total_rank_steps=2500) for p in (1,2)},
                test_used=False,blind_review_done=False,checkpoint_sha256_worker_reported=summary['checkpoint_sha256'],
                limits=['CSA received1250 extra core-only updates; system comparison not matched compute/capacity.',
                        'Off is the jointly adapted baseline after training, not A; it is secondary diagnostic, not a preselected winning arm.',
                        'Runtime/source/seed/sampler policy match A, but historical A did not record actual order/RNG start; no independent order-byte verification against A.',
                        'Stage file layouts differ; absent tensor/checkpoint bytes cannot be rehashed here. Same original predictions/per-image metrics and train targets are verified.',
                        'One seed, repeatedly reused dev, no convergence or all-CSA impossibility conclusion.',
                        'On/off measures removal after coadaptation; no proof of causal linguistic shortcut or calibrated visual grounding.',
                        'CHAIR concerns80COCOobject categories, not relation truth. No new human factuality review.'])


if __name__=='__main__':
    result=audit(Path(sys.argv[1]))
    (HERE/'PILOT_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
