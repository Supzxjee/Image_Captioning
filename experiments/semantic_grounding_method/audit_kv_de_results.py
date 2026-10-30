"""Audit cached ZIP text and paired results; NumPy/metadata only, no neural execution."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from kv_pilot_data import schedules,semantic_metrics
from kv_pilot_evaluate import paired_ci,paired_false
from kv_data import save

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def audit():
    loaded={a:json.loads((HERE/f'review/kv_{a}_endpoint_loaded.json').read_bytes()) for a in ('D','E')}
    members={a:d['members'] for a,d in loaded.items()}
    summaries={a:json.loads(m['summary.json']) for a,m in members.items()}
    prior=json.loads((HERE/'review/kv_preparation_loaded.json').read_bytes())['members']
    meta=json.loads(prior['metadata.json']);aux=json.loads(prior['auxiliary_facts.json'])
    assertions=0
    def check(condition,label):
        nonlocal assertions
        if not condition:raise AssertionError(label)
        assertions+=1
    J=lambda a,n:json.loads(members[a][n])
    expected={r['image_id']:r['coco_id'] for r in meta['rows'][4000:5000]}
    caption_expected=[r['image_id'] for r in meta['rows'][:4000]]
    sched=schedules()
    cider,flags,predictions={}, {}, {}
    for arm in ('D','E'):
        m,s=members[arm],summaries[arm]
        check(s['status']=='ENDPOINT_500_EVALUATED_WAIT_MATCHED_DE' and s['updates']==500,arm+' endpoint')
        check(J(arm,'protocol.json')['policy']==s['policy'],arm+' policy')
        prep=J(arm,'preparation.json')
        for name,h in prep['payload_sha256'].items():
            if name in m:check(hashlib.sha256(m[name].encode()).hexdigest()==h,arm+' prepared '+name)
        for rank in (0,1):
            w=J(arm,f'worker_rank{rank}.json');p=J(arm,f'preflight_rank{rank}.json')
            check(w['status']=='ENDPOINT_500_COMPLETE' and w['updates']==500,arm+' worker')
            for key in ('initial_digest','final_digest','common_text_sha256'):
                check(w[key]==s[key],arm+' rank '+key)
            check(w['initial_frozen_digest']==w['final_frozen_digest']==s['frozen_digest'],arm+' frozen')
            check(p['status']=='PREFLIGHT_PASS' and p['reset_after_benchmark'] and p['null_only_native_parity'],arm+' preflight')
            exposure=J(arm,f'exposure_rank{rank}.json')
            expected_caption=[caption_expected[8*step+offset] for step in range(500) for offset in range(rank*4,(rank+1)*4)]
            check(exposure['caption']==expected_caption,arm+' native caption schedule')
            expected_facts=[aux['partitions']['fit'][i]['fact_id'] for i in sched['bce'][:,rank*4:(rank+1)*4].flatten()]
            check(exposure['bce']==expected_facts and exposure['mp_groups']==sched['mp'],arm+' auxiliary schedule')
            history=[json.loads(line) for line in m[f'history_rank{rank}.jsonl'].splitlines()]
            check([r['step'] for r in history]==[1,2,3]+list(range(25,501,25)),arm+' history steps')
            for row in history:
                check(np.isfinite([row[k] for k in ('loss','grad_norm','scst','kl','bce','mp')]).all(),arm+' finite history')
        for path,h in s['source_assets'].items():
            raw=subprocess.check_output(['git','show',s['source_commit']+':'+path],cwd=ROOT)
            check(hashlib.sha256(raw).hexdigest()==h,arm+' pinned source '+path)
            if 'source/'+path in m:check(m['source/'+path].encode()==raw,arm+' archived source '+path)
        predictions[arm]={};cider[arm]={};flags[arm]={}
        for condition in ('C0','end'):
            rows=J(arm,condition+'_predictions.json')
            check(len(rows)==1000 and {r['image_id']:r['coco_id'] for r in rows}==expected,arm+' complete cohort')
            joined=sorted(J(arm,condition+'_rank0.json')+J(arm,condition+'_rank1.json'),key=lambda r:r['image_id'])
            check(rows==joined,arm+' rank predictions')
            predictions[arm][condition]=rows
            per=J(arm,condition+'_per_image_cider.json')
            check([(r['image_id'],r['coco_id']) for r in rows]==[(r['image_id'],r['coco_id']) for r in per],arm+' CIDEr order')
            scores=np.asarray([r['CIDEr'] for r in per],dtype=float)
            check(np.isfinite(scores).all() and abs(scores.mean()-s['metrics'][condition]['CIDEr'])<1e-12,arm+' CIDEr aggregate')
            cider[arm][condition]=scores
            chair=J(arm,condition+'_chair_output.json');sentences=chair['sentences']
            check(len(sentences)==1000 and {r['image_id'] for r in sentences}==set(expected.values()),arm+' CHAIR cohort')
            by_id={r['image_id']:r for r in sentences}
            f=np.asarray([int(bool(by_id[r['coco_id']]['mscoco_hallucinated_words'])) for r in rows])
            flags[arm][condition]=f
            counts=dict(hallucinated_captions=int(f.sum()),hallucinated_mentions=sum(len(r['mscoco_hallucinated_words']) for r in sentences),
                object_mentions=sum(len(r['mscoco_generated_words']) for r in sentences),
                recalled_unique=sum(len(set(r['mscoco_generated_words'])&set(r['mscoco_gt_words'])) for r in sentences),
                reference_unique=sum(len(set(r['mscoco_gt_words'])) for r in sentences))
            check(counts==s['metrics'][condition]['counts'],arm+' raw CHAIR counts')
            ratios=dict(CHAIRs=counts['hallucinated_captions']/1000,CHAIRi=counts['hallucinated_mentions']/counts['object_mentions'],Recall=counts['recalled_unique']/counts['reference_unique'])
            check(all(abs(v-s['metrics'][condition]['chair'][k])<1e-12 for k,v in ratios.items()),arm+' CHAIR ratios')
            check(abs(np.mean([len(r['caption'].split()) for r in rows])-s['metrics'][condition]['mean_words'])<1e-12,arm+' word counts')
        for metric,values,key in [('CIDEr',cider[arm]['end']-cider[arm]['C0'],'end_minus_C0_CIDEr_paired95'),
                                   ('CHAIRs',flags[arm]['end']-flags[arm]['C0'],'end_minus_C0_CHAIRs_paired95')]:
            check(np.allclose(paired_ci(values),s[key],atol=1e-12,rtol=0),arm+' paired '+metric+' CI')
        check(paired_false(flags[arm]['C0'],flags[arm]['end'])==s['false_caption_pairing'],arm+' McNemar')
        held=J(arm,'held_records.json')
        check(len(held)==99 and {r['fact_id'] for r in held}=={r['fact_id'] for r in aux['partitions']['held']},arm+' held coverage')
        truth={r['fact_id']:r for r in aux['partitions']['held']}
        for r in held:check(all(r[k]==truth[r['fact_id']][k] for k in truth[r['fact_id']]),arm+' held metadata/labels')
        for key in ('pre_score','effective_score','C0_decoder_score','end_decoder_score'):
            sem=semantic_metrics(held,aux['multipositive']['held'],key)
            check(all(sem[k]==s['semantic'][key][k] for k in sem),arm+' semantic '+key)
    for key in ('policy','initial_digest','frozen_digest','common_text_sha256','prepared_receipt_sha256','effective_held_prompt_content_sha256'):
        check(summaries['D'][key]==summaries['E'][key],'matched '+key)
    for rank in (0,1):
        check(J('D',f'runtime_rank{rank}.json')==J('E',f'runtime_rank{rank}.json'),'matched runtime')
        check(J('D',f'exposure_rank{rank}.json')==J('E',f'exposure_rank{rank}.json'),'matched exposures')
        check(J('D',f'text_identity_rank{rank}.json')==J('E',f'text_identity_rank{rank}.json'),'matched text')
    for name in ('text_tokens.json','object_targets.json'):
        check(members['D'][name]==members['E'][name],'matched '+name)
    check(predictions['D']['C0']==predictions['E']['C0'],'identical raw C0 captions')
    check(np.array_equal(cider['D']['C0'],cider['E']['C0']),'identical raw C0 scores')
    source_diffs={k for k,v in summaries['D']['source_assets'].items() if v!=summaries['E']['source_assets'].get(k)}
    check(source_diffs=={'experiments/semantic_grounding_method/kv_data.py','experiments/semantic_grounding_method/kv_pilot_evaluate.py'},'reporting-only source differences')
    recovery=J('D','recovery_receipt.json')
    check(recovery==summaries['D']['recovery'],'D recovery provenance')
    check(not recovery['training_repeated'] and not recovery['model_loaded'] and not recovery['gpu_used'],'D no retraining')
    preserved=0
    for name,h in recovery['original_artifact_sha256'].items():
        if name in members['D']:
            check(hashlib.sha256(members['D'][name].encode()).hexdigest()==h,'D preserved '+name);preserved+=1
    check(summaries['D']['checkpoint_sha256']==recovery['original_artifact_sha256']['checkpoint.bin'],'D checkpoint report')
    comparisons={}
    for left,right in [('D','C0'),('E','C0'),('E','D')]:
        ca=cider['D']['C0'] if right=='C0' else cider[right]['end']
        fa=flags['D']['C0'] if right=='C0' else flags[right]['end']
        delta=cider[left]['end']-ca;fd=flags[left]['end']-fa
        comparisons[left+'-'+right]=dict(CIDEr=float(delta.mean()),CIDEr_paired95=paired_ci(delta),
            CHAIRs=float(fd.mean()),CHAIRs_paired95=paired_ci(fd),false_caption_pairing=paired_false(fa,flags[left]['end']))
    changed_DE=sum(a['caption']!=b['caption'] for a,b in zip(predictions['D']['end'],predictions['E']['end']))
    intervention={}
    for arm in ('D','E'):
        off=J(arm,'reader_off_rank0.json')+J(arm,'reader_off_rank1.json')
        records=J(arm,'intervention_rank0.json')['records']+J(arm,'intervention_rank1.json')['records']
        end_by={r['coco_id']:r['caption'] for r in predictions[arm]['end']}
        check(len(off)==64 and len({r['coco_id'] for r in off})==64,arm+' off coverage')
        changed=sum(r['caption']!=end_by[r['coco_id']] for r in off)
        check(changed==summaries[arm]['reader_off_changed_captions'],arm+' off count')
        intervention[arm]=dict(changed_captions=changed,images=64,
            mean_reader_off_logit_KL=float(np.mean([r['reader_off_logit_kl'] for r in records])),
            max_reader_off_logit_KL=max(r['reader_off_logit_kl'] for r in records))
    result=dict(status='MATCHED_DE_ENDPOINTS_AUDIT_PASS_CONTINUE_GATE_FAIL',assertions=assertions,
        zip_sha256={a:d['zip_sha256'] for a,d in loaded.items()},source_differences=sorted(source_diffs),
        D_recovery_preserved_text_artifacts=preserved,checkpoint_binaries_independently_read=False,
        metrics={a:summaries[a]['metrics']['end'] for a in ('D','E')},C0=summaries['D']['metrics']['C0'],
        comparisons=comparisons,changed_DE_captions=changed_DE,intervention=intervention,
        semantic={a:summaries[a]['semantic'] for a in ('D','E')},
        continue_gate=dict(E_CIDEr_greater_D=bool(comparisons['E-D']['CIDEr']>0),E_CIDEr_greater_C0=bool(comparisons['E-C0']['CIDEr']>0),
            E_false_captions_less_C0=bool(flags['E']['end'].sum()<flags['E']['C0'].sum()),
            E_false_captions_le_D=bool(flags['E']['end'].sum()<=flags['D']['end'].sum()),
            E_semantic_screen=summaries['E']['operational_semantic_screen'],E_recall_length_guard=summaries['E']['recall_length_guard'],passes=False),
        recommendation='Stop scaling/tuning current instantiation; ask external Codex2 to review representation-to-caption gap.',
        limits=['One seed, reused dev1000; no independent test.','No neural/HDF5/checkpoint/scorer rerun locally.',
                'CHAIR object hallucination only; relation-caption truth not established.','Do not infer equivalence/no-effect from CI crossing zero.'])
    save(HERE/'KV_DE_RESULT.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='semantic'},ensure_ascii=False,indent=2))


if __name__=='__main__':audit()
