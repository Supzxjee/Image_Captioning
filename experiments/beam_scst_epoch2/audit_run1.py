"""Audit cached text/counts and replay fixed selection; no Torch or scorer."""
import ast
import hashlib
import json
import math
from pathlib import Path
import sys

from epoch2_contract import PARENT_SHA, PARENT_VERSION, XE_SHA, make_identity, validate_resume, validate_eval_binding

HERE=Path(__file__).resolve().parent
CACHE=HERE/'run1_review_cache'
OLD=HERE.parent/'beam_scst_epoch1'


def audit():
    read=lambda name:json.loads((CACHE/name).read_text(encoding='utf-8'))
    sha=lambda name:hashlib.sha256((CACHE/name).read_bytes()).hexdigest()
    same=lambda a,b:math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    manifest=json.loads((HERE/'SOURCE_MANIFEST.json').read_text())
    sources=read('source_receipt.json')
    for label in ('train','test'):
        assert sources[label]==manifest[label+'_sha256']
        for name,digest in sources[label].items():assert sha('source/'+label+'/'+name)==digest,(label,name)
    for name,digest in read('runtime_input_receipt.json').items():assert sha('source/test/'+name)==digest
    parent=read('checkpoint_source.json');assert parent['version']==PARENT_VERSION and parent['sha256']==PARENT_SHA
    assert parent['seed']==43 and parent['epoch']==1
    parent_receipt=read('source/train/PARENT_CHECKPOINT_RECEIPT.json')
    runtime=read('training_runtime.json');assert runtime['torch_version']==parent_receipt['identity']['torch']
    cfg=read('protocol.json')
    expected=make_identity(cfg,parent_receipt,runtime['torch_version'],parent_receipt['identity']['numpy'],sha('vocabulary.json'))
    full,endpoint=read('checkpoint_receipt.json'),read('pilot_checkpoint_receipt.json')
    for receipt in (full,endpoint):
        validate_resume(receipt,expected)
        assert receipt['complete_epoch'] and receipt['updates']==7082 and receipt['cursor_images']==113287
    for rank in (0,1):
        assert read(f'identity_rank{rank}.json')==expected
        c=read(f'continuity_rank{rank}.json')
        assert (c['parent_version'],c['parent_sha256'],c['epoch'],c['cursor_images'],c['updates'])==(PARENT_VERSION,PARENT_SHA,2,0,3541)
        assert c['optimizer_loaded'] and c['rank_RNG_restored'] and c['optimizer_states']==180
        assert c['Adam_step_min']==c['Adam_step_max']==3541 and not c['preflight_only'] and c['resume'] is None
        guard=read(f'beam_preflight_rank{rank}.json')
        assert guard['passed'] and all(guard['mechanism'].values()) and guard['actual_tail7_update'] and guard['real_eval_paths_match']
        assert guard['trainable_parameters']==53183736 and guard['discarded_optimizer_steps']==6
    log=[json.loads(line) for line in (CACHE/'training.jsonl').read_text().splitlines()]
    assert [x['updates'] for x in log]==list(range(3550,7082,25))+[7082]
    for row in log:
        assert row['cursor_images']==min((row['updates']-3541)*32,113287)
        assert all(math.isfinite(v) for v in row.values() if isinstance(v,(int,float)))
    assert log[-1]['last_real_images']==7
    result=read('result.json');training=read('training_complete.json');status=read('run_status.json')
    assert status['complete'] and status['training_complete'] and status['phase']=='complete' and not status['test_tuning']
    assert result['training']==training and training['complete'] and training['epochs']==2
    assert (training['updates'],training['train_images'],training['cumulative_image_events'])==(7082,113287,226574)
    assert training['parent_sha256']==PARENT_SHA and training['source_sha256']==XE_SHA
    binding=validate_eval_binding(read('evaluation_binding.json'))
    assert binding==read('source/test/EPOCH2_EVAL_LOCK.json') and binding['sha256']==endpoint['sha256']
    assert read('source/test/REPLAY_RECEIPTS.json')['checkpoint_sha256']==endpoint['sha256']
    # VALID: reuse saved outputs, verify seals, per-image means and CHAIR counts.
    valid=read('valid_result.json');assert valid==result['VALID'] and valid['checkpoint_sha256']==endpoint['sha256']
    old_valid=json.loads((OLD/'RUN1_RESULT.json').read_text())['metrics']
    assert valid['epoch1_reference']==old_valid and all(same(v-old_valid[k],valid['delta_vs_epoch1'][k]) for k,v in valid['metrics'].items())
    lock=read('scoring_lock.json');assert lock['checkpoint_sha256']==endpoint['sha256']
    for name,digest in lock['files_sha256'].items():assert sha(name)==digest,name
    ids=read('test_lock.json')['expected_test_ids'];captions=read('captions.json')
    assert len(ids)==len(set(ids))==5000 and [r['coco_id'] for r in captions]==sorted(ids)
    refs=read('test_references.json');assert {r['coco_id'] for r in refs}==set(ids)
    valid_workers=[read(f'worker{r}.json') for r in (0,1)];partition=[]
    for rank,w in enumerate(valid_workers):
        assert w['complete'] and w['images']==2500 and w['first_GPU_beam_path_check_passed']
        assert w['checkpoint_sha256']==endpoint['sha256'] and w['caption_sha256']==sha(f'captions_rank{rank}.json')
        partition+=read(f'captions_rank{rank}.json')
    assert sorted(partition,key=lambda r:r['coco_id'])==captions
    scope=dict(math=math)
    functions=[n for n in ast.parse((CACHE/'source/train/saved_quality_data.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='chair_counts']
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<verified count aggregation>','exec'),scope)
    for prefix,rows,metrics,workers in (('',captions,valid['metrics'],valid_workers),):
        counts_metrics,counts=scope['chair_counts'](read('chair_output.json'),rows)
        assert counts==read('chair_counts.json')
        assert all(same(v,metrics[k]) for k,v in counts_metrics.items())
        per=read('cider_per_image.json');assert len(per)==5000 and {r['coco_id'] for r in per}==set(ids)
        assert same(sum(r['CIDEr'] for r in per)/5000,metrics['CIDEr'])
        assert same(sum(w['NLL_sum'] for w in workers)/sum(w['NLL_tokens'] for w in workers),metrics['NLL'])
        for phase in ('basic','chair'):
            s=read(phase+'.json');assert s['input_sha256']==lock['files_sha256'] and s['images']==5000
            assert all(same(v,metrics[k]) for k,v in s['metrics'].items())
    # TEST: all candidates and CLIP scores, then replay the frozen selector.
    test=result['TEST'];assert test==read('test/result.json')==read('test/rerank/result.json')
    assert test['checkpoint_sha256']==endpoint['sha256'] and test['updates']==7082 and test['epochs_SCST']==2
    assert not test['training'] and not test['tuned_on_test'] and not test['oracle_used_for_selection']
    rr=read('test/rerank/test_lock.json');inputs=read('test/rerank/inference_inputs.json')
    expected_test=read('test/test_lock.json')['expected_test_ids']
    assert len(inputs)==len(set(expected_test))==5000 and [r['coco_id'] for r in inputs]==expected_test
    assert not set(expected_test)&set(ids)
    assert sha('test/rerank/inference_inputs.json')==rr['inference_sha256']
    assert sha('test/rerank/test_references.json')==sha('test/test_references.json')==rr['references_sha256']
    old_test=json.loads((OLD/'TEST_RESULT.json').read_text())
    old_lock=json.loads((OLD/'test199_review_cache/rerank/test_lock.json').read_text())
    assert rr['references_sha256']==old_lock['references_sha256'] and rr['clip']==old_lock['clip']
    assert test['selector']==rr['selector']==manifest['references']['selector']
    assets=read('test/rerank/clip_assets.json');assert all(assets[k]==v for k,v in rr['clip'].items())
    old_env=json.loads((OLD/'test199_review_cache/environment.json').read_text());env=read('test/environment.json')
    for key in ('pycocoevalcap','java_archive_sha256','corenlp_archive_sha256','evaluator_files'):assert env[key]==old_env[key],key
    beams={};scores={};workers=[]
    for rank in (0,1):
        w=read(f'test/worker{rank}.json');workers.append(w)
        assert w['complete'] and w['images']==2500 and w['first_GPU_beam_path_check_passed']
        assert w['checkpoint_sha256']==endpoint['sha256']
        assert w['beams_sha256']==sha(f'test/beams_rank{rank}.json') and w['caption_sha256']==sha(f'test/captions_rank{rank}.json')
        raw=read(f'test/beams_rank{rank}.json');top=read(f'test/captions_rank{rank}.json')
        assert [r['coco_id'] for r in raw]==expected_test[rank::2]
        for b,t in zip(raw,top):
            assert b['coco_id']==t['coco_id'] and len(b['candidates'])==5
            assert b['candidates'][0]==dict(ids=t['ids'],caption=t['caption'],logprob=t['score'])
            assert [c['logprob'] for c in b['candidates']]==sorted([c['logprob'] for c in b['candidates']],reverse=True)
            beams[b['coco_id']]=b
        clip=read(f'test/rerank/worker{rank}.json')
        assert clip['complete'] and clip['images']==2500 and clip['scored_pairs']==12500
        assert clip['scores_sha256']==sha(f'test/rerank/scores_rank{rank}.jsonl')
        rows=[json.loads(line) for line in (CACHE/f'test/rerank/scores_rank{rank}.jsonl').read_text().splitlines()]
        assert [r['coco_id'] for r in rows]==expected_test[rank::2];scores.update({r['coco_id']:r for r in rows})
    assert len(beams)==len(scores)==5000
    rules={}
    exec(compile((CACHE/'source/test/captioning/object_consistency.py').read_text(),'<SHA verified object rules>','exec'),rules)
    nodes=[n for n in ast.parse((CACHE/'source/test/captioning/multiscore_reranking.py').read_text()).body if isinstance(n,ast.FunctionDef)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<SHA verified selector>','exec'),rules)
    chosen=[]
    for row in inputs:
        i=row['coco_id'];s=scores[i];assert row['image_sha256']==s['image_sha256']
        values=s['clipscore'];assert len(values)==5 and all(math.isfinite(v) and 0<=v<=2.50001 for v in values)
        candidates=[dict(c,clipscore=float(v)) for c,v in zip(beams[i]['candidates'],values)]
        selected,_=rules['rerank_multiscore'](dict(candidates=candidates),row['detected'],.4,.3)
        assert selected['original_rank']==s['selected_rank'];chosen.append(selected['original_rank'])
    assert chosen==read('test/rerank/selected_ranks.json')
    nll=sum(w['NLL_sum'] for w in workers)/sum(w['NLL_tokens'] for w in workers);flags={}
    for policy,metrics in test['metrics'].items():
        rows=read('test/rerank/'+policy+'_captions.json')
        assert rows==[dict(coco_id=i,caption=beams[i]['candidates'][0 if policy=='decoder_rank0' else k]['caption']) for i,k in zip(expected_test,chosen)]
        chair=read('test/rerank/'+policy+'_chair_output.json');cm,counts=scope['chair_counts'](chair,rows)
        assert counts==metrics['counts'] and all(same(v,metrics[k]) for k,v in cm.items())
        per=read('test/rerank/'+policy+'_cider_per_image.json')
        assert len(per)==5000 and {r['coco_id'] for r in per}==set(expected_test)
        assert same(sum(r['CIDEr'] for r in per)/5000,metrics['CIDEr']) and same(nll,metrics['NLL'])
        basic=read('test/rerank/'+policy+'_basic_chair.json')
        assert all(same(v,metrics[k]) for k,v in basic.items() if k!='counts')
        meteor=read('test/rerank/'+policy+'_meteor.json')
        assert len(meteor['per'])==5000 and all(math.isfinite(v) for v in meteor['per']) and same(meteor['METEOR'],metrics['METEOR'])
        flags[policy]={r['image_id']:bool(r['mscoco_hallucinated_words']) for r in chair['sentences']}
    a,b=flags.values()
    changes=dict(changed_captions=sum(k!=0 for k in chosen),rank_histogram={str(k):chosen.count(k) for k in range(5)},
                 CHAIR_repairs=sum(a[i] and not b[i] for i in a),CHAIR_new_flags=sum(not a[i] and b[i] for i in a))
    assert changes==test['changes']
    comparison=read('comparison.json');assert comparison['epoch2_metrics']==test['metrics'] and comparison['checkpoint']==binding
    assert comparison['epoch1_metrics']==old_test['SCST_metrics'] and comparison['CE_metrics']==manifest['references']['CE']
    for name,old in (('epoch1',comparison['epoch1_metrics']),('CE',comparison['CE_metrics'])):
        for policy,m in old.items():
            for key,delta in comparison['delta_vs_'+name][policy].items():assert same(test['metrics'][policy][key]-m[key],delta)
    report=dict(status='EPOCH2_V202_COMPLETE_AUDITED',date='2026-11-26',output_version='ducanh3/notebook7bc83c4c76/202',
        parent_source=parent,endpoint_sha256=endpoint['sha256'],full_checkpoint_sha256=full['sha256'],
        training=training,VALID=valid,TEST=test,comparison=comparison,
        verification=dict(source_SHA_files=80,rank_continuity_and_Adam_steps_verified=True,training_log_rows=len(log),
            all5000_fixed_selections_replayed=True,beam_candidates=25000,CLIP_pairs=25000,
            VALID_TEST_disjoint=True,TEST_reference_bytes_and_evaluator_assets_match_epoch1=True,
            CIDEr_means_CHAIR_counts_NLL_verified=True,local_Torch_or_scorer_execution=False),
        timing=dict(training_worker_seconds=training['seconds'],epoch1_training_worker_seconds=old_test.get('training_worker_seconds',11249.117766055),
            TEST_decode_worker_max_seconds=max(w['seconds'] for w in workers),
            CLIP_worker_max_seconds=max(read(f'test/rerank/worker{r}.json')['seconds'] for r in (0,1)),
            not_same_work_speed_benchmark=True,excludes_setup_evaluation_and_quota_billing=True),
        limits=['One SCST seed, reused TEST; no new statistical test or untouched holdout.',
            'Epoch2 adds duration plus the guarded speed candidate; one-update parity does not prove bitwise full-epoch equivalence.',
            'CHAIR is object-only; flags are not human-confirmed corrections.',
            'Binary weights/CLIP/JPEG not in review; SHA and restoration verified through user-run receipts only.',
            'Saved CIDEr/count/NLL/selection aggregation only; no metric scorer rerun. SPICE aggregate present, per-image artifact absent.'],
        recommendation='Retain epoch2+rerank as higher-CIDEr candidate; epoch1+rerank and CE+rerank retain lower CHAIR. No automatic epoch3 or TEST tuning.')
    assert 'torch' not in sys.modules and 'pycocoevalcap' not in sys.modules
    (HERE/'RUN1_RESULT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],metrics=test['metrics'],changes=changes,train_hours=training['seconds']/3600),indent=2))


if __name__=='__main__':audit()
