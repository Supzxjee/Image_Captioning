"""Saved TEST metadata/counts and deterministic selection replay, no model/scorer."""
import ast
import hashlib
import json
import math
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
CACHE=HERE/'test199_review_cache'


def audit():
    read=lambda name:json.loads((CACHE/name).read_text(encoding='utf-8'))
    sha=lambda name:hashlib.sha256((CACHE/name).read_bytes()).hexdigest()
    same=lambda a,b:math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12)
    manifest=json.loads((HERE/'TEST_SOURCE_MANIFEST.json').read_text())
    sources=read('source_receipt.json')['sha256'];assert sources==manifest['sha256']
    for name,digest in sources.items():assert sha('source/'+name)==digest,name
    result=read('result.json');assert result==read('rerank/result.json')
    assert result['status']=='BEAM_SCST_V199_TEST5000_RERANK_COMPLETE'
    assert read('run_status.json')['complete'] and not result['training'] and not result['tuned_on_test']
    assert result['checkpoint_sha256']==manifest['endpoint_sha256']
    assert read('checkpoint_source.json')['version']==manifest['source_version']
    lock=read('rerank/test_lock.json');inputs=read('rerank/inference_inputs.json')
    assert sha('rerank/inference_inputs.json')==lock['inference_sha256']
    assert sha('rerank/test_references.json')==sha('test_references.json')==lock['references_sha256']
    assert lock['selector']==result['selector']==manifest['selector']
    ce=json.loads((HERE.parent/'clip_patch_baseline/CSA_PER_CAPTION_SEED43_RERANK_TEST5000_RESULT.json').read_text())
    assert lock['references_sha256']==ce['rerank_lock']['references_sha256']
    assert lock['clip']==ce['rerank_lock']['clip']
    assets=read('rerank/clip_assets.json')
    assert all(assets[k]==v for k,v in lock['clip'].items())
    expected=read('test_lock.json')['expected_test_ids']
    assert len(inputs)==len(set(expected))==5000 and [x['coco_id'] for x in inputs]==expected
    assert not set(expected)&set(json.loads((HERE/'run1_review_cache/test_lock.json').read_text())['expected_test_ids'])
    beams={};scores={};workers=[]
    for rank in (0,1):
        worker=read(f'worker{rank}.json');workers.append(worker)
        assert worker['complete'] and worker['images']==2500 and worker['first_GPU_beam_path_check_passed']
        assert worker['checkpoint_sha256']==result['checkpoint_sha256']
        assert worker['beams_sha256']==sha(f'beams_rank{rank}.json') and worker['caption_sha256']==sha(f'captions_rank{rank}.json')
        raw=read(f'beams_rank{rank}.json');top=read(f'captions_rank{rank}.json')
        assert [x['coco_id'] for x in raw]==expected[rank::2]
        for b,t in zip(raw,top):
            assert len(b['candidates'])==5 and b['coco_id']==t['coco_id']
            assert b['candidates'][0]==dict(ids=t['ids'],caption=t['caption'],logprob=t['score'])
            beams[b['coco_id']]=b
        clip=read(f'rerank/worker{rank}.json')
        assert clip['complete'] and clip['images']==2500 and clip['scored_pairs']==12500
        assert clip['scores_sha256']==sha(f'rerank/scores_rank{rank}.jsonl')
        rows=[json.loads(line) for line in (CACHE/f'rerank/scores_rank{rank}.jsonl').read_text().splitlines()]
        assert [x['coco_id'] for x in rows]==expected[rank::2]
        scores.update({x['coco_id']:x for x in rows})
    assert len(beams)==len(scores)==5000
    # Only the SHA-verified original rule-based selector is replayed on saved scores.
    scope={}
    exec(compile((CACHE/'source/captioning/object_consistency.py').read_text(),'<verified object rules>','exec'),scope)
    source=(CACHE/'source/captioning/multiscore_reranking.py').read_text()
    nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<verified selector>','exec'),scope)
    chosen=[]
    for row in inputs:
        i=row['coco_id'];score=scores[i]
        assert row['image_sha256']==score['image_sha256']
        values=score['clipscore'];candidates=beams[i]['candidates']
        assert len(values)==5 and all(math.isfinite(x) and 0<=x<=2.50001 for x in values)
        enriched=[dict(c,clipscore=float(s)) for c,s in zip(candidates,values)]
        selected,_=scope['rerank_multiscore'](dict(candidates=enriched),row['detected'],.4,.3)
        assert selected['original_rank']==score['selected_rank']
        chosen.append(selected['original_rank'])
    assert chosen==read('rerank/selected_ranks.json')
    count_scope=dict(math=math)
    functions=[n for n in ast.parse((CACHE/'source/saved_quality_data.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='chair_counts']
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<saved CHAIR aggregation>','exec'),count_scope)
    flags={};metrics=result['metrics']
    nll=sum(w['NLL_sum'] for w in workers)/sum(w['NLL_tokens'] for w in workers)
    for policy in ('decoder_rank0','clip_object_fixed'):
        rows=read(f'rerank/{policy}_captions.json')
        expected_rows=[dict(coco_id=i,caption=beams[i]['candidates'][0 if policy=='decoder_rank0' else k]['caption']) for i,k in zip(expected,chosen)]
        assert rows==expected_rows
        chair=read(f'rerank/{policy}_chair_output.json')
        counts_metrics,counts=count_scope['chair_counts'](chair,rows)
        assert counts==metrics[policy]['counts']
        assert all(same(v,metrics[policy][k]) for k,v in counts_metrics.items())
        per=read(f'rerank/{policy}_cider_per_image.json')
        assert len(per)==5000 and {x['coco_id'] for x in per}==set(expected)
        assert same(sum(x['CIDEr'] for x in per)/5000,metrics[policy]['CIDEr'])
        basic=read(f'rerank/{policy}_basic_chair.json')
        assert all(same(v,metrics[policy][k]) for k,v in basic.items() if k!='counts')
        meteor=read(f'rerank/{policy}_meteor.json')
        assert len(meteor['per'])==5000 and all(math.isfinite(x) for x in meteor['per'])
        assert same(meteor['METEOR'],metrics[policy]['METEOR'])
        assert same(nll,metrics[policy]['NLL'])
        flags[policy]={x['image_id']:bool(x['mscoco_hallucinated_words']) for x in chair['sentences']}
    a,b=flags.values()
    changes=dict(changed_captions=sum(k!=0 for k in chosen),rank_histogram={str(k):chosen.count(k) for k in range(5)},
        CHAIR_repairs=sum(a[i] and not b[i] for i in a),CHAIR_new_flags=sum(not a[i] and b[i] for i in a))
    assert changes==result['changes']
    comparison=read('comparison.json');assert comparison['SCST_metrics']==metrics
    assert comparison['CE_metrics']==manifest['CE_reference']
    for policy,m in comparison['CE_metrics'].items():
        for key,value in m.items():
            assert same(value,ce['result']['metrics'][policy][key])
            assert same(metrics[policy][key]-value,comparison['delta_vs_same_seed_CE'][policy][key])
    limits=['One training seed; TEST was previously used in project development; no formal statistical test.',
        'Recipe changes multiple SCST factors; not an isolated beam-baseline ablation or M3 ablation.',
        'CHAIR flags are object-only, not human-confirmed errors; no equivalence/noninferiority claim versus CE+rerank.',
        'Model, CLIP and JPEG binary hashes are runtime receipts; binaries not reloaded locally.',
        'CIDEr means/CHAIR counts/NLL and fixed selections recomputed; BLEU/ROUGE/SPICE not rescored locally.']
    report=dict(status='TEST200_COMPLETE_AUDITED',date='2026-11-26',output_version='ducanh3/notebook7bc83c4c76/200',
        training_checkpoint_version=manifest['source_version'],checkpoint_sha256=result['checkpoint_sha256'],
        CE_source='ducanh3/notebook7bc83c4c76/179',CE_rerank_output=ce['rerank_output_source'],
        CE_metrics=comparison['CE_metrics'],SCST_metrics=metrics,changes=changes,
        delta_vs_same_seed_CE=comparison['delta_vs_same_seed_CE'],
        rerank_minus_top1={k:metrics['clip_object_fixed'][k]-v for k,v in metrics['decoder_rank0'].items() if isinstance(v,(int,float))},
        verification=dict(source_SHA_files=37,caption_ids=5000,beam_candidates=25000,CLIP_pairs=25000,
            all_fixed_selections_replayed=True,disjoint_from_VALID=True,reference_bytes_match_CE=True,
            CLIP_assets_match_CE=True,CHAIR_counts_verified=True,CIDEr_means_verified=True,NLL_verified=True),
        timing=dict(decode_worker_max_seconds=max(w['seconds'] for w in workers),
            CLIP_worker_max_seconds=max(read(f'rerank/worker{rank}.json')['seconds'] for rank in (0,1)),
            excludes_setup_and_CPU_metrics_and_billing=True),limits=limits,
        next_recommendation='Keep XE+rerank as low-hallucination reference and SCST+rerank as high-CIDEr candidate; no automatic epoch2 or TEST retuning.')
    (HERE/'TEST_RESULT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    assert 'torch' not in sys.modules and 'pycocoevalcap' not in sys.modules
    print(json.dumps(dict(status=report['status'],metrics=metrics,changes=changes),indent=2))


if __name__=='__main__':audit()
