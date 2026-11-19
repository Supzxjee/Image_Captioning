"""Check cached review text and saved-score arithmetic; no models or scorers."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent/'csa_paper'))
from captioning.multiscore_reranking import rerank_multiscore
from saved_quality_data import chair_counts
from scst_kl_valid_compare import gates

XE = 'a51e7ec21be68aa1cda9e0805bd6248add6199914a5c262fce138988066051a5'
CONTROL = '8f1a1fc181b335543d8be117da1315260cdc01b4d4fd4b464d8367a82003d500'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def near(a, b):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b) < 1e-11, (a, b)


def run(cache, version):
    cache = Path(cache)
    manifest = read(cache/'archive_manifest.json')
    for name, info in manifest['files'].items():
        assert sha(cache/name) == info['sha256'], name
    status, result = read(cache/'run_status.json'), read(cache/'result.json')
    assert status == dict(complete=True, training=True, test_used=False, seed=43, updates=500,
        beta=1.0, xe_source='ducanh3/notebook7bc83c4c76/179', control_source='ducanh3/notebook7bc83c4c76/185')
    assert result['status'] == 'SCST_KL500_VALID_COMPLETE'
    binding = read(cache/'cache_checkpoint_binding.json')
    assert binding['xe_checkpoint_sha256'] == XE and not binding['cache_payloads_changed']
    locked = read(cache/'test_lock.json')
    ids = locked['expected_test_ids']
    assert len(ids) == len(set(ids)) == 5000
    assert locked['checkpoint_sha256'] == XE and locked['source_version'].endswith('/179')
    assert locked['beam_width'] == 5 and locked['max_new_tokens'] == 20 and not locked['length_normalization']
    assert read(cache/'scst_cache_receipt.json')['test_used'] is False
    assert sha(cache/'test_references.json') == read(cache/'scst_cache_receipt.json')['files_sha256']['test_references.json']
    refs = read(cache/'test_references.json')
    assert sorted(r['coco_id'] for r in refs) == ids
    for name, expected in [('object_consistency.py','5fbd75b0ff4a35f0475d460a294894009d6007f468e248a0e4b528aaa8fabf73'),
                           ('multiscore_reranking.py','d1baa60e244c86b32e721486d7923ead1391c3a2fb744651c257995f2d0bb443')]:
        local = (HERE.parents[1]/'captioning'/name).read_bytes().replace(b'\r\n',b'\n')
        assert hashlib.sha256(local).hexdigest() == expected  # Windows Git checkout line endings.
    receipt = read(cache/'train/update0500/pilot_checkpoint_receipt.json')
    assert receipt['source_sha256'] == receipt['reference_sha256'] == XE
    assert receipt['updates'] == 500 and receipt['kl_beta'] == 1
    for rank in (0, 1):
        first, last = [read(cache/f'train/kl_{when}_guard_rank{rank}.json') for when in ('start','end')]
        assert abs(first['initial_KL']) < 1e-5
        assert all(first[k] for k in ('teacher_frozen','teacher_outside_optimizer','teacher_rng_unchanged','policy_replay_passed'))
        assert last['updates'] == 500 and last['teacher_versions_unchanged'] and last['teacher_gradients_absent']
    neural = read(cache/'kl_tensor_preflight.json')
    assert all(neural[k] for k in ('equal_policy_zero','beta0_gradient_parity','teacher_detached',
                                 'shifted_policy_positive','padding_and_special_logits_ignored','finite_gradient'))
    logs = [json.loads(line) for line in (cache/'train/training.jsonl').read_text().splitlines()]
    assert [r['updates'] for r in logs] == list(range(25,501,25))
    for row in logs:
        assert row['beta'] == 1
        assert math.isclose(row['loss'],row['SCST']+row['KL'],rel_tol=2e-7,abs_tol=2e-6)  # Logged FP32 addition.
    original = {r['coco_id']: r['caption'] for r in read(cache/'original_XE_VALID.json')}
    all_flags, details, input_seal, ground_truth = {}, {}, None, None
    for name, weight in [('xe', XE), ('scst500', CONTROL), ('kl500', receipt['sha256'])]:
        root = cache/name; rerank = root/'rerank'
        arm = read(root/'result.json')
        assert result['arms'][name] == arm
        assert arm['images'] == 5000 and not arm['test_used'] and arm['checkpoint_sha256'] == weight
        assert read(root/'pilot_checkpoint_receipt.json')['sha256'] == weight
        for file in ('test_lock.json','test_references.json','stage_receipt.json','environment.json','vocabulary.json'):
            assert sha(root/file) == sha(cache/file)
        workers, beams, tops, scored = [], {}, {}, {}
        for rank in (0,1):
            w = read(root/f'worker{rank}.json'); workers.append(w)
            assert w['complete'] and w['images'] == 2500 and w['checkpoint_sha256'] == weight
            assert w['first_GPU_beam_path_check_passed']
            assert not w['runtime_policy']['cuda_wheel_changed']
            for file,key in [('captions','caption_sha256'),('beams','beams_sha256')]:
                assert sha(root/f'{file}_rank{rank}.json') == w[key]
            rows = read(root/f'beams_rank{rank}.json')
            top = read(root/f'captions_rank{rank}.json')
            assert [r['coco_id'] for r in rows] == ids[rank::2]
            assert [r['coco_id'] for r in top] == ids[rank::2]
            beams.update({r['coco_id']:r for r in rows}); tops.update({r['coco_id']:r for r in top})
            sw = read(rerank/f'worker{rank}.json')
            assert sw['complete'] and sw['images'] == 2500 and sw['scored_pairs'] == 12500
            assert sha(rerank/f'scores_rank{rank}.jsonl') == sw['scores_sha256']
            score = [json.loads(line) for line in (rerank/f'scores_rank{rank}.jsonl').read_text().splitlines()]
            assert [r['coco_id'] for r in score] == ids[rank::2]
            scored.update({r['coco_id']:r for r in score})
        nll = sum(w['NLL_sum'] for w in workers)/sum(w['NLL_tokens'] for w in workers)
        lock = read(rerank/'test_lock.json')
        assert sha(rerank/'inference_inputs.json') == lock['inference_sha256']
        assert sha(rerank/'test_references.json') == lock['references_sha256'] == sha(cache/'test_references.json')
        assert lock['checkpoint_sha256'] == weight and lock['selector'] == arm['selector']
        assert tuple(arm['selector'][k] for k in ('decoder_weight','clip_weight','object_weight')) == (.3,.4,.3)
        if input_seal is None: input_seal = lock['inference_sha256']
        assert input_seal == lock['inference_sha256']
        selected = read(rerank/'selected_ranks.json')
        inputs = read(rerank/'inference_inputs.json')
        assert [r['coco_id'] for r in inputs] == ids and len(selected) == 5000
        for row, chosen in zip(inputs, selected):
            i=row['coco_id']; candidates=beams[i]['candidates']; score=scored[i]
            assert len(candidates) == len(score['clipscore']) == 5
            assert candidates[0]['caption'] == tops[i]['caption'] and candidates[0]['ids'] == tops[i]['ids']
            near(candidates[0]['logprob'], tops[i]['score'])
            assert row['image_sha256'] == score['image_sha256']
            assert all(.5 <= v <= 1 for v in row['detected'].values())
            enriched = [dict(c,clipscore=v) for c,v in zip(candidates,score['clipscore'])]
            replay,_ = rerank_multiscore(dict(candidates=enriched),row['detected'],.4,.3)
            assert chosen == score['selected_rank'] == replay['original_rank']
        for policy in ('decoder_rank0','clip_object_fixed'):
            captions=read(rerank/(policy+'_captions.json'))
            assert [r['coco_id'] for r in captions] == ids
            for row,k in zip(captions,selected):
                assert row['caption'] == beams[row['coco_id']]['candidates'][0 if policy=='decoder_rank0' else k]['caption']
            chair=read(rerank/(policy+'_chair_output.json'))
            metrics, counts=chair_counts(chair,captions)
            assert counts == arm['metrics'][policy]['counts']
            for k,v in metrics.items(): near(v,arm['metrics'][policy][k])
            per=read(rerank/(policy+'_cider_per_image.json'))
            assert [r['coco_id'] for r in per] == ids
            near(math.fsum(r['CIDEr'] for r in per)/5000,arm['metrics'][policy]['CIDEr'])
            near(nll,arm['metrics'][policy]['NLL'])
            gt={r['image_id']:set(r['mscoco_gt_words']) for r in chair['sentences']}
            if ground_truth is None:ground_truth=gt
            assert ground_truth == gt
            flags={r['image_id']:bool(r['mscoco_hallucinated_words']) for r in chair['sentences']}
            all_flags[name,policy]=flags
        if name=='xe':
            assert all(tops[i]['caption']==original[i] for i in ids)
        a,b=[all_flags[name,p] for p in ('decoder_rank0','clip_object_fixed')]
        assert sum(a[i] and not b[i] for i in ids)==arm['changes']['CHAIR_repairs']
        assert sum(not a[i] and b[i] for i in ids)==arm['changes']['CHAIR_new_flags']
        details[name]=dict(verified_images=5000,verified_beams=25000,NLL=nll,weights_sha256=weight)
    metrics={n:r['metrics']['clip_object_fixed'] for n,r in result['arms'].items()}
    gate=gates(metrics['xe'],metrics['scst500'],metrics['kl500'])
    assert gate==result['gates'] and all(gate.values())==result['screen_passed']
    transitions={}
    for name in ('xe','scst500'):
        a,b=all_flags[name,'clip_object_fixed'],all_flags['kl500','clip_object_fixed']
        transitions[name]=dict(new_flags=sum(not a[i] and b[i] for i in ids),repairs=sum(a[i] and not b[i] for i in ids))
    report=dict(zip_sha256=manifest['zip_sha256'],source_version=version,run_status=status,result=result,
        audit=dict(arms=details,all_saved_fixed_selections_replayed=15000,reference_sha256=sha(cache/'test_references.json'),
            teacher_guards_passed=True,tensor_preflight_passed=True,original_XE_captions_changed=0,
            KL_rerank_flag_transitions=transitions,training_seconds=logs[-1]['seconds'],
            final_logged_KL=logs[-1]['KL'],peak_memory_bytes=max(read(cache/f'train/kl_end_guard_rank{i}.json')['peak_memory_bytes'] for i in (0,1)),
            local_caption_models_or_caption_metric_scorers_executed=False,bootstrap_intervals_recomputed_locally=False),
        limits=['One seed, reused VALID; not TEST confirmation or seed-level significance.',
                'Checkpoint binaries, raw patch arrays and actual JPEGs absent from review; saved hashes/receipts checked.',
                'BLEU/METEOR/ROUGE/SPICE and bootstrap intervals recorded, not rerun locally.'],
        recommendation='Do not automatically extend KL beta1: caption gain versus SCST500, but locked joint CIDEr/CHAIR screen failed.')
    destination=HERE/'CSA_SEED43_SCST_KL500_VALID5000_RESULT.json'
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    assert 'torch' not in sys.modules
    print(json.dumps(report['audit'],indent=2));print('RESULT:',destination)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cache',type=Path);p.add_argument('--version',required=True)
    a=p.parse_args();run(a.cache,a.version)
