"""Audit saved text/metadata only; no models, pickle or metric scorers."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
CACHE = HERE/'run1_review_cache'


def audit():
    read = lambda name: json.loads((CACHE/name).read_text(encoding='utf-8'))
    sha = lambda name: hashlib.sha256((CACHE/name).read_bytes()).hexdigest()
    same = lambda x, y: math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-12)
    manifest = json.loads((HERE/'V2_SOURCE_MANIFEST.json').read_text())
    files = read('source_receipt.json')['sha256']
    assert files == manifest['sha256']
    for name, digest in files.items():
        assert sha('source/'+name) == digest, name
    result, completed = read('result.json'), read('training_complete.json')
    assert read('run_status.json')['complete'] and completed['complete']
    assert (result['split'], result['images'], result['updates'], result['train_images']) == ('val', 5000, 3541, 113287)
    assert not result['test_used'] and not result['reranking']
    source = read('checkpoint_source.json')
    assert (source['seed'], source['epoch'], source['version'], source['sha256']) == (43, 8, manifest['checkpoint_version'], manifest['checkpoint_sha256'])
    identities = [read(f'identity_rank{rank}.json') for rank in (0, 1)]
    assert identities[0] == identities[1] == result['checkpoint']['identity']
    assert identities[0]['source_hashes'] == files and identities[0]['XE_seed'] == 43
    assert identities[0]['CE_weight'] == 0 and identities[0]['baseline'] == 'image_mean_of_five_beam_rewards'
    for rank in (0, 1):
        guard = read(f'beam_preflight_rank{rank}.json')
        assert guard['passed'] and all(guard['mechanism'].values())
        assert guard['real_eval_paths_match'] and guard['actual_tail7_update']
        assert guard['trainable_parameters'] == 53183736
        assert set(guard['real_gradient_groups_checked']) == {'core', 'object_prompt', 'relation_prompt'}
        assert guard['discarded_optimizer_steps'] == 6 and not guard['training_initializes_from_discarded_guard']
    rows = [json.loads(line) for line in (CACHE/'training.jsonl').read_text().splitlines()]
    assert [x['updates'] for x in rows] == list(range(25, 3541, 25))+[3541]
    for row in rows:
        assert row['cursor_images'] == min(row['updates']*32, 113287)
        assert all(math.isfinite(v) for v in row.values() if isinstance(v, (int, float)))
    assert rows[-1]['last_real_images'] == 7
    endpoint, latest = read('pilot_checkpoint_receipt.json'), read('checkpoint_receipt.json')
    assert endpoint == result['checkpoint']
    for receipt in (endpoint, latest):
        assert (receipt['updates'], receipt['cursor_images'], receipt['source_sha256']) == (3541, 113287, source['sha256'])
    lock = read('scoring_lock.json')
    assert lock['checkpoint_sha256'] == endpoint['sha256']
    for name, digest in lock['files_sha256'].items():
        assert sha(name) == digest, name
    captions = read('captions.json'); ids = [x['coco_id'] for x in captions]
    assert len(ids) == len(set(ids)) == 5000
    assert ids == sorted(read('test_lock.json')['expected_test_ids'])
    refs = read('test_references.json')
    assert len(refs) == 5000 and {x['coco_id'] for x in refs} == set(ids)
    assert all(len(x['references']) >= 5 for x in refs)
    workers = [read(f'worker{rank}.json') for rank in (0, 1)]
    partition = []
    for rank, worker in enumerate(workers):
        assert worker['complete'] and worker['images'] == 2500 and worker['first_GPU_beam_path_check_passed']
        assert worker['checkpoint_sha256'] == endpoint['sha256'] and worker['caption_sha256'] == sha(f'captions_rank{rank}.json')
        partition.extend(read(f'captions_rank{rank}.json'))
    assert sorted(partition, key=lambda x:x['coco_id']) == captions
    metrics = result['metrics']
    for name in ('basic', 'chair'):
        scored = read(name+'.json')
        assert scored['images'] == 5000 and scored['input_sha256'] == lock['files_sha256']
        assert all(same(metrics[k], v) for k, v in scored['metrics'].items())
    per = read('cider_per_image.json')
    assert len(per) == 5000 and {x['coco_id'] for x in per} == set(ids)
    assert same(sum(x['CIDEr'] for x in per)/5000, metrics['CIDEr'])
    assert same(sum(x['NLL_sum'] for x in workers)/sum(x['NLL_tokens'] for x in workers), metrics['NLL'])
    chair = read('chair_output.json'); sentences = chair['sentences']
    assert len(sentences) == 5000 and {x['image_id'] for x in sentences} == set(ids)
    mapping = {x['coco_id']:x['caption'] for x in captions}
    assert all(x['caption'] == mapping[x['image_id']] for x in sentences)
    counts = dict(images=5000, hallucinated_captions=sum(bool(x['mscoco_hallucinated_words']) for x in sentences),
        hallucinated_mentions=sum(len(x['mscoco_hallucinated_words']) for x in sentences),
        object_mentions=sum(len(x['mscoco_generated_words']) for x in sentences),
        recalled_unique=sum(len(set(x['mscoco_generated_words']) & set(x['mscoco_gt_words'])) for x in sentences),
        reference_unique=sum(len(set(x['mscoco_gt_words'])) for x in sentences))
    assert counts == read('chair_counts.json')
    for key, value in dict(CHAIRs=counts['hallucinated_captions']/5000, CHAIRi=counts['hallucinated_mentions']/counts['object_mentions'], Recall=counts['recalled_unique']/counts['reference_unique']).items():
        assert same(metrics[key], value) and same(value, chair['overall_metrics'][key])
    reference = manifest['reference_VALID']
    assert reference == result['CE_reference']
    assert all(same(result['delta'][k], metrics[k]-v) for k, v in reference.items())
    report = dict(status='COMPLETE_AUDITED_METADATA_AND_SAVED_COUNTS', date='2026-11-26',
        output_version='ducanh3/notebook7bc83c4c76/199', checkpoint_source=source,
        endpoint_sha256=endpoint['sha256'], latest_resume_sha256=latest['sha256'],
        audit=dict(source_files=38, scoring_seals_verified=True, caption_and_reference_ids=5000,
            reference_sentences=sum(len(x['references']) for x in refs),
            training_log_rows=len(rows), CIDEr_mean_verified=True, CHAIR_counts_verified=True, NLL_verified=True,
            actual_checkpoint_binary_loaded_locally=False, metric_scorers_run_locally=False),
        metrics=metrics, CE_reference=reference, delta=result['delta'], chair_counts=counts,
        training=completed, mean_caption_words=sum(len(x['caption'].split()) for x in captions)/5000,
        missing_metrics=['METEOR','SPICE'],
        limits=['One seed and one epoch on reused VALID; no TEST or reranking.',
            'Change versus earlier SCST includes beam baseline, optimizer rate, train mode and duration; not a baseline-only ablation.',
            'Checkpoint hashes are user-run receipts; binaries are absent from review ZIP.',
            'Frozen patch header/size/map verified at runtime; full feature content hashes not rehashed.',
            'CHAIR measures object hallucination, not attributes/actions/relations.'],
        recommendation='Preserve output199 endpoint; next consider fixed existing rerank on VALID, not automatic further SCST or TEST.')
    (HERE/'RUN1_RESULT.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], version=report['output_version'], metrics=metrics, counts=counts, train_hours=completed['seconds']/3600),indent=2))


if __name__ == '__main__':
    audit()
