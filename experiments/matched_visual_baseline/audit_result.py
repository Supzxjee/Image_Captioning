"""Audit saved user-run text/metadata. No Torch, caption inference or metric scorer."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = HERE/'review_cache'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a, b, tolerance=1e-10):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b) <= tolerance, (a, b)


def audit():
    receipt = read(CACHE/'ARCHIVE_MANIFEST.json')
    for name, digest in receipt['files'].items():
        assert sha(CACHE/name) == digest, name
    source = read(CACHE/'source_receipt.json')
    expected = read(HERE/'SOURCE_MANIFEST.json')
    assert source['sha256'] == expected['sha256']
    for name, digest in source['sha256'].items():
        assert sha(CACHE/'source'/name) == digest, name
    result = read(CACHE/'result.json')
    state = read(CACHE/'progress.json')
    identity = read(CACHE/'identity.json')
    protocol = read(CACHE/'protocol.json')
    status = read(CACHE/'run_status.json')
    stage = read(CACHE/'stage_summary.json')
    assert status['complete'] and not any(status[k] for k in ('test_used', 'SCST', 'reranking'))
    assert result['status'] == 'MATCHED_VISUAL_PARALLEL3_XE10_COMPLETE'
    assert (state['epoch'], state['batch_cursor'], state['updates'], state['caption_events']) == (11, 0, 177110, 5667470)
    assert (identity['seed'], identity['epochs'], identity['global_batch'], identity['dropout']) == (42, 10, 32, .2)
    assert not identity['sharing'] and not identity['semantic_conditioning'] and not protocol['semantic'] and not protocol['resume_root']
    assert identity['decoder_layers'] == 3 and identity['sampler'] == 'image_grouped_42_plus_epoch'
    assert len(state['history']) == 10 and [r['epoch'] for r in state['history']] == list(range(1, 11))
    assert state['best_epoch'] == max(state['history'], key=lambda r: r['CIDEr'])['epoch'] == 9
    assert result['best_epoch'] == 9
    assert stage['train_images'] == 113287 and stage['valid_images'] == 5000
    assert not stage['test_features_used'] and not stage['feature_arrays_staged'] and 'semantic' not in stage
    assert identity['stage_files'] == stage['files_sha256']
    for rank in (0, 1):
        contract = read(CACHE/f'baseline_contract_rank{rank}.json')
        preflight = read(CACHE/f'preflight_rank{rank}.json')
        runtime = read(CACHE/f'accelerate_runtime_rank{rank}.json')
        assert contract == result['baseline_contract']
        for key in ('passed', 'common_initialization_equal', 'decoder_implementation_equal', 'two_streams_equal_projected_visual', 'prompt_ignored'):
            assert contract[key]
        assert not contract['trained_checkpoint_loaded']
        for key in ('passed', 'causal', 'ordinary_DDP_gradient_parity', 'beam_score_rescore'):
            assert preflight[key]
        assert preflight['initial_tensor_sha256'] == identity['initial_sha256']
        assert preflight['discarded_benchmark_updates'] == 64
        assert runtime['processes'] == 2 and runtime['precision'] == 'no' and runtime['accumulation'] == 1
    parity = read(CACHE/'full_parity.json')
    assert [r['case'] for r in parity] == ['first32', 'tail27']
    assert all(r['passed'] and all(d['passed'] for d in r['gradient_diagnostics'].values()) for r in parity)
    assert sha(CACHE/'training.jsonl') == read(CACHE/'checkpoint_receipt.json')['training_log_sha256']
    log = [json.loads(line) for line in (CACHE/'training.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(log) == 1780 and log[-1]['updates'] == 177110
    assert all(math.isfinite(r[k]) for r in log for k in ('NLL', 'lr', 'grad_norm', 'seconds'))
    assert all(a['updates'] < b['updates'] for a, b in zip(log, log[1:]))
    checked_captions = 0
    valid_ids = None
    for row in state['history']:
        folder = CACHE/'evaluations'/f"epoch_{row['epoch']:02d}"
        for name, digest in row['files_sha256'].items():
            assert sha(folder/name) == digest
        rows = read(folder/'val_rank0.json') + read(folder/'val_rank1.json')
        ids = {r['coco_id'] for r in rows}
        assert len(rows) == len(ids) == 5000
        if valid_ids is None:
            valid_ids = ids
        assert ids == valid_ids
        assert all(r['ids'][0] == 1 and isinstance(r['caption'], str) and math.isfinite(r['score']) for r in rows)
        scores = read(folder/'cider_per_image.json')
        assert len(scores) == 5000 and {r['coco_id'] for r in scores} == ids
        close(sum(r['CIDEr'] for r in scores)/5000, row['CIDEr'])
        workers = [read(folder/f'worker{r}.json') for r in (0, 1)]
        close(sum(w['NLL_sum'] for w in workers)/sum(w['NLL_tokens'] for w in workers), row['NLL'])
        close(sum(len(r['caption'].split()) for r in rows)/5000, row['mean_words'])
        checked_captions += len(rows)
    for epoch in (9, 10):
        folder = CACHE/'quality'/f'epoch_{epoch:02d}'
        rows = read(CACHE/'evaluations'/f'epoch_{epoch:02d}'/'val_rank0.json') + read(CACHE/'evaluations'/f'epoch_{epoch:02d}'/'val_rank1.json')
        mapping = {r['coco_id']: r['caption'] for r in rows}
        chair = read(folder/'chair_output.json')
        sentences = chair['sentences']
        assert len(sentences) == 5000 and {r['image_id'] for r in sentences} == valid_ids
        assert all(r['caption'] == mapping[r['image_id']] for r in sentences)
        counts = dict(images=5000,
            hallucinated_captions=sum(bool(r['mscoco_hallucinated_words']) for r in sentences),
            hallucinated_mentions=sum(len(r['mscoco_hallucinated_words']) for r in sentences),
            object_mentions=sum(len(r['mscoco_generated_words']) for r in sentences),
            recalled_unique=sum(len(set(r['mscoco_generated_words']) & set(r['mscoco_gt_words'])) for r in sentences),
            reference_unique=sum(len(set(r['mscoco_gt_words'])) for r in sentences))
        metric = result['metrics_by_epoch'][str(epoch)]
        assert counts == metric['counts']
        for key, a, b in [('CHAIRs', counts['hallucinated_captions'], 5000),
                          ('CHAIRi', counts['hallucinated_mentions'], counts['object_mentions']),
                          ('Recall', counts['recalled_unique'], counts['reference_unique'])]:
            close(a/b, metric[key])
            close(metric[key], chair['overall_metrics'][key])
        for key, value in read(folder/'result.json').items():
            assert metric[key] == value
    full = read(ROOT/'experiments/clip_patch_baseline/CSA_PER_CAPTION_FULL_XE10_RESULT.json')
    own, other = result['metrics_by_epoch']['9'], full['metrics_by_epoch']['9']
    delta = {k: (other[k]-own[k])*100 for k in ('CIDEr', 'BLEU_1', 'BLEU_4', 'ROUGE_L', 'CHAIRs', 'CHAIRi', 'Recall')}
    complete = read(CACHE/'training_complete.json')
    report = dict(result, source='ducanh3/notebook7bc83c4c76/188',
        archive_sha256=receipt['archive_sha256'], checkpoint_binaries_in_review=False,
        audit=dict(archive_CRC_pass=receipt['CRC_pass'], archive_files=len(receipt['files']), embedded_source_files=len(source['sha256']),
            source_sha_verified=True, evaluation_payload_sha_verified=True, training_log_sha_verified=True,
            caption_rows_checked=checked_captions, CHAIR_captions_checked=10000,
            CIDEr_per_image_means_verified=True, NLL_aggregates_verified=True,
            local_torch_or_model_executed=False, local_metric_scorers_executed=False),
        timing=dict(worker_seconds=complete['worker_seconds'], scope=complete['timing_scope'],
            metadata_prepare_seconds=stage['seconds'], billing_not_measured=True),
        comparison=dict(split='VALID5000', seed=42, decoding='beam top1; no rerank/SCST',
            baseline=own, full_model=other, full_minus_baseline_points=delta,
            full_CHAIRs_relative_reduction=(own['CHAIRs']-other['CHAIRs'])/own['CHAIRs'],
            full_CHAIRi_relative_reduction=(own['CHAIRi']-other['CHAIRi'])/own['CHAIRi'],
            full_checkpoint_sha256=full['receipts']['best_receipt.json']['sha256'],
            limits=['One matched baseline seed; full model has additional parameters.',
                    'Shared initialization verified against reconstructed full constructor, not a stored initial binary.',
                    'Historical full seed42 used local FP16 feature staging; new baseline reads the same raw H5 directly.',
                    'VALID used for checkpoint selection/development; no TEST conclusion or prompt-text-only causal claim.',
                    'Stored scorer outputs/counts audited; no scorers rerun, no significance test.']))
    if (HERE/'RESULT.json').is_file():
        prior = read(HERE/'RESULT.json')
        if 'sheet_update' in prior:
            report['sheet_update'] = prior['sheet_update']
    (HERE/'RESULT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], audit=report['audit'], timing=report['timing'], delta=delta), ensure_ascii=False, indent=2))
    return report


if __name__ == '__main__':
    audit()
