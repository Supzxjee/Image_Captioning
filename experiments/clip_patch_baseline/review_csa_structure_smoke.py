"""Review saved source/log/metadata only; never execute model, scorer or Kaggle."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]


def audit(cache_path):
    cache = json.loads(Path(cache_path).read_text(encoding='utf-8')); files = cache['files']
    get = lambda n: json.loads(files[n]); checks = []
    def check(name, ok):
        if not ok: raise AssertionError(name)
        checks.append(name)
    result = get('result.json'); stage = get('stage_summary.json'); semantic = get('semantic_stage.json')
    status = get('run_status.json'); cfg = get('protocol.json'); source = get('source_receipt.json')
    check('complete smoke only', result['status'] == 'CLIP_PATCH_CSA_STRUCTURE_SMOKE_COMPLETE' and status['complete'] and not result['full_training_complete'] and not result['training_quality_evaluated'])
    for name, obj in [('result', result), ('status', status), ('stage', stage), ('semantic', semantic)]:
        check(name+' no held-out images', not obj['test_used'] and not obj['validation_used'])
    check('no RL/rerank', not result['SCST'] and not result['reranking'] and not status['SCST'] and not status['reranking'])
    check('exact approved source', source['commit'] == '0323747b13122800aa6b5e6ffd9b2d2913054ee8' and len(source['sha256']) == 22)
    for n, h in source['sha256'].items():
        raw = subprocess.check_output(['git', 'show', source['commit']+':'+n], cwd=ROOT)
        check('source '+n, hashlib.sha256(raw).hexdigest() == h)
    baseline = json.loads((HERE/'SMOKE_SOURCE.json').read_text())
    check('same selected raw patch rows as smoke192', stage['raw_selected_row_sha256'] == baseline['raw_selected_row_sha256'])
    check('same cache profile and TRAIN vocab', stage['feature_profile'] == baseline['feature_profile'] and stage['files_sha256']['vocabulary.json'] == baseline['vocabulary_sha256'])
    check('TRAIN scope/counts', stage['images'] == 256 and stage['caption_events'] == 1280 and stage['split'] == 'TRAIN_only' and semantic['images'] == 256 and semantic['objects'] == 779 and semantic['relations'] == 109)
    check('pinned semantic inputs', semantic['prompt_source'] == json.loads((ROOT/'experiments/csa_paper/FULL_PROMPT_FRESH_V175_RESULT.json').read_text())['transport_inventory']['cache_files'] and semantic['bank_source'] == json.loads((ROOT/'experiments/csa_paper/SAVE_BANK_V130_PACKAGE.json').read_text())['files'])
    check('source text vector seal preserved', semantic['files_sha256']['sem_text_vectors.npy'] == semantic['prompt_source']['prompt_embeddings.npy'])
    check('bounded approved config', cfg['updates'] == 200 and cfg['prepare_seconds'] == 2100 and 300 <= cfg['worker_seconds'] <= 2100)
    for rank in (0, 1):
        w = get(f'worker{rank}.json'); g = get(f'preflight_rank{rank}.json'); runtime = get(f'cuda_runtime_rank{rank}.json')
        check(f'rank{rank} result parity', w == result['workers'][rank] and w['complete'] and w['rank'] == rank and w['updates'] == 200 and w['TRAIN_images'] == 256 and w['caption_events'] == 6400)
        check(f'rank{rank} CUDA/NCCL', runtime['cuda_available'] and runtime['device_count'] == 2 and runtime['nccl_available'] and runtime['torch_version'] == '2.11.0+cu130' and w['GPU_name'] == 'Tesla T4')
        check(f'rank{rank} structure', all(w['structure'][k] for k in ('two_object_layers','one_object_guided_relation_layer','object_semantic_carry','separate_memories','packaging_decode_parity','masked_prompt_locality','invalid_slot_invariance','nonzero_output_text_sensitivity','no_strength_gate')) and w['structure']['decoder_layers'] == 3 and w['structure']['dropout'] == .2 and w['structure'] == g['structure'])
        check(f'rank{rank} independent stream response', all(math.isfinite(v) and v > 1e-7 for v in w['structure']['independent_stream_max_logit_change'].values()))
        check(f'rank{rank} gradient and causal checks', g['passed'] and g['causal_future_invariance'] and g['token_normalized_DDP_gradient_parity'] and g['visual_projection_grad_norm'] > 0 and all(math.isfinite(v) and v > 0 for v in g['module_gradient_norms'].values()))
        check(f'rank{rank} probe and learning', len(set(g['probe_caption_events'])) == 32 and all(0 <= v < 1280 for v in g['probe_caption_events']) and g['TRAIN_batch_NLL_before'] == w['TRAIN_probe_NLL_before'] and 0 < w['TRAIN_probe_NLL_after'] < w['TRAIN_probe_NLL_before'])
        check(f'rank{rank} parameters and beam check', g['parameters'] == w['parameters'] == 53183736 and w['beam5_path_replay'])
        check(f'rank{rank} timing finite and throughput arithmetic', all(math.isfinite(w[k]) and w[k] > 0 for k in ('steady_step_median_seconds','steady_step_p90_seconds','peak_allocated_GiB','seconds','beam8_images_seconds')) and math.isclose(w['global_caption_events_per_second'], 32/w['steady_step_median_seconds'], rel_tol=1e-12))
    log = files['notebook.log']; records = [ast.literal_eval(line.split('PATCH XE smoke ', 1)[1]) for line in log.splitlines() if line.startswith('PATCH XE smoke ')]
    check('complete200step progress log', [r['updates'] for r in records] == list(range(25, 201, 25)) and all(math.isfinite(r['TRAIN_NLL']) and r['TRAIN_NLL'] > 0 and r['grad_norm'] > 0 for r in records))
    workers = result['workers']; median = max(w['steady_step_median_seconds'] for w in workers)
    summary = dict(status='CSA_STRUCTURE_SMOKE_REVIEWED_COMPLETE', evaluation_source=dict(notebook_slug='kltnnnnnff', save_version=15, version_user_confirmed=True, notebook_inherited_from_session=True),
                   archive_sha256=cache['archive_sha256'], source_commit=source['commit'], checks_passed=len(checks), images=256, updates=200, global_caption_events=6400,
                   TRAIN_probe_NLL_before=workers[0]['TRAIN_probe_NLL_before'], TRAIN_probe_NLL_after=workers[0]['TRAIN_probe_NLL_after'], trainable_parameters=workers[0]['parameters'],
                   steady_step_median_seconds=median, steady_step_p90_seconds=max(w['steady_step_p90_seconds'] for w in workers), global_caption_events_per_second=32/median,
                   guard_peak_allocated_GiB=max(get(f'preflight_rank{r}.json')['guard_peak_allocated_GiB'] for r in (0,1)), train_and_final_probe_peak_allocated_GiB=max(w['peak_allocated_GiB'] for w in workers),
                   raw_stage_seconds=stage['seconds'], worker_seconds=max(w['seconds'] for w in workers), semantic_objects=semantic['objects'], semantic_relations=semantic['relations'],
                   first_rank_samples=workers[0]['sample_TRAIN_captions'], all_sample_captions_single_a=all(s == 'a' for w in workers for s in w['sample_TRAIN_captions']),
                   approximate_train_only_hours_per_full_epoch=17711*median/3600, approximate_train_only_hours_for10=177110*median/3600,
                   full_training_complete=False, quality_evaluated=False, test_used=False, validation_used=False,
                   limits=['Model/gradient/beam guards reported by user-run pinned code; no local model or binary validation. Metadata/source/log/arithmetic audit only.',
                           'Raw H5/NPY payloads absent from review ZIP; original staged SHA checks reported by runtime, not rehashed locally.',
                           'Eight TRAIN examples from first rank all decode as a; functional PASS is not caption quality, convergence or CIDEr evidence.',
                           'Tiny cached pool, variable caption lengths and early EOS: full-epoch timing is arithmetic only, excludes setup/staging/VALID/scorers/checkpoints and is not an ETA or billing estimate. Beam8 timing is not usable as mature-model VALID ETA.',
                           'Paper structure with CLIP substitutions and extra UPT prompt adapters, not exact full-system CSA. No new training decision or automatic restart.'])
    (HERE/'CSA_STRUCTURE_SMOKE_RESULT.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (HERE/'CSA_STRUCTURE_SMOKE_REVIEW_CHECKS.json').write_text(json.dumps(dict(checks_passed=len(checks), checks=checks, models_scorers_executed_locally=False), indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--cache', required=True); audit(p.parse_args().cache)
