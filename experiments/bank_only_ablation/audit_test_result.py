"""Audit saved TEST output; no Torch, inference, retokenizing or metric scorers."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = HERE/'test_v245_review_cache'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<1e-10, (a,b)


def audit():
    archive = read(CACHE/'ARCHIVE_MANIFEST.json')
    assert archive['CRC_pass']
    for name, digest in archive['files'].items():
        assert sha(CACHE/name)==digest, name
    sources = read(CACHE/'source_receipt.json')
    expected_sources = read(HERE/'TEST_SOURCE_MANIFEST.json')
    assert sources['sha256']==expected_sources['sha256']
    for name, digest in sources['sha256'].items():
        assert sha(CACHE/'source'/name)==digest
    result = read(CACHE/'result.json')
    status = read(CACHE/'run_status.json')
    locked = read(CACHE/'test_lock.json')
    original_lock = read(CACHE/'source/TEST199_LOCK.json')
    checkpoint = read(CACHE/'checkpoint_source.json')
    receipt = read(CACHE/'stage_receipt.json')
    scoring = read(CACHE/'scoring_lock.json')
    expected_sha = expected_sources['checkpoint_sha256']
    assert status['complete'] and not status['training'] and not status['test_tuning']
    assert result['status']=='M2_ONLY_BANK_ALIGNMENT_TEST5000_COMPLETE'
    assert result['images']==5000 and result['epoch']==9 and not result['training'] and not result['reranking']
    assert result['source_version']==checkpoint['version']=='ducanh2403/2-vi-cai-kia-qua-nhieu/245'
    assert status['source_version']==locked['source_version']==result['source_version']
    assert result['seed']==status['seed']==42 and not result['test_tuning']
    assert locked['checkpoint_sha256']==checkpoint['sha256']==result['checkpoint_sha256']==scoring['checkpoint_sha256']==expected_sha
    assert locked['method']=='bank_only_parallel3_xe10_v1' and locked['seed']==42 and locked['updates']==159399
    for key in ('expected_test_ids','test_id_sha256','beam_width','max_new_tokens','length_normalization','reranking'):
        assert locked[key]==original_lock[key]
    assert locked['stage_receipt_sha256']==sha(CACHE/'stage_receipt.json')
    assert receipt['method']==locked['method'] and not receipt['TRAIN_VALID_features_read']
    for name, digest in receipt['output_files_sha256'].items():
        assert sha(CACHE/name)==digest
    for name, digest in scoring['files_sha256'].items():
        assert sha(CACHE/name)==digest
    workers = [read(CACHE/f'worker{r}.json') for r in (0,1)]
    assert all(w['complete'] and w['images']==2500 and not w['training'] and w['first_GPU_beam_path_check_passed'] and w['checkpoint_sha256']==expected_sha for w in workers)
    assert workers[0]['runtime_policy']==workers[1]['runtime_policy']==result['runtime_policy']
    assert result['method']==locked['method']=='bank_only_parallel3_xe10_v1'
    for rank, worker in enumerate(workers):
        guard_path = CACHE/f'bank_only_inference_rank{rank}.json'
        guard = read(guard_path)
        assert sha(guard_path)==worker['bank_guard_sha256']
        assert guard['passed'] and guard['images']==8 and not guard['training']
        assert all(guard[k] for k in ('M3_absent','M2_memories_verified','image_specific_inputs_ignored','bank_shapes_and_L2_verified'))
        assert guard['max_memory_difference']==0 and guard['parameters']==49239288
    rows = []
    words = read(CACHE/'vocabulary.json')
    checked_beams = 0
    for rank, worker in enumerate(workers):
        assert sha(CACHE/f'captions_rank{rank}.json')==worker['caption_sha256']
        assert sha(CACHE/f'beams_rank{rank}.json')==worker['beams_sha256']
        part = read(CACHE/f'captions_rank{rank}.json')
        beams = read(CACHE/f'beams_rank{rank}.json')
        assert len(part)==len(beams)==2500
        mapping = {r['coco_id']:r for r in part}
        for row in beams:
            candidates = row['candidates']
            assert len(candidates)==5
            assert all(candidates[i]['logprob']>=candidates[i+1]['logprob'] for i in range(4))
            for candidate in candidates:
                ids = candidate['ids']
                assert ids[0]==1 and 2<=len(ids)<=21 and math.isfinite(candidate['logprob'])
                tokens = []
                for word in ids[1:]:
                    if word==2:break
                    if word>1:tokens.append(words[word])
                assert candidate['caption']==' '.join(tokens)
                checked_beams += 1
            top = mapping[row['coco_id']]
            assert (top['caption'],top['ids'])==(candidates[0]['caption'],candidates[0]['ids'])
            close(top['score'],candidates[0]['logprob'])
        rows += part
    assert sorted(rows,key=lambda r:r['coco_id'])==read(CACHE/'captions.json')
    ids = {r['coco_id'] for r in rows}
    assert len(rows)==len(ids)==5000 and sorted(ids)==locked['expected_test_ids']
    references = read(CACHE/'test_references.json')
    assert len(references)==5000 and {r['coco_id'] for r in references}==ids
    assert set(read(CACHE/'reference_ptb.json'))==set(read(CACHE/'caption_ptb.json'))=={str(i) for i in ids}
    combined = {}
    for mode in ('basic','meteor','spice','chair'):
        point = read(CACHE/(mode+'.json'))
        assert point['images']==5000 and point['input_sha256']==scoring['files_sha256']
        combined.update(point['metrics'])
    assert set(combined)==set(result['metrics'])-{'NLL'}
    for name, value in combined.items():close(value,result['metrics'][name])
    close(result['metrics']['NLL'],sum(w['NLL_sum'] for w in workers)/sum(w['NLL_tokens'] for w in workers))
    for mode in ('cider','meteor','spice'):
        per = read(CACHE/(mode+'_per_image.json'))
        assert len(per)==5000 and {r['coco_id'] for r in per}==ids
        key = 'CIDEr' if mode=='cider' else 'score'
        assert all(math.isfinite(r[key]) for r in per)
        if mode in ('cider','spice'):close(sum(r[key] for r in per)/5000,result['metrics'][mode.upper() if mode=='spice' else 'CIDEr'])
    chair = read(CACHE/'chair_output.json')
    mapping = {r['coco_id']:r['caption'] for r in rows}
    sentences = chair['sentences']
    assert len(sentences)==5000 and {s['image_id'] for s in sentences}==ids
    assert all(s['caption']==mapping[s['image_id']] for s in sentences)
    counts = dict(images=5000,hallucinated_captions=sum(bool(s['mscoco_hallucinated_words']) for s in sentences),
        hallucinated_mentions=sum(len(s['mscoco_hallucinated_words']) for s in sentences),
        object_mentions=sum(len(s['mscoco_generated_words']) for s in sentences),
        recalled_unique=sum(len(set(s['mscoco_generated_words'])&set(s['mscoco_gt_words'])) for s in sentences),
        reference_unique=sum(len(set(s['mscoco_gt_words'])) for s in sentences))
    assert counts==result['chair_counts']==read(CACHE/'chair_counts.json')
    for key, a, b in [('CHAIRs',counts['hallucinated_captions'],5000),('CHAIRi',counts['hallucinated_mentions'],counts['object_mentions']),('Recall',counts['recalled_unique'],counts['reference_unique'])]:
        close(a/b,result['metrics'][key]);close(a/b,chair['overall_metrics'][key])
    references = {
        'baseline':read(ROOT/'experiments/matched_visual_baseline/TEST_RESULT.json'),
        'M2_only':result,
        'zero_text_M3':read(ROOT/'experiments/prompt_text_ablation/TEST_RESULT.json'),
        'full_prompt':read(ROOT/'experiments/clip_patch_baseline/CSA_PER_CAPTION_BEST9_TEST5000_RESULT.json')}
    for name, reference in references.items():
        for key in ('pycocoevalcap','java_archive_sha256','corenlp_archive_sha256','corenlp_version','evaluator_files'):
            assert reference['environment'][key]==result['environment'][key], (name,key)
        assert reference['beam_width']==5 and reference['max_new_tokens']==20
        assert reference['chair_counts']['reference_unique']==counts['reference_unique']
    metrics={k:v['metrics'] for k,v in references.items()}
    differences={}
    for before,after in [('baseline','M2_only'),('M2_only','zero_text_M3'),('zero_text_M3','full_prompt'),('M2_only','full_prompt'),('baseline','full_prompt')]:
        delta={k:(metrics[after][k]-metrics[before][k])*(1 if k=='NLL' else 100) for k in result['metrics']}
        differences[after+'_minus_'+before]=dict(points=delta,
            CHAIRs_relative_reduction=(metrics[before]['CHAIRs']-metrics[after]['CHAIRs'])/metrics[before]['CHAIRs'],
            CHAIRi_relative_reduction=(metrics[before]['CHAIRi']-metrics[after]['CHAIRi'])/metrics[before]['CHAIRi'],
            hallucinated_captions_difference=references[after]['chair_counts']['hallucinated_captions']-references[before]['chair_counts']['hallucinated_captions'])
    report=dict(result,test_output_source='ducanh2403/2-vi-cai-kia-qua-nhieu/248',
        archive_sha256=archive['archive_sha256'],
        audit=dict(archive_CRC_pass=True,archive_files=len(archive['files']),source_files=len(sources['sha256']),
            all_source_payload_sha_verified=True,scoring_and_worker_hashes_verified=True,
            caption_rows_checked=5000,beam_candidates_checked=checked_beams,CHAIR_captions_checked=5000,
            CIDEr_SPICE_means_and_NLL_verified=True,CHAIR_counts_verified=True,
            same_fixed_TEST_cohort_and_decode=True,same_evaluator_file_digests_across_four=True,
            bank_only_guard_hashes_and_real_GPU_receipts_verified=True,
            local_torch_or_inference=False,local_scorers_rerun=False,checkpoint_binary_present=False),
        timing=dict(stage_seconds=receipt['seconds'],max_worker_seconds=max(w['seconds'] for w in workers),
            worker_scope='decode + teacher-forced NLL, excludes setup/patch staging/CPUmetrics; not quota billing'),
        comparison=dict(split='Karpathy TEST5000',seed=42,caption_policy='top1 beam before SCST/rerank',
            metrics=metrics,counts={k:v['chair_counts'] for k,v in references.items()},deltas=differences,
            parameters=dict(baseline=29553424,M2_only=49239288,zero_text_M3=53183736,full_prompt=53183736),
            M2_only_mean_words=sum(len(r['caption'].split()) for r in rows)/5000,
            checkpoint_sha256={k:v['checkpoint_sha256'] for k,v in references.items()},
            limits=['One seed for these component ablations; no significance or formal noninferiority test.',
                'Baseline/M2-only/full counts differ; zero text keeps declared count but deactivates some input-weight columns.',
                'Best epochs9/9/8/9 all selected using maximum VALID CIDEr after ten CE epochs.',
                'Historical fullseed42 uses different patch I/O staging; new arms use direct H5.',
                'Previous project TEST development exposure; no untouched final holdout claim.',
                'CHAIR measures objects; no attribute/action/relation correctness claim.',
                'Component additions can interact; successive metric differences are not independent causal shares.',
                'Prior result JSON reused from earlier audits; no previous binaries or scorers rerun.']))
    if (HERE/'TEST_RESULT.json').is_file():
        old=read(HERE/'TEST_RESULT.json')
        if 'sheet_update' in old:report['sheet_update']=old['sheet_update']
    (HERE/'TEST_RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(audit=report['audit'],metrics=metrics,deltas=differences,timing=report['timing']),ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':audit()
