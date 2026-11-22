"""Audit saved TEST output; no Torch, inference, retokenizing or metric scorers."""
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = HERE/'test_review_cache'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<1e-10, (a,b)


def audit():
    archive = read(CACHE/'ARCHIVE_MANIFEST.json')
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
    assert result['status']=='BASELINE_MATCHED_PARALLEL3_TEST5000_COMPLETE'
    assert result['images']==5000 and result['epoch']==9 and not result['training'] and not result['reranking']
    assert result['source_version']==checkpoint['version']=='ducanh3/notebook7bc83c4c76/188'
    assert locked['checkpoint_sha256']==checkpoint['sha256']==result['checkpoint_sha256']==scoring['checkpoint_sha256']==expected_sha
    assert locked['method']=='matched_visual_parallel3_xe10_v1' and locked['seed']==42 and locked['updates']==159399
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
    full = read(ROOT/'experiments/clip_patch_baseline/CSA_PER_CAPTION_BEST9_TEST5000_RESULT.json')
    rerank = read(ROOT/'experiments/clip_patch_baseline/CSA_PER_CAPTION_BEST9_RERANK_TEST5000_RESULT.json')
    for key in ('pycocoevalcap','java_archive_sha256','corenlp_archive_sha256','corenlp_version','evaluator_files'):
        assert full['environment'][key]==result['environment'][key]
    assert result['beam_width']==full['beam_width']==5 and result['max_new_tokens']==full['max_new_tokens']==20
    assert full['chair_counts']['reference_unique']==counts['reference_unique']
    b, f = result['metrics'], full['metrics']
    delta = {k:(f[k]-b[k])*(1 if k=='NLL' else 100) for k in b}
    report = dict(result, test_output_source='ducanh3/notebook7bc83c4c76/189',
        archive_sha256=archive['archive_sha256'],
        audit=dict(archive_CRC_pass=True,archive_files=len(archive['files']),source_files=len(sources['sha256']),
            all_source_payload_sha_verified=True,scoring_and_worker_hashes_verified=True,
            caption_rows_checked=5000,beam_candidates_checked=checked_beams,CHAIR_captions_checked=5000,
            CIDEr_SPICE_means_and_NLL_verified=True,CHAIR_counts_verified=True,
            same_fixed_TEST_cohort_and_decode=True,same_evaluator_file_digests_as_full=True,
            local_torch_or_inference=False,local_scorers_rerun=False,checkpoint_binary_present=False),
        timing=dict(stage_seconds=receipt['seconds'],max_worker_seconds=max(w['seconds'] for w in workers),
            worker_scope='decode + teacher-forced NLL, excludes setup/patch staging/CPUmetrics; not quota billing'),
        comparison=dict(split='Karpathy TEST5000',seed=42,caption_policy='beam top1 before SCST and rerank',
            baseline_metrics=b,full_metrics=f,full_minus_baseline_points=delta,
            full_checkpoint_sha256=full['checkpoint_sha256'],baseline_counts=counts,full_counts=full['chair_counts'],
            full_CHAIRs_relative_reduction=(b['CHAIRs']-f['CHAIRs'])/b['CHAIRs'],
            full_CHAIRi_relative_reduction=(b['CHAIRi']-f['CHAIRi'])/b['CHAIRi'],
            baseline_mean_words=sum(len(r['caption'].split()) for r in rows)/5000,
            full_mean_words=rerank['metrics']['decoder_rank0']['mean_words'],
            full_plus_fixed_rerank_metrics=rerank['metrics']['clip_object_fixed'],
            baseline_parameters=29553424,full_parameters=53183736,
            limits=['One matched baseline seed; full model has 23,630,312 additional parameters.',
                    'Evaluates entire semantic addition, not isolated prompt text or M2/M3 individually.',
                    'Historical full seed42 used local FP16 patch staging; new baseline trained on direct H5.',
                    'Previous project TEST development exposure; no claim of an untouched final holdout.',
                    'CHAIR covers objects; no action/relation correctness claim or significance test.',
                    'Historical full result reused from previously audited result JSON; no old binary or scorer rerun.']))
    if (HERE/'TEST_RESULT.json').is_file():
        old = read(HERE/'TEST_RESULT.json')
        if 'sheet_update' in old:report['sheet_update']=old['sheet_update']
    (HERE/'TEST_RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(audit=report['audit'],delta=delta,CHAIRs_relative_reduction=report['comparison']['full_CHAIRs_relative_reduction'],timing=report['timing']),ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':
    audit()
