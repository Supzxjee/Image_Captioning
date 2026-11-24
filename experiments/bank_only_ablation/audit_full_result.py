"""Audit completed M2-only CE10 from saved metadata/captions; no models/scorers."""
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CACHE=HERE/'resume243_review_cache'
PRIOR=HERE/'run1_review_cache'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def close(a,b):assert math.isfinite(a) and math.isfinite(b) and abs(a-b)<1e-10,(a,b)


def audit():
    transport=read(CACHE/'ARCHIVE_MANIFEST.json')
    assert transport['CRC_pass']
    for name,digest in transport['files'].items():assert sha(CACHE/name)==digest
    source=read(CACHE/'source_receipt.json');expected=read(HERE/'SOURCE_MANIFEST.json')
    assert source==read(PRIOR/'source_receipt.json') and source['sha256']==expected['sha256']
    for name,digest in source['sha256'].items():assert sha(CACHE/'source'/name)==digest
    result=read(CACHE/'result.json');state=read(CACHE/'progress.json');identity=read(CACHE/'identity.json')
    status=read(CACHE/'run_status.json');parent=read(PRIOR/'progress.json');resumed=read(CACHE/'resume_source.json')
    assert result['status']=='BANK_ONLY_PARALLEL3_XE10_COMPLETE' and status['complete']
    assert result['epochs']==10 and result['updates']==177110 and result['caption_events']==5667470
    assert result['method']==status['method']==identity['method'] and result['seed']==status['seed']==42
    assert not any(status[k] for k in ('test_used','SCST','reranking'))
    assert (state['epoch'],state['batch_cursor'],state['phase'],state['updates'],state['caption_events'])==(11,0,'train',177110,5667470)
    assert identity==read(PRIOR/'identity.json') and identity['method']=='bank_only_parallel3_xe10_v1' and not identity['image_specific_prompts']
    assert resumed['version']=='ducanh2403/2-vi-cai-kia-qua-nhieu/243' and resumed['resume_updates']==147304 and resumed['remaining_updates']==29806
    assert resumed['checkpoint_sha256']==read(PRIOR/'checkpoint_receipt.json')['sha256']
    assert resumed['best_sha256']==read(PRIOR/'best_receipt.json')['sha256']
    assert sha(CACHE/'resume_helper.py')==resumed['wrapper_verifier_sha256']
    assert state['history'][:8]==parent['history'] and [r['epoch'] for r in state['history']]==list(range(1,11))
    assert state['best_epoch']==result['best_epoch']==max(state['history'],key=lambda r:r['CIDEr'])['epoch']==9
    for name in ('best_receipt.json','checkpoint_receipt.json'):
        assert result['receipts'][name]==read(CACHE/name)
    best=read(CACHE/'best_receipt.json');latest=read(CACHE/'checkpoint_receipt.json')
    assert best['epoch']==9 and best['CIDEr']==state['best_CIDEr']
    assert latest['updates']==177110 and latest['epoch']==11 and latest['phase']=='train'
    completed=read(CACHE/'training_complete.json')
    assert all(completed[k]==v for k,v in state.items())
    assert sha(CACHE/'training.jsonl')==read(CACHE/'checkpoint_receipt.json')['training_log_sha256']
    before=(PRIOR/'training.jsonl').read_bytes();after=(CACHE/'training.jsonl').read_bytes()
    assert after.startswith(before)
    old_log=[json.loads(l) for l in before.decode('utf-8').splitlines()]
    log=[json.loads(l) for l in after.decode('utf-8').splitlines()]
    assert len(old_log)==1480 and len(log)==1780 and log[-1]['updates']==177110
    assert log[len(old_log)]['updates']==147388 and log[len(old_log)]['updates']>parent['updates']
    assert log[len(old_log)]['lr']<old_log[-1]['lr'] and log[-1]['lr']==1e-5
    assert all(math.isfinite(r[k]) for r in log for k in ('NLL','lr','grad_norm','seconds'))
    assert all(a['updates']<b['updates'] for a,b in zip(log,log[1:]))
    for rank in (0,1):
        guard=read(CACHE/f'bank_only_contract_rank{rank}.json');pre=read(CACHE/f'preflight_rank{rank}.json')
        assert guard==read(PRIOR/f'bank_only_contract_rank{rank}.json') and guard['passed']
        assert guard['trainable_parameters']==49239288 and guard['full_parameters']==53183736
        assert pre['passed'] and pre['initial_tensor_sha256']==identity['initial_sha256']
    checked=0;cohort=None
    for row in state['history']:
        folder=CACHE/'evaluations'/f"epoch_{row['epoch']:02d}"
        for name,digest in row['files_sha256'].items():assert sha(folder/name)==digest
        rows=read(folder/'val_rank0.json')+read(folder/'val_rank1.json');ids={r['coco_id'] for r in rows}
        assert len(rows)==len(ids)==5000
        if cohort is None:cohort=ids
        assert ids==cohort
        per=read(folder/'cider_per_image.json');assert len(per)==5000 and {r['coco_id'] for r in per}==ids
        close(sum(r['CIDEr'] for r in per)/5000,row['CIDEr'])
        workers=[read(folder/f'worker{rank}.json') for rank in (0,1)]
        close(sum(w['NLL_sum'] for w in workers)/sum(w['NLL_tokens'] for w in workers),row['NLL'])
        close(sum(len(r['caption'].split()) for r in rows)/5000,row['mean_words']);checked+=len(rows)
    for epoch in (9,10):
        folder=CACHE/'quality'/f'epoch_{epoch:02d}'
        rows=read(CACHE/'evaluations'/f'epoch_{epoch:02d}'/'val_rank0.json')+read(CACHE/'evaluations'/f'epoch_{epoch:02d}'/'val_rank1.json')
        mapping={r['coco_id']:r['caption'] for r in rows};chair=read(folder/'chair_output.json');sentences=chair['sentences']
        assert len(sentences)==5000 and {r['image_id'] for r in sentences}==cohort
        assert all(r['caption']==mapping[r['image_id']] for r in sentences)
        counts=dict(images=5000,hallucinated_captions=sum(bool(r['mscoco_hallucinated_words']) for r in sentences),
            hallucinated_mentions=sum(len(r['mscoco_hallucinated_words']) for r in sentences),
            object_mentions=sum(len(r['mscoco_generated_words']) for r in sentences),
            recalled_unique=sum(len(set(r['mscoco_generated_words'])&set(r['mscoco_gt_words'])) for r in sentences),
            reference_unique=sum(len(set(r['mscoco_gt_words'])) for r in sentences))
        metric=result['metrics_by_epoch'][str(epoch)]
        assert metric['epoch']==epoch
        for key in ('CIDEr','NLL','mean_words'):
            close(metric[key],state['history'][epoch-1][key])
        assert counts==metric['counts'] and read(folder/'result.json')==metric
        for key,a,b in [('CHAIRs',counts['hallucinated_captions'],5000),('CHAIRi',counts['hallucinated_mentions'],counts['object_mentions']),('Recall',counts['recalled_unique'],counts['reference_unique'])]:
            close(a/b,metric[key]);close(metric[key],chair['overall_metrics'][key])
    refs = {
        'baseline': read(ROOT/'experiments/matched_visual_baseline/RESULT.json'),
        'M2_only': result,
        'zero_text_M3': read(ROOT/'experiments/prompt_text_ablation/FULL_XE10_RESULT.json'),
        'full_prompt': read(ROOT/'experiments/clip_patch_baseline/CSA_PER_CAPTION_FULL_XE10_RESULT.json')}
    comparisons = {}
    keys = ('CIDEr','BLEU_1','BLEU_4','ROUGE_L','CHAIRs','CHAIRi','Recall','NLL')
    for label in ('best_by_VALID_CIDEr','same_endpoint10'):
        rows = {name:r['metrics_by_epoch'][str(r['best_epoch'] if label=='best_by_VALID_CIDEr' else 10)] for name,r in refs.items()}
        delta = {}
        for before,after in [('baseline','M2_only'),('M2_only','zero_text_M3'),('zero_text_M3','full_prompt'),('M2_only','full_prompt')]:
            delta[after+'_minus_'+before] = {k:(rows[after][k]-rows[before][k])*(1 if k=='NLL' else 100) for k in keys}
        comparisons[label] = dict(split='Karpathy VALID5000',metrics=rows,deltas=delta)
    report = dict(result,source_version='ducanh2403/2-vi-cai-kia-qua-nhieu/245',parent_source_version=resumed['version'],
        archive_sha256=transport['archive_sha256'],history=[{k:v for k,v in r.items() if k!='files_sha256'} for r in state['history']],
        audit=dict(CRC_pass=True,archive_files=len(transport['files']),source_files=len(source['sha256']),
            source_identity_and_run1_history_preserved=True,training_log_prefix_preserved=True,
            source_and_evaluation_hashes_verified=True,caption_rows_checked=checked,CHAIR_caption_rows_checked=10000,
            CIDEr_NLL_and_CHAIR_aggregates_verified=True,actual_parent_API_declared='exactv243',
            local_Torch_or_model_or_scorer_executed=False,checkpoint_binary_present=False),
        resume_worker_seconds=read(CACHE/'training_complete.json')['worker_seconds'],
        comparisons=comparisons,comparison_limits=[
            'One seed for baseline/M2-only/zero-text; reused VALID is used for checkpoint selection, not independent TEST.',
            'M2-only49,239,288 vs baseline29,553,424 vs full/zero53,183,736 parameters: component ablation, not capacity matched.',
            'Zero-text/full retain equal declared count but text0 deactivates corresponding input-weight columns.',
            'Historical fullseed42 used different patch I/O staging; new baseline/M2-only/zero use direct H5.',
            'Best epochs9/9/8/9 under the same maxVALIDCIDEr rule; also report common endpoint10.',
            'No statistical significance, formal equivalence or action/relation hallucination claim from CHAIR.',
            'METEOR/SPICE and M2-only TEST not measured; keep existing best checkpoint selection unchanged.'])
    if (HERE/'FULL_XE10_RESULT.json').is_file():
        prior=read(HERE/'FULL_XE10_RESULT.json')
        if 'sheet_update' in prior:report['sheet_update']=prior['sheet_update']
    (HERE/'FULL_XE10_RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(source=report['source_version'],audit=report['audit'],best=comparisons['best_by_VALID_CIDEr'],resume_worker_seconds=report['resume_worker_seconds']),ensure_ascii=False,indent=2))
    return report


if __name__=='__main__':audit()
