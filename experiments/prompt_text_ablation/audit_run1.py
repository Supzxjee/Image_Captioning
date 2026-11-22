"""Audit user-run partial XE metadata/captions only. No model or scorer execution."""
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
CACHE=HERE/'review_cache'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    transport=read(CACHE/'ARCHIVE_MANIFEST.json')
    for name,digest in transport['files'].items():assert sha(CACHE/name)==digest
    source=read(CACHE/'source_receipt.json');expected=read(HERE/'SOURCE_MANIFEST.json')
    assert source['sha256']==expected['sha256']
    for name,digest in source['sha256'].items():assert sha(CACHE/'source'/name)==digest
    state=read(CACHE/'progress.json');identity=read(CACHE/'identity.json');status=read(CACHE/'run_status.json')
    receipt=read(CACHE/'checkpoint_receipt.json');best=read(CACHE/'best_receipt.json')
    assert not status['complete'] and not any(status[k] for k in ('test_used','SCST','reranking'))
    assert (state['epoch'],state['batch_cursor'],state['updates'],state['caption_events'])==(8,17456,141433,4525821)
    assert receipt['updates']==state['updates'] and receipt['epoch']==8 and receipt['phase']=='train'
    assert state['updates']==(state['epoch']-1)*17711+state['batch_cursor']
    assert state['caption_events']==(state['epoch']-1)*566747+state['batch_cursor']*32
    assert identity['method']=='full_zero_prompt_text_xe10_v1' and identity['prompt_text_mode']=='zero' and identity['seed']==42
    assert identity['source']==source and identity['global_batch']==32 and identity['epochs']==10 and identity['dropout']==.2
    assert [r['epoch'] for r in state['history']]==list(range(1,8))
    assert state['best_epoch']==best['epoch']==max(state['history'],key=lambda r:r['CIDEr'])['epoch']==7
    assert state['best_CIDEr']==best['CIDEr']
    assert sha(CACHE/'training.jsonl')==receipt['training_log_sha256']
    log=[json.loads(line) for line in (CACHE/'training.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(log)==1420 and log[-1]['updates']<state['updates'] and state['updates']-log[-1]['updates']<100
    assert all(math.isfinite(r[k]) for r in log for k in ('NLL','lr','grad_norm','seconds'))
    assert all(a['updates']<b['updates'] for a,b in zip(log,log[1:]))
    contracts=[]
    for rank in (0,1):
        contract=read(CACHE/f'zero_text_contract_rank{rank}.json');guard=read(CACHE/f'preflight_rank{rank}.json')
        assert contract['trainable_parameters']==contract['full_trainable_parameters']==53183736
        for key in ('passed','same_state_keys_and_initial_tensors','M1_M2_banks_unchanged','ROI_geometry_masks_unchanged',
                    'input_prompt_dict_not_mutated','text_change_invariant','equals_full_receiving_zero_text','original_state_unchanged'):
            assert contract[key]
        assert contract['positive_control_mean_memory_change']>1e-7 and not contract['trained_checkpoint_loaded']
        assert all(guard[k] for k in ('passed','causal','ordinary_DDP_gradient_parity','beam_score_rescore'))
        assert guard['initial_tensor_sha256']==identity['initial_sha256'] and guard['discarded_benchmark_updates']==64
        contracts.append(contract)
    assert contracts[0]==contracts[1]
    parity=read(CACHE/'full_parity.json')
    assert [r['case'] for r in parity]==['first32','tail27'] and all(r['passed'] for r in parity)
    ids=None;checked_rows=0
    for row in state['history']:
        folder=CACHE/'evaluations'/f"epoch_{row['epoch']:02d}"
        for name,digest in row['files_sha256'].items():assert sha(folder/name)==digest
        rows=read(folder/'val_rank0.json')+read(folder/'val_rank1.json');cohort={r['coco_id'] for r in rows}
        assert len(rows)==len(cohort)==5000
        if ids is None:ids=cohort
        assert cohort==ids
        per=read(folder/'cider_per_image.json');assert len(per)==5000 and {r['coco_id'] for r in per}==ids
        assert abs(sum(r['CIDEr'] for r in per)/5000-row['CIDEr'])<1e-10
        workers=[read(folder/f'worker{rank}.json') for rank in (0,1)]
        assert abs(sum(w['NLL_sum'] for w in workers)/sum(w['NLL_tokens'] for w in workers)-row['NLL'])<1e-10
        assert abs(sum(len(r['caption'].split()) for r in rows)/5000-row['mean_words'])<1e-10
        checked_rows+=len(rows)
    log_text=(CACHE/'notebook.log').read_text(encoding='utf-8')
    assert "TimeoutError: Saved before parent/worker cap" in log_text
    assert not (CACHE/'result.json').exists() and not (CACHE/'training_complete.json').exists()
    report=dict(status='PARTIAL_CE10_CHECKPOINT_SAVED_AT_CAP',source_version='ducanh3/notebook7bc83c4c76/192',
        archive_sha256=transport['archive_sha256'],method=identity['method'],seed=42,prompt_text_mode='zero',
        completed_epochs=7,cursor_epoch=8,batch_cursor=17456,updates=141433,caption_events=4525821,
        target_updates=177110,remaining_updates=35677,remaining_epoch8_updates=255,
        best_epoch_so_far=7,best_VALID_CIDEr=best['CIDEr'],best_VALID_NLL=state['history'][-1]['NLL'],
        latest_checkpoint_sha256=receipt['sha256'],best_checkpoint_sha256=best['sha256'],training_log_sha256=receipt['training_log_sha256'],
        history=[{k:v for k,v in r.items() if k!='files_sha256'} for r in state['history']],
        failure_reason='Safety stop at worker deadline; checkpoint saved before parent cap. Not CUDA/model failure.',
        contracts=contracts[0],audit=dict(CRC_pass=True,archive_files=len(transport['files']),source_files=len(source['sha256']),
            payloads_and_source_and_trainlog_sha_verified=True,evaluation_hashes_and_CIDEr_NLL_means_verified=True,
            caption_rows_checked=checked_rows,training_log_rows=len(log),DDP_first32_and_tail27_pass=True,
            local_Torch_model_or_scorer_executed=False,checkpoint_binary_in_review=False),
        limits=['Incomplete10epoch arm; do not compare best7 versus fullbest9 as a final ablation.',
                'CHAIR and other best/end metrics not yet scored; no TEST.',
                'Actual binary/optimizer/RNG restore pending user-run; receipt-only checkpoint verification.',
                'Do not restart fresh, change the training schedule, or substitute the best7 snapshot for latestcursor.'])
    if (HERE/'RUN1_RESULT.json').is_file():
        prior=read(HERE/'RUN1_RESULT.json')
        if 'sheet_update' in prior:report['sheet_update']=prior['sheet_update']
    (HERE/'RUN1_RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','updates','remaining_updates','best_epoch_so_far','best_VALID_CIDEr','audit')},indent=2))
    return report


if __name__=='__main__':audit()
