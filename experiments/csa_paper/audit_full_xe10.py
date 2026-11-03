"""Audit cached review JSON/logs only. No Torch, feature, checkpoint or scorer run."""
import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
from full_xe_data import STEPS,TRAIN_EVENTS,validate_state,validate_predictions,learning_rate

HERE=Path(__file__).parent

def audit():
    cache=json.loads((HERE/'review/full_xe10_v2_loaded.json').read_text(encoding='utf-8'));f=cache['files'];checks=0
    def check(value,label):
        nonlocal checks
        if not value:raise AssertionError(label)
        checks+=1
    r=f['result.json'];p=f['progress.json'];source=f['source_receipt.json'];identity=f['input_identity.json']
    check(r['status']=='FULL_XE_10_COMPLETE' and f['run_status.json']==dict(complete=True,requested_epochs=10,test_used=False),'complete')
    check((r['completed_epochs'],r['updates'],r['caption_events'])==(10,177110,5667470),'endpoint counters')
    validate_state(p);check(p['history']==r['history'] and p['epoch']==11 and p['batch_cursor']==0,'cursor means next epoch, not11 trained')
    check((r['train_images'],r['val_images'],r['test_images'])==(113287,5000,0),'split counts')
    check(not r['feature_sharing'] and not r['rl'] and not r['chair_computed'],'run scope')
    for name,h in source['sha256'].items():
        raw=subprocess.check_output(['git','show',source['commit']+':experiments/csa_paper/'+name],cwd=HERE.parents[1])
        check(hashlib.sha256(raw).hexdigest()==h,'Git source '+name)
    check(identity['source']==source['sha256'],'source identity')
    stage=f['stage_summary.json'];check(stage['source_package']==json.loads((HERE/'SAVE_FULL_ROI_V134_PACKAGE.json').read_text(encoding='utf-8')),'full input package')
    check(identity['stage_sha256']==stage['files_sha256'],'stage identity')
    check(identity['bank_package']==f['bank_identity.json']==json.loads((HERE/'SAVE_BANK_V130_PACKAGE.json').read_text(encoding='utf-8')),'bank identity')
    check(identity['software']==f['runtime.json']['software'] and f['runtime.json']['world_size']==2,'runtime')
    refs=f['validation_references.json'];ids=[x['coco_id'] for x in refs]
    check(len(ids)==len(set(ids))==5000 and sum(len(x['references']) for x in refs)==25010,'reference counts')
    manifest=json.loads((HERE/'full_roi_review_text/manifest.json').read_text(encoding='utf-8'))
    offset=0
    for shard in manifest['shards']:
        if shard['group']=='val':
            group=ids[offset:offset+shard['images']];offset+=len(group)
            check(hashlib.sha256(json.dumps(group).encode()).hexdigest()==shard['id_order_sha256'],'exact val shard IDs')
    check(offset==5000,'all five val shards')
    for name,key in [('validation_references.json','validation_references_sha256'),('validation_ptb.json','ptb_sha256')]:
        raw=(json.dumps(f[name],indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()
        check(hashlib.sha256(raw).hexdigest()==identity[key],'saved JSON identity '+name)
    check(set(map(int,f['validation_ptb.json']))==set(ids),'PTB reference IDs')
    check(f['beam_preflight.json']['passed'] and not f['beam_preflight.json']['batched_decoder'],'original beam guard')
    check(f['metric_preflight.json']['passed'] and f['metric_preflight.json']['reference_images']==5000,'metric guard')
    preflight=f['preflight/preflight.json'];check(len(preflight)==6 and f['preflight/result.json']['preflight_cases']==6,'DDP guards')
    check(all(abs(x['global_nll']-x['reference_nll'])<=5e-5 and all(g['relative_l2']<=.005 and g['max_abs']<=3e-5 for g in x['gradients'].values()) for x in preflight),'DDP tolerances')
    vocab=f['vocabulary.json'];epoch_rows=[]
    for epoch,h in enumerate(r['history'],1):
        metric=f[f'metrics_epoch{epoch:02d}.json'];a=f[f'val_epoch{epoch:02d}_rank0.json'];b=f[f'val_epoch{epoch:02d}_rank1.json'];rows=a+b
        check(len(a)==len(b)==2500 and not ({x['coco_id'] for x in a}&{x['coco_id'] for x in b}),'rank coverage')
        validate_predictions(rows,refs)
        check(h['epoch']==epoch and h['updates']==epoch*STEPS,'epoch/update order')
        check(all(h[k]==v for k,v in metric.items()),'standalone metrics agree')
        check(all(math.isfinite(h[k]) for k in ('cider','val_nll','observed_train_nll','validation_seconds')),'finite metrics')
        check(h['validation_images']==5000 and h['references']==25010 and h['val_target_tokens']==286871,'all-val coverage')
        check(h['beam']==5 and h['max_new_tokens']==20 and h['decoder']=='model.beam_search per-image','decode policy')
        captions=[];eos=0;bad=0
        for row in rows:
            seq=row['ids'];ended=2 in seq;content=seq[:seq.index(2)] if ended else seq
            if ended:
                bad+=int(any(t!=0 for t in seq[seq.index(2)+1:]))
            else:bad+=int(len(seq)!=20)
            bad+=int(not 1<=len(seq)<=20 or any(type(t)!=int or not 2<=t<len(vocab) for t in content))
            text=' '.join(vocab[t] for t in content if t!=0)
            bad+=int(text!=row['caption'] or ended!=row['eos']);captions.append(text);eos+=int(ended)
        check(bad==0,'IDs/EOS/PAD/rendered caption agreement')
        mean=sum(len(x.split()) for x in captions)/5000
        check(eos==h['eos_images'] and abs(mean-h['mean_words'])<1e-12,'recomputed caption summaries')
        epoch_rows.append(dict(epoch=epoch,cider=h['cider'],val_nll=h['val_nll'],train_nll=h['observed_train_nll'],
            mean_words=mean,eos=eos,empty=sum(not x for x in captions),unique_captions=len(set(captions)),
            one_or_two_words=sum(len(x.split())<=2 for x in captions),validation_seconds=h['validation_seconds']))
    best=max(r['history'],key=lambda x:x['cider']);br=f['best_receipt.json'];cr=f['checkpoint_receipt.json']
    check((best['epoch'],best['cider'])==(r['best_epoch'],r['best_cider'])==(p['best_epoch'],p['best_cider'])==(br['validated_epoch'],br['cider']),'best selection')
    check(br['epoch']==best['epoch']+1 and br['updates']==best['epoch']*STEPS and br['caption_events']==best['epoch']*TRAIN_EVENTS,'best next-epoch cursor')
    check(cr['epoch']==11 and cr['updates']==177110 and cr['caption_events']==5667470 and cr['phase']=='train' and cr['batch_cursor']==0,'latest receipt')
    logs=[ast.literal_eval(line[6:]) for line in f['notebook.log'].splitlines() if line.startswith('TRAIN {')]
    check([x['step'] for x in logs]==list(range(100,177111,100)),'all 1771 progress logs')
    check(all(x['epoch']==(x['step']-1)//STEPS+1 and math.isclose(x['lr'],learning_rate(x['step']),rel_tol=1e-12) and all(math.isfinite(x[k]) for k in ('nll','grad_norm','seconds')) for x in logs),'logged epoch/warmup/finite traces')
    check('Traceback (most recent call last)' not in f['notebook.log'] and 'FULL XE10 COMPLETE' in f['notebook.log'],'clean completed log')
    result=dict(status='FULL_XE_10_REVIEW_COMPLETE',zip_sha256=cache['zip_sha256'],source_commit=source['commit'],
        source_files=len(source['sha256']),metadata_checks=checks,epochs=epoch_rows,best_epoch=best['epoch'],best_cider=best['cider'],
        best_checkpoint_sha256_reported=br['sha256'],latest_checkpoint_sha256_reported=cr['sha256'],
        worker_hours=r['session_seconds']/3600,validation_minutes=sum(x['validation_seconds'] for x in r['history'])/60,
        epoch_phases_minus_validation_hours=sum(x['phase_seconds_this_session']-x['validation_seconds'] for x in r['history'])/3600,
        chair_computed=False,official_cider_rerun_locally=False,checkpoint_binaries_verified_locally=False,model_run_by_agent=False)
    (HERE/'FULL_XE_10_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':audit()
