"""Audit user-run XE15 stored metadata/text/counts, not models or scorers."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
from real_bank_data import HERE,read,save
from saved_quality_data import check_rows,chair_counts
from run_best9_decode import saved_text,tokens


def main():
    cache=read(HERE/'review/xe15_loaded.json');entries=cache['entries'];get=lambda n:json.loads(entries[n]);checks=0
    def check(v):
        nonlocal checks
        if not v:raise ValueError('Metadata check failed '+str(checks+1))
        checks+=1
    r=get('result.json');q=get('quality_result.json');status=get('run_status.json');source=get('source_receipt.json')
    check(r['status']=='FULL_XE_15_COMPLETE' and q['status']=='XE15_SAVED_QUALITY_COMPLETE')
    check(all(status[k] for k in ('complete','training_complete','quality_complete')) and not status['test_used'])
    check((r['completed_epochs'],r['updates'],r['initial_updates_this_session'],r['added_updates_this_session'],r['caption_events'])==(15,265665,177110,88555,8501205))
    check((r['train_images'],r['val_images'],r['test_images'])==(113287,5000,0) and not r['rl'])
    check('failure.txt' not in entries)
    for n,h in source['sha256'].items():
        raw=subprocess.check_output(['git','show',source['commit']+':experiments/csa_paper/'+n],cwd=HERE.parents[1])
        check(hashlib.sha256(raw).hexdigest()==h)
    old=read(HERE/'full_xe10_review_text/result.json');check(r['history'][:10]==old['history'])
    check([h['epoch'] for h in r['history']]==list(range(1,16)))
    check(all(h['updates']==h['epoch']*17711 and h['validation_images']==5000 and h['beam']==5 and h['max_new_tokens']==20 for h in r['history']))
    check(r['best_epoch']==max(r['history'],key=lambda h:h['cider'])['epoch']==15)
    best=get('best_receipt.json');last=get('checkpoint_receipt.json')
    check(best['sha256']==last['sha256'] and best['validated_epoch']==15 and last['epoch']==16 and last['updates']==265665)
    check(math.isclose(best['cider'],r['best_cider'],rel_tol=0,abs_tol=1e-12))
    migration=get('resume_migration.json');check(migration['kind']=='exact139-latest10-to15')
    check(migration['checkpoint_sha256']==read(HERE/'SAVE_XE139_PACKAGE.json')['files']['checkpoint.bin'])
    raw,_=saved_text();refs=get('validation_references.json');vocab=get('vocabulary.json')
    check(refs==json.loads(raw['validation_references.json']) and len(refs)==5000 and len({x['coco_id'] for x in refs})==5000)
    check(vocab==json.loads(raw['vocabulary.json']))
    baseline=read(HERE/'saved_quality_review_text/comparison.json')['metrics']['full5000']['CSA9'];check(q['metrics']['9']['counts']==baseline['counts'])
    check(all(abs(q['metrics']['9'][k]-baseline[k])<1e-7 for k in ('CIDEr','Bleu_4','ROUGE_L','CHAIRs','CHAIRi','Recall','mean_words')))
    flags={}
    for epoch in (9,11,12,13,14,15):
        rows=get(f'val_epoch{epoch:02d}_rank0.json')+get(f'val_epoch{epoch:02d}_rank1.json');check_rows(rows,{x['coco_id'] for x in refs})
        check(all(x['caption']==' '.join(vocab[t] for t in tokens(x['ids']) if t!=2) for x in rows))
        if epoch==9:check(rows==json.loads(raw['val_epoch09_rank0.json'])+json.loads(raw['val_epoch09_rank1.json']))
        ch=get(f'quality_epoch{epoch:02d}.chair_output.json');metrics,counts=chair_counts(ch,rows);point=q['metrics'][str(epoch)]
        check(counts==point['counts']);check(all(abs(point[k]-v)<1e-12 for k,v in metrics.items()))
        lengths=[len(x['caption'].split()) for x in rows]
        check(abs(sum(lengths)/len(rows)-point['mean_words'])<1e-12 and sum(x<=2 for x in lengths)==point['short_le2'] and sum(x==0 for x in lengths)==point['empty'])
        check(abs(point['CIDEr']-r['history'][epoch-1]['cider'])<1e-7)
        flags[epoch]={x['image_id']:bool(x['mscoco_hallucinated_words']) for x in ch['sentences']}
    a,b=flags[9],flags[15];paired=dict(repaired=sum(a[i] and not b[i] for i in a),introduced=sum(not a[i] and b[i] for i in a))
    out=dict(status='XE15_REVIEW_COMPLETE',zip_sha256=cache['zip_sha256'],source_commit=source['commit'],source_hashes_verified=len(source['sha256']),metadata_checks=checks,
        metrics=q['metrics'],history=r['history'],best_receipt=best,latest_receipt=last,paired9_to15=paired,session_seconds=r['session_seconds'],
        checkpoint_bytes_locally_verified=False,local_models_or_scorers_executed=False)
    save(HERE/'XE15_RESULT.json',out)
    dest=HERE/'xe15_review_text';dest.mkdir(exist_ok=True)
    for n in ('result.json','quality_result.json','run_status.json','best_receipt.json','checkpoint_receipt.json','source_receipt.json','runtime.json'):
        if n in entries:(dest/n).write_text(entries[n],encoding='utf-8')
    print('PASS',checks,'metadata checks',len(source['sha256']),'Git source hashes;',paired,'session hours',r['session_seconds']/3600)


if __name__=='__main__':main()
