"""Stored metadata/count/text audit only. No neural model or scorer execution."""
import hashlib
import json
import math
import subprocess
from real_bank_data import HERE,read,save
from run_best9_decode import saved_text,tokens
from saved_quality_data import check_rows,chair_counts


def main():
    cached={n:read(HERE/'review'/(n+'_loaded.json')) for n in ('plain_full_xe_10','csa_epoch10_karpathy_test')}
    entries={n:c['entries'] for n,c in cached.items()};p=entries['plain_full_xe_10'];t=entries['csa_epoch10_karpathy_test']
    get=lambda e,n:json.loads(e[n]);checks=0
    def check(v):
        nonlocal checks
        if not v:raise ValueError('Stored review check failed '+str(checks+1))
        checks+=1
    pr=get(p,'result.json');tr=get(t,'result.json');source=get(p,'source_receipt.json')
    check(source==get(t,'source_receipt.json'))
    for n,h in source['sha256'].items():
        raw=subprocess.check_output(['git','show',source['commit']+':experiments/csa_paper/'+n],cwd=HERE.parents[1]);check(hashlib.sha256(raw).hexdigest()==h)
    check(get(p,'run_status.json')['complete'] and get(t,'run_status.json')['complete'])
    check('failure.txt' not in p and 'failure.txt' not in t)
    check(pr['status']=='PLAIN_XE_10_COMPLETE' and tr['status']=='CSA10_KARPATHY_TEST_COMPLETE')
    check((pr['updates'],pr['caption_events'],pr['initial_updates_this_session'],pr['test_images'])==(177110,5667470,0,0))
    check(pr['completed_epochs']==10 and pr['arm']=='plain' and not pr['rl'])
    check([r['epoch'] for r in pr['history']]==list(range(1,11)))
    check(all(r['updates']==r['epoch']*17711 and r['validation_images']==5000 and r['beam']==5 and r['max_new_tokens']==20 for r in pr['history']))
    check(pr['best_epoch']==10 and pr['best_cider']==pr['history'][9]['cider'])
    raw,_=saved_text();old=json.loads(raw['input_identity.json']);identity=get(p,'input_identity.json')
    for key in ('stage_sha256','bank_package','validation_references_sha256','ptb_sha256','software','cuda','world_size','global_batch','seed','precision','epochs'):
        check(identity[key]==old[key])
    check(identity['schema']=='plain_full_xe10_v1' and identity['arm']=='plain')
    smoke=get(p,'plain_smoke.json');check(smoke['passed'] and smoke['common_initial_weights_exact'] and smoke['all_gradients_finite'] and not smoke['semantic_bank_used'])
    check(get(p,'preflight/result.json')['preflight_cases']==6)
    refs=get(p,'validation_references.json');vocab=get(p,'vocabulary.json')
    check(refs==json.loads(raw['validation_references.json']) and vocab==json.loads(raw['vocabulary.json']))
    val_ids={r['coco_id'] for r in refs}
    for epoch in range(1,11):
        rows=get(p,f'val_epoch{epoch:02d}_rank0.json')+get(p,f'val_epoch{epoch:02d}_rank1.json');check_rows(rows,val_ids)
        check(all(r['caption']==' '.join(vocab[x] for x in tokens(r['ids']) if x!=2) for r in rows))
        check(abs(sum(len(r['caption'].split()) for r in rows)/5000-pr['history'][epoch-1]['mean_words'])<1e-12)
    receipt=get(t,'test_receipt.json');metadata=get(t,'test_metadata.json');ids={r['coco_id'] for r in metadata}
    check(len(metadata)==len(ids)==5000 and not ids&val_ids and all(r['source_split']=='test' for r in metadata))
    check(hashlib.sha256(t['test_metadata.json'].encode()).hexdigest()==receipt['files']['test_metadata.json'])
    check(hashlib.sha256(t['vocabulary.json'].encode()).hexdigest()==receipt['files']['vocabulary.json'])
    check(get(t,'vocabulary.json')==vocab)
    profile=read(HERE/'full_roi_review_text/input_receipt.json');check(receipt['feature_profile']==profile['feature_profile'])
    check(receipt['dataset_json_sha256']==profile['dataset_json_sha256'] and receipt['detector_sha256']==profile['detections_sha256'])
    check(receipt['group']=='test' and receipt['training_images']==receipt['validation_images']==0)
    rows=get(t,'test_rank0.json')+get(t,'test_rank1.json');check_rows(rows,ids)
    check(all(r['caption']==' '.join(vocab[x] for x in tokens(r['ids']) if x!=2) for r in rows))
    ch,counts=chair_counts(get(t,'test_chair_output.json'),rows);point=tr['metrics']
    check(counts==point['counts'] and all(abs(point[k]-v)<1e-12 for k,v in ch.items()))
    check(abs(sum(len(r['caption'].split()) for r in rows)/5000-point['mean_words'])<1e-12)
    check(sum(len(r['caption'].split())<=2 for r in rows)==point['short_le2'])
    workers=[]
    for rank in (0,1):
        w=get(t,f'test_worker{rank}.json');workers.append(w)
        check(w['complete'] and w['images']==2500 and w['optimizer_steps']==0 and w['epoch']==10)
        check([r['coco_id'] for r in get(t,f'test_rank{rank}.json')]==[r['coco_id'] for r in metadata][rank::2])
        check(w['nll_tokens']==sum(sum(x!=0 for x in s[1:]) for r in metadata[rank::2] for s in r['sequences']))
    check(abs(sum(w['nll_sum'] for w in workers)/sum(w['nll_tokens'] for w in workers)-point['NLL'])<1e-12)
    check(tr['checkpoint_sha256']==read(HERE/'SAVE_XE139_PACKAGE.json')['files']['checkpoint.bin'] and not tr['training'])
    check(get(t,'test_lock.json')['epoch']==10)
    best=get(p,'best_receipt.json');last=get(p,'checkpoint_receipt.json')
    check(best['sha256']==last['sha256'] and last['updates']==177110 and best['validated_epoch']==10)
    baseline_summary=dict(epoch=10,validation=pr['history'][9],checkpoint=last,best_receipt=best,parameters=get(p,'runtime.json')['trainable_parameters'],session_seconds=pr['session_seconds'],test_available=False)
    result=dict(status='PLAIN10_TRAIN_AND_CSA10_TEST_REVIEW_COMPLETE',metadata_checks=checks,source_hashes_verified=len(source['sha256']),
        source_commit=source['commit'],zip_sha256={n:c['zip_sha256'] for n,c in cached.items()},plain=baseline_summary,csa_test=tr,
        csa_validation10=read(HERE/'full_xe10_review_text/result.json')['history'][9],test_cache_receipt=receipt,
        local_models_scorers_or_binary_arrays_read=False,baseline_test_still_required=True)
    save(HERE/'PLAIN10_CSA10_RESULT.json',result)
    for name,e,names in [('plain10_review_text',p,('result.json','run_status.json','checkpoint_receipt.json','best_receipt.json','runtime.json','plain_smoke.json','input_identity.json','source_receipt.json')),
                         ('csa10_test_review_text',t,('result.json','run_status.json','test_lock.json','test_receipt.json','test_worker0.json','test_worker1.json','source_receipt.json'))]:
        out=HERE/name;out.mkdir(exist_ok=True)
        for n in names:(out/n).write_text(e[n],encoding='utf-8')
    print('PASS',checks,'metadata/text/count checks;',len(source['sha256']),'source Git hashes. Baseline TEST still required.')


if __name__=='__main__':main()
