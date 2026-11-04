"""Stored text/count/provenance audit, no model/scorer execution."""
import gzip
import hashlib
import json
import subprocess
from real_bank_data import HERE,read,save
from plain_test_data import PLAIN_SHA,check_test_equivalence
from saved_quality_data import check_rows,chair_counts
from run_best9_decode import tokens


def main():
    cache=read(HERE/'review/plain_self_test_loaded.json');e=cache['entries'];get=lambda n:json.loads(e[n]);checks=0
    def check(v):
        nonlocal checks
        if not v:raise ValueError('Stored TEST review failed at check '+str(checks+1))
        checks+=1
    r=get('result.json');c=get('comparison.json');status=get('run_status.json');source=get('source_receipt.json')
    check(status['complete'] and not status['training'] and status['test_used'] and not status['foreign_cache_api'])
    check(r['status']=='PLAIN10_KARPATHY_TEST_COMPLETE' and c['status']=='PLAIN10_VS_CSA10_TEST_COMPLETE')
    check('failure.txt' not in e)
    for n,h in source['sha256'].items():
        data=subprocess.check_output(['git','show',source['commit']+':experiments/csa_paper/'+n],cwd=HERE.parents[1]);check(hashlib.sha256(data).hexdigest()==h)
    check(r['checkpoint_sha256']==PLAIN_SHA and (r['epoch'],r['images'],r['beam'],r['max_new_tokens'])==(10,5000,5,20) and not r['training'])
    receipt=get('test_receipt.json');check_test_equivalence(receipt);checks+=1
    for n in ('test_metadata.json','vocabulary.json'):check(hashlib.sha256(e[n].encode()).hexdigest()==receipt['files'][n])
    metadata=get('test_metadata.json');ids={r['coco_id'] for r in metadata};vocab=get('vocabulary.json')
    check(len(ids)==len(metadata)==5000 and all(r['source_split']=='test' for r in metadata))
    rows=get('test_rank0.json')+get('test_rank1.json');check_rows(rows,ids);checks+=1
    check(all(r['caption']==' '.join(vocab[t] for t in tokens(r['ids']) if t!=2) for r in rows))
    metric,counts=chair_counts(get('test_chair_output.json'),rows);point=r['metrics']
    check(point['counts']==counts and all(abs(point[k]-v)<1e-12 for k,v in metric.items()))
    lengths=[len(r['caption'].split()) for r in rows]
    check(abs(sum(lengths)/5000-point['mean_words'])<1e-12 and sum(x<=2 for x in lengths)==point['short_le2'])
    workers=[]
    for rank in (0,1):
        w=get(f'test_worker{rank}.json');workers.append(w)
        check(w['complete'] and w['checkpoint_sha256']==PLAIN_SHA and w['optimizer_steps']==0 and w['images']==2500)
        check([r['coco_id'] for r in get(f'test_rank{rank}.json')]==[r['coco_id'] for r in metadata][rank::2])
        check(w['nll_tokens']==sum(sum(t!=0 for t in s[1:]) for r in metadata[rank::2] for s in r['sequences']))
    check(abs(sum(w['nll_sum'] for w in workers)/sum(w['nll_tokens'] for w in workers)-point['NLL'])<1e-12)
    old=read(HERE/'csa10_test_review_text/result.json')['metrics'];check(c['plain']==point and c['csa']==old)
    check(all(abs(c['csa_minus_plain'][k]-(old[k]-point[k]))<1e-12 for k in c['csa_minus_plain']))
    check(all(abs(c['metric_replay'][name][k]-target[k])<1e-7 for name,target in (('plain',point),('csa',old)) for k in ('CIDEr','Bleu_4','ROUGE_L')))
    saved=json.loads(gzip.decompress((HERE/'csa10_test_saved_text.json.gz').read_bytes()));flags=saved['hallucination']
    actual={str(r['image_id']):bool(r['mscoco_hallucinated_words']) for r in get('test_chair_output.json')['sentences']}
    paired=dict(plain_to_csa_repaired=sum(actual[str(i)] and not flags[str(i)] for i in ids),plain_to_csa_introduced=sum(not actual[str(i)] and flags[str(i)] for i in ids))
    check(paired==c['paired_hallucination'])
    result=dict(status='PLAIN_VS_CSA10_TEST_REVIEW_COMPLETE',zip_sha256=cache['zip_sha256'],metadata_checks=checks,source_commit=source['commit'],
        source_hashes_verified=len(source['sha256']),comparison=c,baseline_checkpoint_sha256=PLAIN_SHA,plain_test_cache_receipt=receipt,
        worker_seconds=[w['seconds'] for w in workers],checkpoint_or_npz_bytes_locally_verified=False,local_models_or_scorers_executed=False)
    save(HERE/'PLAIN_VS_CSA10_TEST_RESULT.json',result)
    out=HERE/'plain_self_test_review_text';out.mkdir(exist_ok=True)
    for n in ('result.json','comparison.json','run_status.json','test_lock.json','test_receipt.json','source_receipt.json','test_worker0.json','test_worker1.json'):
        (out/n).write_text(e[n],encoding='utf-8')
    print('PASS',checks,'metadata/text/count checks;',len(source['sha256']),'Git pins. TEST comparison complete.')


if __name__=='__main__':main()
