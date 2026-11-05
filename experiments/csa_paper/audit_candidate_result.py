"""Audit cached beam text/counts only. Never imports models or metric scorers."""
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from candidate_data import baseline,check_record,choose_oracle,CHECKPOINT_SHA
from real_bank_data import read,sha,save
from saved_quality_data import chair_counts

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


def run():
    folder=HERE/'candidates161_review_text';checks=0
    def check(value):
        nonlocal checks
        if not value:raise ValueError('Candidate audit failed at check'+str(checks+1))
        checks+=1
    result=read(folder/'result.json');saved=baseline();source=read(folder/'source_receipt.json')
    check(read(folder/'saved_text_receipt.json')['crc_pass'])
    check(read(folder/'run_status.json')==dict(complete=True,training=False,test_used=False,actual_reranking=False))
    check(result['status']=='CSA_CANDIDATE_DIAGNOSTIC_COMPLETE' and result['images']==5000 and result['oracle_uses_references'])
    check(not any(result[k] for k in ('test_used','training','actual_reranking')))
    check(result['checkpoint_sha256']==CHECKPOINT_SHA and source['gpu']==dict(built=True,available=True,count=2))
    for n,h in source['sha256'].items():
        check(hashlib.sha256(subprocess.check_output(['git','show',source['commit']+':experiments/csa_paper/'+n],cwd=ROOT)).hexdigest()==h)
    check(read(folder/'checkpoint_source.json')==dict(source=saved['source'],sha256=CHECKPOINT_SHA))
    check(read(folder/'validation_references.json')==saved['references'] and read(folder/'vocabulary.json')==saved['vocabulary'])
    check(read(folder/'metric_preflight.json')['passed'] and read(folder/'metric_preflight.json')['metrics']=={k:v for k,v in saved['metrics'].items() if k!='NLL' and k!='mean_words'})
    check(read(folder/'prepared.json')['checkpoint_sha256']==CHECKPOINT_SHA)
    rows=[];workers=[]
    for rank in (0,1):
        path=folder/f'candidates_rank{rank}.jsonl';part=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
        w=read(folder/f'worker{rank}.json');workers.append(w)
        check(w['complete'] and w['images']==w['rank0_replayed']==2500 and w['max_score_error']==0)
        check(w['checkpoint_sha256']==CHECKPOINT_SHA and sha(path)==w['candidates_sha256'])
        check([r['coco_id'] for r in part]==saved['order'][rank::2]);rows.extend(part)
    check(len(rows)==5000 and len({r['coco_id'] for r in rows})==5000)
    check([r['coco_id'] for r in rows]==[r['coco_id'] for r in saved['predictions']])
    for row,old in zip(rows,saved['predictions']):
        if check_record(row,old,saved['vocabulary'])!=0:raise ValueError('Rank0 score differs')
    check(True)
    cider=read(folder/'candidate_cider.json')
    check([r['coco_id'] for r in cider]==[r['coco_id'] for r in rows])
    values=np.asarray([r['CIDEr'] for r in cider]).T;check(values.shape==(5,5000) and np.isfinite(values).all())
    old_per={r['coco_id']:r['CIDEr'] for r in read(HERE/'scst_full1_ce_review_text/evaluations/end/cider_per_image.json')}
    check(all(abs(values[0,i]-old_per[r['coco_id']])<1e-12 for i,r in enumerate(rows)))
    selected,choices,upper=choose_oracle(rows,values)
    check(choices==[r['oracle_rank'] for r in cider])
    check(read(folder/'reference_oracle_diagnostic_only_captions.json')==selected)
    check(abs(upper-result['metrics']['reference_oracle_diagnostic_only']['CIDEr'])<1e-12)
    check(abs(upper-values[0].mean()-result['CIDEr_oracle_headroom'])<1e-12)
    check({str(k):choices.count(k) for k in range(5)}==result['oracle_rank_counts'])
    diversity={str(k):sum(len({c['caption'] for c in r['candidates']})==k for r in rows) for k in range(1,6)}
    check(diversity==result['unique_candidate_histogram'])
    check(sum(all(len(c['caption'].split())<=2 for c in r['candidates']) for r in rows)==result['all_five_short'])
    chair_data={};lengths={}
    for k,name in enumerate(['rank'+str(k) for k in range(5)]+['reference_oracle_diagnostic_only']):
        predictions=read(folder/(name+'_captions.json'))
        expected=selected if k==5 else [dict(coco_id=r['coco_id'],caption=r['candidates'][k]['caption']) for r in rows]
        check(predictions==expected)
        check(read(folder/(name+'_chair_input.json'))==[dict(image_id=r['coco_id'],caption=r['caption']) for r in predictions])
        chair=read(folder/(name+'_chair_output.json'));metric,counts=chair_counts(chair,predictions);m=result['metrics'][name]
        check(counts==m['counts'] and all(abs(v-m[key])<1e-12 for key,v in metric.items()))
        check(abs(m['CIDEr']-(upper if k==5 else values[k].mean()))<1e-12)
        lengths[name]=np.array([len(r['caption'].split()) for r in predictions])
        check(int((lengths[name]<=2).sum())==m['short_le2'])
        chair_data[name]={r['image_id']:r for r in chair['sentences']}
    base_chair=chair_data['rank0'];oracle_chair=chair_data['reference_oracle_diagnostic_only'];ids=[r['coco_id'] for r in rows]
    check(all(set(base_chair[i]['mscoco_gt_words'])==set(chair_data[n][i]['mscoco_gt_words']) for n in chair_data for i in ids))
    a=np.array([bool(base_chair[i]['mscoco_hallucinated_words']) for i in ids]);b=np.array([bool(oracle_chair[i]['mscoco_hallucinated_words']) for i in ids])
    changes=dict(changed_captions=sum(k!=0 for k in choices),strictly_higher_per_image_cider=int((values.max(0)>values[0]).sum()),
        CHAIR_repairs=int((a&~b).sum()),CHAIR_new_flags=int((~a&b).sum()),
        short_to_long=int(((lengths['rank0']<=2)&(lengths['reference_oracle_diagnostic_only']>2)).sum()),
        long_to_short=int(((lengths['rank0']>2)&(lengths['reference_oracle_diagnostic_only']<=2)).sum()))
    check(changes['CHAIR_repairs']-changes['CHAIR_new_flags']==18)
    check(changes['short_to_long']-changes['long_to_short']==346)
    log=(folder/'notebook.log').read_text(encoding='utf-8')
    check('Ran 7 tests' in log and 'exact161 routing mock PASS' in log and 'Historical5000 metric replay PASS' in log and 'Traceback' not in log)
    report=dict(status='CANDIDATE_STORED_TEXT_AUDIT_PASS',checks=checks,source=source['commit'],
        zip_sha256=read(folder/'saved_text_receipt.json')['zip_sha256'],metrics=result['metrics'],
        oracle_headroom_publication_points=result['CIDEr_oracle_headroom']*100,changes=changes,
        oracle_rank_counts=result['oracle_rank_counts'],unique_candidate_histogram=diversity,all_five_short=result['all_five_short'],
        workers=workers,mean_words={k:float(v.mean()) for k,v in lengths.items()},test_used=False,
        actual_reranking=False,oracle_uses_references=True,
        limitations=['No author model/scorer execution; BLEU/ROUGE oracle values are stored user-run outputs, not independently recomputed.',
            'Checkpoint/feature binary payloads absent from ZIP; SHA checks are recorded user-run evidence.',
            'Oracle is reference-dependent CIDEr upper bound within these five beams, not an inference-only method or upper bound for minimum hallucination.',
            'CHAIR evaluates object mentions, not all relation/attribute correctness; no human quality confirmation.'])
    save(HERE/'CANDIDATE_DIAGNOSTIC_RESULT.json',report)
    print(json.dumps(dict(checks=checks,changes=changes,oracle_headroom=report['oracle_headroom_publication_points'],mean_words=report['mean_words']),indent=2))


if __name__=='__main__':run()
