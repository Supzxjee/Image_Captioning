"""Audit stored text/counts only. No Torch, models, or metric scorer execution."""
import hashlib
import json
import math
from pathlib import Path
import subprocess
from real_bank_data import read,save,HERE
from run_best9_decode import BEST_SHA,tokens,summary,saved_text
from saved_quality_data import check_rows,chair_counts


def main():
    cached=read(HERE/'review/best9_decode_loaded.json');entries=cached['entries'];get=lambda n:json.loads(entries[n])
    r=get('result.json');checks=0
    def check(value):
        nonlocal checks
        if not value:raise ValueError('Stored result audit failed at check '+str(checks+1))
        checks+=1
    check(r['status']=='BEST9_DECODE_COMPLETE' and get('run_status.json')['complete'])
    check(not r['training'] and not r['test_used'] and r['optimizer_steps']==0)
    check(r['checkpoint_sha256']==BEST_SHA and r['epoch']==9 and r['max_new_tokens']==20)
    check('failure.txt' not in entries)
    source=get('source_receipt.json')
    for n,h in source['sha256'].items():
        raw=subprocess.check_output(['git','show',source['commit']+':experiments/csa_paper/'+n],cwd=HERE.parents[1])
        check(hashlib.sha256(raw).hexdigest()==h)
    raw,_=saved_text();beam=get('beam5.json');greedy=get('greedy_rank0.json')+get('greedy_rank1.json')
    check(beam==json.loads(raw['val_epoch09_rank0.json'])+json.loads(raw['val_epoch09_rank1.json']))
    refs=get('validation_references.json');vocab=get('vocabulary.json');all_ids={x['coco_id'] for x in refs}
    check(refs==json.loads(raw['validation_references.json']) and len(all_ids)==5000)
    check(vocab==json.loads(raw['vocabulary.json']))
    for rank in (0,1):
        worker=get(f'worker_rank{rank}.json');rows=get(f'greedy_rank{rank}.json');replay=get(f'replay_rank{rank}.json')
        check(worker['complete'] and worker['optimizer_steps']==0 and worker['summary']==summary(rows))
        check([x['coco_id'] for x in rows]==[x['coco_id'] for x in refs][rank::2])
        check(len(replay)==8)
        for p in replay:
            check(tokens(p['actual_ids'])==tokens(p['expected_ids']) and math.isclose(p['actual_score'],p['expected_score'],rel_tol=1e-5,abs_tol=1e-4))
    expected=read(HERE/'saved_quality_review_text/comparison.json')['metrics'];paired={}
    for cohort in ('full5000','common4000'):
        ids={x['image_id'] for x in get(cohort+'_beam5_chair_output.json')['sentences']}
        check(len(ids)==(5000 if cohort=='full5000' else 4000))
        flags={}
        for name,all_rows in (('beam5',beam),('greedy',greedy)):
            rows=[x for x in all_rows if x['coco_id'] in ids];check_rows(rows,ids)
            ch=get(cohort+'_'+name+'_chair_output.json');metrics,counts=chair_counts(ch,rows);point=r['metrics'][cohort][name]
            check(counts==point['counts'])
            check(all(abs(v-point[k])<1e-12 for k,v in {**metrics,**summary(rows)}.items()))
            check(all(tokens(x['ids']) and x['caption']==' '.join(vocab[t] for t in tokens(x['ids']) if t!=2) and x['eos']==(2 in x['ids']) for x in rows))
            per=get(cohort+'_'+name+'_per_image.json');check_rows([dict(coco_id=x['coco_id'],caption='') for x in per],ids)
            check(abs(sum(x['CIDEr'] for x in per)/len(per)-point['CIDEr'])<1e-12)
            if name=='beam5':
                check(all(abs(point[k]-expected[cohort]['CSA9'][k])<=1e-7 for k in ('CIDEr','Bleu_4','ROUGE_L','CHAIRs','CHAIRi','Recall','mean_words')))
                check(counts==expected[cohort]['CSA9']['counts'])
            flags[name]={x['image_id']:bool(x['mscoco_hallucinated_words']) for x in ch['sentences']}
        b,g=flags['beam5'],flags['greedy']
        paired[cohort]=dict(beam_to_greedy_repaired=sum(b[i] and not g[i] for i in ids),beam_to_greedy_introduced=sum(not b[i] and g[i] for i in ids))
    b={x['coco_id']:x['caption'] for x in beam};check(sum(x['caption']!=b[x['coco_id']] for x in greedy)==r['changed_captions'])
    result=dict(status='BEST9_DECODE_REVIEW_COMPLETE',zip_sha256=cached['zip_sha256'],metadata_checks=checks,
        source_commit=source['commit'],source_hashes_verified=len(source['sha256']),result=r,paired_hallucinations=paired,
        worker_seconds=[get(f'worker_rank{i}.json')['seconds'] for i in (0,1)],local_model_or_scorer_execution=False)
    save(HERE/'BEST9_DECODE_RESULT.json',result)
    out=HERE/'best9_decode_review_text';out.mkdir(exist_ok=True)
    for n in ('result.json','run_status.json','worker_rank0.json','worker_rank1.json','source_receipt.json','prepared.json','saved_text_receipt.json'):
        (out/n).write_text(entries[n],encoding='utf-8')
    print('PASS',checks,'metadata checks;',len(source['sha256']),'source pins;',paired)


if __name__=='__main__':main()
