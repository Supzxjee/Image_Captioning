"""Cached JSON/count/statistic audit only; no NLP, neural model or metric engine."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import numpy as np
from saved_quality_data import check_rows,chair_counts

HERE=Path(__file__).parent

def audit():
    cache=json.loads((HERE/'review/saved_quality_v2_loaded.json').read_text(encoding='utf-8'));f=cache['files']
    r=f['comparison.json'];source=f['source_receipt.json'];checks=0
    def check(v,label):
        nonlocal checks
        if not v:raise AssertionError(label)
        checks+=1
    check(r['status']=='SAVED_CAPTION_COMPARISON_COMPLETE' and f['run_status.json']['complete'],'completion')
    check(r['replay_mismatches']==f['replay_mismatches.json']==[] and r['cider_replay_valid'] and r['chair_replay_valid'],'strict replay passed')
    check(not r['model_executed'] and not r['checkpoints_loaded'] and not r['new_generation'] and not r['test_used'],'CPU-only scope')
    for name,h in source['sha256'].items():
        raw=subprocess.check_output(['git','show',source['commit']+':experiments/csa_paper/'+name],cwd=HERE.parents[1])
        check(hashlib.sha256(raw).hexdigest()==h,'source '+name)
    b=json.loads(gzip.decompress((HERE/'comparison_baseline4000.json.gz').read_bytes()))
    original=json.loads((HERE/'review/full_xe10_v2_loaded.json').read_text(encoding='utf-8'))['files']
    old_ids=set(b['cohort_ids']);all_ids={x['coco_id'] for x in original['validation_references.json']}
    snapshots={};vectors={}
    for cohort,ids in [('common4000',old_ids),('full5000',all_ids)]:
        groups=r['metrics'][cohort];expected_names={'A','C0','C','CSA9','CSA10'} if cohort=='common4000' else {'CSA9','CSA10'}
        check(set(groups)==expected_names,'cohort conditions')
        check(f[cohort+'_reference_tokenization_check.json']['images_differing_from_stored_CSA_tokens']==0,'tokenized reference replay')
        expected_refs={str(b['legacy_image_ids'][str(i)] if cohort=='common4000' else i):original['validation_ptb.json'][str(i)][:5 if cohort=='common4000' else None] for i in ids}
        check(f[cohort+'_references_ptb.json']==expected_refs,'actual tokenized references')
        vectors[cohort]={}
        for name,m in groups.items():
            label=cohort+'_'+name;rows=f[label+'_captions.json'];points=f[label+'_per_image.json']
            check_rows(rows,ids);check(len(points)==len(rows) and [x['coco_id'] for x in points]==[x['coco_id'] for x in rows],'caption/vector order')
            ch=f[label+'_chair_output.json'];metric,counts=chair_counts(ch,rows)
            check(metric=={k:m[k] for k in metric} and counts==m['counts'],'raw CHAIR reaggregation')
            check(f[label+'_metrics.json']['point']==m and not f[label+'_metrics.json']['replay_mismatches'],'per-condition replay')
            check(abs(sum(len(x['caption'].split()) for x in rows)/len(rows)-m['mean_words'])<1e-12,'caption length')
            check(abs(float(np.mean([x['CIDEr'] for x in points]))-m['CIDEr'])<1e-12,'per-image CIDEr mean')
            flags={x['image_id']:bool(x['mscoco_hallucinated_words']) for x in ch['sentences']}
            check(all(x['hallucination']==flags[x['coco_id']] and math.isfinite(x['CIDEr']) for x in points),'per-image false flags')
            if name in ('A','C0','C'):
                expected={x['coco_id']:x['caption'] for x in b['models'][name]}
                check({x['coco_id']:x['caption'] for x in rows}==expected,'old captions unchanged')
                historical=b['expected_metrics'][name]
                check(all(abs(m[k]-historical[k])<=1e-7 for k in ('CIDEr','CHAIRs','CHAIRi','Recall','Bleu_4','ROUGE_L','mean_words')),'old metrics reproduced')
            else:
                epoch=int(name[3:]);old=original[f'val_epoch{epoch:02d}_rank0.json']+original[f'val_epoch{epoch:02d}_rank1.json']
                expected={x['coco_id']:x['caption'] for x in old if x['coco_id'] in ids}
                check({x['coco_id']:x['caption'] for x in rows}==expected,'CSA captions unchanged')
                if cohort=='full5000':check(abs(m['CIDEr']-original['result.json']['history'][epoch-1]['cider'])<1e-12,'stored full5000 CIDEr')
            vectors[cohort][name]=points
        snapshots[cohort]=groups
    check(r['csa_loss_history']==original['result.json']['history'] and r['old_comparable_val_nll'] is None and r['shared_word_types']==6916,'loss boundary')
    rng=np.random.default_rng(20261103);paired={}
    for before,after in [('A','CSA9'),('C','CSA9'),('CSA9','CSA10')]:
        a=vectors['common4000'][before];brows=vectors['common4000'][after]
        check([x['coco_id'] for x in a]==[x['coco_id'] for x in brows],'paired IDs')
        d=np.array([y['CIDEr']-x['CIDEr'] for x,y in zip(a,brows)]);means=[]
        for _ in range(100):means.extend(d[rng.integers(0,len(d),size=(100,len(d)))].mean(1).tolist())
        key=after+'_minus_'+before;p=r['paired'][key]
        repaired=sum(x['hallucination'] and not y['hallucination'] for x,y in zip(a,brows));introduced=sum(not x['hallucination'] and y['hallucination'] for x,y in zip(a,brows))
        check((repaired,introduced)==(p['repaired'],p['introduced']),'paired repairs/new errors')
        check(abs(float(d.mean())-p['CIDEr_delta'])<1e-12 and np.allclose(np.quantile(means,[.025,.975]),p['paired95_CIDEr'],rtol=0,atol=1e-12),'paired CIDEr bootstrap')
        n=repaired+introduced
        tail=math.fsum(math.exp(math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)-n*math.log(2)) for k in range(min(repaired,introduced)+1)) if n else .5
        paired[key]={**p,'exact_McNemar_p':min(1.,2*tail)}
    result=dict(status='SAVED_QUALITY_REVIEW_COMPLETE',zip_sha256=cache['zip_sha256'],source_commit=source['commit'],
        source_files=len(source['sha256']),metadata_checks=checks,metrics=snapshots,paired=paired,
        loss_history=[dict(epoch=x['epoch'],train_nll=x['observed_train_nll'],val_nll=x['val_nll']) for x in r['csa_loss_history']],
        old_comparable_val_nll=None,official_metrics_rerun_locally=False,model_executed_by_agent=False)
    (HERE/'SAVED_QUALITY_RESULT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(metadata_checks=checks,paired=paired),indent=2))

if __name__=='__main__':audit()
