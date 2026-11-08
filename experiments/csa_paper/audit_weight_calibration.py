"""Audit cached CPU190 outputs; no official metric/model execution."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
from weight_calibration_data import ORIGINAL,split_ids,grid,choose,chair_counts,metrics,guards,mean_cider,fit,accept

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'valid_rerank_diagnostic'

def audit(path):
    cached=json.loads(path.read_text(encoding='utf-8'));files=cached['files'];checks=0
    def check(ok,label):
        nonlocal checks
        if not ok:raise AssertionError(label)
        checks+=1
    def read(n):return json.loads(files[n])
    def equal(x,y,label):
        if isinstance(x,dict):
            check(set(x)==set(y),label+' keys')
            for k,v in x.items():equal(v,y[k],label+'/'+k)
        elif isinstance(x,list):
            check(len(x)==len(y),label+' length')
            for i,(a,b) in enumerate(zip(x,y)):equal(a,b,label+'/'+str(i))
        elif isinstance(x,float):check(math.isfinite(x) and abs(x-y)<1e-12,label)
        else:check(x==y,label)
    receipt=read('source_receipt.json')
    check(receipt['commit']=='b2ebae2e75aff02bba77c777e0e24940f8bacfa4','source commit')
    for n,h in receipt['sha256'].items():
        raw=subprocess.check_output(['git','show',receipt['commit']+':'+n],cwd=ROOT)
        check(hashlib.sha256(raw).hexdigest()==h,'source SHA '+n)
    bundle=json.loads(gzip.decompress((OUT/'VALID_CANDIDATES.json.gz').read_bytes()))
    parity=json.loads(gzip.decompress((OUT/'CALIBRATION_PARITY.json.gz').read_bytes()))
    splits=read('split.json');ids=[r['coco_id'] for r in bundle['arms']['CONTROL']['records']]
    equal(split_ids(ids),splits,'split')
    check(hashlib.sha256(files['split.json'].encode()).hexdigest()==read('selection_lock.json')['split_sha256'],'split SHA')
    check(read('replay.json')['status']=='PASS','historical replay')
    equal(read('run_status.json'),dict(complete=True,training=False,decoding=False,test_used=False),'status')
    rows={a:{r['coco_id']:r for r in s['records']} for a,s in bundle['arms'].items()}
    chairs={a:{i:[None]*5 for i in ids} for a in rows}
    for arm in rows:
        for rank in range(5):
            data=read(f'{arm}_CHAIR_rank{rank}.json');sentences=data['sentences']
            check(len(sentences)==5000 and {s['image_id'] for s in sentences}==set(ids),'CHAIR cohort')
            counts=[0]*5
            for s in sentences:
                i=s['image_id'];check(s['caption']==rows[arm][i]['candidates'][rank]['caption'],'caption identity')
                c=chair_counts(s);chairs[arm][i][rank]=c
                for j,v in enumerate(c):counts[j]+=v
                if rank==0:equal(c,parity['arms'][arm][str(i)]['rank0'],'rank0 historical CHAIR')
                if rank==rows[arm][i]['selected_rank']:equal(c,parity['arms'][arm][str(i)]['selected'],'selected historical CHAIR')
            equal(data['overall_metrics'],dict(CHAIRs=counts[0]/5000,CHAIRi=counts[1]/counts[2],Recall=counts[3]/counts[4]),'CHAIR aggregate')
    def split_scores(name):
        data={}
        for arm in rows:
            data[arm]={int(i):v for i,v in read(f'{name}_{arm}_CIDEr.json').items()}
            check(set(data[arm])==set(splits[name]),'CIDEr cohort')
            for v in data[arm].values():check(len(v)==5 and all(math.isfinite(x) for x in v),'finite five CIDEr')
        return data
    def aggregate(data,name,w):return {a:metrics(rows[a],data[a],chairs[a],splits[name],w) for a in rows}
    tune=split_scores('TUNE');saved=read('tune_grid.json')
    check([tuple(x['weights']) for x in saved]==grid(),'66-rule grid/order')
    recalculated=[dict(weights=list(w),arms=aggregate(tune,'TUNE',w)) for w in grid()]
    equal(saved,recalculated,'full TUNE grid replay')
    selected=fit(recalculated);lock=read('selection_lock.json')
    equal(lock['weights'],selected['weights'],'selected weights');equal(lock['tune'],selected['arms'],'selection TUNE')
    check(lock['bundle_sha256']==parity['bundle_sha256'] and lock['test_used'] is False,'lock input/scope')
    result=read('result.json')
    check(hashlib.sha256(files['selection_lock.json'].encode()).hexdigest()==result['selection_lock_sha256'],'lock SHA')
    data=split_scores('CHECK');base=aggregate(data,'CHECK',ORIGINAL);proposed=aggregate(data,'CHECK',lock['weights'])
    equal(result['CHECK_original'],base,'CHECK baseline');equal(result['CHECK_proposed'],proposed,'CHECK proposal')
    equal(result['check_accept'],accept(proposed,base),'acceptance decision')
    equal(result['retained_weights'],list(ORIGINAL),'retained original')
    check(result['status']=='VALID_WEIGHT_CALIBRATION_COMPLETE' and not any(result[k] for k in ('test_used','training','decoding')),'completion/scope')
    tune_base=next(x['arms'] for x in saved if tuple(x['weights'])==ORIGINAL)
    eligible=[x for x in saved if guards(x['arms'],tune_base)]
    best=max(saved,key=lambda x:mean_cider(x['arms']))
    output=dict(status='VALID_WEIGHT_CALIBRATION_REVIEW_COMPLETE',source_version=190,source_notebook='ducanh2403/2-vi-cai-kia-qua-nhieu',
        archive_sha256=cached['archive_sha256'],checks=checks,grid_rules=len(saved),eligible_rules=len(eligible),
        selected_weights=dict(decoder=1-lock['weights'][0]-lock['weights'][1],CLIP=lock['weights'][0],object=lock['weights'][1]),
        unchanged=True,TUNE_original=tune_base,CHECK=base,unconstrained_TUNE_max_mean=best,
        decision='Retain original weights; no new improvement under predeclared joint guards.',
        limitation='Reused VALID internal split, one model pair. Does not prove global optimality or impossibility of better reranking. No official scorer/model executed locally.')
    (OUT/'CALIBRATION_RESULT.json').write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('AUDIT PASS',checks,'checks;',len(eligible),'eligible rule; retain original.')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cache',type=Path);audit(p.parse_args().cache)
