"""Replay saved candidate scores/metadata only; no official metric or model."""
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
from collections import Counter
from score_valid_candidate_oracle import oracle_summary

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'valid_rerank_diagnostic'


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    cached=read(ROOT.parent/'.review_cache/valid_candidate_oracle_v1/archive_text.json')
    files=cached['files'];result=json.loads(files['result.json']);status=json.loads(files['run_status.json'])
    src=json.loads(files['source_receipt.json']);progress=json.loads(files['progress.json'])
    payload=(OUT/'VALID_CANDIDATES.json.gz').read_bytes();bundle=json.loads(gzip.decompress(payload))
    checks=[];arms={};cases={}
    def check(name,ok):
        if not ok:raise ValueError(name)
        checks.append(name)
    check('complete',result['status']=='VALID_CANDIDATE_ORACLE_COMPLETE' and status==dict(complete=True,training=False,decoding=False,test_used=False,selector_changed=False))
    check('no-model-change',not any(result[k] for k in ('training','decoding','test_used','selector_changed')))
    check('bundle-sha',hashlib.sha256(payload).hexdigest()==result['bundle_sha256']==read(OUT/'RESULT.json')['bundle_sha256'])
    check('source-bundle',src['sha256']['experiments/csa_paper/valid_rerank_diagnostic/VALID_CANDIDATES.json.gz']==result['bundle_sha256'])
    for path,digest in src['sha256'].items():
        raw=subprocess.check_output(['git','show',src['commit']+':'+path],cwd=ROOT)
        check('source/'+path,hashlib.sha256(raw).hexdigest()==digest)
    check('arms',set(result['arms'])==set(bundle['arms'])=={'CONTROL','PROMPT'} and set(progress['complete_arms'])=={'CONTROL','PROMPT'})
    reference_ids=None
    for arm in ('CONTROL','PROMPT'):
        records=bundle['arms'][arm]['records'];byid={r['coco_id']:r for r in records}
        refs=bundle['arms'][arm]['references'];ids={r['coco_id'] for r in refs}
        if reference_ids is None:reference_ids=ids
        else:check('common-refs',reference_ids==ids and refs==bundle['arms']['CONTROL']['references'])
        stored=json.loads(files[arm+'_per_candidate_CIDEr.json']);scored={r['coco_id']:r for r in stored}
        check(arm+'/coverage',len(scored)==len(stored)==len(records)==len(byid)==len(ids)==5000 and set(scored)==set(byid)==ids)
        for i,r in byid.items():
            s=scored[i];v=s['CIDEr'];k=r['selected_rank']
            check(arm+f'/{i}/shape',len(v)==5 and all(isinstance(x,(int,float)) and math.isfinite(x) and x>=0 for x in v))
            check(arm+f'/{i}/selected',s['selected_rank']==k and abs(v[0]-r['rank0_CIDEr'])<1e-7 and abs(v[k]-r['selected_CIDEr'])<1e-7)
        values=[scored[r['coco_id']]['CIDEr'] for r in records]
        reproduced=oracle_summary(records,values)
        for key,v in reproduced.items():check(arm+'/'+key,abs(v-result['arms'][arm][key])<1e-9)
        check(arm+'/source-version',result['arms'][arm]['source_version']==bundle['arms'][arm]['source_version'])
        check(arm+'/progress',progress['results'][arm]==result['arms'][arm])
        extra=dict(oracle_rank_counts=dict(Counter(max(range(5),key=lambda k:v[k]) for v in values)),
                   selected_below_best_images=sum(v[r['selected_rank']]<max(v)-1e-10 for r,v in zip(records,values)),
                   oracle_gap_quantiles_raw=sorted(max(v)-v[r['selected_rank']] for r,v in zip(records,values)))
        gaps=extra.pop('oracle_gap_quantiles_raw');extra['oracle_gap_quantiles_raw']={str(p):gaps[round((len(gaps)-1)*p)] for p in (0,.25,.5,.75,.9,1)}
        # Fixed component-only probes, not tuned weights and not deployed policies.
        for name,key in [('decoder','decoder_score'),('clip','clip_score_normalized'),('object','object_score')]:
            extra[name+'_argmax_CIDEr_x100']=100*sum(v[max(range(5),key=lambda k:r['candidates'][k][key])] for r,v in zip(records,values))/5000
        arms[arm]={**result['arms'][arm],**extra}
        examples=read(OUT/'review_candidates.json')
        cases[arm]=[dict(coco_id=r['coco_id'],group=r['group'],selected_rank=r['selected_rank'],
            selected_CIDEr=scored[r['coco_id']]['CIDEr'][r['selected_rank']],
            oracle_rank=max(range(5),key=lambda k:scored[r['coco_id']]['CIDEr'][k]),
            all_five_CIDEr=scored[r['coco_id']]['CIDEr']) for r in examples if r['arm']==arm]
    report=dict(status='VALID_CANDIDATE_ORACLE_REVIEW_COMPLETE',checks=len(checks),archive_sha256=cached['archive_sha256'],
       source_commit=src['commit'],bundle_sha256=result['bundle_sha256'],arms=arms,example_oracles=cases,
       scoring_run_version=189,scoring_run_notebook='ducanh2403/2-vi-cai-kia-qua-nhieu',
       limits=['Oracle uses VALID references; not deployable, not achievable-gain guarantee, not visual truth.',
               'Selected-is-best counts allow CIDEr ties; not caption accuracy. Capture ratio is fraction of CIDEr headroom, not accuracy.',
               'CIDEr oracle does not optimize CHAIR, relation truth or linguistic quality.',
               'Same generation/weights; no new model training. VALID reused, one seed; no official scorer/model local.',
               'Component-only argmax probes are fixed diagnostics, no weight search or TEST tuning.'])
    (OUT/'ORACLE_RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PASS',len(checks));print(json.dumps(arms,indent=2))


if __name__=='__main__':main()
