"""Audit cached user-run SPICE values; no official scorer/model execution."""
import csv
import hashlib
import json
import subprocess
from pathlib import Path
import numpy as np
from score_spice_subsets import summarize, clean, CATEGORIES
from compare_spice_subsets import paired

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
CACHE=ROOT.parent/'.review_cache/spice_subsets_v2'


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    manifest=read(HERE/'PROMPT_TEST_EXTRA_INPUTS.json')
    historical=read(HERE/'PROMPT_TEST_EXTRA_RESULT.json')
    checks=[]; arms={}; rows={}; refs=[]
    def check(name,ok):
        if not ok:raise ValueError(name)
        checks.append(name)
    for arm,version in [('CONTROL',188),('PROMPT',64)]:
        cached=read(CACHE/(arm+'.json'));files=cached['files'];rd=lambda n:json.loads(files[n])
        result=rd('result.json');receipt=rd('input_receipt.json');source=rd('source_receipt.json')
        check(arm+'/complete',result['status']=='SPICE_SUBSETS_COMPLETE' and result['arm']==arm and result['images']==5000)
        check(arm+'/runstatus',rd('run_status.json')==dict(complete=True,arm=arm,training=False,decoding=False,reranking=False))
        check(arm+'/smoke',rd('subset_smoke.json')['passed'] and rd('smoke.json')['passed'])
        check(arm+'/no-training',not any(result[k] for k in ('training','decoding','reranking','test_tuning')))
        check(arm+'/receipt',result['inputs']==receipt and receipt['payload_sha256']==manifest['arms'][arm]['payload_sha256'])
        check(arm+'/caption-source',result['source_version']==manifest['arms'][arm]['version'])
        for name,digest in receipt['payload_sha256'].items():
            matches=[value for key,value in files.items() if key.startswith('saved_inputs/') and Path(key).name==name]
            check(arm+'/source/'+name,len(matches)==1 and hashlib.sha256(matches[0].encode()).hexdigest()==digest)
        for name,digest in receipt['tokenized_sha256'].items():
            check(arm+'/tokenized/'+name,hashlib.sha256(files[name].encode()).hexdigest()==digest)
        for name,digest in source['sha256'].items():
            raw=subprocess.check_output(['git','show',source['commit']+':experiments/csa_paper/'+name],cwd=ROOT)
            check(arm+'/git/'+name,hashlib.sha256(raw).hexdigest()==digest)
        reference=rd('references_ptb.json');refs.append(reference)
        check(arm+'/ids',len(reference)==5000 and len({int(i) for i in reference})==5000)
        for policy in manifest['policies']:
            stored=rd(policy+'_spice_per_image.json');raw=rd(policy+'_spice_raw.json')
            check(arm+'/'+policy+'/coverage',len(stored)==5000 and len({r['coco_id'] for r in stored})==5000 and {r['coco_id'] for r in stored}=={int(i) for i in reference})
            check(arm+'/'+policy+'/raw',raw['arm']==arm and raw['policy']==policy and raw['tokenized_sha256']==receipt['tokenized_sha256'] and len(raw['scores'])==5000)
            expected=[dict(coco_id=int(i),scores=clean(s)) for i,s in zip(sorted(reference),raw['scores'])]
            check(arm+'/'+policy+'/raw-to-rows',expected==stored)
            computed=summarize(stored)
            for cat in CATEGORIES:
                a,b=computed[cat],result['subsets'][policy][cat]
                check(arm+'/'+policy+'/'+cat,a['finite_images']==b['finite_images'] and a['undefined_images']==b['undefined_images'] and abs(a['f_x100']-b['f_x100'])<1e-10)
            check(arm+'/'+policy+'/all-replay',abs(computed['All']['f_x100']/100-historical['arms'][arm]['metrics'][policy]['SPICE'])<1e-10 and abs(raw['point']-computed['All']['f_x100']/100)<1e-10)
            rows[arm,policy]=stored
        arms[arm]=dict(archive_sha256=cached['archive_sha256'],scoring_version=version,
            scoring_notebook='ducanh2403/2-vi-cai-kia-qua-nhieu' if arm=='CONTROL' else 'ducanh3/notebook7bc83c4c76',
            caption_source=result['source_version'],subsets=result['subsets'])
    check('shared-reference',refs[0]==refs[1])
    comparisons={}
    for policy in manifest['policies']:
        comparisons[policy]=paired(rows['CONTROL',policy],rows['PROMPT',policy])
        maps={a:{r['coco_id']:r for r in rows[a,policy]} for a in arms}
        for cat in CATEGORIES:
            ids=[i for i in sorted(maps['CONTROL']) if all(maps[a][i]['scores'][cat]['f'] is not None for a in arms)]
            check(policy+'/'+cat+'/same-valid-ids',all({i for i,r in maps[a].items() if r['scores'][cat]['f'] is not None}==set(ids) for a in arms))
            delta=np.array([maps['PROMPT'][i]['scores'][cat]['f']-maps['CONTROL'][i]['scores'][cat]['f'] for i in ids])
            rng=np.random.default_rng(42)
            draws=np.array([100*delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(2000)])
            comparisons[policy][cat]['CI95_points']=np.quantile(draws,[.025,.975]).tolist()
    result=dict(status='SPICE_SUBSETS_REVIEW_COMPLETE',checks=len(checks),arms=arms,comparisons=comparisons,
        limits=['Conditional paired bootstrap 2000 draws seed42; one trained seed, not training variability; multiple subcategory comparisons unadjusted.',
                'Prior TEST exposure; C10/P9 start and adapter parameter count differ.',
                'SPICE reference scene-graph overlap is not region localization or visual relation verification.',
                'No model/Java/official scorer run by author. Source SaveVersions188/64 user-confirmed; not fetched again.'])
    (HERE/'SPICE_SUBSETS_RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (HERE/'SPICE_SUBSETS_TABLE.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['arm','policy','All','Objects','Relations','Attributes','N_All','N_Objects','N_Relations','N_Attributes','scoring_version'])
        for policy in manifest['policies']:
            for arm in arms:
                values=arms[arm]['subsets'][policy]
                w.writerow([arm,policy,*[values[c]['f_x100'] for c in CATEGORIES],*[values[c]['finite_images'] for c in CATEGORIES],arms[arm]['scoring_version']])
    print('PASS',len(checks));print(json.dumps(comparisons,indent=2))


if __name__=='__main__':main()
