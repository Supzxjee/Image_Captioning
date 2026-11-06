"""Audit saved CPU baseline metrics; do not run scorers or models."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda b:hashlib.sha256(b).hexdigest()
cached=read(ROOT.parent/'.review_cache/plain_csa10_TEST_extra/archive_text.json')
f=cached['files'];rd=lambda n:json.loads(f[n])
m=read(HERE/'PLAIN_CSA10_EXTRA_INPUTS.json')
r=rd('result.json');receipt=rd('input_receipt.json');checks=[]


def check(name,condition):
    if not condition:raise ValueError(name)
    checks.append(name)


check('status',r['status']=='PLAIN_CSA10_TEST_EIGHT_METRICS_COMPLETE' and r['images']==5000 and r['equal_xe_epochs']==10)
check('run',rd('run_status.json')==dict(complete=True,training=False,decoding=False,reranking=False))
check('no-model',all(r[k] is False for k in ('training','decoding','reranking','test_tuning')))
check('input',receipt==r['inputs'] and receipt['payload_sha256']==m['arms']['BASELINES']['payload_sha256'])
check('source-identities',r['historical_sources']==receipt['historical_sources']==m['sources'])
source=rd('source_receipt.json')
for n,h in source['sha256'].items():
    raw=subprocess.check_output(['git','show',source['commit']+':experiments/csa_paper/'+n],cwd=ROOT)
    check('source/'+n,sha(raw)==h)
for n,h in receipt['tokenized_sha256'].items():check('ptb/'+n,sha(f[n].encode())==h)
refs=rd('references_ptb.json')
check('references',len(refs)==5000 and sorted(map(len,refs.values()))==[5]*4990+[6]*10)
prompt_cache=read(ROOT.parent/'.review_cache/prompt_TEST_extra/CONTROL.json')['files']
check('same-refs-as-current-pipeline',refs==json.loads(prompt_cache['references_ptb.json']))
env=rd('environment.json');oldenv=json.loads(prompt_cache['environment.json'])
check('environment',env==r['environment'] and env['pycocoevalcap']=='1.2' and env['corenlp_version']=='3.6.0')
for k in ('java_archive_sha256','corenlp_archive_sha256','evaluator_files'):
    check('same-evaluator/'+k,env[k]==oldenv[k])
smoke=rd('smoke.json');check('smoke',smoke['passed'] and all(math.isfinite(v) and v>0 for v in smoke['scores'].values()))
rows=list(csv.DictReader(io.StringIO(f['metrics_x100.csv'])))
check('csv',len(rows)==2 and {v['policy'] for v in rows}==set(m['policies']))
keys={'Bleu_1','Bleu_4','METEOR','ROUGE_L','CIDEr','SPICE','CHAIRs','CHAIRi'}
for policy in m['policies']:
    hyp=rd(policy+'_ptb.json');check(policy+'/coverage',set(hyp)==set(refs) and all(len(v)==1 for v in hyp.values()))
    points={}
    for phase in ('basic','meteor','spice'):
        part=rd(policy+'_'+phase+'.json')
        check(policy+'/'+phase,part['images']==5000 and part['tokenized_sha256']==receipt['tokenized_sha256'])
        points.update(part['metrics'])
    expected=m['arms']['BASELINES']['metrics'][policy]
    points.update({k:expected[k] for k in ('CHAIRs','CHAIRi')})
    check(policy+'/points',set(points)==keys and points==r['metrics'][policy] and all(math.isfinite(v) for v in points.values()))
    for k in ('Bleu_4','ROUGE_L','CIDEr','CHAIRs','CHAIRi'):
        check(policy+'/historical/'+k,abs(points[k]-expected[k])<1e-7)
    csvrow=next(v for v in rows if v['policy']==policy)
    check(policy+'/csv-scale',all(abs(float(csvrow[k])-100*v)<1e-10 and abs(r['display_x100'][policy][k]-100*v)<1e-10 for k,v in points.items()))
out=dict(status='PLAIN_CSA10_EXTRA_REVIEW_COMPLETE',checks=len(checks),archive_sha256=cached['archive_sha256'],
    source_commit=source['commit'],snapshot_sha256=m['snapshot_sha256'],metrics=r['metrics'],display_x100=r['display_x100'],
    csa_minus_plain_x100={k:100*(r['metrics']['CSA10'][k]-r['metrics']['NoCSA10'][k]) for k in keys},
    historical_sources=m['sources'],extra_save_version=186,extra_version_provenance='User confirmed186; notebookducanh2403/2-vi-cai-kia-qua-nhieu inferred from continuing source, not independently API fetched.',
    limits=['CHAIR carried from exact-caption historical audit, not recomputed.',
            'Single seed, different encoder capacity; NoCSA has many short captions.',
            'No author local scorer/model execution; no significance claim from point estimates.'])
(HERE/'PLAIN_CSA10_EXTRA_RESULT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(len(checks),'stored checks PASS')
print(json.dumps(r['display_x100'],indent=2))
