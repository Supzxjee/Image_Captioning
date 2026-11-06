"""Read cached uploaded text once; verify metadata, never execute scorers."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CACHE = ROOT.parent / '.review_cache/prompt_TEST_extra'
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
digest = lambda raw: hashlib.sha256(raw).hexdigest()
manifest = read(HERE/'PROMPT_TEST_EXTRA_INPUTS.json')
checks = []
results = {}
environments = {}


def check(name, condition):
    if not condition:
        raise ValueError(name)
    checks.append(name)


for arm in ('CONTROL', 'PROMPT'):
    cached = read(CACHE/(arm+'.json'))
    f = cached['files']
    rd = lambda name: json.loads(f[name])
    result, receipt = rd('result.json'), rd('input_receipt.json')
    spec = manifest['arms'][arm]
    check(arm+'/status', result['status']=='PROMPT_TEST_EIGHT_METRICS_COMPLETE' and result['arm']==arm
          and result['images']==5000 and result['source_version']==spec['version'])
    check(arm+'/no-model', all(result[k] is False for k in ('training','decoding','reranking','test_tuning')))
    check(arm+'/run', rd('run_status.json')==dict(complete=True,arm=arm,training=False,decoding=False,reranking=False))
    check(arm+'/receipt', result['inputs']==receipt and receipt['payload_sha256']==spec['payload_sha256'])
    for name, expected in spec['payload_sha256'].items():
        matches=[n for n in f if n.endswith('/'+name) and digest(f[n].encode())==expected]
        check(arm+'/saved/'+name, len(matches)==1)
    for name, expected in receipt['tokenized_sha256'].items():
        check(arm+'/tokenized/'+name, digest(f[name].encode())==expected)
    refs=rd('references_ptb.json')
    check(arm+'/5000references',len(refs)==5000 and sorted(map(len,refs.values()))==[5]*4990+[6]*10)
    source=rd('source_receipt.json')
    for name, expected in source['sha256'].items():
        actual=subprocess.check_output(['git','show',source['commit']+':experiments/csa_paper/'+name],cwd=ROOT)
        check(arm+'/source/'+name,digest(actual)==expected)
    smoke=rd('smoke.json')
    check(arm+'/actual-smoke',smoke['passed'] and all(math.isfinite(v) and v>0 for v in smoke['scores'].values()))
    csv_rows=list(csv.DictReader(io.StringIO(f['metrics_x100.csv'])))
    check(arm+'/csv',len(csv_rows)==2 and {r['policy'] for r in csv_rows}==set(manifest['policies']))
    for policy in manifest['policies']:
        hyp=rd(policy+'_ptb.json')
        check(arm+'/'+policy+'/coverage',set(hyp)==set(refs) and all(len(v)==1 for v in hyp.values()))
        combined={}
        for mode in ('basic','meteor','spice'):
            phase=rd(policy+'_'+mode+'.json')
            check(arm+'/'+policy+'/'+mode,phase['images']==5000 and phase['tokenized_sha256']==receipt['tokenized_sha256'])
            combined.update(phase['metrics'])
        combined.update({k:spec['metrics'][policy][k] for k in ('CHAIRs','CHAIRi')})
        check(arm+'/'+policy+'/eight',set(combined)==set(('Bleu_1','Bleu_4','METEOR','ROUGE_L','CIDEr','SPICE','CHAIRs','CHAIRi'))
              and combined==result['metrics'][policy] and all(math.isfinite(v) for v in combined.values()))
        for k in ('Bleu_4','ROUGE_L','CIDEr','CHAIRs','CHAIRi'):
            check(arm+'/'+policy+'/replay/'+k,abs(combined[k]-spec['metrics'][policy][k])<1e-7)
        row=next(r for r in csv_rows if r['policy']==policy)
        check(arm+'/'+policy+'/csv-scale',all(abs(float(row[k])-100*v)<1e-10 and abs(result['display_x100'][policy][k]-100*v)<1e-10 for k,v in combined.items()))
    env=rd('environment.json')
    check(arm+'/environment',result['environment']==env and env['pycocoevalcap']=='1.2' and env['corenlp_version']=='3.6.0'
          and env['java_archive_sha256']=='6f7fb5fd640a0fd00837344b0920cbc4b9b9284b50e66f33789e3b250446a16e')
    environments[arm]=env
    results[arm]=dict(archive_sha256=cached['archive_sha256'],source_version=spec['version'],
        checkpoint_sha256=spec['checkpoint_sha256'],metrics=result['metrics'],display_x100=result['display_x100'],
        source_commit=source['commit'],extra_save_version=None)

check('same-evaluator-and-models',environments['CONTROL']['evaluator_files']==environments['PROMPT']['evaluator_files']
      and environments['CONTROL']['corenlp_archive_sha256']==environments['PROMPT']['corenlp_archive_sha256'])
delta={p:{k:100*(results['PROMPT']['metrics'][p][k]-results['CONTROL']['metrics'][p][k])
          for k in results['CONTROL']['metrics'][p]} for p in manifest['policies']}
out=dict(status='PROMPT_TEST_EXTRA_REVIEW_COMPLETE',checks=len(checks),arms=results,prompt_minus_control_x100=delta,
    chair_provenance='Carried from audited scores using identical raw caption/reference/result SHA; not recomputed.',
    limits=['One seed; unequal selected initial XE epochs C10/P9 and added adapter capacity.',
            'Prior TEST exposure; no new tuning or causal isolation claim.',
            'METEOR/SPICE differences are point estimates; no new significance test.',
            'Author audited stored metadata, did not rerun Java, scorers or models.'])
(HERE/'PROMPT_TEST_EXTRA_RESULT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(len(checks),'stored checks PASS')
print(json.dumps({a:r['display_x100'] for a,r in results.items()},indent=2))
