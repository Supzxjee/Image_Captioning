"""Audit saved JSON/source bytes only; never execute archive code or load model weights."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import zipfile
from make_notebook import HERE, ROOT, BASE, sources
sys.path.insert(0,str(ROOT/'experiments/semantic_evidence_probe'))
from oracle_data import summarize
from probe_data import identity


def audit(source):
    with zipfile.ZipFile(source) as z:
        assert z.testzip() is None
        read=lambda n:json.loads(z.read(n))
        cfg,manifest,records,summary=[read(n) for n in ('config.json','cases_manifest.json','records.json','summary.json')]
        assert z.read('cases_manifest.json')==(HERE/'cases_manifest.json').read_text(encoding='utf-8').encode('utf-8')
        assert cfg['project_commit']==BASE
        for name,code in sources().items():
            raw=z.read('source/'+name)
            assert raw==code.encode('utf-8'),name
            assert hashlib.sha256(raw).hexdigest()==cfg['source_sha256'][name],name
        assert hashlib.sha256(z.read('cases_manifest.json')).hexdigest()==cfg['manifest_sha256']==summary['manifest_sha256']
        assert identity(manifest['cases'])==manifest['cases_sha256']
        assert summary['checkpoint_sha256']==manifest['checkpoint_sha256']
        ranks=[read(f'rank{i}.json') for i in (0,1)]
        assert ranks[0]['model_sha256']==ranks[1]['model_sha256']
        merged=[]
        for rank,data in enumerate(ranks):
            assert data['rank']==rank and data['no_training'] and data['parameters_unchanged']
            assert data['manifest_sha256']==cfg['manifest_sha256'] and data['checkpoint_sha256']==manifest['checkpoint_sha256']
            assert data['runtime']['gpu']['name']=='Tesla T4' and data['runtime']['deterministic_algorithms']
            assert [r['case_id'] for r in data['records']]==[r['case_id'] for r in manifest['cases'][rank::2]]
            merged+=data['records']
        assert sorted(merged,key=lambda r:r['case_id'])==records
        for row,case in zip(records,manifest['cases']):
            assert all(row[k]==v for k,v in case.items())
            assert 'images/'+row['filename'] in z.namelist()
            assert row['matcher_replay_max_abs_error']<1e-4
            assert all(math.isfinite(v) and v>0 for v in row['evidence_change'].values())
            for cond in row['conditions'].values():
                p=row['positive_index']
                assert cond['target_descriptions']==[row['candidates'][i]['text'].rstrip('.') for i in (p,1-p)]
                assert all(n>0 for n in cond['target_tokens'])
                assert all(math.isfinite(x) for x in cond['target_logp']+cond['target_mean_logp'])
                for total,mean,n in zip(cond['target_logp'],cond['target_mean_logp'],cond['target_tokens']):
                    assert abs(total/n-mean)<1e-5
                assert abs(cond['positive_minus_negative_logp']-(cond['target_logp'][0]-cond['target_logp'][1]))<1e-5
                assert abs(cond['positive_minus_negative_mean_logp']-(cond['target_mean_logp'][0]-cond['target_mean_logp'][1]))<1e-5
        assert summarize(records)==summary['metrics'] and summary['no_training'] and not summary['test_used']
        assert not summary['human_semantic_verdicts_available']
        setup=z.read('setup.log').decode()
        assert 'Ran 3 tests' in setup and 'Ran 1 test' in setup and setup.count('\nOK')==2
        delta=[r['conditions']['annotation_positive']['positive_minus_negative_logp']-r['conditions']['annotation_negative']['positive_minus_negative_logp'] for r in records]
        pairs={k:[] for k in ('same_preference_as_off','positive_negative_same_caption')}
        for r in records:
            c=r['conditions']
            if all((v['positive_minus_negative_logp']>0)==(c['off']['positive_minus_negative_logp']>0) for v in c.values()):pairs['same_preference_as_off'].append(r['case_id'])
            if c['annotation_positive']['caption']==c['annotation_negative']['caption']:pairs['positive_negative_same_caption'].append(r['case_id'])
        result=dict(status='SAVED_ARTIFACT_AUDIT_PASS',source_zip_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
            cases=20,images=20,source_files_verified=len(sources()),checkpoint_sha256=summary['checkpoint_sha256'],
            manifest_sha256=cfg['manifest_sha256'],model_sha256=ranks[0]['model_sha256'],
            no_training=True,test_used=False,kaggle_data_tests=3,kaggle_synthetic_model_tests=1,
            metrics=summary['metrics'],
            additional=dict(mean_absolute_preference_shift=statistics.mean(map(abs,delta)),
                median_absolute_preference_shift=statistics.median(map(abs,delta)),max_absolute_preference_shift=max(map(abs,delta)),
                positive_preference_shifts=sum(x>0 for x in delta),negative_preference_shifts=sum(x<0 for x in delta),
                R_token_relative_L2_min=min(r['evidence_change']['R_token_relative_L2'] for r in records),
                R_token_relative_L2_max=max(r['evidence_change']['R_token_relative_L2'] for r in records),
                max_matcher_replay_error=max(r['matcher_replay_max_abs_error'] for r in records),
                same_preference_as_off=len(pairs['same_preference_as_off'])),
            checks=['CRC, exact manifest and all embedded sources; matching checkpoint identity and full-model hashes across ranks',
                    '20 unique fixed cases, two completed T4 workers, saved unchanged-weight flags, matcher replay tolerance',
                    'All summary metrics recomputed from rank records; description ordering and likelihood arithmetic checked',
                    'Setup logs report three data tests and one synthetic model test passed'],
            decision='Do not scale F or tune its alignment-loss weight on this evidence. Design a relation-conditioned representation and caption-coupled learning objective before another pilot.',
            limits=summary['limits']+['Local audit reads saved artifacts; does not replay inference or independently inspect tensor checkpoint/runtime internals. No human labels produced.'])
        (HERE/'RESULT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
        table=[dict(coco_id=r['coco_id'],case_id=r['case_id'],descriptions=r['conditions']['annotation_positive']['target_descriptions'],
            captions={k:v['caption'] for k,v in r['conditions'].items()},evidence_change=r['evidence_change'],preference_shift=d) for r,d in zip(records,delta)]
        (HERE/'CASE_CAPTIONS.json').write_text(json.dumps(table,indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',required=True)
    audit(parser.parse_args().source)
