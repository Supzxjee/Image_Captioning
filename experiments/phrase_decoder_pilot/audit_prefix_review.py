"""Recompute saved diagnostic statistics; never load adapters or execute archived code."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
from phrase_data import identity
from compare_reviews import read_review


def audit(path,ce,cross):
    originals={'CE':read_review(ce),'CROSS':read_review(cross)}
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        cfg=json.loads(z.read('protocol.json'));m=json.loads(z.read('locked_manifest.json'))
        assert identity({k:v for k,v in cfg.items() if k!='protocol_sha256'})==cfg['protocol_sha256']
        assert hashlib.sha256(z.read('locked_manifest.json')).hexdigest()==cfg['manifest_sha256']
        for name,sha in cfg['source_sha256'].items():
            raw=z.read('source/'+name);assert hashlib.sha256(raw).hexdigest()==sha
            assert raw.decode()==(Path(__file__).parent/name).read_text(encoding='utf-8'),name
        assert identity(json.loads(z.read('stage_receipt.json')))==cfg['stage_tensor_sha256']
        by={g['group_id']:g for g in m['groups'] if g['same_image']};assert len(by)==4
        findings={}
        for arm,original in originals.items():
            assert m==original['manifest'] and cfg['stage_tensor_sha256']==original['cfg']['stage_tensor_sha256']
            d=json.loads(z.read(f'prefix_{arm}.json'))
            assert d['status']=='EVALUATION_ONLY_REPLAY_PASS' and d['weights_unchanged']
            assert d['model_sha256']==original['summary']['final_sha256']
            assert {g['group_id'] for g in d['groups']}==set(by)
            old={g['group_id']:g for g in original['phases']['after']};rows=[]
            for row in d['groups']:
                g=by[row['group_id']];assert row['positive_indices']==g['positive_indices'] and row['descriptions']==g['descriptions']
                for name,c in row['conditions'].items():
                    details=c['teacher_forced_token_logp_event_major'];assert len(details)==4
                    scores=np.array(c['scores']);assert scores.shape==(2,2)
                    recomputed=[np.mean([v['log_probability'] for v in seq if v['token']!='<eos>']) for seq in details]
                    np.testing.assert_allclose(scores,np.array(recomputed).reshape(2,2),atol=2e-6,rtol=2e-6)
                    for i,prefix in enumerate(c['after_a_person']):
                        assert [v['token'] for v in details[i*2]][:2]==['a','person']
                        np.testing.assert_allclose([v['log_probability'] for v in prefix['branch_tokens']],
                            [details[i*2+j][2]['log_probability'] for j in range(2)],atol=1e-5,rtol=1e-5)
                    if name in ('natural','evidence_off'):
                        prior=old[row['group_id']]['conditions']['on' if name=='natural' else 'off']
                        np.testing.assert_allclose(scores,prior['scores'],rtol=1e-5,atol=1e-5)
                        assert c['from_bos']['captions']==prior['captions']
                c=row['conditions']['natural'];branch=c['after_a_person']
                rows.append(dict(group_id=g['group_id'],image_id=row['image_id'],partition=row['partition'],
                    correct_template_preferences=sum(s[i]>s[1-i] for s,i in zip(c['scores'],g['positive_indices'])),
                    correct_first_branch_preferences=sum(v['branch_tokens'][i]['log_probability']>v['branch_tokens'][1-i]['log_probability'] for v,i in zip(branch,g['positive_indices'])),
                    prefix_changes_caption=c['from_bos']['captions']!=c['from_a_person']['captions'],
                    captions=c['from_bos']['captions'],branch_tokens=branch))
            findings[arm]=rows
        return dict(status='SAVED_ARTIFACT_AUDIT_PASS',archive_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                    findings=findings,limits=['Saved records verified, no independent inference replay',
                    'Natural outputs already start with a person: prefix equality cannot establish an exposure-bias mechanism',
                    'Two previously inspected holdout groups; no full-caption or semantic accuracy claim'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--zip',required=True);p.add_argument('--ce',required=True);p.add_argument('--cross',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.zip,a.ce,a.cross);a.output.write_text(json.dumps(r,indent=2,ensure_ascii=False),encoding='utf-8');print(r['status'])
