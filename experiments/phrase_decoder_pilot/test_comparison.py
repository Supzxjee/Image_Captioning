"""Synthetic artifact audit tests; do not run neural models or create real result claims."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from make_notebooks import HERE,BASE,sources
from phrase_data import identity
from run_phrase import metrics,BASE_SHA
from compare_reviews import read_review,compare


def fixture(arm):
    m=json.loads((HERE/'locked_manifest.json').read_text(encoding='utf-8'));raw=json.dumps(m).encode()
    cfg=dict(arm=arm,manifest_sha256=hashlib.sha256(raw).hexdigest(),checkpoint_sha256=BASE_SHA,
        dataset_sha256='synthetic',vocabulary_sha256='synthetic',stage_tensor_sha256='synthetic',
        optimizer={'name':'synthetic'},base_sha256='synthetic',project_commit=BASE,
        source_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in sources().items()})
    cfg['protocol_sha256']=identity(cfg);phases={}
    for phase in ('before','after'):
        rows=[]
        for g in m['groups']:
            good=phase=='after' and arm=='CROSS';margin=1. if good else 0.
            c=dict(scores=[[0.,0.],[0.,0.]],positive_margins=[margin,margin],agreement=1. if good else .5,
                both_directions_correct=good,captions=['synthetic fixture']*2,truncated=[False]*2,positive_nll=1.)
            for i,j in enumerate(g['positive_indices']):c['scores'][i][j]=margin
            rows.append(dict(group_id=g['group_id'],conditions={'on':copy.deepcopy(c),'off':copy.deepcopy(c)}))
        phases[phase]=sorted(rows,key=lambda r:r['group_id'])
    summary=dict(arm=arm,test_used=False,semantic_human_evaluation_available=False,
        metrics={p:{s:metrics(rows,keys) for s,keys in m['partitions'].items()} for p,rows in phases.items()})
    files={'protocol.json':cfg,'locked_manifest.json':m,'summary.json':summary}
    for rank in (0,1):
        files[f'complete_rank{rank}.json']=dict(rank=rank,updates=300,original_parameters_unchanged=True,
            protocol_sha256=cfg['protocol_sha256'],initial_sha256='synthetic initial',final_sha256=arm+'synthetic final',frozen_sha256='synthetic frozen',runtime={'fixture':True})
        files[f'ce_warmup_rank{rank}.json']=dict(update=60,model_sha256='synthetic warmup',optimizer_sha256='synthetic optimizer')
        for phase,rows in phases.items():files[f'{phase}_rank{rank}.json']=rows[rank::2]
    for phase,rows in phases.items():files[f'{phase}_records.json']=rows
    return files


def archive(path,files):
    with zipfile.ZipFile(path,'w') as z:
        for n,data in files.items():z.writestr(n,json.dumps(data))
        for n,code in sources().items():z.writestr('source/'+n,code)


class Checks(unittest.TestCase):
    def test_matched_pair_and_recomputed_delta(self):
        with tempfile.TemporaryDirectory(prefix='phrase_data_check_compare_',dir=HERE) as folder:
            a=Path(folder)/'ce.zip';b=Path(folder)/'cross.zip';archive(a,fixture('CE'));archive(b,fixture('CROSS'))
            result=compare(read_review(a),read_review(b))
            self.assertEqual(result['holdout_groups'],7);self.assertEqual(result['holdout_CROSS_minus_CE_agreement'],.5)
            self.assertFalse(result['semantic_human_verdicts_available'])

    def test_confounded_warmup_rejected(self):
        with tempfile.TemporaryDirectory(prefix='phrase_data_check_compare_',dir=HERE) as folder:
            a=Path(folder)/'ce.zip';b=Path(folder)/'cross.zip';left=fixture('CE');right=fixture('CROSS')
            for rank in (0,1):right[f'ce_warmup_rank{rank}.json']['model_sha256']='different'
            archive(a,left);archive(b,right)
            with self.assertRaises(AssertionError):compare(read_review(a),read_review(b))

    def test_incorrect_scoring_summary_rejected(self):
        with tempfile.TemporaryDirectory(prefix='phrase_data_check_compare_',dir=HERE) as folder:
            path=Path(folder)/'bad.zip';files=fixture('CE');files['summary.json']['metrics']['after']['fit']['on']['agreement']=.123
            archive(path,files)
            with self.assertRaises(AssertionError):read_review(path)


if __name__=='__main__':unittest.main(verbosity=2)
