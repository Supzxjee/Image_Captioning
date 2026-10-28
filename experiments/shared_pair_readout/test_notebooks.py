"""Static and mocked notebook flow; no process launch, Torch import or network."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
import unittest
import zipfile

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]


class Checks(unittest.TestCase):
    def test_mock_bootstrap_failure_archives_log(self):
        nb=json.loads((HERE/'SHARED_PAIR_READOUT_I_2GPU.ipynb').read_bytes())
        code=[c['source'] for c in nb['cells'] if c['cell_type']=='code'][2]
        parent=ROOT/'experiments/relation_frames_inventory/review'
        with tempfile.TemporaryDirectory(prefix='shared-bootstrap-mock-',dir=parent) as temp:
            root=Path(temp);log=root/'log';log.write_text('Mock git failure');review=root/'review.zip'
            def fail(*args,**kwargs):raise subprocess.CalledProcessError(128,['mockgit'])
            env=dict(run=fail,REPO=root,LOG=log,REVIEW=review,ARM='I',SOURCE_COMMIT='mock',json=json,zipfile=zipfile,traceback=traceback)
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(subprocess.CalledProcessError):exec(code,env)
            with zipfile.ZipFile(review) as z:
                self.assertEqual(json.loads(z.read('execution_status.json'))['stage'],'bootstrap')
                self.assertIn('bootstrap_error.txt',z.namelist())

    def test_pinned_notebooks_and_single_config_difference(self):
        package=json.loads((HERE/'NOTEBOOK_PACKAGE.json').read_bytes());notebooks=[]
        for r in package['notebooks']:
            p=HERE/r['file'];self.assertLess(p.stat().st_size,1000000)
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),r['sha256'])
            nb=json.loads(p.read_bytes());code=[c['source'] for c in nb['cells'] if c['cell_type']=='code'];notebooks.append(code)
            for c in nb['cells']:
                if c['cell_type']=='code':compile(c['source'],p.name,'exec');self.assertIsNone(c['execution_count']);self.assertEqual(c['outputs'],[])
            self.assertIn(package['source_commit'],''.join(code))
        self.assertEqual(notebooks[0][1:],notebooks[1][1:])
        self.assertEqual(notebooks[0][0].replace("ARM='I'","ARM='J'"),notebooks[1][0])
        self.assertFalse(package['local_neural_execution']);self.assertFalse(package['kaggle_launched'])

    def test_mock_completed_worker_and_evaluator(self):self.flow(None)
    def test_mock_worker_failure_prevents_evaluation(self):self.flow('worker')
    def test_mock_evaluator_failure_is_not_success(self):self.flow('evaluate')

    def flow(self,fail):
        notebook=json.loads((HERE/'SHARED_PAIR_READOUT_I_2GPU.ipynb').read_bytes())
        code=[c['source'] for c in notebook['cells'] if c['cell_type']=='code'][-1]
        parent=ROOT/'experiments/relation_frames_inventory/review'
        with tempfile.TemporaryDirectory(prefix='shared-reader-mock-',dir=parent) as temp:
            root=Path(temp);out=root/'out';config=root/'config.json';config.write_text('{}')
            log=root/'log.txt';log.write_text('Mock only; no model or process.');review=root/'review.zip';phases=[]
            def run(args,**kwargs):
                phase=next((x for x in ('prepare','worker','evaluate') if x in args),'test');phases.append(phase)
                if phase==fail:raise subprocess.CalledProcessError(1,args)
                if phase=='prepare':out.mkdir();(out/'inputs.h5').write_bytes(b'mock bytes, not features')
                if phase=='worker':(out/'checkpoint.bin').write_bytes(b'mock checkpoint, not a model')
                if phase=='evaluate':(out/'summary.json').write_text(json.dumps(dict(status='ENDPOINT_500_EVALUATED_WAIT_MATCHED_I_J')))
            env=dict(run=run,sys=sys,HERE=root,REPO=root,CONFIG=config,LOG=log,REVIEW=review,
                OUTPUT=out,ARM='I',SOURCE_ASSETS={},SOURCE_COMMIT='mock',zipfile=zipfile,json=json)
            with contextlib.redirect_stdout(io.StringIO()):
                if fail:
                    with self.assertRaises(subprocess.CalledProcessError):exec(code,env)
                else:exec(code,env)
            with zipfile.ZipFile(review) as z:
                self.assertEqual(json.loads(z.read('execution_status.json'))['success'],fail is None)
                self.assertNotIn('inputs.h5',z.namelist());self.assertNotIn('checkpoint.bin',z.namelist())
                self.assertIn('notebook_config.json',z.namelist())
            if fail=='worker':self.assertNotIn('evaluate',phases)


if __name__=='__main__':unittest.main(verbosity=2)
