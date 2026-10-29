"""Generated notebook/source checks and archive control-flow mocks; no GPU."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import zipfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

class Checks(unittest.TestCase):
    def notebooks(self):return [json.loads((HERE/f'CONCEPT_MASS_{a}_2GPU.ipynb').read_text(encoding='utf8')) for a in ('C','G')]
    def test_small_unexecuted_and_compile(self):
        for nb in self.notebooks():
            for c in nb['cells']:
                if c['cell_type']=='code':
                    self.assertIsNone(c['execution_count']);self.assertEqual(c['outputs'],[]);compile(c['source'],'cell','exec')
            self.assertLess(len(json.dumps(nb).encode()),1000000)
    def test_only_arm_differs_in_code(self):
        a,b=self.notebooks()
        code=lambda nb:'\n'.join(c['source'] for c in nb['cells'] if c['cell_type']=='code')
        self.assertEqual(code(a).replace("ARM='C'","ARM='G'"),code(b))
    def test_pinned_git_assets_and_no_local_neural_import(self):
        p=json.loads((HERE/'NOTEBOOK_PACKAGE.json').read_text(encoding='utf8'))
        for name,want in p['source_assets'].items():
            self.assertEqual(hashlib.sha256(subprocess.check_output(['git','show',p['source_commit']+':'+name],cwd=ROOT)).hexdigest(),want)
        self.assertNotIn('torch',sys.modules)
    def test_dev_labels_only_imported_in_post_training_diagnostic(self):
        run=(HERE/'run_concept.py').read_text(encoding='utf8');stage=(HERE/'prepare_concept.py').read_text(encoding='utf8')
        self.assertNotIn('DEV_DIAGNOSTIC_GT',run+stage)
        self.assertGreater(run.index('from diagnostic_concept import diagnostic'),run.index('for step in range(1,501)'))
    def test_current_kernel_does_not_reuse_old_policy_module(self):
        for nb in self.notebooks():
            setup=nb['cells'][3]['source']
            self.assertIn('spec_from_file_location',setup)
            self.assertNotIn('from concept_data import POLICY',setup)
    def flow(self,fail=False):
        source=self.notebooks()[0]['cells'][4]['source'];calls=[]
        parent=HERE/'review';parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            root=Path(temp);assert root.resolve().is_relative_to(parent.resolve());out=root/'output';out.mkdir();cfg=root/'config.json';cfg.write_text('{}');log=root/'log';log.write_text('mock')
            review=root/'review.zip'
            def run(args,**kwargs):
                args=list(map(str,args));calls.append(args)
                if fail and 'worker' in args:raise RuntimeError('mock worker failure')
                if 'evaluate' in args:(out/'summary.json').write_text(json.dumps(dict(status='ENDPOINT_500_EVALUATED_WAIT_MATCHED_C_G')))
            env=dict(run=run,sys=sys,HERE=root/'source',REPO=root,CONFIG=cfg,OUTPUT=out,LOG=log,REVIEW=review,zipfile=zipfile,json=json,ARM='C',SOURCE_COMMIT='mock',SOURCE_ASSETS={})
            with contextlib.redirect_stdout(io.StringIO()):
                if fail:
                    with self.assertRaisesRegex(RuntimeError,'mock worker failure'):exec(source,env)
                else:exec(source,env)
            with zipfile.ZipFile(review) as z:status=json.loads(z.read('execution_status.json'))
            self.assertEqual(status['success'],not fail)
            if fail:self.assertFalse(any('evaluate' in a for a in calls))
            self.assertFalse(status['assistant_launched_job'])
    def test_worker_failure_archived_not_success(self):self.flow(True)
    def test_success_requires_completed_evaluation(self):self.flow(False)

if __name__=='__main__':unittest.main()
