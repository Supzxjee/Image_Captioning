"""Compile and mock notebook orchestration; no notebook/API/model execution."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]

class Checks(unittest.TestCase):
 def notebook(self,save=False):return json.loads((HERE/('SAVE_CONCEPT_C_C0_V111_V112_DATASET.ipynb' if save else 'CONCEPT_MASS_EXPANDED_EVAL_4000_2GPU.ipynb')).read_text(encoding='utf-8'))
 def test_small_unexecuted_compile_and_asset_pins(self):
  package=json.loads((HERE/'EXPANDED_NOTEBOOK_PACKAGE.json').read_text(encoding='utf-8'))
  for name,h in package['source_assets'].items():self.assertEqual(hashlib.sha256(subprocess.check_output(['git','show',package['source_commit']+':'+name],cwd=ROOT)).hexdigest(),h)
  for entry in package['notebooks']:
   raw=(HERE/entry['file']).read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),entry['sha256']);self.assertLess(len(raw),1000000)
   for c in json.loads(raw)['cells']:
    if c['cell_type']=='code':compile(c['source'],'cell','exec');self.assertIsNone(c['execution_count']);self.assertEqual(c['outputs'],[])
  self.assertNotIn('torch',sys.modules)
 def test_export_routes_only_exact_versions_and_patterns(self):
  nb=self.notebook(True);source='\n'.join(c['source'] for c in nb['cells'] if c['cell_type']=='code')
  self.assertIn('version_label=v111',source);self.assertIn('version_label=v112',source)
  self.assertNotIn('version_label=v58',source);self.assertIn('licenses=[dict(name=',source)
  expression=next(n.value for n in ast.walk(ast.parse(nb['cells'][4]['source'])) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='pattern' for t in n.targets))
  for name in ['C','C0']:
   pattern=eval(compile(ast.Expression(expression),'pattern','eval'),{'name':name})
   self.assertTrue(re.search(pattern,'x/concept_mass_'+name+'_v2/checkpoint.bin'))
   self.assertFalse(re.search(pattern,'x/concept_mass_'+name+'_v2/checkpointXbin'))
  self.assertIn(".replace(token,'[REDACTED]')",source)
 def test_eval_inputs_and_caps(self):
  nb=self.notebook();source='\n'.join(c['source'] for c in nb['cells'] if c['cell_type']=='code')
  self.assertNotIn('DETECTIONS',source);self.assertIn("WEIGHTS/'C.bin'",source);self.assertIn("WEIGHTS/'C0.bin'",source)
  self.assertIn('PARENT_DEADLINE=time.monotonic()+8400',source)
  self.assertIn("HERE/'expanded_data.py'",source);self.assertIn('POLICY=policy_module.EVAL_POLICY',source)
  self.assertIn('reported_remaining_gpu_hours=REPORTED_REMAINING_GPU_HOURS',source)
 def flow(self,fail):
  source=self.notebook()['cells'][4]['source'];parent=HERE/'review';parent.mkdir(exist_ok=True)
  with tempfile.TemporaryDirectory(dir=parent) as temp:
   root=Path(temp);assert root.resolve().is_relative_to(parent.resolve());out=root/'output';out.mkdir()
   cfg=root/'config.json';cfg.write_text('{}');log=root/'log';log.write_text('mock');review=root/'review.zip';calls=[]
   def run(args,**kw):
    args=list(map(str,args));calls.append(args)
    if fail and 'worker' in args:raise RuntimeError('mock worker failed')
    if 'evaluate' in args:(out/'summary.json').write_text(json.dumps(dict(status='EXPANDED_EVALUATION_COMPLETE',count=4000)))
   env=dict(run=run,time=time,PARENT_DEADLINE=time.monotonic()+8400,sys=sys,HERE=root/'source',REPO=root,CONFIG=cfg,OUTPUT=out,LOG=log,REVIEW=review,zipfile=zipfile,json=json,SOURCE_COMMIT='mock',SOURCE_ASSETS={})
   with contextlib.redirect_stdout(io.StringIO()):
    if fail:
     with self.assertRaisesRegex(RuntimeError,'mock worker failed'):exec(source,env)
    else:exec(source,env)
   with zipfile.ZipFile(review) as z:status=json.loads(z.read('execution_status.json'))
   self.assertEqual(status['success'],not fail);self.assertFalse(status['training']);self.assertFalse(status['assistant_launched_job'])
   if fail:self.assertFalse(any('evaluate' in a for a in calls))
 def test_worker_failure_archived_and_no_subset_scoring(self):self.flow(True)
 def test_completed_evaluation_required(self):self.flow(False)

if __name__=='__main__':unittest.main()
