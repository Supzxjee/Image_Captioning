"""Synthetic transport checks; no Kaggle API/model execution."""
import ast,contextlib,copy,hashlib,io,json,os,subprocess,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from package_dataset import package,sha
from make_dataset_notebook import build,HERE,VERSION_PREFLIGHT

class Checks(unittest.TestCase):
 def fixture(self,root):
  inv=json.loads((HERE/'upload_inventory.json').read_text());src=root/'source';src.mkdir();(src/'panel').mkdir()
  for n in inv['files']:
   (src/n).write_text('synthetic '+n)
  receipt=dict(status='PREPARATION_COMPLETE',images=2000,full_pilot_run=False,panel_sha256={'one.png':hashlib.sha256(b'pixels').hexdigest()})
  (src/'receipt.json').write_text(json.dumps(receipt));(src/'panel/one.png').write_bytes(b'pixels')
  inv['files']={n:dict(sha256=sha(src/n)) for n in inv['files']};return src,inv
 def test_copy_resume_and_changed_destination_preserved(self):
  with tempfile.TemporaryDirectory(dir=HERE) as tmp,contextlib.redirect_stdout(io.StringIO()):
   root=Path(tmp);src,inv=self.fixture(root);dest=root/'dest';package(src,dest,inv);package(src,dest,inv)
   self.assertEqual(len(list(dest.glob('*.h5'))),3);self.assertEqual((dest/'panel__one.png').read_bytes(),b'pixels')
   (dest/'aux_rank1.h5').write_bytes(b'preserve')
   with self.assertRaises(ValueError):package(src,dest,inv)
   self.assertEqual((dest/'aux_rank1.h5').read_bytes(),b'preserve')
 def test_bad_hdf5_or_panel_stops_before_copy(self):
  for filename in ('aux_rank0.h5','panel/one.png'):
   with tempfile.TemporaryDirectory(dir=HERE) as tmp,contextlib.redirect_stdout(io.StringIO()):
    root=Path(tmp);src,inv=self.fixture(root);(src/filename).write_bytes(b'corrupt')
    with self.assertRaises(ValueError):package(src,root/'dest',inv)
    self.assertFalse((root/'dest').exists())
 def test_wrong_receipt_and_unsafe_names_rejected(self):
  with tempfile.TemporaryDirectory(dir=HERE) as tmp,contextlib.redirect_stdout(io.StringIO()):
   root=Path(tmp);src,inv=self.fixture(root);inv['files']['../unsafe']={}
   with self.assertRaises(ValueError):package(src,root/'dest',inv)
   inv['files'].pop('../unsafe');r=json.loads((src/'receipt.json').read_text());r['full_pilot_run']=True;(src/'receipt.json').write_text(json.dumps(r));inv['files']['receipt.json']['sha256']=sha(src/'receipt.json')
   with self.assertRaises(ValueError):package(src,root/'dest',inv)
 def test_notebook_pinned98_private_empty_and_hash_bound(self):
  with contextlib.redirect_stdout(io.StringIO()):p=build()
  b=json.loads(p.read_text(encoding='utf-8'));allcode=''
  for c in b['cells']:
   if c['cell_type']=='code':
    code=''.join(c['source']);ast.parse(code);allcode+=code;self.assertFalse(c['outputs']);self.assertIsNone(c['execution_count'])
  self.assertIn('/98',allcode);self.assertIn('v98',VERSION_PREFLIGHT);self.assertIn('public=False',allcode);self.assertIn("licenses=[dict(name='other')]",allcode)
  self.assertIn("INVENTORY['files']['receipt.json']['sha256']",allcode);self.assertLess(p.stat().st_size,1000000)
  self.assertEqual(b['metadata']['kaggle']['accelerator'],'none')
 def test_secret_redaction_and_no_credential_on_pip_child(self):
  b=json.loads((HERE/'SAVE_STALE_PROMPT_2000_DATASET.ipynb').read_text(encoding='utf-8'))
  code=next(''.join(c['source']) for c in b['cells'] if c['cell_type']=='code' and 'def run_cli(' in ''.join(c['source']))
  fn=next(n for n in ast.parse(code).body if isinstance(n,ast.FunctionDef));ns=dict(os=os,subprocess=subprocess,token='synthetic-secret')
  exec(compile(ast.Module(body=[fn],type_ignores=[]),'<test>','exec'),ns)
  with patch.object(subprocess,'Popen') as mock,contextlib.redirect_stdout(io.StringIO()) as output:
   process=mock.return_value.__enter__.return_value;process.stdout=['synthetic-secret'];process.wait.return_value=0
   ns['run_cli'](['python','upload']);self.assertNotIn('synthetic-secret',output.getvalue());self.assertEqual(mock.call_args.kwargs['env']['KAGGLE_API_TOKEN'],'synthetic-secret')
   ns['run_cli'](['python','pip'],authenticated=False);self.assertNotIn('KAGGLE_API_TOKEN',mock.call_args.kwargs['env'])

if __name__=='__main__':unittest.main(verbosity=2)
